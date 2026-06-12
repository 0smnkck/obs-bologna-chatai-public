"""
model — RAG Dil Modeli Yöneticisi (BFloat16 Native / GH200)
============================================================
Bu modül, Gemma-4 31B büyük dil modelinin (LLM) GH200 GPU üzerinde
BFloat16 hassasiyetiyle yüklenmesini, VRAM bütçe yönetimini,
MetricsCollector entegrasyonunu ve RAG tabanlı streaming inference'ı yönetir.

Mimari Kararlar ve Optimizasyonlar (Akademik Kurulum):
    - **BFloat16 (BF16) Hassasiyeti**: Model yüklemesinde donanımsal ``torch.bfloat16`` 
      kullanılır. GH200/Blackwell mimarisindeki Tensor Çekirdekleri ile tam uyumludur. 
      FP16'ya kıyasla daha geniş sayısal aralık sunarak NaN (Not a Number) patlamalarını 
      önler, üstelik bellek ayakizini FP32'ye kıyasla %50 oranında düşürür.
    - **JIT Derleme (torch.compile)**: PyTorch Inductor backend'i ile model 
      işlemleri "reduce-overhead" modunda graf tabanlı derlenir. Sabitlerin katlanması 
      (constant folding) ve GPU kernel füzyonu ile TPS (Tokens Per Second) artırılır.
    - **KV Cache Quantization (quanto)**: Uzun bağlam penceresinde (Context Window) 
      oluşan devasa KV (Key-Value) tensör matrisleri, VRAM taşmasını (OOM) engellemek 
      amacıyla ``quanto`` backend'i ile dinamik olarak 8-bit hassasiyetine sıkıştırılır.
    - **OOM İzolasyonu (ThreadWithException)**: Streaming üretimi sırasında oluşabilecek CUDA OutOfMemory 
      hataları, izole bir katman üzerinden ana süreç çökmeden yakalanır ve ardışık geri-çekilme 
      (fallback) mekanizmasını (Retry & Half-Budget) tetikler.

Sınıflar:
    - ``RAGModel``: Model yükleme, bağlam kırpma ve streaming inference
      orkestratörü.

Bağımlılıklar:
    - ``torch>=2.6.0`` (GH200/Hopper CUDA 12.4+ desteği)
    - ``transformers>=5.7.0`` (Gemma4 model_type desteği)
    - ``flash-attn>=2.6.3`` (Hopper BF16 desteği)
"""

import threading
import os
import gc
import queue
from typing import Any, List, Optional

import torch
import torch._inductor.config as inductor_config
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_core.documents import Document

# ──────────────────────────────────────────────────────────────
# FA3 MODÜL TANIMA (MONKEY-PATCH)
# ──────────────────────────────────────────────────────────────
# flash-attn-3 paketi "flash_attn_interface" olarak import edilir,
# ancak transformers v5.x "flash_attn_3" modülünü arar.
try:
    import transformers.utils.import_utils as _import_utils
    import transformers.utils as _utils
    
    # Orijinal fonksiyonu yedekle (varsa)
    _orig_fa3_check = getattr(_import_utils, "is_flash_attn_3_available", None)
    
    def _patched_is_flash_attn_3_available():
        try:
            # flash-attn-3 (Hopper/Blackwell) kontrolü
            import flash_attn_interface
            return True
        except ImportError:
            # Paket yoksa orijinal transformers kontrolüne dön (varsa)
            if _orig_fa3_check is not None:
                return _orig_fa3_check()
            return False

    # Tüm olası giriş noktalarını yamala
    _import_utils.is_flash_attn_3_available = _patched_is_flash_attn_3_available
    if hasattr(_utils, "is_flash_attn_3_available"):
        setattr(_utils, "is_flash_attn_3_available", _patched_is_flash_attn_3_available)
    
    print("[INIT] FlashAttention-3 monkey-patch (v2/robust) uygulandı.")
except Exception as e:
    print(f"[INIT] FA3 patch uygulanamadı: {e}")

try:
    from src.utils import setup_logger, profile_time_and_memory, aggressive_vram_cleanup, get_vram_info, compact_text
    logger = setup_logger("RAGModel")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAGModel")
    # Fallback definitions to prevent NameError if src.utils is missing
    def aggressive_vram_cleanup():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def get_vram_info(): return {"free": 0}
    def compact_text(t): return t
    def profile_time_and_memory(f): return f

try:
    from src.metrics import MetricsCollector, detect_gpu
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False



class ThreadWithException(threading.Thread):
    """Asenkron inference sırasında bellek taşmalarını (OOM) yöneten yalıtılmış süreç yöneticisi.
    
    Metodolojik İşlevi:
    RAG mimarisinde LLM üretimi (generation) ana iş parçacığını (thread) bloklamadan 
    ve kesintisiz streaming akışı sağlamak amacıyla arka planda çalıştırılır. Devasa bağlam pencereleri 
    kullanıldığında anlık VRAM pikleri ``torch.cuda.OutOfMemoryError`` hatasına yol açabilir. 
    Bu sınıf, donanım düzeyinde fırlatılan GPU hatalarını Python bağlamında güvenle yakalayarak, 
    ana sürecin çökmesini engeller ve dinamik bağlam kırpma (Dynamic Context Truncation) algoritmasının 
    devreye girmesini sağlar.
    """
    def __init__(self, target, kwargs):
        super().__init__(target=target, kwargs=kwargs)
        self.exception = None

    def run(self):
        try:
            if self._target:
                self._target(**self._kwargs)
        except Exception as e:
            self.exception = e


class RAGModel:
    """Retrieval-Augmented Generation (RAG) dil modeli yöneticisi.

    Bu sınıf, büyük dil modelinin GPU üzerinde BFloat16 ile yüklenmesini,
    bağlam belgelerinin token bütçesine göre kırpılmasını ve streaming
    modunda MetricsCollector entegreli cevap üretilmesini yönetir.

    Attributes:
        model_name: Hugging Face model kimliği.
        tokenizer: Model tokenizer nesnesi.
        model: Yüklenmiş ``AutoModelForCausalLM`` nesnesi.
        device: Aktif cihaz (``"cuda"`` veya ``"cpu"``).
        load_mode: Yükleme modunun açıklaması.
        gpu_name: Çalışma zamanı GPU adı (metrik kayıtları için).
        MAX_CONTEXT_TOKENS: VRAM-güvenli maksimum bağlam token sayısı.
    """

    # ──────────────────────────────────────────────────────────────
    # SINIF SABİTLERİ
    # ──────────────────────────────────────────────────────────────

    # Blackwell/Hopper (96GB): Teorik 128K. Çalışma zamanında dinamik güncellenir.
    MAX_CONTEXT_TOKENS: int = 65_536  # Güvenli başlangıç değeri

    # ──────────────────────────────────────────────────────────────
    # BAŞLATMA
    # ──────────────────────────────────────────────────────────────

    def __init__(self, model_name: str = "google/gemma-4-31b-it", summary_path: str = "data/summary.json") -> None:
        """RAGModel nesnesini başlatır ve modeli belleğe yükler.

        Args:
            model_name: Hugging Face Hub'daki model kimliği.
            summary_path: summary.json istatistik dosyasının yolu.
        """
        self.model_name = model_name
        self.summary_path = summary_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_mode = "bilinmiyor"
        self.gpu_name = detect_gpu() if _METRICS_AVAILABLE else "GH200"
        self.is_warmed_up = False # JIT derleme durumunu takip etmek için

        logger.info(f"RAG Modeli yükleme başlatılıyor. Cihaz: {self.device.upper()} | GPU: {self.gpu_name}")
        
        # 0. JIT Cache Hazırlığı
        self._setup_persistent_jit_cache()
        
        self._load_model_and_tokenizer()

    def _setup_persistent_jit_cache(self) -> bool:
        """Google Drive tabanlı kalıcı torch.compile önbelleğini yapılandırır."""
        # 1. Öncelik: Halihazırda set edilmiş ortam değişkeni
        env_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if env_cache:
            os.makedirs(env_cache, exist_ok=True)
            logger.info(f"[JIT] Mevcut önbellek yolu kullanılıyor: {env_cache}")
            return True

        # 2. İkincil: Çalışma dizini Drive üzerindeyse otomatik klasör oluştur
        cwd = os.getcwd()
        if "/content/drive/" in cwd:
            drive_cache = os.path.join(cwd, "torch_cache")
            try:
                os.makedirs(drive_cache, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = drive_cache
                # Graph cache ve Triton tuning ayarları
                inductor_config.fx_graph_cache = True
                inductor_config.triton.unique_kernel_names = True
                logger.info(f"[JIT] Drive üzerinde otomatik önbellek oluşturuldu: {drive_cache}")
                return True
            except Exception as e:
                logger.warning(f"[JIT] Drive cache hazırlanamadı: {e}")
        
        logger.info("[JIT] Yerel geçici önbellek kullanılacak (Drive tespiti başarısız).")
        return False

    # ──────────────────────────────────────────────────────────────
    # YÜKLEME KADEMELERİ (BF16 + FA2 → BF16 + SDPA → CPU)
    # ──────────────────────────────────────────────────────────────

    def _try_load_bfloat16_fa(self):
        """Kademe 1: BFloat16 + FlashAttention3 — Blackwell/Hopper optimal yolu.

        Hopper/Blackwell mimarilerinde FlashAttention3, prefill hızını ve bellek verimliliğini artırır.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_3",
        )
        return model, "BFloat16 Native (GPU + FlashAttention3)"

    def _try_load_bfloat16_fa2(self):
        """Kademe 1.5: BFloat16 + FlashAttention2 — Hopper/Ampere fallback."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        return model, "BFloat16 Native (GPU + FlashAttention2)"

    def _try_load_bfloat16_sdpa(self):
        """Kademe 2: BFloat16 + SDPA — FA2 kurulu değilse veya hata verirse.

        PyTorch native SDPA; head-dim sınırı yok, tüm GPU'larda çalışır.

        Returns:
            Tuple[model, str]: Yüklenen model ve mod açıklaması.

        Raises:
            torch.cuda.OutOfMemoryError: VRAM yetersizse.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        return model, "BFloat16 Native (GPU + SDPA)"

    def _try_load_cpu(self):
        """Kademe 3: CPU Float32 — tüm GPU yolları başarısız olursa.

        Returns:
            Tuple[model, str]: Yüklenen model ve mod açıklaması.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.device = "cpu"
        return model, "Float32 CPU (Fallback)"

    @profile_time_and_memory
    def _load_model_and_tokenizer(self) -> None:
        """Tokenizer ve modeli kademeli fallback stratejisiyle yükler.

        Yükleme sırası:
            1. Tokenizer (başarısız olursa erken çıkış)
            2. BFloat16 + FlashAttention2 (Hopper optimal)
            3. BFloat16 + SDPA (universal fallback)
            4. CPU Float32 (son çare)
        """
        # 1. Tokenizer
        try:
            logger.info(f"Tokenizer yükleniyor: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"TOKENIZER YÜKLEME HATASI: {str(e)}")
            return

        # 2. Mimari Bazlı Strateji Seçimi
        cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
        
        loaders = []
        if cc[0] >= 10:
            # Blackwell (SM 12.0): SDPA + JIT en kararlı ve hızlı yoldur.
            logger.info("[HARDWARE] Blackwell (cc>=10) algılandı. SDPA + Persistent JIT optimize ediliyor.")
            loaders = [
                ("BF16+SDPA", self._try_load_bfloat16_sdpa),
                ("CPU",       self._try_load_cpu),
            ]
        elif cc[0] == 8:
            # Ampere / A100 (SM 8.0): FlashAttention-2 + JIT optimaldir.
            logger.info("[HARDWARE] Ampere/A100 (cc=8.0) algılandı. FA2 + Persistent JIT optimize ediliyor.")
            loaders = [
                ("BF16+FA2",  self._try_load_bfloat16_fa2),
                ("BF16+SDPA", self._try_load_bfloat16_sdpa),
                ("CPU",       self._try_load_cpu),
            ]
        else:
            # Diğer (V100, T4 vb.): SDPA, JIT yok.
            loaders = [
                ("BF16+SDPA", self._try_load_bfloat16_sdpa),
                ("CPU",       self._try_load_cpu),
            ]

        for label, loader_fn in loaders:
            try:
                logger.info(f"Model yükleniyor [{label}]...")
                self.model, self.load_mode = loader_fn()
                logger.info(f"✅ Model başarıyla yüklendi: {self.load_mode}")
                
                # 3. VRAM Bütçesi & JIT Uygulaması
                if self.device == "cuda":
                    # JIT sadece modern GPU'larda (Ampere+) anlamlıdır
                    if cc[0] >= 8:
                        logger.info(f"[OPTIMIZATION] {self.gpu_name} için torch.compile aktif ediliyor...")
                        try:
                            # 'default' modu: Triton kernel fusion + constant folding.
                            # Streaming inference ile uyumlu (CUDA graph kullanmaz).
                            self.model = torch.compile(self.model, mode="default")
                            logger.info("✅ torch.compile başarıyla kuruldu (veya cache'den yüklendi).")
                        except Exception as e:
                            logger.warning(f"torch.compile hatası: {e}. Standart modda devam ediliyor.")

                    # Başlangıç bütçesini hesapla
                    self.MAX_CONTEXT_TOKENS = self._estimate_safe_token_budget()
                
                return
            except ImportError as e:
                logger.warning(f"[{label}] import hatası (paket eksik?): {str(e)} — sonraki kademeye geçiliyor.")
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"[{label}] OOM — VRAM yetersiz: {str(e)} — sonraki kademeye geçiliyor.")
                aggressive_vram_cleanup()
            except Exception as e:
                logger.warning(f"[{label}] başarısız: {str(e)} — sonraki kademeye geçiliyor.")

        logger.error("❌ TÜM YÜKLEME KADEMELERİ BAŞARISIZ! Model yüklenemedi.")

    # ──────────────────────────────────────────────────────────────
    # INFERENCE — TOKEN-AWARE CONTEXT BUDGETING
    # ──────────────────────────────────────────────────────────────

    def _estimate_safe_token_budget(self) -> int:
        """Mevcut boş VRAM'e göre güvenli bir token bütçesi hesaplar.
        
        Blackwell SM 12.0 SDPA performansı için agresif bir güvenlik tamponu kullanır.
        Prefill aşamasındaki O(L^2) bellek sıçramalarını önlemek için 
        GB başına ~1500 token (muhafazakar) baz alır.
        """
        vram = get_vram_info()
        if vram["total"] <= 0:
            return 8192
            
        # Tamponlar: Model ağırlıkları (~62GB) + Sistem/Overhead (5GB)
        usable_vram = max(vram["free"] - 5.0, 0)
        
        # Dinamik Katsayı: FA2 bellek erişiminde daha verimlidir (IO-bound minimize)
        # FA2 (Ampere): 2500 tokens/GB — O(L) bellek, yüksek throughput
        # SDPA (Blackwell/Other): 1000 tokens/GB — O(L²) prefill spike koruması
        ratio = 2500 if "FlashAttention2" in self.load_mode else 1000
        
        safe_budget = int(usable_vram * ratio)
        
        # FA2 varsa 131K (O(L) bellek), SDPA varsa 32K (O(L²) prefill koruması)
        max_cap = 131072 if "FlashAttention2" in self.load_mode else 32768
        safe_budget = min(max(safe_budget, 4096), max_cap)
        
        logger.info(f"[PROBE] Mod: {self.load_mode} | Boş VRAM: {vram['free']:.2f}GB | Bütçe: {safe_budget} token")
        return safe_budget

    def _truncate_context_to_budget(self, context_str: str, limit: Optional[int] = None) -> tuple[str, int]:
        """Bağlam metnini VRAM-güvenli token sınırında kırpar.

        Args:
            context_str: RAG pipeline'ından gelen ham bağlam metni.
            limit: Opsiyonel özel limit (OOM retry senaryoları için).

        Returns:
            Tuple[kırpılmış_metin, gerçek_token_sayısı]
        """
        target_limit = limit if limit is not None else self.MAX_CONTEXT_TOKENS
        
        # Gereksiz boşlukları temizle
        context_str = compact_text(context_str)
        
        token_ids = self.tokenizer.encode(context_str, add_special_tokens=False)
        actual_tokens = len(token_ids)

        if actual_tokens <= target_limit:
            logger.info(f"[BUDGET] Bağlam: {actual_tokens} token (limit: {target_limit}) — OK")
            return context_str, actual_tokens

        # Akıllı Kırpma: Sadece doküman içeriğini kısalt, sonu (soru) korumak için 
        # (Not: RAG'da soru genellikle context_str dışında eklenir ama burada güvenli kesim yapıyoruz)
        truncated_ids = token_ids[:target_limit]
        truncated_str = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        logger.warning(
            f"[BUDGET] Bağlam kırpıldı: {actual_tokens} → {target_limit} token"
        )
        return truncated_str, target_limit

    def _is_analytic_query(self, query: str) -> bool:
        """Sorgunun global istatistik veya listeleme sorusu olup olmadığını belirler."""
        analytic_keywords = [
            "toplam", "sayı", "tüm hocalar", "tüm dersler", "ortalama", "en çok", "en az", "istatistik",
            "hocalar", "öğretim üyesi", "öğretim görevlisi", "listeleme", "hangi dersler", "bölümde",
            "müfredat", "programa", "kaçıncı yarıyıl", "kaç ders", "kaç hoca"
        ]
        q_lower = query.lower()
        return any(kw in q_lower for kw in analytic_keywords)

    def _estimate_response_budget(self, query: str, retrieved_doc_count: int) -> int:
        """Sorgunun karmaşıklığına ve getirilen belge sayısına göre max_new_tokens bütçesi belirler."""
        q_lower = query.lower()
        
        long_keywords = ["haftalık", "plan", "içerik", "konu", "değerlendirme", "kriter", "sınav", "ara sınav", "final", "ödev", "listele", "dersleri", "müfredat", "program"]
        is_long_query = any(kw in q_lower for kw in long_keywords)
        is_analytic = self._is_analytic_query(query)
        
        if is_analytic:
            budget = 2048
        elif is_long_query:
            budget = 1024
        else:
            # Kısa sorgu (AKTS, kredi, zorunlu/seçmeli vb.)
            budget = 512
            
        logger.info(f"[BUDGET] Sorgu bütçesi belirlendi: {budget} tokens (Docs: {retrieved_doc_count}, LongQuery: {is_long_query}, Analytic: {is_analytic})")
        return budget

    def _load_summary_stats(self) -> str:
        """summary.json dosyasını yükler, hoca isimlerindeki URL/e-posta kirliliğini temizler ve istatistikleri döndürür."""
        if not hasattr(self, "summary_path") or not os.path.exists(self.summary_path):
            return "Global özet dosyası bulunamadı."
        
        try:
            import json
            import re
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            toplam_ders = data.get("toplam_ders", 0)
            dersler = data.get("dersler", [])
            
            cleaned_teachers = set()
            clean_dersler_list = []
            
            # Ünvan ve isim dışındaki kirli kısımları (URL, e-posta) temizle
            clean_pattern = re.compile(r'https?://\S+|www\.\S+|\S+@\S+', re.IGNORECASE)
            
            for ders in dersler:
                hoca_raw = ders.get("hoca", "")
                # URL/e-posta temizliği
                hoca_clean = clean_pattern.sub("", hoca_raw).strip()
                hoca_clean = " ".join(hoca_clean.split()).title()
                
                # Temizlenmiş hoca isminden sonra unvan kelimeleri dışındaki
                # temiz hocaları set içerisine ekle
                if hoca_clean:
                    cleaned_teachers.add(hoca_clean)
                
                clean_dersler_list.append({
                    "kod": ders.get("kod", ""),
                    "ad": ders.get("ad", ""),
                    "akts": ders.get("akts", ""),
                    "hoca": hoca_clean
                })
            
            # Benzersiz öğretmen listesini alfabetik sırala
            sorted_teachers = sorted(list(cleaned_teachers))
            
            # İstatistik metni oluştur
            stats = []
            stats.append(f"Toplam Ders Sayısı: {toplam_ders}")
            stats.append(f"Toplam Öğretim Üyesi/Görevlisi Sayısı: {len(sorted_teachers)}")
            stats.append("\nÖğretim Üyeleri ve Görevlileri Listesi:")
            for idx, teacher in enumerate(sorted_teachers, 1):
                stats.append(f"{idx}. {teacher}")
            
            stats.append("\nTüm Dersler Listesi (Kod - Adı - AKTS - Hoca):")
            for ders in clean_dersler_list:
                stats.append(f"- {ders['kod']}: {ders['ad']} ({ders['akts']} AKTS) - {ders['hoca']}")
                
            return "\n".join(stats)
        except Exception as e:
            logger.error(f"Global istatistik yükleme hatası: {str(e)}")
            return f"Global istatistik yüklenirken hata oluştu: {str(e)}"

    # ──────────────────────────────────────────────────────────────
    # INFERENCE — STREAMING CEVAP ÜRETİMİ
    # ──────────────────────────────────────────────────────────────

    def generate_answer_stream(
        self,
        query: str,
        retrieved_docs: List[Any],
        metrics_collector: Optional[Any] = None,
    ):
        """RAG bağlamıyla zenginleştirilmiş sorguya streaming cevap üretir.

        Args:
            query: Kullanıcının doğal dil sorgusu.
            retrieved_docs: RAG pipeline'ından dönen ``Document`` listesi.
            metrics_collector: Opsiyonel ``MetricsCollector`` instance'ı.
                Verilmezse metrik toplanmaz (geriye dönük uyumlu).

        Yields:
            str: Kümülatif olarak büyüyen cevap metni.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("generate_answer_stream çağrıldı ama model/tokenizer yok!")
            yield (
                "❌ **Model yüklenemedi.** Lütfen Colab log'larını kontrol edin.\n\n"
                "**Olası nedenler:**\n"
                "- VRAM yetersiz → Runtime'ı yeniden başlatın (Runtime > Restart)\n"
                "- Model indirme hatası → HF token ve bağlantıyı kontrol edin"
            )
            return

        logger.info(f"[INFERENCE] [{self.load_mode}] Sorgu: '{query[:60]}'")
        
        # ── 0. VRAM Ön Hazırlık & Proaktif Limit ──
        # Dinamik bütçe: gerçek zamanlı VRAM durumuna göre limit hesapla
        # (main.py finally bloğu sorgu sonrası temizliği zaten yapıyor)
        current_limit = self._estimate_safe_token_budget()
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                # ── 1. Bağlam metni oluşturma & Proaktif Belge Sınırlama ──
                if retrieved_docs:
                    context_parts = []
                    current_tokens = 0
                    for i, doc in enumerate(retrieved_docs, start=1):
                        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        text_part = f"[Belge {i}]: {content}\n"
                        # Hızlı token tahmini (chars/4) - tam tokenizasyon truncate'de yapılacak
                        current_tokens += len(text_part) // 4
                        
                        if current_tokens > current_limit and i > 1:
                            logger.warning(f"[GUARD] {i}. belgede bütçe aşıldı, geri kalanı atlanıyor.")
                            break
                        context_parts.append(text_part)
                    context_str = "\n".join(context_parts)
                else:
                    context_str = "İlgili belge bulunamadı."

                # ── 2. Kesin Token Kırpma ──
                context_str, context_token_count = self._truncate_context_to_budget(context_str, limit=current_limit)

                # ── 3. Prompt & Tokenizasyon ──
                system_content = (
                    "Sen Osmaniye Korkut Ata Üniversitesi (OKÜ) Bologna Bilgi Paketi resmi asistanısın. "
                    "Görevin, paylaşılan ders içeriklerini analiz ederek kullanıcıya kesin ve doğru bilgi sunmaktır.\n\n"
                    "KURALLAR:\n"
                    "1. YALNIZCA sağlanan bağlamdaki (Belgelerdeki) bilgileri kullan. Bağlam dışı bilgi verme.\n"
                    "2. Eğer sorunun cevabı bağlamda (veya sağlanmışsa Global İstatistiklerde) yoksa, 'Verilen belgelerde bu bilgiye ulaşılamadı' de. Asla tahminde bulunma.\n"
                    "3. Cevabı bulursan DOĞRUDAN cevabı ver. 'Belgelerde bulunmamaktadır ancak istatistiklere göre...' gibi giriş cümleleri KULLANMA.\n"
                    "4. Akademik, profesyonel ve yardımcı bir dil kullan.\n"
                    "5. Tablosal verileri (Haftalık akış vb.) Markdown tablosu veya düzenli liste şeklinde sun.\n"
                    "6. Birden fazla ders veya hoca eşleşiyorsa hepsini ayrı başlıklar altında net bir şekilde listele."
                )

                if self._is_analytic_query(query):
                    stats = self._load_summary_stats()
                    system_content += f"\n\n📊 GLOBAL İSTATİSTİKLER:\n{stats}"

                system_content += """

--- ÖRNEK SORU-CEVAP ---

Soru: BMB203 dersinin değerlendirme kriterleri nelerdir?
Cevap:
**BMB203 — Ayrık Matematik** dersinin değerlendirme kriterleri:

| Değerlendirme | Ağırlık |
|:---|:---|
| Ara Sınav | %35 |
| Ödev | %25 |
| Yarıyıl Sonu Sınavı | %40 |

Soru: Erhan Turan hocanın dersleri nelerdir?
Cevap:
**Dr. Öğr. Üyesi Erhan TURAN** aşağıdaki dersleri vermektedir:
1. **BMB304** — Otomata Teorisi (5 AKTS)
2. **BMB308** — Mikroişlemciler (5 AKTS)
3. **BMB422** — Derin Öğrenme (5 AKTS)

--- ÖRNEK BİTTİ ---
"""
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": f"Bağlam:\n{context_str}\n\nSoru: {query}"},
                ]

                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

                # ── 4. Streaming Üretim (BF16 Native KV Cache) ──
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.05)
                
                generation_kwargs = dict(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=self._estimate_response_budget(query, len(retrieved_docs)),
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                if metrics_collector:
                    metrics_collector.start_inference(query, context_token_count, len(retrieved_docs))

                # ThreadWithException kullanarak arka plandaki OOM'ları yakalıyoruz
                thread = ThreadWithException(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                generated_text = ""
                ttft_measured = False
                
                # Streamer'dan veri beklerken aynı zamanda thread exception kontrolü yap
                while thread.is_alive() or not streamer.empty():
                    try:
                        # Timeout koyarak thread'in ölmesini (OOM) kontrol edebiliyoruz
                        new_text = next(streamer)
                        
                        if not ttft_measured and metrics_collector:
                            metrics_collector.record_first_token()
                            ttft_measured = True
                            self.is_warmed_up = True # İlk token geldiyse warmup bitmiştir
                        
                        if metrics_collector:
                            metrics_collector.record_token(new_text)
                        
                        generated_text += new_text
                        yield generated_text
                    except queue.Empty:
                        if thread.exception:
                            raise thread.exception
                        continue
                    except StopIteration:
                        break

                if thread.exception:
                    raise thread.exception

                if metrics_collector:
                    # Analitik sorgularda system prompt'a enjekte edilen global
                    # istatistikleri de context olarak dahil et, böylece
                    # Faithfulness hesabı gerçek bağlamı yansıtsın.
                    effective_context = list(retrieved_docs)
                    if self._is_analytic_query(query):
                        stats_text = self._load_summary_stats()
                        effective_context.append(
                            Document(page_content=stats_text)
                        )
                    metrics_collector.end_inference(generated_text, effective_context)
                
                return # Başarılı çıkış

            except torch.cuda.OutOfMemoryError:
                retry_count += 1
                logger.warning(f"OOM Yakalandı! Retry {retry_count}/{max_retries}. Bütçe %50 düşürülüyor...")
                aggressive_vram_cleanup()
                current_limit = int(current_limit * 0.5) 
                
                if retry_count > max_retries:
                    yield "❌ **Üzgünüm, çok geniş bağlam nedeniyle bellek yetersiz.** Lütfen sorunuzu daha spesifik hale getirin (örn. bölüm adı belirterek)."
                    break
            except Exception as e:
                logger.error(f"Beklenmedik hata: {str(e)}")
                yield f"❌ **İşlem sırasında bir hata oluştu:** {str(e)}"
                return
