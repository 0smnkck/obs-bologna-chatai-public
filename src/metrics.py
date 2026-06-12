r"""metrics — Model-Agnostik RAG Metrik Toplayıcı
================================================
Tüm inference süreci boyunca performans ve kalite metriklerini
toplar, hesaplar ve JSON formatında diske kaydeder.

Tasarım Felsefesi:
    Model-agnostik: Hangi model kullanılırsa kullanılsın (Gemma-4 31B,
    Gemma-4 E2B, Qwen vb.) aynı MetricsCollector sınıfı çalışır.
    Böylece modeller arası karşılaştırmalar doğrudan yapılabilir.

Toplanan Metrikler (Deneysel Değerlendirme Protokolü):
    - **TTFT (Time To First Token)**: $TTFT = T_{first\_token} - T_{request\_start}$. Modelin ilk anlamlı tepkiyi verme gecikmesini ölçer.
    - **TPS (Tokens Per Second)**: $TPS = \frac{N_{tokens}}{T_{end} - T_{start}}$. Üretim hızının (throughput) verimliliğini değerlendirir.
    - **Faithfulness**: Halüsinasyon (hallucination) oranını düşürmeye yönelik bir tahmin metodudur.
      $Faithfulness = \frac{| W_{response} \cap W_{context} |}{| W_{response} |}$
    - **ROUGE-L (Longest Common Subsequence)**: RAG çıktısının referans metin ile kelime sırası bütünlüğünü ölçer.
      $F_{LCS} = \frac{(1 + \beta^2) R_{LCS} P_{LCS}}{R_{LCS} + \beta^2 P_{LCS}}$
    - **Total Inference Time**: Uçtan uca RAG ardışık işlem süresidir.

Çıktı:
    ``data/metrics/YYYY-MM-DD_HH-MM-SS_<model-adı>.json``

Sınıflar:
    - ``MetricsCollector``: Ana metrik toplama ve kayıt sınıfı.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Any, List, Optional

try:
    import snowballstemmer
except ImportError:
    snowballstemmer = None

try:
    from src.utils import setup_logger
    logger = setup_logger("Metrics")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Metrics")


class MetricsCollector:
    """RAG inference pipeline için model-agnostik metrik toplayıcı.

    Her sorgu için tek bir ``MetricsCollector`` instance'ı oluşturulur.
    ``start_inference`` ile ölçüm başlar, ``end_inference`` ile biter.
    ``save_to_json`` çağrıldığında ``data/metrics/`` klasörüne yazılır.

    Args:
        model_name: Hugging Face model kimliği (örn: ``"google/gemma-4-31b-it"``).
        gpu_name: Çalışma zamanı GPU adı (örn: ``"GH200"``). Otomatik
            tespit de yapılabilir; ``detect_gpu()`` yardımcısına bakın.

    Attributes:
        OUTPUT_DIR: Metrik JSON dosyalarının yazıldığı klasör yolu.
    """

    OUTPUT_DIR: str = "data/metrics"
    BENCHMARK_VERSION: str = "v4.2"

    def __init__(self, model_name: str, gpu_name: str = "GH200") -> None:
        self.model_name = model_name
        self.gpu_name = gpu_name
        self._reset()

    # ──────────────────────────────────────────────────────────────
    # DURUM SIFIRLA
    # ──────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._query: str = ""
        self._context_token_count: int = 0
        self._retrieved_doc_count: int = 0
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._token_count: int = 0
        self._end_time: Optional[float] = None
        self._response: str = ""
        self._context_docs: List[Any] = []

    # ──────────────────────────────────────────────────────────────
    # YAŞAM DÖNGÜSÜ ÇAĞIRILARI
    # ──────────────────────────────────────────────────────────────

    def start_inference(
        self, query: str, context_token_count: int, retrieved_doc_count: int
    ) -> None:
        """Ölçümü başlatır. ``generate_answer_stream`` girişinde çağrılır."""
        self._reset()
        self._query = query
        self._context_token_count = context_token_count
        self._retrieved_doc_count = retrieved_doc_count
        self._start_time = time.perf_counter()

    def record_first_token(self) -> None:
        """İlk token üretildiğinde çağrılır (TTFT hesabı için)."""
        if self._first_token_time is None and self._start_time is not None:
            self._first_token_time = time.perf_counter()

    def record_token(self, new_text: str = "") -> None:
        """Her yeni token/chunk üretildiğinde çağrılır (TPS sayacı için)."""
        # Yaklaşık token sayımı: boşluk-ayrımlı kelime sayısı
        self._token_count += len(new_text.split()) if new_text.strip() else 1

    def end_inference(self, full_response: str, context_docs: List[Any] = None) -> None:
        """Inference tamamlandığında çağrılır."""
        self._end_time = time.perf_counter()
        self._response = full_response
        self._context_docs = context_docs or []

    # ──────────────────────────────────────────────────────────────
    # KALİTE METRİKLERİ
    # ──────────────────────────────────────────────────────────────

    # Türkçe stopword listesi: Faithfulness hesabında gürültü yaratan
    # yaygın bağlaç, edat ve dolgu kelimeleri filtrelenir.
    _TURKISH_STOPWORDS: set = {
        "için", "olan", "olarak", "gibi", "daha", "sonra", "kadar",
        "ancak", "fakat", "veya", "ile", "bir", "olan", "olup",
        "ders", "adet", "katkı", "oranı", "sahip", "olmak",
        "aşağıdaki", "şeklinde", "üzere", "yönelik", "ilgili",
        "bulunmaktadır", "vermektedir", "edilmektedir", "yapılmaktadır",
        "değerlendirme", "belgelerde", "belgelerdeki", "bilgileri",
        "kullanıcıya", "sunmaktır", "analiz", "ederek",
        "this", "that", "with", "from", "have", "been",
        "bulunmamaktadır", "sistem", "genelindeki", "göre", "tür", 
        "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür"
    }

    # Sorgu kalıplarında sıkça geçen ve model cevabında tekrarlandığında
    # faithfulness değerini düşüren / saptıran soru kelimeleri.
    _QUERY_FORM_STOPWORDS: set = {
        "nelerdir", "nedir", "neden", "niçin", "nasıl", "kaçtır", 
        "hangi", "hangileridir", "kimdir", "kaç", "ne", "mi", "mı", 
        "mu", "mü"
    }

    def compute_faithfulness(self) -> float:
        """Yanıttaki kelimelerin context'te bulunma oranını (Faithfulness) hesaplar.
        
        Akademik Formülasyon:
            $Faithfulness = \\frac{|W_{response} \\cap W_{context}|}{|W_{response}|}$
        
        Burada $W_{response}$ yanıtta üretilen kelime kümesini, $W_{context}$
        ise retrieval katmanından dönen belgelerin kelime kümesini ifade eder. Bu metrik,
        üretilen metnin verilen kaynağa ne kadar sadık olduğunu ölçerek model 
        halüsinasyonunu tespit etmeye yardımcı olur.

        Returns:
            0.0 ile 1.0 arasında faithfulness skoru (float).
        """
        if not self._response or not self._context_docs:
            return 0.0

        context_text = " ".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in self._context_docs
        ).lower()

        # Alfanumerik + Türkçe karakter, ≥2 uzunluk
        _TOKEN_PATTERN = r"[a-z0-9çşüöığ]{2,}"
        raw_response_words = re.findall(_TOKEN_PATTERN, self._response.lower())
        raw_context_words = re.findall(_TOKEN_PATTERN, context_text)

        # Ham filtreleme (Stemming öncesi)
        filtered_response = [
            w for w in raw_response_words
            if w not in self._TURKISH_STOPWORDS and w not in self._QUERY_FORM_STOPWORDS
        ]
        filtered_context = [
            w for w in raw_context_words
            if w not in self._TURKISH_STOPWORDS
        ]

        # Stemmer ilklendir
        stemmer = None
        if snowballstemmer is not None:
            try:
                stemmer = snowballstemmer.stemmer("turkish")
            except Exception:
                pass

        # Stem et
        if stemmer is not None:
            try:
                stemmed_response = stemmer.stemWords(filtered_response)
            except Exception:
                stemmed_response = filtered_response

            try:
                stemmed_context = stemmer.stemWords(filtered_context)
            except Exception:
                stemmed_context = filtered_context
        else:
            stemmed_response = filtered_response
            stemmed_context = filtered_context

        # Post-normalization (Sesli düşürme ve yumuşama düzeltmesi)
        def post_normalize(w: str) -> str:
            vowels = "aeıioöuü"
            while len(w) > 3 and w[-1] in vowels:
                w = w[:-1]
            if len(w) > 3 and w[-1] == 'ğ':
                w = w[:-1] + 'k'
            return w

        # Set dönüşümü ve normalizasyon
        response_words = {post_normalize(w) for w in stemmed_response}
        context_words = {post_normalize(w) for w in stemmed_context}

        # Stemming sonrası filtreleme (Stem edilmiş & normalize edilmiş stopwords için)
        if stemmer is not None:
            try:
                stemmed_stopwords = stemmer.stemWords(list(self._TURKISH_STOPWORDS))
                normalized_stopwords = {post_normalize(w) for w in stemmed_stopwords}
            except Exception:
                normalized_stopwords = self._TURKISH_STOPWORDS

            try:
                stemmed_query_stopwords = stemmer.stemWords(list(self._QUERY_FORM_STOPWORDS))
                normalized_q_stopwords = {post_normalize(w) for w in stemmed_query_stopwords}
            except Exception:
                normalized_q_stopwords = self._QUERY_FORM_STOPWORDS

            response_words -= normalized_stopwords
            response_words -= normalized_q_stopwords
            context_words -= normalized_stopwords

        if not response_words:
            return 0.0

        overlap = response_words & context_words
        return round(len(overlap) / len(response_words), 4)

    def compute_rouge_l(self, reference: str) -> Optional[float]:
        """ROUGE-L (Longest Common Subsequence) F1 skorunu hesaplar.
        
        Akademik Formülasyon:
            $F_{LCS} = \\frac{(1 + \\beta^2) R_{LCS} P_{LCS}}{R_{LCS} + \\beta^2 P_{LCS}}$
        
        Bu metrik, RAG sisteminin ürettiği yanıtın beklenen referans yanıta 
        göre sıralama (sequence) korunarak ne kadar örtüştüğünü niceliksel
        olarak değerlendirir.

        Args:
            reference: Altın standart referans cevap metni.

        Returns:
            ROUGE-L F1 skoru (0.0–1.0) veya paket yoksa ``None``.
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            scores = scorer.score(reference, self._response)
            return round(scores["rougeL"].fmeasure, 4)
        except ImportError:
            logger.warning("rouge-score yüklü değil (pip install rouge-score). ROUGE-L atlandı.")
            return None
        except Exception as e:
            logger.warning(f"ROUGE-L hesaplama hatası: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # KAYIT VE DIŞA AKTARMA
    # ──────────────────────────────────────────────────────────────

    def build_record(self, reference: str = None) -> dict:
        """Tüm metrikleri tek bir dict'e toplar.

        Args:
            reference: Opsiyonel referans cevap (ROUGE-L için).

        Returns:
            Tüm metrik alanlarını içeren dict.
        """
        ttft = None
        total_time = None
        tps = None

        if self._start_time is not None and self._first_token_time is not None:
            ttft = round(self._first_token_time - self._start_time, 3)

        if self._start_time is not None and self._end_time is not None:
            total_time = round(self._end_time - self._start_time, 3)

        if total_time and self._token_count > 0:
            tps = round(self._token_count / total_time, 2)

        faithfulness = self.compute_faithfulness()
        rouge = self.compute_rouge_l(reference) if reference else None

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "benchmark_version": self.BENCHMARK_VERSION,
            "model_name": self.model_name,
            "gpu": self.gpu_name,
            "query": self._query,
            "context_token_count": self._context_token_count,
            "retrieved_doc_count": self._retrieved_doc_count,
            "ttft_seconds": ttft,
            "total_time_seconds": total_time,
            "tokens_generated": self._token_count,
            "tokens_per_second": tps,
            "faithfulness_score": faithfulness,
            "rouge_l_f1": rouge,
            "response_length_chars": len(self._response),
        }

        logger.info(
            f"[METRICS] TTFT={ttft}s | Süre={total_time}s | "
            f"TPS={tps} tok/s | Faithfulness={faithfulness}"
        )
        return record

    def save_to_json(self, reference: str = None) -> str:
        """Metrik kaydını ``data/metrics/`` altına JSON olarak yazar.

        Args:
            reference: Opsiyonel referans cevap (ROUGE-L için).

        Returns:
            Yazılan dosyanın tam yolu.
        """
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_short = self.model_name.split("/")[-1]
        filename = f"{ts}_{model_short}.json"
        path = os.path.join(self.OUTPUT_DIR, filename)

        record = self.build_record(reference)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.info(f"[METRICS] Kaydedildi → {path}")
        return path


# ──────────────────────────────────────────────────────────────
# YARDIMCI
# ──────────────────────────────────────────────────────────────

def detect_gpu() -> str:
    """Çalışma zamanındaki GPU adını otomatik tespit eder.

    Returns:
        GPU adı (örn: ``"NVIDIA GH200 96GB"``) veya ``"CPU"``.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"
