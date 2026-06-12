"""VectorStore — Hibrit RAG (BM25 + Semantik + Metadata Filtresi)
==============================================================
Üç katmanlı retrieval mimarisi ile akademik Bologna verilerine
yüksek doğrulukta erişim sağlar.

Katmanlar:
    1. **Metadata Filtresi**: Soruda ders kodu, hoca adı veya ders adı
       varsa ilgili dokümanı DOĞRUDAN döndürür (embedding'e gerek yok).
       Türkçe karakter normalizasyonu ile karakter-duyarsız eşleşme yapar.
    2. **BM25 (Keyword)**: Okuda–Robertson BM25 algoritması ile
       klasik TF-IDF tabanlı kesin kelime eşleşmesi.
    3. **Semantik (FAISS)**: MiniLM-L12 embedding modeli ile
       vektörel benzerlik araması.

Son sıralama: Reciprocal Rank Fusion (RRF) ile BM25 ve FAISS
sonuçlarının birleştirilmesi.

Sınıflar:
    - ``LocalTransformersEmbeddings``: CPU tabanlı yerel embedding üretici.
    - ``BM25``: Okuda–Robertson BM25 arama motoru.
    - ``MetadataIndex``: Ders kodu / hoca adı / ders adı kesin eşleşme indeksi.
    - ``VectorStore``: Tüm retrieval katmanlarını orkestre eden ana sınıf.
"""

import os
import re
import json
import math
from typing import List, Tuple, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# BAAI/bge-reranker-base için CrossEncoder benzeri yapı
class LocalReranker:
    """Cross-Encoder tabanlı (bge-reranker-base) yeniden sıralama (reranking) modülü.

    Bi-Encoder (FAISS) ve Lexical (BM25) modellerinden dönen kaba aday listesini, 
    sorgu ve dokümanı tek bir Transformer modeline (Cross-Encoder) vererek 
    derin semantik etkileşim (attention) ile yeniden sıralar. Bu sayede 
    Recall yüksek tutulurken Precision maksimize edilir. GPU (VRAM) tasarrufu 
    amacıyla CPU üzerinde asenkron batch-processing ile çalışır.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cpu" # VRAM tasarrufu için CPU
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, documents: List[Document], k: int = 10) -> List[Document]:
        if not documents: return []
        pairs = [[query, doc.page_content] for doc in documents]
        
        with torch.no_grad():
            # Batch size ile CPU yükünü dengele
            batch_size = 8
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs).logits.view(-1).float()
                all_scores.extend(outputs.tolist())
            
        # Skorlara göre sırala
        combined = list(zip(documents, all_scores))
        ranked = sorted(combined, key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:k]]

try:
    from src.utils import setup_logger, normalize_text
    logger = setup_logger("VectorStore")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VectorStore")
    def normalize_text(t): return t.lower()


# ──────────────────────────────────────────────────────────────
# BÖLÜM 1: EMBEDDİNG MODELİ
# ──────────────────────────────────────────────────────────────

from langchain_core.embeddings import Embeddings

class LocalTransformersEmbeddings(Embeddings):
    """CPU tabanlı yerel Transformer embedding üretici.

    Sentence-Transformers modellerini kullanarak metin listelerini
    sabit boyutlu vektörlere dönüştürür. Mean pooling stratejisi
    uygulanır (son gizli katmanın token ortalaması alınır).

    Args:
        model_name: Model adı. ``"/"`` içermiyorsa otomatik olarak
            ``sentence-transformers/`` prefix'i eklenir.

    Attributes:
        tokenizer: Model tokenizer'ı (max 512 token).
        model: Transformer encoder modeli.
        device: Her zaman ``"cpu"`` (GPU, ana LLM için ayrılmıştır).
    """

    def __init__(self, model_name: str) -> None:
        full_name = f"sentence-transformers/{model_name}" if "/" not in model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(full_name)
        self.model = AutoModel.from_pretrained(full_name)
        self.device = "cpu"
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Metin listesini vektör listesine dönüştürür (mean pooling)."""
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Tek bir sorgu metnini vektöre dönüştürür."""
        return self.embed_documents([text])[0]

    def __call__(self, text):
        """FAISS uyumluluk arayüzü — str veya list kabul eder."""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)


# ──────────────────────────────────────────────────────────────
# BÖLÜM 2: BM25
# ──────────────────────────────────────────────────────────────

class BM25:
    """Okuda-Robertson BM25 anahtar kelime arama motoru.

    Klasik TF-IDF (Term Frequency-Inverse Document Frequency) tabanlı sıralama 
    algoritmasının, Belge Uzunluğu Normalizasyonu (b) ve Terim Frekansı Doygunluğu 
    (k1) ile geliştirilmiş halidir. Kesin eşleşme (exact match) gerektiren kod ve 
    terim aramalarında Bi-Encoder'ların eksikliklerini giderir.

    Matematiksel Formülasyon:
    Score(D, Q) = Σ [ IDF(q_i) * (f(q_i, D) * (k_1 + 1)) / (f(q_i, D) + k_1 * (1 - b + b * (|D| / avgdl))) ]

    Args:
        k1: Terim frekansı doygunluk parametresi (varsayılan: 1.5). Yüksek k1, tekrarlanan kelimelere daha fazla puan verir.
        b: Belge uzunluğu normalizasyon parametresi (varsayılan: 0.75). b=1 tam normalizasyon, b=0 normalizasyon yok demektir.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized = []
        self.df = defaultdict(int)
        self.idf = {}
        self.avg_dl = 0.0
        self.N = 0

    def clear(self):
        self.corpus.clear()
        self.tokenized.clear()
        self.df.clear()
        self.idf.clear()
        self.avg_dl = 0.0
        self.N = 0

    def _tokenize(self, text: str):
        return [t.lower() for t in re.findall(r'[a-zçşüöığA-ZÇŞÜÖİĞ0-9]+', text) if len(t) >= 2]

    def fit(self, documents):
        self.corpus = documents
        self.N = len(documents)
        self.tokenized = []
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self.tokenized.append(tokens)
            for token in set(tokens):
                self.df[token] += 1
        total_len = sum(len(t) for t in self.tokenized)
        self.avg_dl = total_len / self.N if self.N > 0 else 1.0
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        logger.info(f"BM25 indeksi: {self.N} doküman, {len(self.df)} unique token")

    def get_scores(self, query: str):
        query_tokens = self._tokenize(query)
        scores = []
        for idx, tokens in enumerate(self.tokenized):
            tf_dict = defaultdict(int)
            for t in tokens:
                tf_dict[t] += 1
            dl = len(tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = tf_dict.get(qt, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += self.idf[qt] * (numerator / denominator)
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 5):
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.corpus[idx], score) for idx, score in ranked[:k] if score > 0]


# ──────────────────────────────────────────────────────────────
# BÖLÜM 3: METADATA INDEX
# ──────────────────────────────────────────────────────────────

class MetadataIndex:
    """Ders kodu, hoca adı ve ders adı bazlı kesin eşleşme indeksi.

    FAISS/BM25'e düşmeden doğrudan metadata üzerinden sonuç döndürür.
    Tüm anahtarlar ``normalize_text()`` ile Türkçe karakter-duyarsız
    hale getirilir (Kakız == Kakiz).

    Attributes:
        code_to_docs: Ders kodu → Document listesi eşleşmesi.
        name_to_docs: Ders adı → Document listesi eşleşmesi.
        professor_to_docs: Normalize hoca adı varyantı → Document listesi.
    """

    def __init__(self) -> None:
        self.code_to_docs: dict = {}
        self.name_to_docs: dict = {}
        self.professor_to_docs: dict = {}

    def build(self, documents):
        code_pattern = re.compile(r'\b([A-ZÇŞÜÖİĞ]{2,4}\s*\d+[A-Z]?)\b')
        name_pattern = re.compile(r'Ders Adı:\s*(.+)')

        for doc in documents:
            content = doc.page_content

            # --- Ders kodu indeksleme ---
            codes = code_pattern.findall(content)
            for code in codes:
                key = code.strip().upper().replace(" ", "")
                if key not in self.code_to_docs:
                    self.code_to_docs[key] = []
                if doc not in self.code_to_docs[key]:
                    self.code_to_docs[key].append(doc)

            # --- Ders adı indeksleme ---
            m = name_pattern.search(content)
            if m:
                name = m.group(1).strip().lower()
                if name not in self.name_to_docs:
                    self.name_to_docs[name] = []
                if doc not in self.name_to_docs[name]:
                    self.name_to_docs[name].append(doc)

            # --- Hoca adı indeksleme (professor_to_docs) ---
            # Önce temizlenmiş liste metadata'sını kontrol et (çoklu hoca desteği)
            prof_list = doc.metadata.get("ogretim_uyesi_listesi", [])
            if not prof_list:
                # Fallback: ham metinden çıkar
                prof_raw = doc.metadata.get("ogretim_uyesi", "")
                if prof_raw:
                    prof_clean = re.sub(
                        r'(Dr\.|Doç\.|Prof\.|Öğr\.|Üyesi|Arş\.|Gör\.|Yrd\.|Öğretim|Görevlisi)',
                        '', prof_raw, flags=re.IGNORECASE
                    ).strip().lower()
                    prof_list = [' '.join(prof_clean.split())]

            for prof_clean in prof_list:
                prof_clean = prof_clean.strip()
                if not prof_clean:
                    continue
                
                # Varyantlar: tam ad, soyad, ad+soyad
                parts = prof_clean.split()
                v_list = [prof_clean]
                if len(parts) >= 2:
                    v_list.append(parts[-1])
                    v_list.append(f"{parts[0]} {parts[-1]}")

                for v in v_list:
                    # NORMALİZE EDİLMİŞ anahtar kullan (ı, i -> i)
                    v_norm = normalize_text(v)
                    if len(v_norm) >= 3:
                        if v_norm not in self.professor_to_docs:
                            self.professor_to_docs[v_norm] = []
                        if doc not in self.professor_to_docs[v_norm]:
                            self.professor_to_docs[v_norm].append(doc)

        logger.info(
            f"Metadata indeksi: {len(self.code_to_docs)} ders kodu, "
            f"{len(self.name_to_docs)} ders adı, "
            f"{len(self.professor_to_docs)} hoca varyantı"
        )

    def lookup(self, query: str):
        query_norm = normalize_text(query)
        results = []
        seen_docs = set()

        def add_docs(docs):
            for d in docs:
                if id(d) not in seen_docs:
                    seen_docs.add(id(d))
                    results.append(d)

        # --- Katman 1a: Ders kodu tam eşleşmeleri ---
        code_pattern = re.compile(r'\b([A-ZÇŞÜÖİĞ]{2,4}\s*\d+[A-Z]?)\b', re.IGNORECASE)
        found_codes = code_pattern.findall(query)
        for code in found_codes:
            key = code.strip().upper().replace(" ", "")
            if key in self.code_to_docs:
                logger.info(f"[METADATA HIT] Ders kodu: {key}")
                add_docs(self.code_to_docs[key])

        # --- Katman 1b: Hoca adı eşleşmeleri (NORMALIZE MATCH) ---
        found_profs = []
        for v_norm, docs in self.professor_to_docs.items():
            if len(v_norm) >= 4:
                if v_norm in query_norm or query_norm in v_norm:
                    is_redundant = False
                    for existing in found_profs:
                        if v_norm in existing:
                            is_redundant = True
                            break
                    if not is_redundant:
                        found_profs.append(v_norm)
                        add_docs(docs)
        
        if found_profs:
            logger.info(f"[METADATA HIT] Hocalar (Norm): {found_profs}")
            # Hoca bazlı sorgularda limitleri esnet (Tüm dersleri sayabilmesi için)
            return results

        # --- Katman 1c: Ders adı eşleşmesi (NORMALIZE MATCH) ---
        if not results:
            for name, docs in self.name_to_docs.items():
                name_norm = normalize_text(name)
                if len(name_norm) >= 5 and name_norm in query_norm:
                    logger.info(f"[METADATA HIT] Ders adı: {name}")
                    add_docs(docs)

        return results


class SemanticJSONSplitter:
    """JSON tabanlı eğitim materyalleri için anlamsal metin bölütleyici.

    Bologna veri formatının yapısal bütünlüğünü (tablolar, haftalık planlar) 
    koruyarak, veriyi LLM bağlam penceresine (context window) uygun parçalara 
    ayırır. Bölütleme işlemi RegEx lookahead (`(?=...)`) ile başlık tespiti 
    yaparak gerçekleştirilir.

    Matematiksel Boyut Sınırı:
    L(C_i) <= MAX_CHUNK_SIZE (Eğer yapı tablosal değilse)
    """

    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 250):
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split(self, content: str) -> List[str]:
        sections = re.split(r'\n(?=[A-ZÇŞÜÖİĞ][a-zçşüöığA-ZÇŞÜÖİĞ\s\(\)]+:)', content)
        semantic_chunks = []
        current_chunk = ""

        for section in sections:
            section = section.strip()
            if not section: continue
            
            # Tablo tespiti (Haftalık program genellikle rakamlarla başlar veya düzenli satırlardır)
            is_table = bool(re.search(r'^\d+\s*[\.\-]', section, re.MULTILINE))
            
            if len(section) > self.max_chunk_size and not is_table:
                if current_chunk:
                    semantic_chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_chunks = self.fallback_splitter.split_text(section)
                semantic_chunks.extend(sub_chunks)
            elif len(current_chunk) + len(section) < self.max_chunk_size:
                current_chunk += section + "\n\n"
            else:
                if current_chunk:
                    semantic_chunks.append(current_chunk.strip())
                current_chunk = section + "\n\n"
        
        if current_chunk:
            semantic_chunks.append(current_chunk.strip())
        
        return semantic_chunks

# ──────────────────────────────────────────────────────────────
# BÖLÜM 4: ANA VECTORSTORE
# ──────────────────────────────────────────────────────────────

class VectorStore:
    """Hibrit RAG retrieval orkestratörü.

    Üç katmanlı arama pipeline'ını (Metadata → BM25 → FAISS)
    tek bir ``search_similar_documents`` arayüzünde birleştirir.
    JSON verisini okur, chunk'lara ayırır, indeksler ve arama yapar.

    Args:
        data_dir: Normalize edilmiş JSON dosyalarının bulunduğu klasör.
        embedding_model: Sentence-Transformers model adı.

    Attributes:
        vector_db: FAISS vektör veritabanı.
        bm25: BM25 anahtar kelime indeksi.
        metadata_index: Ders kodu / hoca adı kesin eşleşme indeksi.
        all_documents: Tüm chunk'lanmış Document nesneleri.
    """

    def __init__(self, data_dir: str = "data/normalized", embedding_model: str = "BAAI/bge-m3", use_master_chunking: bool = True) -> None:
        self.data_dir = data_dir
        self.use_master_chunking = use_master_chunking
        self.vector_db = None
        self.bm25 = None
        self.metadata_index = None
        self.reranker = None # Reranker placeholder
        self.all_documents: List[Document] = []

        try:
            logger.info(f"Embedding modeli yükleniyor: {embedding_model}")
            self.embeddings = LocalTransformersEmbeddings(model_name=embedding_model)
            logger.info("Embedding modeli başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Embedding modeli hatası: {str(e)}")
            self.embeddings = None

        try:
            logger.info("Reranker modeli yükleniyor: BAAI/bge-reranker-base")
            self.reranker = LocalReranker()
            logger.info("Reranker modeli başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Reranker modeli hatası: {str(e)}")

    def process_files(self):
        """Normalize edilmiş dosyaları okur ve chunk'lara böler.
        
        Colab ve yerel çalışma dizinlerini otomatik tespit eder.
        `summary.json` varsa genel bağlam olarak enjekte eder.
        """
        logger.info("Normalize edilmiş veriler yükleniyor...")
        documents = []

        # Colab/Drive Yolu Tespiti
        possible_paths = [
            self.data_dir,
            os.path.abspath(self.data_dir),
            os.path.join(os.getcwd(), self.data_dir),
            f"/content/{self.data_dir}"
        ]
        
        target_path = None
        for p in possible_paths:
            if os.path.exists(p):
                target_path = p
                break
        
        if not target_path:
            logger.error(f"Veri klasörü bulunamadı! Aranan yollar: {possible_paths}")
            return []
        
        self.data_dir = target_path
        logger.info(f"Veri kaynağı: {self.data_dir}")

        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        if not json_files:
            logger.error("JSON dosyası bulunamadı!")
            return []

        # Metadata çıkarma patternları
        ders_kodu_pattern = re.compile(r'Ders Kodu:\s*([A-ZÇŞÜÖİĞ]{2,4}\s*\d+[A-Z]?)')
        ders_adi_pattern = re.compile(r'Ders Adı:\s*(.+)')
        ogretim_pattern = re.compile(r'(?:retim [Üü]yesi|Dersi Verenler):\s*(.+?)(?:\n|$)')
        title_words = ['Dr', 'Doç', 'Prof', 'Öğr', 'Gör', 'Arş', 'Yrd', 'Üyesi', 'Öğretim', 'Görevlisi', 'Üretim']
        title_clean_pattern = re.compile(r'\b(?:' + '|'.join(title_words) + r')\.?\s*', re.IGNORECASE)

        for filename in json_files:
            if filename == "summary.json": continue
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                content = data.get("content", "")
                title = data.get("title", filename.replace('_bologna.json', ''))
                if not content:
                    continue

                ders_kodu = ""
                ders_adi = ""
                ogretim_uyesi_list_meta = []
                ogretim_uyesi_raw = ""

                # 1. Ders Kodu ve Adı Çıkarma
                km = ders_kodu_pattern.search(content)
                if km:
                    ders_kodu = km.group(1).strip().replace(" ", "")
                am = ders_adi_pattern.search(content)
                if am:
                    ders_adi = am.group(1).strip()

                # 2. Hoca Bilgisi Çıkarma ve Temizleme
                om = ogretim_pattern.search(content)
                if om:
                    ogretim_uyesi_raw = om.group(1).strip()
                    clean_raw = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', ogretim_uyesi_raw)
                    clean_raw = ' '.join(clean_raw.split())
                    unvan_boundary = re.compile(r'(?=\b(?:Dr\.|Doç\.|Prof\.|Arş\.|Gör\.|Yrd\.|[A-Z][a-z]+\s+Gör\.))', re.IGNORECASE)
                    parts = [p.strip() for p in unvan_boundary.split(clean_raw) if p.strip()]
                    if not parts: parts = [clean_raw]
                    for part in parts:
                        clean_name = title_clean_pattern.sub('', part).strip()
                        clean_name = ' '.join(clean_name.split())
                        if len(clean_name) > 3:
                            ogretim_uyesi_list_meta.append(clean_name)

                if self.use_master_chunking:
                    prefix = f"[DERS: {ders_kodu or title} - {ders_adi or title}"
                    if ogretim_uyesi_list_meta:
                        prefix += f" | HOCA: {', '.join(ogretim_uyesi_list_meta)}"
                    prefix += "]\n---\n"
                    
                    enriched = prefix + content
                    doc = Document(
                        page_content=enriched,
                        metadata={
                            "source": filename,
                            "title": title,
                            "ders_kodu": ders_kodu,
                            "ders_adi": ders_adi,
                            "ogretim_uyesi": ogretim_uyesi_raw,
                            "ogretim_uyesi_listesi": ogretim_uyesi_list_meta
                        }
                    )
                    documents.append(doc)
                else:
                    # 3. Anlamsal (Semantic) Chunking
                    splitter = SemanticJSONSplitter(max_chunk_size=1500, chunk_overlap=250)
                    semantic_chunks = splitter.split(content)

                    for chunk in semantic_chunks:
                        prefix = f"[DERS: {ders_kodu or title} - {ders_adi or title}"
                        if ogretim_uyesi_list_meta:
                            prefix += f" | HOCA: {', '.join(ogretim_uyesi_list_meta)}"
                        prefix += "]\n---\n"
                        
                        enriched = prefix + chunk
                        doc = Document(
                            page_content=enriched,
                            metadata={
                                "source": filename,
                                "title": title,
                                "ders_kodu": ders_kodu,
                                "ders_adi": ders_adi,
                                "ogretim_uyesi": ogretim_uyesi_raw,
                                "ogretim_uyesi_listesi": ogretim_uyesi_list_meta
                            }
                        )
                        documents.append(doc)

            except Exception as e:
                logger.error(f"Dosya hatası ({filename}): {str(e)}")

        # 4. Summary Injection (Global Context)
        summary_path = os.path.join(self.data_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                summary_text = json.dumps(summary_data, ensure_ascii=False, indent=2)
                summary_doc = Document(
                    page_content=f"[GLOBAL ÖZET - SİSTEM BİLGİSİ]\n---\n{summary_text}",
                    metadata={"source": "summary.json", "title": "Global Özet"}
                )
                documents.append(summary_doc)
                logger.info("Global summary (summary.json) bağlama eklendi.")
            except Exception as e:
                logger.error(f"Summary yükleme hatası: {str(e)}")

        logger.info(f"Toplam {len(documents)} chunk oluşturuldu.")
        self.all_documents = documents
        return documents

    def build_faiss_index(self, documents):
        if not documents or not self.embeddings:
            logger.error("İndeks oluşturulamadı!")
            return False

        try:
            logger.info("FAISS indeksi oluşturuluyor...")
            self.vector_db = FAISS.from_documents(documents, self.embeddings)
            logger.info("FAISS tamamlandı.")
        except Exception as e:
            logger.error(f"FAISS hatası: {str(e)}")
            return False

        try:
            logger.info("BM25 indeksi oluşturuluyor...")
            if self.bm25 is not None:
                try:
                    self.bm25.clear()
                except Exception:
                    pass
            self.bm25 = BM25()
            self.bm25.fit(documents)
        except Exception as e:
            logger.error(f"BM25 hatası: {str(e)}")

        try:
            logger.info("Metadata indeksi oluşturuluyor...")
            self.metadata_index = MetadataIndex()
            self.metadata_index.build(documents)
        except Exception as e:
            logger.error(f"Metadata indeksi hatası: {str(e)}")

        return True

    def _rrf_fusion(self, bm25_results, faiss_results, k_rrf: int = 60):
        """Reciprocal Rank Fusion (RRF) algoritması ile hibrit sıralama.

        FAISS (Dense/Semantik) ve BM25 (Sparse/Lexical) katmanlarından gelen
        sıralamaları birleştirir. Skorlama her bir doküman için sıralama tersi 
        olarak hesaplanır.
        
        RRFScore(d) = Σ ( 1 / (k_rrf + rank_i(d)) )

        Args:
            bm25_results: BM25 arama sonuçları ve skorları.
            faiss_results: FAISS arama sonuçları.
            k_rrf: Düzenleme katsayısı (genellikle 60).
        """
        scores = defaultdict(float)
        doc_map = {}
        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content[:100]
            scores[key] += 1.0 / (k_rrf + rank + 1)
            doc_map[key] = doc
        for rank, doc in enumerate(faiss_results):
            key = doc.page_content[:100]
            scores[key] += 1.0 / (k_rrf + rank + 1)
            doc_map[key] = doc
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def search_similar_documents(self, query: str, k: int = 5):
        logger.info(f"Hibrit arama başlatıldı: '{query[:60]}' (k={k})")

        # Katman 1: Metadata filtresi — kesin eşleşme
        if self.metadata_index:
            meta_results = self.metadata_index.lookup(query)
            if meta_results:
                # Akademik sorgu (hoca/tüm dersler) tespiti
                is_academic = any(kw in query.lower() for kw in ["tüm", "liste", "hoca", "kim veriyor", "dersleri"])
                
                # VRAM dostu ama kapsamlı limit (Blackwell SM 12.0 için 80 doc max)
                MAX_DOCS = 80 if is_academic else 30
                capped = meta_results[:MAX_DOCS]
                
                # Global özet enjeksiyonu (Akademik sorgularda her zaman rank 0)
                summary_docs = [d for d in self.all_documents if d.metadata.get("source") == "summary.json"]
                if summary_docs:
                    # Summary zaten meta_results içinde olabilir, mükerrer önle
                    if id(summary_docs[0]) not in [id(d) for d in capped]:
                        capped = [summary_docs[0]] + capped
                
                if len(meta_results) > MAX_DOCS:
                    logger.warning(f"[METADATA] {len(meta_results)} → {MAX_DOCS} doküman (VRAM kısıtı)")
                else:
                    logger.info(f"[METADATA] {len(capped)} doküman döndürülüyor")
                return capped

        # Katman 2+3: Hibrit BM25 + FAISS
        # Metadata eşleşmesi yoksa daha geniş arama yap
        effective_k = max(k, 15)
        return self._hybrid_search(query, k=effective_k)

    def _hybrid_search(self, query: str, k: int = 5):
        bm25_results = []
        faiss_results = []

        if self.bm25:
            try:
                # Reranking için daha geniş bir aday kümesi al (Top 25)
                bm25_results = self.bm25.search(query, k=25)
            except Exception as e:
                logger.error(f"BM25 arama hatası: {str(e)}")

        if self.vector_db:
            try:
                faiss_results = self.vector_db.similarity_search(query, k=25)
            except Exception as e:
                logger.error(f"FAISS arama hatası: {str(e)}")

        if not bm25_results and not faiss_results:
            return []
            
        # Hibrit sonuçları birleştir (RRF)
        initial_results = self._rrf_fusion(bm25_results, faiss_results)
        
        # Katman 4: Reranker (Cross-Encoder) ile yeniden sırala
        if self.reranker:
            try:
                logger.info(f"[RERANKER] {len(initial_results)} aday yeniden sıralanıyor...")
                return self.reranker.rerank(query, initial_results, k=k)
            except Exception as e:
                logger.error(f"Reranker hatası: {str(e)}")
                return initial_results[:k]

        return initial_results[:k]
