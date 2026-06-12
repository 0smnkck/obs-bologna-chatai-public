"""
main — OKÜ Bologna RAG Asistanı Ana Giriş Noktası
====================================================
Bu modül, tüm RAG pipeline bileşenlerini (VectorStore, RAGModel,
MetricsCollector) başlatır ve Gradio web arayüzü üzerinden sunar.

İş Akışı:
    1. CUDA bellek konfigürasyonu (OOM önleyici)
    2. VectorStore: JSON verilerini yükle → Chunk → FAISS + BM25 + Metadata
    3. RAGModel: Gemma-4 31B'yi BFloat16 native olarak GPU'ya yükle
    4. Gradio ChatInterface ile kullanıcı arayüzünü başlat

Dinamik Routing (Query Routing) Stratejisi:
    Kullanıcı sorgusunun (query) semantik genişliğine göre getirme (retrieval) 
    katmanındaki 'k' parametresi otonom olarak ayarlanır:
    - Akademik/Global Sorgu ("tüm", "liste", vb.) → K=100: Kapsamlı döküman 
      taraması gerektiren listeleyici sorgular için genişletilmiş arama.
    - Hedefli Sorgu (Spesifik bir ders veya konu) → K=10: Nokta atışı 
      bilgi çekimine yönelik daraltılmış ve yüksek hassasiyetli (precision) arama.

Metrik Kaydı (Telemetry):
    Her sorgu döngüsünde TTFT, TPS, Faithfulness gibi akademik metrikler toplanarak 
    JSON formatında ``data/metrics/`` dizininde kalıcı hale getirilir.
"""

__version__ = "6.1.0"
__author__ = "OKÜ Academic Research"
__description__ = "Bologna RAG Hybrid Search System"


import os

# VRAM Fragmantasyonunu önlemek için PyTorch bellek konfigürasyonu.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc

import gradio as gr
import torch

from src.metrics import MetricsCollector, detect_gpu
from src.model import RAGModel
from src.vector_store import VectorStore


def initialize_rag_system():
    """Tüm RAG pipeline bileşenlerini başlatır ve döndürür.

    Returns:
        Tuple[VectorStore, RAGModel, str]: Retrieval, inference nesneleri
        ve GPU adı.
    """
    print("\n[INFO] Vektör Veritabanı Yükleniyor (Hibrit: BM25 + FAISS + Metadata)...")
    vector_store = VectorStore(data_dir="data/normalized")
    chunks = vector_store.process_files()

    if chunks:
        vector_store.build_faiss_index(chunks)
    else:
        print("[UYARI] Chunk oluşturulamadı! data/normalized klasörünü kontrol edin.")

    print("\n[INFO] Dil Modeli Yükleniyor (BFloat16 Native / GH200)...")
    rag_model = RAGModel(summary_path="data/summary.json")

    gpu_name = detect_gpu()

    # torch.compile (OptimizedModule) truthiness kontrolünü (if model:) desteklemez, 'is not None' şarttır.
    if rag_model.model is not None:
        print(f"\n[INFO] ✅ Model hazır: {rag_model.load_mode} | GPU: {gpu_name}")
    else:
        print("\n[UYARI] ❌ Model yüklenemedi! Tüm sorgular hata mesajı döndürecek.")
        print("[UYARI] Çözüm: Runtime > Restart and run all adımını deneyin.")

    return vector_store, rag_model, gpu_name


# Sistemi global olarak tek seferlik yükle
vector_store_instance, rag_model_instance, GPU_NAME = initialize_rag_system()


def respond(message, history):
    """Gradio ChatInterface callback fonksiyonu.

    Her sorgu için bir ``MetricsCollector`` oluşturur, dinamik routing
    ile K değerini belirler, VectorStore'dan belge getirir ve RAGModel
    ile streaming cevap üretir. Üretim tamamlandığında metrik JSON'a kaydedilir.

    Args:
        message: Kullanıcının doğal dil sorgusu.
        history: Gradio tarafından sağlanan konuşma geçmişi.

    Yields:
        str: Kümülatif olarak büyüyen cevap metni.
    """
    if not message.strip():
        yield "Lütfen geçerli bir soru sorunuz."
        return

    try:
        mesaj_kucuk = message.lower()

        # Dinamik Routing — sorgu tipine göre arama genişliği ayarlanır
        if any(kw in mesaj_kucuk for kw in ["tüm", "bütün", "hepsi", "özet", "liste", "kaç ders", "program", "hoca"]):
            # summary.json zaten sistem promptuna ekleneceği için K=100 yerine K=25 yeterlidir.
            # Bu, doğruluk kaybı olmadan 45K tokenlik OOM problemini çözer.
            k_hedef = 25 
            print(f"\n[ROUTING] Akademik/Global Arama (K={k_hedef}): '{message[:50]}'")
        else:
            k_hedef = 10
            print(f"\n[ROUTING] Hedefli Arama (K={k_hedef}): '{message[:50]}'")

        retrieved_docs = vector_store_instance.search_similar_documents(query=message, k=k_hedef)

        # Her sorgu için bağımsız bir MetricsCollector instance'ı
        metrics = MetricsCollector(
            model_name=rag_model_instance.model_name,
            gpu_name=GPU_NAME,
        )

        # Warm-up (JIT) bildirimi — Blackwell RTX 6000 için ilk sorgu optimizasyonu
        if not getattr(rag_model_instance, "is_warmed_up", True):
            yield "⏳ **JIT Derleme Başlatıldı (Blackwell G4 Optimizing...):** İlk sorgunuz için model GPU üzerinde optimize ediliyor. Bu işlem donanımınıza bağlı olarak 2-5 dakika sürebilir. Lütfen bekleyiniz...\n\n---\n\n"

        for bot_message_chunk in rag_model_instance.generate_answer_stream(
            query=message,
            retrieved_docs=retrieved_docs,
            metrics_collector=metrics,
        ):
            yield bot_message_chunk

        try:
            metrics.save_to_json()
        except Exception as e:
            print(f"[METRICS] Hata: Metrikler kaydedilemedi: {e}")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[MEMORY] VRAM cache temizlendi.")


print("\n[INFO] Gradio arayüzü başlatılıyor...")

demo = gr.ChatInterface(
    fn=respond,
    title="OKÜ Bologna Bilgi Sistemi — Hibrit RAG Asistanı",
    description=(
        "Osmaniye Korkut Ata Üniversitesi ders içerikleri ve AKTS bilgilerini sorgulayabilirsiniz.\n"
        "- **Model:** Gemma-4 31B (google/gemma-4-31b-it)\n"
        "- **Donanım:** NVIDIA Blackwell (RTX PRO 6000) - JIT Derleme Aktif\n"
        "- **Hızlandırma:** Native SDPA + torch.compile (Inductor)\n"
        "- **Not:** İlk sorguda 2-3 dakikalık bir 'warm-up' (derleme) süresi gerekebilir.\n"
        "- **Bağlam Penceresi:** 128K token (Blackwell Optimized)\n"
    ),
    examples=[
        "BMB416 dersinin AKTS kredisi kaçtır?",
        "Ayrık Matematik dersinin kaynak kitapları nelerdir?",
        "Bilgisayar Ağları dersini kim veriyor?",
        "BMB203 dersinin değerlendirme kriterleri nelerdir?",
        "Erhan Turan hocanın tüm dersleri nelerdir?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
