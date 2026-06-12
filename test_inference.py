"""
test_inference — RAG Pipeline Benchmark ve Doğrulama
=====================================================
Tek seferlik inference testi + MetricsCollector ile benchmark modu.

Kullanım::

    python test_inference.py                  # Tek sorgu testi
    python test_inference.py --benchmark      # Tüm benchmark sorguları
"""

import argparse
import os
import logging

# Gereksiz kütüphane loglarını sustur (HTTP GET/HEAD kirliliğini önlemek için)
for lib in ["httpx", "urllib3", "httpcore", "openai", "transformers"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.metrics import MetricsCollector, detect_gpu
from src.model import RAGModel
from src.vector_store import VectorStore

# ──────────────────────────────────────────────────────────────
# BENCHMARK SORGU SETİ (Deneysel Değerlendirme Protokolü)
# ──────────────────────────────────────────────────────────────
# Bu sorgu seti, RAG sisteminin farklı bilişsel yeteneklerini test etmek üzere 
# stratejik olarak seçilmiştir:
# 1. Hoca-Ders İlişkisi (Entity Relation & Aggregation)
# 2. Spesifik Sayısal Veri Çıkarımı (Exact Fact Retrieval)
# 3. Metin İçi Detay Çıkarımı (Deep Span Extraction)
# 4. Kapsamlı Sayım/Liste (Global Context & Counting)
# 5. Yapılandırılmış Veri Çıkarımı (Structured Data/Table Reading)

BENCHMARK_QUERIES = [
    {
        "query": "Erhan Turan hocanın dersleri nelerdir?",
        "reference": "Dr. Öğr. Üyesi Erhan Turan'ın verdiği dersler: BMB304 (Otomata Teorisi), BMB308 (Mikroişlemciler), BMB310 (Bilgisayar Mühendisliği Projesi-1), BMB422 (Derin Öğrenme), BMB434 (Doğal Dil İşleme).",
    },
    {
        "query": "BMB416 dersinin AKTS kredisi kaçtır?",
        "reference": "BMB416 (Hesaplama Teorisi) dersinin AKTS kredisi 5'tir.",
    },
    {
        "query": "Ayrık Matematik dersinin kaynak kitapları nelerdir?",
        "reference": "Ayrık Matematik (BMB203) dersinin kaynakları: H.Rosen, Ayrık Matematik ve Uygulamaları, Mc.Graw Hill, 2015. Ayrıca ders notları, internet ve Teams üzerinden paylaşılan dökümanlar kullanılmaktadır.",
    },
    {
        "query": "Bilgisayar Mühendisliği bölümünde kaç ders var?",
        "reference": "Bilgisayar Mühendisliği bölümünde toplam 104 ders bulunmaktadır.",
    },
    {
        "query": "BMB203 dersinin değerlendirme kriterleri nelerdir?",
        "reference": "BMB203 (Ayrık Matematik) dersinin değerlendirme kriterleri: 1 adet Ara Sınav (%35), 4 adet Ödev (%25) ve 1 adet Yarıyıl Sonu Sınavı (%40) şeklindedir.",
    },
    {
        "query": "BMB302 dersinin 5. hafta konusu nedir?",
        "reference": "BMB302 (Makine Öğrenmesi) dersinin 5. hafta konusu: ML'de temel kavramlar, öğrenme süreçleri ve veri setleri.",
    },
    {
        "query": "BMB101 ve BMB102 derslerinin AKTS kredileri kaçtır?",
        "reference": "BMB101 (Matematik - I) dersinin AKTS kredisi 5, BMB102 (Matematik- II) dersinin AKTS kredisi ise 5'tir.",
    },
    {
        "query": "BMB416 dersi zorunlu mu seçmeli mi?",
        "reference": "BMB416 (Hesaplama Teorisi) dersi seçmeli bir derstir.",
    },
    {
        "query": "BMB999 dersinin içeriği nedir?",
        "reference": "Verilen belgelerde bu bilgiye ulaşılamadı.",
    },
    {
        "query": "BMB302 dersinin değerlendirme kriterleri nelerdir?",
        "reference": "BMB302 (Makine Öğrenmesi) dersinin değerlendirme kriterleri: 1 adet Ara Sınav (%40), 2 adet Ödev (%10) ve 1 adet Yarıyıl Sonu Sınavı (%50) şeklindedir.",
    },
    {
        "query": "Özcan Yırtıcı hocanın dersleri nelerdir?",
        "reference": "Dr. Öğr. Üyesi Özcan Yırtıcı'nın verdiği dersler: BMB101 (Matematik - I), BMB102 (Matematik- II), BMB103 (Mühendislik Fiziği), BMB106 (Lineer Cebir), BMB201 (Diferansiyel Denklemler), BMB202 (Sayısal Yöntemler), BMB312 (Gönüllülük Çalışmaları), BMB411 (Kompleks Analize Giriş), BMB413 (Yenilenebilir Enerji Kaynakları).",
    },
    {
        "query": "BMB304 dersinin haftalık ders saati kaçtır?",
        "reference": "BMB304 (Otomata Teorisi) dersinin haftalık ders saati Teori=3, Uygulama=0, Lab=0 şeklindedir.",
    },
    {
        "query": "BMB416 dersi kaçıncı yarıyılda?",
        "reference": "BMB416 (Hesaplama Teorisi) dersi 8. yarıyıldadır.",
    },
    {
        "query": "Makine Öğrenmesi dersinin içeriği nedir?",
        "reference": "Makine Öğrenmesi (BMB302) dersinin içeriği: Öğrenmenin tanımı, sık kullanılan öğrenme paradigmaları, veri türleri, makine öğrenmesinin tanımı ve temel özellikleri, tarihsel gelişimi, makine öğrenmesinde kullanılan programlama dilleri, Anaconda kurulumu, Python tabanlı ML kütüphaneleri, makine öğrenmesinde temel kavramlar ve süreçler, öğrenme türleri (denetimli, denetimsiz, yarı denetimli), temel istatistik bilgisi, olasılık kavramları, Bayes teoremi, veri ön işleme süreçleri, Scikit-learn kütüphanesi, regresyon ve sınıflandırma modellemeleri, değerlendirme metrikleri ve uygulama örneklerini kapsamaktadır.",
    },
    {
        "query": "Bahar Gezici Geçer hocanın dersleri nelerdir?",
        "reference": "Dr. Öğr. Üyesi Bahar Gezici Geçer'in verdiği dersler: BMB214 (Yapay Zekaya Giriş), BMB306 (Yazılım Mühendisliği), BMB405 (İşletme Ekonomisi), BMB406 (Mühendislik Ekonomisi), BMB435 (Açıklanabilir ve Yorumlanabilir Yapay Zekâ: Teknikler ve Uygulamalar).",
    },
]


def run_single_query(vector_store: VectorStore, rag_model: RAGModel,
                     query: str, reference: str = None, gpu_name: str = "GH200"):
    """Tek bir sorguyu çalıştırır ve metriği kaydeder."""
    print(f"\n{'='*60}")
    print(f"[SORGU] {query}")
    print("=" * 60)

    retrieved_docs = vector_store.search_similar_documents(query=query, k=10)
    print(f"[RETRIEVAL] {len(retrieved_docs)} doküman bulundu.")

    metrics = MetricsCollector(
        model_name=rag_model.model_name,
        gpu_name=gpu_name,
    )

    result = ""
    print("[CEVAP] ", end="", flush=True)
    for chunk in rag_model.generate_answer_stream(
        query=query,
        retrieved_docs=retrieved_docs,
        metrics_collector=metrics,
    ):
        # Son delta'yı hesapla ve yazdır (streaming konsol çıktısı)
        delta = chunk[len(result):]
        print(delta, end="", flush=True)
        result = chunk

    print()  # newline

    # ROUGE-L için referans varsa kaydet
    saved_path = metrics.save_to_json(reference=reference)
    print(f"\n[METRICS] → {saved_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Testi ve Benchmark")
    parser.add_argument("--benchmark", action="store_true",
                        help="Tüm benchmark sorgularını çalıştır")
    args = parser.parse_args()

    print("[INFO] Vektör Veritabanı Yükleniyor...")
    vector_store = VectorStore(data_dir="data/normalized")
    chunks = vector_store.process_files()
    if chunks:
        print(f"[INFO] Toplam oluşturulan master-chunk doküman sayısı: {len(chunks)}")
        vector_store.build_faiss_index(chunks)
    else:
        print("[HATA] Chunk oluşturulamadı!")
        return

    print("\n[INFO] Dil Modeli Yükleniyor...")
    rag_model = RAGModel(summary_path="data/summary.json")

    # OptimizedModule support: Explicit None check required
    if rag_model.model is None:
        print("[HATA] Model yüklenemedi!")
        return

    gpu_name = detect_gpu()
    print(f"[INFO] GPU: {gpu_name} | Mod: {rag_model.load_mode}")

    # 🚀 FAZ 4: ANALİTİK SORGU VE TEMİZLİK DOĞRULAMASI
    print(f"\n{'='*60}\n⚙️ FAZ 4: ANALİTİK SORGU VE TEMİZLİK DOĞRULAMASI\n{'='*60}")
    test_query = "Bilgisayar Mühendisliği bölümünde toplam kaç ders var?"
    is_analytic = rag_model._is_analytic_query(test_query)
    print(f"Örnek Sorgu: '{test_query}'")
    print(f"Analitik Sorgu Tespiti: {is_analytic}")
    if is_analytic:
        stats = rag_model._load_summary_stats()
        print("\n📊 Temizlenmiş Global İstatistikler (İlk 10 satır):")
        print("\n".join(stats.split("\n")[:12]))
    print(f"{'='*60}\n")

    if args.benchmark:
        print(f"\n[BENCHMARK] {len(BENCHMARK_QUERIES)} sorgu çalıştırılıyor...")
        for item in BENCHMARK_QUERIES:
            run_single_query(
                vector_store, rag_model,
                query=item["query"],
                reference=item["reference"],
                gpu_name=gpu_name,
            )
        print(f"\n[BENCHMARK] Tamamlandı. Metrikler: data/metrics/")
    else:
        # Tek sorgu modu
        run_single_query(
            vector_store, rag_model,
            query=BENCHMARK_QUERIES[0]["query"],
            reference=BENCHMARK_QUERIES[0]["reference"],
            gpu_name=gpu_name,
        )


if __name__ == "__main__":
    main()
