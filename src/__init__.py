"""
src — OKÜ Bologna RAG Asistanı Kaynak Paketi (v6.1.0 Academic Edition)
========================================================================
Bu paket, Bologna Bilgi Paketi RAG asistanının tüm çekirdek
modüllerini barındırır.

Modüller:
    - ``scraper``: Selenium tabanlı otonom veri madenciliği ve DOM Expand.
    - ``vector_store``: Hibrit retrieval (BM25 + FAISS + Metadata Index).
    - ``model``: Gemma-4 E2B BFloat16 inference, Token Budgeting ve Streaming.
    - ``metrics``: Formel değerlendirme (TTFT, TPS, ROUGE-L, Faithfulness).
    - ``utils``: Merkezi loglama, metin normalizasyonu ve VRAM Garbage Collection.
"""

__all__ = ["scraper", "vector_store", "model", "metrics", "utils"]
