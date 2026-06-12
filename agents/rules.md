# 🤖 OBS/Bologna ChatAI - Sistem & Ajan Kuralları (System Prompt)

**[AMAÇ]:** Osmaniye Korkut Ata Üni. Bologna sistemini kazıyan (Scraper), verileri JSON formatında kaydeden, bu verileri FAISS tabanlı Vektör Veritabanı'na (Vector Store) işleyen ve Blackwell/GH200/A100 GPU üzerinde yerel (BFloat16 native) olarak çalışan Gemma-4-31B-it modeli vasıtasıyla akademik danışmanlık hizmeti sunan bir asistan inşa etmektir.

---

## 🎯 1. Kritik Donanım ve Model Mimarisi (Asla Değiştirme)
* **Model Kimliği:** `google/gemma-4-31b-it` (Native BFloat16 formatında yüklenir).
* **Donanım Hedefi:** Google Colab Blackwell / GH200 / A100 GPU.
* **Yükleme Mantığı (Sıfır Sıkıştırma):** `load_in_4bit`, `load_in_8bit` veya `bitsandbytes` kütüphanesi KESİNLİKLE KULLANILMAYACAKTIR. Model doğrudan native BFloat16 formatında ve `device_map="auto"` veya ilgili CUDA aygıtına kilitlenerek yüklenir.
* **KV Cache:** 8-bit quantized KV cache kullanılacaktır (quanto backend).
* **Bağımlılık Kısıtlaması:** CUDA 12.4+ uyumu ve modern donanım hızlandırma desteği için Pytorch kütüphanesi `torch>=2.6.0` versiyonunda olmalıdır.
* **Hızlandırma & Optimizasyon:** Native SDPA / FlashAttention-3 ve `torch.compile` (Inductor) JIT önbellek katmanı aktif olmalıdır.

## 💻 2. Modüler Yazılım Mimarisi (OOP Mantığı)
Sistem tamamen Sınıflara (Class) dayalı Nesne Yönelimli Programlama (OOP) ile tasarlanmıştır. Tüm geliştirmeler bu modüler düzende yapılmalıdır:
* **`src/scraper.py` (BolognaCrawler):** Web kazıma, DOM işlemleri ve JSON kayıt yönetiminden sorumludur (kodlar boşluksuz normalize edilir).
* **`src/vector_store.py` (VectorStore):** JSON verilerini okur, metin parçalama (Semantic Splitter) yapar, CPU batch-processing ile embedding üretip FAISS indeksine yazar ve BM25 + Metadata + Reranker (bge-reranker-base) ile hibrit arama orkestrasyonu sağlar.
* **`src/model.py` (RAGModel):** Cihaz kontrolü yapar, Modeli BFloat16 olarak yükler, gelen kullanıcı sorgusuyla FAISS belgelerini (Bağlam) harmanlar (RAG) ve cevabı bir iş parçacığında (ThreadWithException) VRAM-aware context bütçeleme ile üretir.
* **`src/utils.py`:** Merkezi loglama (logging), VRAM temizleme ve metin normalizasyonu barındırır.

## 🕷️ 3. Otonom Web Kazıma (Scraping) Standartları
* **Araçlar:** Veriler Selenium (`webdriver-manager` destekli, `--headless=new` görünmez Chrome) ve BeautifulSoup kullanılarak çekilir. HTML verileri yapılandırılmış `JSON` dosyalarına çevrilip `data/processed/` dizinine yazılmalıdır.
* **Ders Kodları:** Kaydedilen dosya isimlerinde ve veri yapılarında ders kodları boşluksuz olmalıdır (örn: `BMB101_bologna.json`).
* **DOM Genişletme & Gizli Dersler (Kritik Bypass):** Sayfadaki tüm `span.expandCollapse i.fa-plus-square` elementleri tıklanarak gizli DOM öğeleri açığa çıkarılmalıdır.
* **Atlama (Skip) Mekanizması:** Performans için, `data/processed/{DersKodu}_bologna.json` dosyasının varlığı kontrol edilir; mevcutsa ders es geçilir.

## 🧠 4. RAG Veri İşleme ve Üretim (Inference)
* **Metadata Enjeksiyonu (Data Integrity):** Vector Store, chunk üretirken bağlam bütünlüğü için her metnin başına zorla `Ders Kodu` ve `Ders Adı` metadatalarını (*ör: "[DERS: BMB416 - Veri Madenciliği] ..."*) eklemelidir.
* **Embedding Standardı:** Projenin içerisindeki `LocalTransformersEmbeddings` sınıfı kullanılır (OOM önlemek için batch_size=32 ile çalışır).
* **Dinamik K-Routing:** Kullanıcı sorgusunun tipine göre (akademik sorgularda K=100, hedefli sorgularda K=10) dinamik limit ayarlanır ve VRAM bütçesine göre capping uygulanır (Blackwell için maks 80 doc).
* **Asenkron Canlı Çıktı (Streaming):** `TextIteratorStreamer` kullanılır. Model üretimi `ThreadWithException` üzerinde başlatılır.
* **Bellek Temizliği:** İşlem döngüleri bittiğinde mutlak suretle `torch.cuda.empty_cache()` ve `gc.collect()` çağırımları yapılmalıdır.

## 🛡️ 5. Güvenlik ve Hata Yönetimi (Zero-Crash)
* **Hata Toleransı:** Tüm fonksiyonlar `try-except` bloklarıyla sarmalanmalı, hatalar `print` ile değil, merkezi `logger` ile gösterilmelidir. Windows ortamlarında konsol encoding uyumsuzluğunu gidermek için `utf-8` reconfigure edilmelidir.
* **Versiyon Kontrol ve Temizlik:** Geçici çalışma klasörleri, metrics kayıtları (`data/metrics/`) ve `.gitignore` dosyası güncel tutulmalıdır.

**[TALİMAT]:** Bu dosya projenin teknik anayasasıdır. Asistan, gelecekteki geliştirmelerinde token tasarrufu gözeterek bu kuralları ihlal edecek öneriler sunmamalı ve modüler yapıya sadık kalmalıdır.
