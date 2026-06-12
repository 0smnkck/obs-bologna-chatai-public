"""
utils — Proje Geneli Yardımcı Araçlar
======================================
Bu modül, OKÜ Bologna RAG Asistanı projesinin tüm alt modülleri
tarafından ortaklaşa kullanılan yardımcı fonksiyonları barındırır.

Sağlanan Araçlar:
    - ``normalize_text``: Türkçe → ASCII karakter normalizasyonu
      (metadata eşleştirmede karakter duyarsız arama sağlar).
    - ``setup_logger``: Merkezi loglama konfigürasyonu
      (tüm modüllerde tutarlı log formatı garanti eder).

Not:
    Bu dosyadaki fonksiyonlar **stateless** (durumsuz) tasarlanmıştır;
    herhangi bir sınıf bağımlılığı yoktur ve doğrudan import edilebilir.
"""

import logging
import sys
import time
import gc
from functools import wraps


def normalize_text(text: str) -> str:
    """Türkçe karakterleri ASCII karşılıklarına dönüştürür ve küçültür.

    Bu fonksiyon, kullanıcı sorgusundaki Türkçe karakter farklılıklarını
    (örn: "Kakız" vs "Kakiz", "Şeker" vs "Seker") ortadan kaldırarak
    MetadataIndex'teki hoca ve ders adı eşleştirmelerinde karakter-duyarsız
    (case-insensitive & diacritic-insensitive) arama yapılmasını sağlar.

    Dönüşüm tablosu::

        İ → i, I → i, Ş → s, ş → s, Ğ → g, ğ → g,
        Ç → c, ç → c, Ö → o, ö → o, Ü → u, ü → u, ı → i

    Args:
        text: Normalize edilecek ham metin.

    Returns:
        Tüm Türkçe karakterleri ASCII'ye çevrilmiş, küçük harfli,
        baş/son boşlukları temizlenmiş metin. Boş girdi için boş
        string döner.

    Examples:
        >>> normalize_text("Dr. Öğr. Üyesi Muhammet Talha KAKIZ")
        'dr. ogr. uyesi muhammet talha kakiz'
        >>> normalize_text("Kakız")
        'kakiz'
    """
    if not text:
        return ""
    table = str.maketrans("İIŞşĞğÇçÖöÜüı", "iissggccoouui")
    return text.translate(table).lower().strip()


def setup_logger(name: str) -> logging.Logger:
    """Proje genelinde tutarlı log formatı sağlayan logger fabrikası.

    Her modül bu fonksiyonu çağırarak kendi isim alanına sahip bir
    ``logging.Logger`` nesnesi alır. Aynı isimle tekrar çağrılırsa
    mevcut logger döndürülür (handler duplikasyonu önlenir).

    Log formatı::

        2026-04-28 14:30:00,123 - VectorStore - INFO - Mesaj

    Args:
        name: Logger'ın isim alanı (örn: ``"VectorStore"``, ``"RAGModel"``).

    Returns:
        Yapılandırılmış ``logging.Logger`` nesnesi (INFO seviyesi,
        stdout handler, zaman damgalı format).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Gereksiz kütüphane loglarını sustur (HTTP GET/HEAD kirliliğini önlemek için)
    for lib in ["httpx", "urllib3", "httpcore"]:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = False

    return logger


def profile_time_and_memory(func):
    """Fonksiyonun çalışma süresini ve GPU VRAM ayak izini ölçer.
    
    Akademik Performans Profilleme:
    Bu dekoratör, bir işlemin (örn. model yükleme, embedding çıkarma) süresini 
    (Time Complexity) ve bellek tüketimini (Space Complexity - VRAM Footprint) 
    analiz eder. CUDA asenkronizasyonunu yöneterek gerçek GPU kullanım 
    tepelerini (Peak VRAM) kaydeder.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("Profiler")
        try:
            import torch
            has_torch = torch.cuda.is_available()
        except ImportError:
            has_torch = False
            
        if has_torch:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        if has_torch:
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"[{func.__name__}] Süre: {duration:.3f} sn | VRAM Değişimi: {mem_after - mem_before:.3f} GB | Tepe VRAM: {peak_mem:.3f} GB")
        else:
            logger.info(f"[{func.__name__}] Süre: {duration:.3f} sn")
            
        return result
    return wrapper


def aggressive_vram_cleanup():
    """Olası bellek sızıntılarını (memory leaks) önlemek için CUDA VRAM'i zorla temizler.
    
    Metodolojik İşlevi:
    Python'un yerleşik çöp toplayıcısını (Garbage Collector) `gc.collect()` ile tetikleyerek 
    referansı kalmayan CPU nesnelerini temizler. Eşzamanlı olarak `torch.cuda.empty_cache()` ve 
    `torch.cuda.ipc_collect()` aracılığıyla PyTorch'un caching allocator'ında serbest bırakılmış 
    ancak işletim sistemine iade edilmemiş VRAM bloklarını parçalanmayı (fragmentation) 
    azaltmak amacıyla GPU havuzundan zorla siler.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def get_vram_info() -> dict:
    """Mevcut GPU bellek durumunu (Total, Allocated, Reserved, Free) hesaplar.
    
    Durum Değişkenleri:
    - `total`: Fiziksel VRAM kapasitesi (GB).
    - `allocated`: Tensörler tarafından aktif olarak kullanılan VRAM (GB).
    - `reserved`: PyTorch bellek yöneticisi (caching allocator) tarafından bloke edilen VRAM (GB).
    - `free`: İşletim sistemi üzerinden tespit edilen ve modelin anında kullanabileceği
       tamamen boş (unreserved) VRAM miktarı (GB).
       
    Bu bilgi, Dynamic Context Truncation algoritmalarında OOM engellemek için kritik veridir.
    """
    info = {"total": 0.0, "allocated": 0.0, "reserved": 0.0, "free": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            # Blackwell SM 12.0 gibi kartlarda torch.cuda.mem_get_info daha doğrudur
            free_raw, total_raw = torch.cuda.mem_get_info(0)
            info = {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free_raw / (1024**3)
            }
    except Exception:
        pass
    return info


def compact_text(text: str) -> str:
    """Metindeki gereksiz boşlukları temizleyerek token tasarrufu sağlar.
    
    Markdown yapılarını (tablolar vb.) bozmadan sadece whitespace optimizasyonu yapar.
    """
    if not text:
        return ""
    # Satır sonlarını koruyarak her satırdaki fazla boşlukları temizle
    lines = [line.strip() for line in text.split("\n")]
    # Boş satırları teke indir
    compacted_lines = []
    last_empty = False
    for line in lines:
        if not line:
            if not last_empty:
                compacted_lines.append("")
                last_empty = True
        else:
            compacted_lines.append(line)
            last_empty = False
    return "\n".join(compacted_lines)
