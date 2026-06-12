import os
import json
from typing import List, Dict

def lint_data(data_dir: str = "data/normalized") -> Dict:
    """Bologna veri setini kalite kontrolünden geçirir."""
    report = {
        "total_files": 0,
        "valid_files": 0,
        "errors": [],
        "warnings": [],
        "missing_fields": {}
    }

    if not os.path.exists(data_dir):
        report["errors"].append(f"Dizin bulunamadı: {data_dir}")
        return report

    files = [f for f in os.listdir(data_dir) if f.endswith(".json") and f != "summary.json"]
    report["total_files"] = len(files)

    required_fields = ["title", "content", "metadata"]
    metadata_fields = ["ders_kodu", "ders_adi", "ogretim_uyesi"]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. Temel Alan Kontrolü
            missing = [field for field in required_fields if field not in data]
            if missing:
                report["errors"].append(f"{filename}: Eksik alanlar -> {missing}")
                continue

            # 2. Metadata Kontrolü
            meta = data.get("metadata", {})
            missing_meta = [field for field in metadata_fields if field not in meta or not meta[field]]
            if missing_meta:
                report["warnings"].append(f"{filename}: Eksik metadata -> {missing_meta}")

            # 3. İçerik Uzunluğu
            if len(data.get("content", "")) < 500:
                report["warnings"].append(f"{filename}: Çok kısa içerik ({len(data['content'])} karakter)")

            report["valid_files"] += 1

        except Exception as e:
            report["errors"].append(f"{filename}: Okuma hatası -> {str(e)}")

    return report

if __name__ == "__main__":
    import sys
    # Windows konsolunda emoji yazdırma hatasını önlemek için stdout'u UTF-8 yapıyoruz.
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    # Colab uyumu için farklı yolları dene
    paths = ["data/normalized", os.path.join(os.getcwd(), "data/normalized"), "/content/data/normalized"]
    data_path = "data/normalized"
    for p in paths:
        if os.path.exists(p):
            data_path = p
            break
            
    print(f"🔍 Veri denetimi başlatılıyor: {data_path}")
    res = lint_data(data_path)
    
    print(f"\n📊 Özet Rapor:")
    print(f"- Toplam Dosya: {res['total_files']}")
    print(f"- Geçerli: {res['valid_files']}")
    print(f"- Hatalı: {len(res['errors'])}")
    print(f"- Uyarı: {len(res['warnings'])}")

    if res["errors"]:
        print("\n❌ KRİTİK HATALAR:")
        for err in res["errors"][:10]:
            print(f"  - {err}")

    if res["warnings"]:
        print("\n⚠️ UYARILAR:")
        for warn in res["warnings"][:5]:
            print(f"  - {warn}")
