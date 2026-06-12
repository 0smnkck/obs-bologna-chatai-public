"""
normalize_json — Bologna JSON Normalizer (OOP)
================================================
Ham scraper çıktısı olan JSON dosyalarını, LLM'in anlayabileceği
düz-dil, anahtar-değer formatına dönüştürür.

Dönüşüm Örneği::

    ÖNCE:  "1 BMB107 Bilgisayar Mühendisliğine Giriş 2+0+0 2 5 24.10.2025"
    SONRA: "Ders Adı: Bilgisayar Mühendisliğine Giriş
            Ders Kodu: BMB107 | Kredi: 2 | AKTS: 5 | Yarıyıl: 1 | ..."

Sınıflar:
    - ``BolognaNormalizer``: Tüm parse, extract ve normalize işlemlerini
      kapsayan ana normalizer sınıfı.

Kullanım::

    python normalize_json.py
"""

import json
import os
import re
from typing import Optional, List


class BolognaNormalizer:
    """Ham Bologna JSON verilerini LLM-uyumlu formata dönüştüren normalizer.

    Bu sınıf, Deterministic Finite Automaton (DFA) tabanlı bir durum makinesi (state machine) 
    ve Düzenli İfadeler (Regex) heuristikleri kullanarak, yarı-yapılandırılmış 
    Bologna Bilgi Paketi verilerinden semantik bilgi çıkarımı yapar. Gürültülü 
    (noisy) scraper çıktıları filtrelenerek, LLM'lerin Retrieval-Augmented Generation 
    (RAG) katmanında yüksek doğrulukla (high precision) kullanabileceği, bilgi 
    kaybı minimize edilmiş düz-metin formata dönüştürülür.

    Attributes:
        input_dir: Ham JSON dosyalarının bulunduğu kaynak klasör.
        output_dir: Normalize edilmiş JSON'ların yazılacağı hedef klasör.
    """

    # ──────────────────────────────────────────────────────────────
    # BÖLÜM 1: YARDIMCI METOTLAR
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Baş/son boşlukları ve fazla whitespace'i temizler.

        Args:
            text: Temizlenecek ham metin.

        Returns:
            Fazla boşlukları kaldırılmış, trim edilmiş metin.
        """
        return " ".join(text.split()).strip()

    @staticmethod
    def _parse_header_line(lines: list[str]) -> Optional[dict]:
        """Ders başlık satırını Regex heuristikleri ile yapısal formata dönüştürür.

        Extraction işlemi şu RegEx deseni üzerinden O(1) zaman karmaşıklığı ile çalışır:
        `^(\d+)\s+([A-ZÇŞÜÖİĞ]{2,4}\s*\d+[A-Z]?)\s+(.+?)\s+(\d+\+\d+\+\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)$`
        Bu sayede yarıyıl, ders kodu, ders adı, kredi (AKTS) gibi kritik veriler 
        sabit indekslerle (O(1)) güvenilir bir şekilde parse edilir.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            Parse edilen alanları içeren sözlük veya uygun başlık bulunamazsa ``None``.
        """
        pattern = re.compile(
            r'^(\d+)\s+'
            r'([A-ZÇŞÜÖİĞ]{2,4}\s*\d+[A-Z]?)\s+'
            r'(.+?)\s+'
            r'(\d+\+\d+\+\d+)\s+'
            r'(\d+)\s+'
            r'(\d+)\s+'
            r'([\d.]+)$'
        )
        for line in lines[:10]:
            m = pattern.match(line.strip())
            if m:
                tul = m.group(4).split('+')
                return {
                    "yariyil": m.group(1),
                    "ders_kodu": BolognaNormalizer._clean(m.group(2)),
                    "ders_adi": BolognaNormalizer._clean(m.group(3)),
                    "teori_saati": tul[0],
                    "uygulama_saati": tul[1],
                    "lab_saati": tul[2],
                    "kredi": m.group(5),
                    "akts": m.group(6),
                    "son_guncelleme": m.group(7),
                }
        return None

    @staticmethod
    def _extract_kv_fields(lines: list[str]) -> dict:
        """Ders detayları bölümündeki anahtar-değer çiftlerini state-machine ile çıkarır.

        Belirli anahtar kelimeleri (trigger keys) tespit eden ve bölüm sonlandırıcı 
        (terminator) token'ları görene kadar ardışık metinleri biriktiren lineer O(N) 
        zaman karmaşıklığına sahip bir durum makinesi (state machine) kullanır.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            Anahtar-değer çiftlerini içeren sözlük.
        """
        kv_keys = [
            "Dersin Dili", "Dersin Düzeyi", "Bölümü / Programı",
            "Öğrenim Türü", "Dersin Türü", "Dersin Öğretim Şekli",
            "Dersin Amacı", "Dersin İçeriği", "Dersin Yöntem ve Teknikleri",
            "Ön Koşulları", "Dersin Koordinatörü", "Dersi Verenler",
            "Dersin Yardımcıları", "Dersin Staj Durumu",
        ]

        result = {}
        current_key = None
        current_val_lines = []

        section_terminators = {
            "Ders Kaynakları", "Ders Yapısı", "Planlanan Öğrenme Aktiviteleri",
            "Değerlendirme Ölçütleri", "AKTS Hesaplama", "Dersin Öğrenme Çıktıları",
            "Ders Konuları", "Sürdürülebilir Kalkınma", "Dersin Program Çıktılarına",
        }

        for line in lines:
            stripped = line.strip()

            if any(stripped.startswith(t) for t in section_terminators):
                if current_key:
                    result[current_key] = BolognaNormalizer._clean(" ".join(current_val_lines))
                    current_key = None
                    current_val_lines = []
                break

            matched_key = None
            for key in kv_keys:
                if stripped.startswith(key + " ") or stripped.startswith(key + "\t"):
                    matched_key = key
                    break

            if matched_key:
                if current_key:
                    result[current_key] = BolognaNormalizer._clean(" ".join(current_val_lines))
                current_key = matched_key
                current_val_lines = [stripped[len(matched_key):].strip()]
            elif current_key and stripped:
                current_val_lines.append(stripped)

        if current_key and current_val_lines:
            result[current_key] = BolognaNormalizer._clean(" ".join(current_val_lines))

        return result

    @staticmethod
    def _extract_resources(lines: list[str]) -> dict:
        """'Ders Kaynakları' bölümündeki kitap/not listelerini O(N) karmaşıklıkla çıkarır.

        Durum makinesi (state machine), "Ders Kaynakları" token'ı ile tetiklenir (in_section=True) 
        ve kaynak metinlerini sınıflandırarak listelere ayırır.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            ``{"kitaplar": [...], "ders_notlari": [...]}`` formatında sınıflandırılmış sözlük.
        """
        resources = {"kitaplar": [], "ders_notlari": []}
        in_section = False

        section_end = {
            "Ders Yapısı", "Planlanan Öğrenme", "Değerlendirme",
            "AKTS Hesaplama", "Dersin Öğrenme", "Ders Konuları",
        }

        for line in lines:
            stripped = line.strip()
            if stripped == "Ders Kaynakları":
                in_section = True
                continue
            if not in_section:
                continue
            if any(stripped.startswith(e) for e in section_end):
                break

            if stripped.startswith("Kaynaklar "):
                val = stripped[len("Kaynaklar "):].strip()
                if val:
                    resources["kitaplar"].append(val)
            elif stripped.startswith("Ders Notları "):
                val = stripped[len("Ders Notları "):].strip()
                if val:
                    resources["ders_notlari"].append(val)
            elif stripped and in_section and not stripped.startswith("Ders Kaynakları"):
                resources["kitaplar"].append(stripped)

        return resources

    @staticmethod
    def _extract_evaluation(lines: list[str]) -> List[dict]:
        """'Değerlendirme Ölçütleri' bölümündeki sınav/ödev oranlarını çıkarır.

        Düzenli ifadeler kullanılarak, `^(.+?)\s+(\d+)\s+%\s*(\d+)$` örüntüsü ile 
        sınav türü, adedi ve yüzde katkı payı ayrıştırılır.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            Değerlendirme ölçütlerini (tür, sayı, katkı) içeren sözlük listesi.
        """
        items = []
        in_section = False
        eval_pattern = re.compile(r'^(.+?)\s+(\d+)\s+%\s*(\d+)$')

        section_end = {"AKTS Hesaplama", "Dersin Öğrenme", "Ders Konuları", "Sürdürülebilir"}

        for line in lines:
            stripped = line.strip()
            if "Değerlendirme Ölçütleri" in stripped:
                in_section = True
                continue
            if not in_section:
                continue
            if any(stripped.startswith(e) for e in section_end):
                break

            m = eval_pattern.match(stripped)
            if m and m.group(1) not in ("Yarıyıl Çalışmaları", "Toplam"):
                items.append({
                    "tur": BolognaNormalizer._clean(m.group(1)),
                    "sayi": m.group(2),
                    "katki": f"%{m.group(3)}"
                })

        return items

    @staticmethod
    def _extract_learning_outcomes(lines: list[str]) -> List[str]:
        """'Dersin Öğrenme Çıktıları' bölümündeki maddeleri ayrıştırır.

        Liste ögelerini Regex `^(\d+)\s+(.+)$` kullanarak tespit eder ve indeksleme 
        hatalarını önlemek için temizler.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            Öğrenme çıktısı metinlerinin listesi.
        """
        outcomes = []
        in_section = False
        outcome_pattern = re.compile(r'^(\d+)\s+(.+)$')

        section_end = {"Ders Konuları", "Sürdürülebilir", "Dersin Program"}

        for line in lines:
            stripped = line.strip()
            if "Dersin Öğrenme Çıktıları" in stripped:
                in_section = True
                continue
            if not in_section:
                continue
            if stripped in ("Sıra No Açıklama",):
                continue
            if any(stripped.startswith(e) for e in section_end):
                break

            m = outcome_pattern.match(stripped)
            if m:
                outcomes.append(BolognaNormalizer._clean(m.group(2)))

        return outcomes

    @staticmethod
    def _strip_source_from_topic(raw: str) -> str:
        """Konu+kaynak karışımı satırlarından yalnızca konu kısmını heuristik olarak çıkarır.

        Eğitim materyallerinde sık rastlanan yazar adları, yayıncılar ve akademik referans 
        kalıpları (Örn: Tanenbaum, Cormen, W3Schools) Regex tabanlı bir kara liste (blacklist) 
        ile filtrelenerek, salt bilgi çıkarımı sağlanır.

        Args:
            raw: Konu + olası kaynak bilgisi içeren ham metin.

        Returns:
            Yalnızca konu kısmını içeren, referanslardan arındırılmış temizlenmiş metin.
        """
        kaynak_patterns = [
            r'\b[A-ZÇŞÜÖİĞ][a-zçşüöığ]{2,},\s+[A-ZÇŞÜÖİĞ]\.',
            r'\bÖğretim Üyesi\b',
            r'\bDers [Nn]otları?\b',
            r'\bKurum ve\b',
            r'https?://',
            r'\bW3Schools\b',
            r'\bMDN\b',
            r'\bToros Rifat\b',
            r'\bDuckett\b',
            r'\bTanenbaum\b',
            r'\bSilberschatz\b',
            r'\bForouzan\b',
            r'\bKleinberg\b',
            r'\bKnuth\b',
            r'\bCormen\b',
        ]
        earliest = len(raw)
        for p in kaynak_patterns:
            m = re.search(p, raw)
            if m and m.start() > 5:
                earliest = min(earliest, m.start())
        return raw[:earliest].strip().rstrip(',').strip() or raw

    @staticmethod
    def _extract_weekly_topics(lines: list[str]) -> List[str]:
        """'Ders Konuları' bölümündeki haftalık konuları çıkarır.

        Kaynak/ön hazırlık bilgilerini konudan ayırır.

        Args:
            lines: Dosyanın satır listesi.

        Returns:
            ``"Hafta N: Konu"`` formatında metin listesi.
        """
        topics = []
        in_section = False
        topic_pattern = re.compile(r'^(\d+)\s+(.+)$')

        section_end = {"Sürdürülebilir", "Dersin Program", "https://"}

        for line in lines:
            stripped = line.strip()
            if stripped == "Ders Konuları" or "  Ders Konuları" in line:
                in_section = True
                continue
            if not in_section:
                continue
            if stripped in ("Hafta Konu Ön Hazırlık Dökümanlar",):
                continue
            if any(stripped.startswith(e) for e in section_end):
                break

            m = topic_pattern.match(stripped)
            if m and int(m.group(1)) <= 16:
                konu = BolognaNormalizer._strip_source_from_topic(m.group(2))
                topics.append(f"Hafta {m.group(1)}: {konu}")

        return topics

    # ──────────────────────────────────────────────────────────────
    # BÖLÜM 2: ANA NORMALIZE METODU
    # ──────────────────────────────────────────────────────────────

    def normalize_content(self, raw_content: str, title: str) -> tuple[str, dict, dict]:
        """Ham content string'ini LLM için okunabilir metin bloğuna dönüştürür.

        Tüm extract metotlarını sırayla çağırır ve sonuçları yapılandırılmış
        bir metin formatında birleştirir.

        Args:
            raw_content: Scraper'dan gelen ham sayfa metni.
            title: Ders başlığı (header parse edilemezse yedek bilgi).

        Returns:
            Tuple: (normalize_content_str, metadata_dict, sections_dict)
        """
        lines = raw_content.split('\n')

        header = self._parse_header_line(lines)
        kv = self._extract_kv_fields(lines)
        resources = self._extract_resources(lines)
        evaluation = self._extract_evaluation(lines)
        outcomes = self._extract_learning_outcomes(lines)
        topics = self._extract_weekly_topics(lines)

        # ── Normalize edilmiş metni oluştur ──
        parts = []

        # Başlık bloğu
        parts.append(f"=== DERS BİLGİ PAKETİ ===")
        if header:
            parts.append(f"Ders Adı: {header['ders_adi']}")
            parts.append(f"Ders Kodu: {header['ders_kodu']}")
            parts.append(f"Yarıyıl: {header['yariyil']}. Yarıyıl")
            parts.append(f"Kredi: {header['kredi']}")
            parts.append(f"AKTS Kredisi: {header['akts']}")
            parts.append(f"Ders Saati: Teori={header['teori_saati']}, Uygulama={header['uygulama_saati']}, Lab={header['lab_saati']}")
            parts.append(f"Son Güncelleme: {header['son_guncelleme']}")
        else:
            parts.append(f"Ders Başlığı: {title}")

        parts.append("")

        # Ders detayları
        parts.append("--- DERS DETAYLARI ---")
        detail_map = {
            "Dersin Dili": "Dil",
            "Dersin Düzeyi": "Düzey",
            "Bölümü / Programı": "Program",
            "Öğrenim Türü": "Öğrenim Türü",
            "Dersin Türü": "Ders Türü",
            "Dersin Öğretim Şekli": "Öğretim Şli",
            "Ön Koşulları": "Ön Koşul",
            "Dersin Koordinatörü": "Koordinatör",
            "Dersi Verenler": "Öğretim Üyesi",
            "Dersin Staj Durumu": "Staj Durumu",
        }
        # Repair typo in detail_map keys/values if needed - wait, Dersin Öğretim Şekli is Öğretim Şekli
        detail_map["Dersin Öğretim Şekli"] = "Öğretim Şekli"
        
        for raw_key, label in detail_map.items():
            if raw_key in kv and kv[raw_key] and kv[raw_key] != "Yok":
                parts.append(f"{label}: {kv[raw_key]}")

        parts.append("")

        # Amaç ve içerik
        if "Dersin Amacı" in kv and kv["Dersin Amacı"]:
            parts.append("--- DERS AMACI ---")
            parts.append(kv["Dersin Amacı"])
            parts.append("")

        if "Dersin İçeriği" in kv and kv["Dersin İçeriği"]:
            parts.append("--- DERS İÇERİĞİ ---")
            parts.append(kv["Dersin İçeriği"])
            parts.append("")

        if "Dersin Yöntem ve Teknikleri" in kv and kv["Dersin Yöntem ve Teknikleri"]:
            parts.append(f"Öğretim Yöntemleri: {kv['Dersin Yöntem ve Teknikleri']}")
            parts.append("")

        # Kaynaklar
        if resources["kitaplar"] or resources["ders_notlari"]:
            parts.append("--- DERS KAYNAKLARI ---")
            for kitap in resources["kitaplar"]:
                parts.append(f"• {kitap}")
            for not_ in resources["ders_notlari"]:
                parts.append(f"• Ders Notu: {not_}")
            parts.append("")

        # Değerlendirme
        if evaluation:
            parts.append("--- DEĞERLENDİRME ---")
            for item in evaluation:
                parts.append(f"{item['tur']}: {item['sayi']} adet, katkı oranı {item['katki']}")
            parts.append("")

        # Öğrenme çıktıları
        if outcomes:
            parts.append("--- ÖĞRENME ÇIKTILARI ---")
            for i, outcome in enumerate(outcomes, 1):
                parts.append(f"{i}. {outcome}")
            parts.append("")

        # Haftalık konular
        if topics:
            parts.append("--- HAFTALIK KONU PLANI ---")
            for topic in topics:
                parts.append(topic)
            parts.append("")

        normalized_text = "\n".join(parts)

        # metadata dictionary
        metadata = {}
        if header:
            metadata["ders_kodu"] = header.get("ders_kodu", "").replace(" ", "")
            metadata["ders_adi"] = header.get("ders_adi", "")
            metadata["yariyil"] = header.get("yariyil", "")
            metadata["kredi"] = header.get("kredi", "")
            metadata["akts"] = header.get("akts", "")
            metadata["son_guncelleme"] = header.get("son_guncelleme", "")
        else:
            code_match = re.search(r'\b[A-Z]{3,4}\s*\d{3}\b', title)
            metadata["ders_kodu"] = code_match.group(0).replace(" ", "") if code_match else ""
            metadata["ders_adi"] = title
            metadata["yariyil"] = ""
            metadata["kredi"] = ""
            metadata["akts"] = ""
            metadata["son_guncelleme"] = ""
        metadata["ogretim_uyesi"] = kv.get("Dersi Verenler", "")

        # sections dictionary
        sections = {
            "amac": kv.get("Dersin Amacı", ""),
            "icerik": kv.get("Dersin İçeriği", ""),
            "yontemler": kv.get("Dersin Yöntem ve Teknikleri", ""),
            "kaynaklar": resources,
            "degerlendirme": evaluation,
            "ogrenme_cikti": outcomes,
            "haftalik_plan": topics
        }

        return normalized_text, metadata, sections

    # ──────────────────────────────────────────────────────────────
    # BÖLÜM 3: TOPLU İŞLEM
    # ──────────────────────────────────────────────────────────────

    def normalize_all(self, input_dir: str, output_dir: str) -> None:
        """Belirtilen klasördeki tüm JSON dosyalarını normalize eder.

        Her dosya okunur, ``normalize_content`` ile dönüştürülür ve
        hedef klasöre aynı isimle yazılır. Başarı/hata istatistikleri
        terminale yazdırılır.

        Args:
            input_dir: Ham JSON dosyalarının bulunduğu kaynak klasör.
            output_dir: Normalize edilmiş dosyaların yazılacağı hedef klasör.
        """
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        success = 0
        errors = []

        for fname in files:
            try:
                input_path = os.path.join(input_dir, fname)
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                raw_content = data.get("content", "")
                title = data.get("title", fname.replace('_bologna.json', ''))

                normalized, metadata, sections = self.normalize_content(raw_content, title)

                new_data = {
                    "title": title,
                    "metadata": metadata,
                    "sections": sections,
                    "content": normalized,
                    "source_file": data.get("source_file", ""),
                    "_normalized": True,
                    "_version": "2.0"
                }

                output_path = os.path.join(output_dir, fname)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

                success += 1

            except Exception as e:
                errors.append((fname, str(e)))

        print(f"\n[BASARILI] Basariyla normalize edilen: {success}/{len(files)} dosya")
        if errors:
            print(f"[HATA] Hatali dosyalar:")
            for fname, err in errors:
                print(f"   - {fname}: {err}")


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    BASE = os.getcwd()
    INPUT = os.path.join(BASE, "data", "processed")
    OUTPUT = os.path.join(BASE, "data", "normalized")

    print("Bologna JSON Normalizer başlatılıyor...")
    print(f"Kaynak: {INPUT}")
    print(f"Hedef:  {OUTPUT}")
    print()

    normalizer = BolognaNormalizer()
    normalizer.normalize_all(INPUT, OUTPUT)

    # Örnek çıktı göster
    sample_file = os.path.join(OUTPUT, "bmb107_bologna.json")
    if os.path.exists(sample_file):
        with open(sample_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        print("\n─── ÖRNEK ÇIKTI (BMB107) ───")
        print(d["content"])
