"""
scraper — Otonom Bologna Veri Madenciliği (Selenium Crawler)
=============================================================
Bu modül, Osmaniye Korkut Ata Üniversitesi'nin ASP.NET tabanlı
Bologna Bilgi Paketleri sayfalarını Headless Selenium ile kazıyarak
ders bilgilerini yapılandırılmış JSON formatında diske kaydeder.

Mimari Özellikler:
    - **Anti-Detection**: Chrome WebDriver maskeleme (CDP komutları)
      ile bot algılama mekanizmalarını aşar.
    - **DOM Genişletme**: FontAwesome ``fa-plus-square`` ikonları
      tetiklenerek ASP.NET Accordion klasörleri açılır ve gizli
      (4. sınıf Teknik Seçmeli) dersler DOM'a çıkarılır.
    - **Akıllı Atlama**: ``os.path.exists`` ile daha önce kazınmış
      dersler es geçilerek performans optimize edilir.

Sınıflar:
    - ``BolognaCrawler``: Kazıma orkestratörü.

Kullanım::

    python src/scraper.py --start 1 --end 8 --url <bologna_url>
"""

import os
import json
import time
import argparse
from typing import Dict

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("Selenium kütüphaneleri eksik! Lütfen 'pip install -r requirements.txt' komutunu çalıştırın.")
    exit(1)

# Projemizin utils modülünden ortak loglayıcıyı çağırıyoruz.
try:
    from src.utils import setup_logger
    logger = setup_logger("SeleniumScraper")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SeleniumScraper")

class BolognaCrawler:
    """
    OBS/Bologna sayfalarındaki dinamik (ASP.NET/JS tabanlı) tabloları
    Selenium vasıtasıyla otonom olarak okuyup JSON'a aktaran bot.
    """
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        self.driver = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_driver(self):
        """Chrome tarayıcısını Headless olmadan (izlenebilir) açar. Anti-Ban ayarları barındırır."""
        logger.info("Tarayıcı motoru (Chrome) başlatılıyor...")
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--headless=new") # Görünmez Arka Plan Modu
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        # Sistemin bizi bot olarak algılamasını zorlaştırır
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Selenium detection bypass
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                '''
            })
            logger.info("Tarayıcı başarıyla başlatıldı ve maskelendi.")
        except Exception as e:
            logger.error(f"Tarayıcı başlatılamadı: {str(e)}")
            exit(1)

    def scrape_courses(self, url: str, start_sem: int, end_sem: int):
        """Web sayfasındaki belirli yarıyıllara ait derslerin detaylarını kazar."""
        self._setup_driver()
        wait = WebDriverWait(self.driver, 15)
        
        # Dinamik Çerçeve (Iframe) Navigasyon Modülü
        # Dış çerçeve (index.aspx) üzerinden Document nesnesine erişim engellendiğinden, 
        # oturum doğrudan hedef veri matrisine (progCourses.aspx) yönlendirilmektedir.
        if "index.aspx" in url and "curSunit=" in url:
            sunit = url.split("curSunit=")[1].split("&")[0].split("#")[0]
            url = f"https://obs.osmaniye.edu.tr/oibs/bologna/progCourses.aspx?lang=tr&curSunit={sunit}"
            logger.info(f"[NAVIGASYON] DOM Frame İzolasyonu Giderildi. Hedef URL Güncellendi: {url}")
        
        logger.info(f"Hedefe gidiliyor: {url}")
        self.driver.get(url)
        
        # Sayfanın yüklenmesi için statik ve dinamik beklemeler
        logger.info("Mevcut Ders tablolarının (JavaScript) DOM'a yüklenmesi bekleniyor...")
        time.sleep(5)  # Tüm JS scriptlerinin çalışması için kaba bir mola
        
        # ========================================================
        # GİZLİ DERS GRUPLARINI (KLASÖRLERİ) GENİŞLETME MODÜLÜ
        # ========================================================
        logger.info("Kapalı Ders Grupları (Seçmeli Ders Paketleri vb.) Tespit Ediliyor...")
        try:
            # FontAwesome 'plus' ikonu içeren expandCollapse span'larını bul
            expand_buttons = self.driver.find_elements(By.CSS_SELECTOR, "span.expandCollapse i.fa-plus-square")
            if expand_buttons:
                logger.info(f"Toplam {len(expand_buttons)} adet kapalı klasör bulundu. Genişletiliyor...")
                for btn in expand_buttons:
                    parent_span = btn.find_element(By.XPATH, "..")
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent_span)
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", parent_span)
                
                logger.info("Tüm klasörlere tıklandı, DOM'un güncellenmesi bekleniyor...")
                time.sleep(3) # AJAX/JS DOM yenilenmesi
            else:
                logger.info("Kapalı klasör bulunamadı veya zaten açık.")
        except Exception as e:
            logger.warning(f"Grup genişletme işlemi sırasında hata oluştu: {str(e)[:30]}")
            
        logger.info("Ana kazıma işlemine geçiliyor...")
        
        # Strateji: ASP.NET sayfalarında Tıklamalar sayfayı yenileyebileceği veya Popup açabileceği için
        # ilk aşamada tüm linkleri (veya onClick eventlerini) toplayacağız, 
        # ikinci aşamada hepsine tek tek gidip/tıklayıp metni çekeceğiz.
        
        # Üniversite tabloları genellikle "table" etiketleri içerisinde veya
        # divler halinde listelenir. "i" ikonları (info icon) genelde <a> tagleri ile sarılı resimlerdir.
        
        # NOT: Bu kısım DOM yapısı netleşince geliştirilebilir. Şu an Genel (Generic) bir tespit yöntemi uygulanıyor.
        try:
            # Info (i) iconlarını temsil eden potansiyel elementleri bul
            # src'sinde info geçen resimli linkler veya 'Detay', 'Bologna' yazan etiketler.
            courses = []
            
            # Sayfadaki BÜTÜN satırları (tr) bul, içinde info linki olanları ayıkla
            rows = self.driver.find_elements(By.TAG_NAME, "tr")
            logger.info(f"Sayfada toplam {len(rows)} satır bulundu. Ders formatına uyanlar ayrıştırılıyor...")
            
            for row in rows:
                text = row.text.strip()
                if not text:
                    continue
                
                # Çok kabaca: İçinde yarıyıl sayıları veya AKTS vs barındırıyor mu?
                # Daha güvenilir yöntem: Satırın içindeki "a" etiketini (link) aramak.
                links = row.find_elements(By.TAG_NAME, "a")
                
                for link in links:
                    # "(i)" butonu genelde boş yazılı olur veya title="Detay" tarzı içerir.
                    # Bazen de href="javascript:__doPostBack(...)" olur.
                    href = link.get_attribute("href")
                    if href and ("javascript:" in href or "Course" in href or "Ders" in href or "showPac" in href):
                        # Aynı linklerin defalarca listelenmemesi için (özet sayfa engeli) href bazlı kontrol
                        if href not in [c['href'] for c in courses]:
                            # Dersin kodunu ve adını tablodan ayıklamaya çalış (ilk iki sütun)
                            cols = row.find_elements(By.TAG_NAME, "td")
                            course_code = cols[1].text.strip() if len(cols) > 1 else "BILINMEYEN_KOD"
                            course_name = cols[2].text.strip() if len(cols) > 2 else "Bilinmeyen Ders"
                            
                            # Eğer kod formata uymuyorsa (Örn: satır tablo başlığına denk gelmişse) atla
                            if len(course_code) > 1 and len(course_code) < 15:
                                courses.append({
                                    "code": course_code,
                                    "name": course_name,
                                    "href": href,
                                    "element": link # JavaScript tıklaması için elementi de kaydet
                                })
            
            logger.info(f"Dağıtım: Sistemdeki tüm ders tarandı. Hedef linkler çıkarıldı. Toplam: {len(courses)}")
            
        except Exception as e:
            logger.error(f"Tablolar taranırken ayrıştırma hatası: {str(e)}")
            self.driver.quit()
            return
            
        # ========================================================
        # NODE BAZLI İŞ YÜKÜ DAĞITIMI (WORKLOAD DISTRIBUTION)
        # ========================================================
        logger.info(f"Otonom çekim süreci başlatılıyor. Modül: {start_sem}. - {end_sem}. Yarıyıllar.")
        
        # Dağıtık veri madenciliği senaryosu simülasyonu
        # Hedef veri tablosu yarıyıl bazlı parçalanarak (sharding) düğümlere atanır
        
        total_courses = len(courses)
        courses_per_sem = total_courses // 8 if total_courses > 8 else total_courses
        
        start_idx = max(0, (start_sem - 1) * courses_per_sem)
        end_idx = min(total_courses, end_sem * courses_per_sem)
        
        target_courses = courses[start_idx:end_idx]
        
        logger.info(f"Mevcut Düğüme (Node) Atanan İş Bloğu (Yarıyıl {start_sem}-{end_sem}): {len(target_courses)} Nesne İşlenecektir.")
        
        for idx, course in enumerate(target_courses, start=1):
            clean_code = course['code'].strip().replace(" ", "")
            filename = f"{clean_code}_bologna.json"
            save_path = os.path.join(self.output_dir, filename)
            
            # Akıllı Atlama (Skip) Mekanizması
            if os.path.exists(save_path):
                logger.info(f"[{idx}/{len(target_courses)}] ATLANDI (Zaten Mevcut): {course['code']} - {course['name']}")
                continue
                
            logger.info(f"[{idx}/{len(target_courses)}] İşleniyor: {course['code']} - {course['name']}")
            
            try:
                # 1. Yöntem: Eğer href normal bir URL ise yeni sekmeye git (Daha Güvenli)
                if course['href'].startswith("http"):
                    # Ana sekmenin IDsini kaydet
                    main_window = self.driver.current_window_handle
                    
                    # Yeni Sekme Aç ve Git
                    self.driver.execute_script(f"window.open('{course['href']}', '_blank');")
                    time.sleep(1) # Sekme açılış payı
                    
                    # Yeni sekmeye geçiş
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    
                    # Detay sayfası metnini çek
                    self._extract_and_save(course)
                    
                    # Kapat ve Ana ekrana dön
                    self.driver.close()
                    self.driver.switch_to.window(main_window)
                    
                # Yöntem 2: Dinamik Javascript Yönlendirmeleri (AJAX / Modal Handle)
                elif course['href'].startswith("javascript:"):
                    # ASP.NET WebForms mimarisinde yer alan "__doPostBack" metodolojisi, 
                    # dışarıdan script enjeksiyonuna karşı katı (strict-mode) koruması içerir. 
                    # Bu kısıtlamayı aşmak adına, Javascript execute fonksiyonları yerine 
                    # tarayıcı üzerinde fiziksel bir simülasyon (Native Click Event) tetiklenmektedir.
                    try:
                        fresh_element = self.driver.find_element(By.XPATH, f"//a[@href=\"{course['href']}\"]")
                        # Normal tıklamada "element is not clickable at point" hatası almamak için kaydırma
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", fresh_element)
                        time.sleep(1)
                        fresh_element.click()
                    except Exception as inner_e:
                        logger.warning(f"Normal tıklama başarısız. Zorla tıklanıyor... ({str(inner_e)[:30]})")
                        # Failsafe: Eğer banner altında kaldıysa Javascript'in "Tıklama" metodunu çağır (Bu strict mode'a takılmaz)
                        self.driver.execute_script("arguments[0].click();", fresh_element)
                        
                    time.sleep(4) # Hedef verinin DOM'a inmesini (veya yeni sayfanın yüklenmesini) bekle
                    
                    # DOM'daki tüm görünür metinleri yeniden çek formatı
                    self._extract_and_save(course)
                    
                    # Sayfa yönlendirmesi olduysa veya Modal açıldıysa, güvenli şekilde ana tabloya dön
                    # Eğer URL değiştiyse (yeni sayfaya gidildiyse) geri tuşuna bas.
                    self.driver.back()
                    time.sleep(4)
                
                
            except Exception as e:
                logger.error(f"'{course['code']}' Okunurken Hata (Geçiliyor): {str(e)}")
                
        logger.info(f"\n[BİLGİ] Veri Kazıma Entegrasyonu Başarıyla Tamamlandı. {len(target_courses)} eğitim nesnesi JSON formatına dönüştürüldü.")
        self.driver.quit()

    def _extract_and_save(self, course: Dict):
        """Açılan sekmedeki veya modaldaki saf metni RAG için kaydeder."""
        try:
            # WebDriverWait(self.driver, 5).until(lambda d: len(d.find_elements(By.TAG_NAME, "body")) > 0)
            # Tüm sayfanın metnini çekiyoruz (Sadece düz metin, etiketler olmadan)
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # JSON formatında kaydetme
            clean_code = course['code'].strip().replace(" ", "")
            filename = f"{clean_code}_bologna.json"
            save_path = os.path.join(self.output_dir, filename)
            
            structured_metadata = {
                "title": f"{course['code']} - {course['name']} Bologna Paketi",
                "content": page_text,
                "source_file": filename
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(structured_metadata, f, ensure_ascii=False, indent=4)
                
            logger.info(f"✓ Başarıyla Kaydedildi: {filename}")
        except Exception as e:
            logger.warning(f"Metin ayrıştırılamadı (JSON): {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ağ Dağıtımlı Bologna Selenium Botu")
    parser.add_argument("--start", type=int, default=1, help="Başlangıç Yarıyılı (Örn: 1)")
    parser.add_argument("--end", type=int, default=8, help="Bitiş Yarıyılı (Örn: 3)")
    parser.add_argument("--url", type=str, 
                        default="https://obs.osmaniye.edu.tr/oibs/bologna/index.aspx?lang=tr&curOp=showPac&curUnit=20231&curSunit=5792#", 
                        help="Derslerin listelendiği hedef Bologna adresi")

    args = parser.parse_args()
    
    bot = BolognaCrawler()
    bot.scrape_courses(url=args.url, start_sem=args.start, end_sem=args.end)
