# Emprime / Serigrafi Renk (Kanal) Ayırma Rehberi

> **Okuma Kuralı:** Fihristten ilgili başlığı (`## X.`) bulun ve yalnızca o bölümü okuyun.

## Alfabetik Bölüm ve Metot Fihristi (A-Z)
| Başlık | Satır Aralığı | İçerik Özeti |
| :--- | :--- | :--- |
| [A. Alt Baskı (Underbase), Üst Beyaz ve Choke/Spread Kuralları](#a-alt-baskı-underbase-üst-beyaz-ve-chokespread-kuralları) | L30-L44 | Koyu kumaş beyaz taban, choke 2–3 px, üst beyaz ve siyah altı fon ayrımı |
| [B. AM Tramlama, Dot-on-Dot Yığılma ve Dot Gain Kuralları](#b-am-tramlama-dot-on-dot-yığılma-ve-dot-gain-kuralları) | L48-L60 | 45 LPI, 15°, yuvarlak nokta, dot-on-dot overprint ve sürekli ton tavanı |
| [C. Atölye Standartları ve Makine Kısıtları](#c-atölye-standartları-ve-makine-kısıtları) | L64-L68 | Maksimum 6 kafa, zıt renk kısıtları ve kurutma süreleri |
| [D. Dithering ve Difüzyon Yöntemleri](#d-dithering-ve-difüzyon-yöntemleri) | L72-L76 | Index color, 254 DPI dither + 2x büyütme standardı |
| [E. Doğrulanmayanlar ve Sınırlar](#e-doğrulanmayanlar-ve-sınırlar) | L80-L88 | Fiziksel film, elek tutma, gerçek moiré ve ölçüm sınırları |
| [F. Film Çıktısı Standartları](#f-film-çıktısı-standartları) | L92-L105 | 1-bit bitmap, pHYs etiketi ve dosya adlandırma |
| [G. İş Akışı: Tek Aşamalı vs İki Aşamalı Yöntem](#g-iş-akışı-tek-aşamalı-vs-iki-aşamalı-yöntem) | L109-L121 | bolge_am.py vs hibrit2.py adım adım yürütme |
| [H. Kalite Kontrol ve Hata Ölçümü](#h-kalite-kontrol-ve-hata-ölçümü) | L125-L142 | ΔE ≤ 12, kenar pürüzlülüğü ≤ 0.03 ve kalite_kontrol.py |
| [I. Kaynaklar ve Literatür](#i-kaynaklar-ve-literatür) | L146-L158 | MYK, Ryonet, Fresener, Anatol ve ISO-13660 standartları |
| [J. Tonlu Nesne ve Alt Taban Çalışma Stili](#j-tonlu-nesne-ve-alt-taban-çalışma-stili) | L162-L169 | Ton dilimli spot kanallar, sert alt taban ve şablon mantığı |
| [K. Önerilen Boru Hattı ve Ön Analiz Karar Eşikleri](#k-önerilen-boru-hattı-ve-ön-analiz-karar-eşikleri) | L173-L225 | 5 adımlı Python pipeline kodu ve analiz.py karar eşikleri |
| [L. Photoshop JSX Otomasyon Araçları](#l-photoshop-jsx-otomasyon-araçları) | L229-L240 | psd_olustur_tek_adim.jsx ile tek tıkla Multichannel PSD |
| [M. Sınama Sonuçları (Sentetik 2000×2000 px, 508 PPI)](#m-sınama-sonuçları-sentetik-20002000-px-508-ppi) | L244-L259 | Raggedness, sahte saçak, choke ve dither sentetik test tablosu |
| [N. Terimler Sözlüğü](#n-terimler-sözlüğü) | L263-L283 | MYK tanımları, mesh, tram, LPI, spot, choke/spread ve flash cure |
| [O. Tramsız Düz Renk Ayrımı](#o-tramsız-düz-renk-ayrımı) | L287-L296 | tramsiz.py ile vektörel grafik ayrımı ve trapping |
| [P. Emprime Boya Kimyası ve Kumaş Eşleşmesi](#p-emprime-boya-kimyası-ve-kumaş-eşleşmesi) | L300-L316 | Beyazlı, Pat ve Şeffaf boya farkları, tıkanma, tuşe ve kumaş seçimi |

---

**Doğrulama durumu:** Sayılar sentetik görselde Python ile ölçüldü. Fiziksel film, gerçek elek, nokta tutma, gerçek moiré ve Photoshop'ta açılış **sınanmadı** (Bölüm E). Kaynakların çoğu ABD tişört serigrafi sektöründen (plastisol ağırlıklı).

---

## A. Alt Baskı (Underbase), Üst Beyaz ve Choke/Spread Kuralları

1. **Koyu Kumaş İhtiyacı ve Beyaz LAP Fon Zorunluluğu:** Koyu/renkli kumaşlarda üst renklerin canlı çıkması için alta opak beyaz astar (`fon` / underbase, `Solidity = 100%`) basılır; flash cure sonrası üst renkler basılır. Fon rengi istisnai durumlar (30 desende en fazla 2) hariç her zaman **beyaz** ve **`%100` LAP** olmalıdır: üst renklerin oturtulup zayıf kalmaması ve beyaz boyanın tramlı kalıpta koyulaşıp tıkanmaması için şarttır [usta]. Açık zeminli beyaz kumaşta fon basılmaz, kumaşın kendi rengi zemin kabul edilir.
2. **Choke (İçeri Daraltma) Standardı (699–702 atölye ölçümleriyle kesinleşti):**
   - **Standart choke:** **`2 px`** (`~0,10 mm @ 508 PPI`, ince detaylı ve genel işler; 699/700/701 ölçümü: `1,91 px`).
   - **Geniş düz (LAP) alan choke:** **`3 px`** (`~0,15 mm @ 508 PPI`; 700 ölçümü: `2,32–2,87 px`).
   - Yalnızca **dışa bakan kenarlarda** uygulanır; iç kenarlar daraltılmaz.
3. **İnce Çizgi ve Detay Choke İstisnası:**
   - 1 pt (`~7 px @ 508 PPI`) ve altındaki ince çizgilerde 3 px üstü choke alt tabanı eritir.
   - Kural: İnce detaylarda **choke `1 px`** uygulanır veya üst renk **spread `1–2 px`** (`0,05–0,10 mm`) şişirilir.
4. **Üst Beyaz (Highlight White / Çift Beyaz — `699.psd`):**
   - Koyu kumaşta (bordo, siyah, lacivert) tek kat `fon` beyazı canlı kalmaz. `fon` (tüm açık/renkli alan altı, `2 px choke`, `Solidity 100%`) üzerine yalnızca tasarımda **saf beyaz** görünecek alanlar için ayrı bir **`beyaz`** kanalı (`0 px choke`, `Solidity 16%`) eklenir.
5. **Siyah Altı Fon Kuralı (Kanal Türü ve Kumaşa Göre Ayrılır):**
   - **Tramlı siyah gölge/kontur:** Altından fon **OYULMAZ** (`701.psd`: siyah tramın `%95,75`'i `fon` üstündedir). Oyulursa ince siyah noktaların çevresinde kayıt kaçıklığı beyaz halkası çıkar.
   - **Geniş düz (LAP) koyu/siyah blok:** Kumaş koyu ve harf/blok görünürlüğü gerekiyorsa fon konur (`700.psd`: `1 fon`); kumaş açık/orta tondaysa veya koyu antrasit doğrudan kumaşa basılacaksa fon oyulur (`699.psd`: `antrasit ∩ fon = %2,08`; `700.psd`: `1 fon copy` varyantı). Belirsiz durumlarda çift fon varyantı (`1_fon` + `1_fon_koyusuz`) birlikte sunulabilir.

---

## B. AM Tramlama, Dot-on-Dot Yığılma ve Dot Gain Kuralları

1. **Atölye Tramlama Standartları (651, 654, 699, 700, 701 + 2026-11-19 usta değerlendirmesi):**
   - **Çözünürlük:** **508 PPI** (`200 dpcm`, `pHYs = 20000 px/m`).
   - **Tram Frekansı:** Spot renk ayrımında **`45 LPI`** (hücre periyodu `≈11,3–11,47 px @ 508 PPI`); CMYK'da (baskı dilinde **"fotoğraf baskı"**) **`55 LPI`** daha iyi sonuç verir [usta].
   - **Tram Açısı ve Nokta Formu:** **Yuvarlak (Round)** nokta; tüm kanallarda aynı açı ve aynı faz (`dot-on-dot`). **0° ve 90° kesinlikle yasaktır.** CMYK dışı spot ayrımlarda açı atölye/kişisel tercihe göre seçilebilir (`15°`, `45°`, **`72°`** — 699–701 atölyesi `15°`, Londra ustası `72°` kullanır) [atölye, usta].
2. **Bölge Saflığı + Aile İçi "Açık LAP + Üst %50–60 Koyu Tram" + Degrade "%85 Alt Bindirme":**
   - Farklı renk aileleri birbirine karışmaz (`< %1` çakışma; ör. `700`: yeşil ∩ mavi `%0,95`, `701`: kırmızı ∩ sarı `%0,56`).
   - **Aile İçi Doluluk Kuralı [usta]:** Tek başına tram alttan beyaz fon sızdırıp deseni zayıf bırakır. Canlı ve dolu baskı için her renk ailesinde altta açık ton **`%100` LAP** (veya LAP'a çok yakın yoğunluk) basılır, üstüne koyu ton tram olarak bindirilir. Altta LAP varken üst tram yoğunluğu ortalama **`%50–60`'ı geçmemelidir** (`%80` olursa aşırı koyulaşır; alt ton gereğinden fazla açık, üst tram çok yoğun olursa iki ton birbirinden kopar). Siyah tram için **CMYK `K`** kanalı yeterlidir.
   - **İki Renkli Degradede "%85 Alt Bindirme" (Underlap) [usta]:** İki renkli geçişte (ör. Kırmızı ↔ Sarı) aynı tram açısıyla renkler yalnız göründüğü kadar alınırsa orta geçişte beyaz fon parlayıp geçişi soldurur. Açık/taban renk (Sarı), komşu koyu rengin (Kırmızı) **en az `%85` dolu olduğu alana kadar altına bindirilerek** başlatılır; böylece beyaz fon kapanır ve geçiş dolu çıkar.
   - **Profesyonel Ayrım Önce, Renk Azaltma Kullanıcıda [usta]:** Baştan 6 kafaya sıkmak profesyonel baskıda eksik sonuç verir. Önce her ana renk ailesine **2 ton** (geniş/detaylı yüzeylerde **3 tona kadar**) + tek `fon` + tek `siyah` (ör. 8 kafa) üretilip renk düşürme işlemi kullanıcıya bırakılır.
   - Alternatif olarak piksel başına tek mürekkep istenen işlerde çakışmasız (indeks) AM (`bolge_am.py`) kullanılabilir.
3. **Kalıp İncelmesi (%5) & Baskı Yayılması (Dot Gain) Kompanzasyonu:**
   - Sürekli ton → 1-bit AM tram dönüşümü **1:1 doğrusal** yapılır (`diff = ±0,1%`).
   - Film → kalıp aşamasında (emülsiyon, ışık, yıkama) tram noktaları **`~%5` incelir**; baskıda boya kumaşa geçerken genişleyip yayılır. Bunu dengelemek için çalışma orijinalden **`~%5` daha açık** tonlarda hazırlanır ve gerekirse layer ortamında hafif parlaklaştırılıp siyah yoğunluğu azaltılır (toplam **`%5–10`** etki, abartılmamalı) [usta].
   - Ayrıca sürekli ton kanallarının üst yoğunluğu tavanlanır: altta LAP yoksa renk `%75–85`, altta açık LAP varsa üst koyu tram **`%50–60`**, siyah `%85–90`. Çok zayıf uç noktalar (`< %3–4`) isteğe bağlı kırpılabilir.

---

## C. Atölye Standartları ve Makine Kısıtları

1. **6 Kafa Sınırı:** Atölye ahtapot baskı makinesi standart 6 kafalıdır (ara kurutma flash cure durumuna göre 7-8). Gereksiz kafa sayısından kaçınılır.
2. **Zıt Renk Karıştırma Yasağı:** Lab ton farkı > 110° olan renk çiftlerinin (örneğin lacivert ile turuncu) yan yana degrade karışımı çamurlaşma yaratır. Bu alanlar fiziksel olarak ayrı basılır veya siyah çizgiyle ayrılır.
3. **Flash Cure Kurutma:** Renkler arası ara kurutma yüzey sıcaklığı ≈93–110 °C, temas süresi 3–8 saniyedir.

---

## D. Dithering ve Difüzyon Yöntemleri

1. **Index Color / Difüzyon:** Eşit boyutlu kare noktalarla stokastik dağılım.
2. **Çözünürlük Kuralı:** Doğrudan 508 PPI'da dither yapılırsa noktaların %80'i tekil piksel (0.05 mm) kalır ve kalıpta tutmaz.
   - Çözüm: **254 DPI'da Floyd-Steinberg dither + x2 Nearest Neighbor büyütme** ile en küçük nokta 0.10 mm'ye çekilir; ΔE ton doğruluğu korunur.

---

## E. Doğrulanmayanlar ve Sınırlar

- Filmlerin Photoshop'ta Bitmap modunda ve 508 PPI olarak açılması.
- RIP/film yazıcı çıktısı, fiziksel film Dmax.
- Gerçek elekte 0,05 / 0,10 / 0,15 mm dither noktaları ile %5 / %8 halftone noktalarının tutması (sayılar fiziksel tutma kanıtı değildir).
- Gerçek moiré (yalnız simülasyon denendi).
- Choke 5 px'in gerçek baskıdaki kayıt toleransına yetmesi; spread ile şişen alt baskının üst rengin altında görünmez kalması.
- Gerçek görseller (yalnız sentetik denendi); PSD spot kanal tesliminin resmî Adobe belgesi; imagesetter çözünürlükleri (1270/2540 dpi).
- Kenar pürüzlülüğü için "0–1 oran" tanımlı kaynaklı bir ölçü bulunamadı. ISO/IEC 13660 raggedness: kenar R60 eşiğinde, uydurulan doğruya dik sapmaların standart sapması (pictorial işleri ve saydam çıktıları kapsamaz; uyarlama).

---

## F. Film Çıktısı, Solidity ve Master PSD Standartları

1. **1-Bit Mod:** Her nihai renk filmi `mode="1"` (siyah-beyaz bitmap) olarak kaydedilir.
2. **Film Mantığı:** Basılacak mürekkep alanı **SİYAH (0)**, basılmayacak kumaş alanı **BEYAZ (1)** olarak yazılır.
3. **pHYs Etiketi:** 508 PPI piksel yoğunluğu PNG header'ına `pHYs = 20000 piksel/metre` olarak işlenir; Pillow'da `save(..., dpi=(508,508))` bu etiketi yazar. RIP yazılımları filmi doğru fiziksel boyutta açar.
4. **Adlandırma:** `01_fon.png`, `02_krem_lap.png`, `03_kahve_tram.png` şeklinde baskı sırasına göre (açıktan koyuya) numaralandırılır.
5. **Solidity (Opaklık) Standardı (699–701 atölye standardı):**
   - **Kumaş/Tişört kanalı:** Tam tuval dolu (`min=0, max=0`), **`Solidity = 100%`**.
   - **Zemin beyazı (`fon`):** **`Solidity = 100%`**.
   - **Tüm üst renk mürekkepleri ve üst beyaz (`beyaz`):** **`Solidity = 16%`** (Photoshop'ta yarı şeffaf tekstil mürekkebi overprint önizlemesi için).
6. **2 Katlı Master PSD Mimarisi (`700.psd`, `701.psd` + usta kuralı):**
   - Belgenin `Layers` panelindeki **orijinal görsel katmanı asla silinmez** (değişiklik ve geri dönüş referansı olarak korunur) [usta]. Kanal panelinde üstte `[Kumaş Kanalı]` + **8-bit sürekli tonlu çalışma kanalları**, altta `[2. Kumaş Kanalı ayırıcısı]` + **numaralı 1-bit AM tramlı / LAP nihai baskı kanalları** yer alır. Dolu LAP siyah kumaş kanalı koyu zemin önizlemesi ve kanal gruplama ayırıcısı olarak kullanılır [usta].
7. **Üretim Zarfı (Kayıt Krosu + 4 Alanlı Etiket + Ön/Arka Yerleşim — `tools/film_etiket_ve_kros.py`):**
   - Arka (sırt) ve ön (göğüs) baskı tek tuvalde yerleştirilebilir; her baskı bölgesinin altında sol tarafta 4 bölmeli bilgi bloğu (`beden/yüz | kanal adı | Xr iş_no ad | tarih`) ve sağda `+` kayıt krosu bulunur. Etiket ve kroslar **saf 1-bit (anti-aliasing kapalı, 0/255)** basılır.

---

## G. İş Akışı: Tek Aşamalı vs İki Aşamalı Yöntem

Her işin başında kullanıcıya hangi yöntemin uygulanacağı sorulur:

1. **Tek Aşamalı Yöntem — Yığılmış Ton + LAP (Önerilen Atölye Standardı):**
   - 8-bit sürekli ton kanallarını (dot-gain tavanlı), bölge alt-taban LAP'ını ve 1-bit `45 LPI 15°` dot-on-dot AM filmlerini tek seferde üretir:
   ```bash
   python tools/lap_overprint_am.py girdi.png cikti_klasoru --renk 6 --kumas "#4f6061" --choke 2
   ```
   - Alternatif çakışmasız (indeks) AM için: `python tools/bolge_am.py girdi.png cikti_klasoru 6`
2. **İki Aşamalı Yöntem (Kontrollü):**
   - **Yol A (2 Katlı Sürekli Ton PSD — Önerilen):** `lap_overprint_am.py` çıktısındaki 8-bit sürekli ton çalışma kanalları Photoshop'ta `Levels/Curves/Brush` ile düzenlenir, ardından tek tıkla 1-bit `45 LPI 15°` baskı setine çevrilir.
   - **Yol B (Indexed Önizleme):** `ASAMA=onizleme python tools/bolge_am.py girdi.png cikti 6` → kullanıcı Photoshop'ta onaylar → `python tools/onizlemeden_ayir.py ...`.

---

## H. Kalite Kontrol ve Hata Ölçümü

1. **Otomatik Film Kalite Denetimi (`tools/kalite_kontrol.py`):**
   ```bash
   python tools/kalite_kontrol.py cikti_klasoru kaynak.png
   ```
   - `kanallar_hibrit.json` içindeki `mod` alanına göre profil seçer:
     - `indeks`: Mürekkepler arası çakışma `> %0,1` uyarıdır.
     - `yigilmali` (dot-on-dot / LAP): Aynı aile içi yığılma ve siyah-üstü-renk çakışması doğaldır; yalnızca **zıt renk ailesi çakışması** (Lab ton farkı `> 110°`, kroma `≥ 5`) ve zıt kaynak beneği uyarılır.
     - `tramsiz`: Bilinçli trap payı kadar çakışma beklenir.
2. **Çok Kanallı PSD Derin Denetçi (`tools/psd_incele.py`):**
   - 200–600 MB spot kanallı PSD'leri tam çözünürlükte okuyarak PPI, Solidity, 2D FFT LPI/açı, `fon` choke piksel mesafesi, sürekli-ton→tram ton transferi ve `dot-on-dot` faz oranını raporlar.
3. **Kritik Eşik Değerleri:**
   - **Çözünürlük:** `508 PPI` (±1).
   - **Renk Sapması:** Düz bölgelerde film rengi kaynaktan **ΔE ≤ 12**.
   - **Kenar Pürüzlülüğü (Raggedness):** Düz bölgede `≤ 0,03` (20° test kenarında `≤ 0,03 mm`).
   - **Mikro Nokta (`≤3 px`):** Düz/LAP kanallarda `≤ %1`; renkli tramda `≤ %10`; siyah gölge tramında `> %10` yalnızca bilgi (gerçek 700/701 siyah tramlarında `%20` civarı doğal uç nokta görülür, `--min-dot 0.04` ile azaltılabilir).
   - **Kayıp Detay:** 1 pt altındaki çizgilerde kopukluk bulunmamalıdır.

---

## I. Kaynaklar ve Literatür

- MYK, Serigrafi Baskı Operatörü Seviye 3 (13UMS0309-3), 2013: https://myk.gov.tr/images/articles/editor/Rega290513/Serigrafi_3_REGA_Ekli.pdf
- Ryonet, "A Crash Course in Halftones…", 24.02.2025: https://screenprinting.com/blogs/news/a-crash-course-in-halftones-for-screen-printing
- Solutions for Screen Printers, "Halftone Angles Demystified" (güncelleme 25.03.2026): https://support.solutionsforscreenprinters.com/hc/en-us/articles/26564295432212
- UMBC, "Bitmap for Screen Printing" (2022): https://artprint.umbc.edu/files/2022/04/Bitmap-for-Screen-Printing.pdf
- Fresener, "Index Color Separations…", 11.05.2011: https://t-biznetwork.com/?p=2298
- Fresener, "Choking, Trapping…", 01.01.2017: https://t-biznetwork.com/computergraphics/choking-trapping-cleaning-up-type-in-photoshop-by-scott-fresener/
- Fresener, "Simulated Process… Part Two", 10.04.2018: https://t-biznetwork.com/computergraphics/colorseparations/simulatedseps-part2/
- Anatol, "The Tricky Business of Screen Printing Underbase", 27.09.2018: https://anatol.com/blog/the-tricky-business-of-screen-printing-underbase
- Briggs ve ark., "Applications of ISO-13660", 1999: https://www.qea.com/wp-content/uploads/2015/04/Paper_1999_ISJ-JH_Applications_of_ISO-13660-newaddr.pdf
- Sharma, Wu, Dalal, CIEDE2000 (2005): https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
- Basak, Saritha, Paul, J. Graphic Eng. & Design 16(1), 2025: https://jged.uns.ac.rs/index.php/jged/article/download/1595/1460/2554

---

## J. LAP (Katı Dolgu), Tonlu Nesne ve Alt Taban Çalışma Stili

1. **LAP Kesin Tanımı (`700.psd`, `701.psd`):** LAP = tramsız, **`%100` dolu katı (solid) renk kalıbı**. İki biçimde kullanılır:
   - *Bölge Alt-Taban LAP'ı:* Sıcak/belirli bir renk ailesinin tamamına alttan düz renk basılır (ör. yeşil yaprak & kahve dalların altına `2 krem`, kırmızı güllerin altına `açık kırmızı`); üstüne koyu tonlar `45 LPI 15°` AM tram olarak biner. **Kritik Kural:** Sıcak alt-taban LAP'ı soğuk/zıt renk ailesinin altına konmaz (ör. mavi çiçekler doğrudan beyaz `fon` üstüne basılır).
   - *Bağımsız Katı Grafik/Yazı LAP'ı:* Büyük harf, logo veya keskin yazı blokları tramsız tek kalıp olarak basılır (ör. `6 koyu kahve lap`).
2. **Ton Dilimli / Yığılmış Spot Kanal:** Tonlu nesneler tek kanala sıkıştırılmaz; açık→orta→koyu tonlar ayrı sürekli tonlu kanallara ayrılıp aynı fazda (`dot-on-dot`, `45 LPI 15°`) üst üste basılır.
3. **Sert Alt Taban (`fon`):** Koyu/renkli tişörtte alt taban tonlu değil, basılan alanların `2–3 px` choke edilmiş sert birleşim silüetidir (`Solidity = 100%`).
4. **Kanal İsimleri:** Baskı sırasıyla numaralandırılır (`1 fon`, `2 krem`, `3 kahve tram`, ..., `6 koyu kahve lap`, `7 siyah tram`); ASCII dosya adları kullanılır.

---

## K. Önerilen Boru Hattı ve Ön Analiz Karar Eşikleri

1. **Önerilen 5 Adımlı Boru Hattı:**
   - **Palet:** Lab k-means ile öner, birleşecek renkleri kullanıcı seçsin. Üst renk sayısı = makine sınırı (koyu tişörtte −1).
   - **Atama:** 4× Lanczos üst örnekle → en yakın renk → 4×4 çoğunluk oyu (`assign_ss`).
   - **Fotoğraflı bölüm:** 254 dpi'de dither + ×2 en yakın komşu büyütme (veya AM halftone 22,5°, en açık ton ≈%8).
   - **Koyu tişört:** alt baskı = basılan üst renklerin birleşimi (tramlı siyah oyulmaz), choke `2 px` (`3 px` geniş lap), ince detaya choke `1 px` (veya spread `1–2 px`); saf beyaz alanlar için üst `beyaz` (`0 px` choke, `Solidity 16%`).
   - **Yaz:** 1-bit PNG, basılacak yer siyah, 508 PPI (`pHYs` = 20000 px/m), isteğe bağlı `+` kayıt krosu ve saf 1-bit film etiketi (`tools/film_etiket_ve_kros.py`).

```python
import numpy as np, cv2
from PIL import Image
from skimage.color import rgb2lab

def to_lab(u8): return rgb2lab(u8.astype(np.float32) / 255.0).astype(np.float32)

def assign_ss(rgb, pal_lab, ss=4):
    """4x Lanczos ust ornek -> en yakin palet rengi -> ss x ss cogunluk oyu."""
    h, w = rgb.shape[:2]; out = np.empty((h, w), np.int16)
    for y in range(0, h, 250):
        y0, y1 = max(0, y - 8), min(h, y + 258)
        up = cv2.resize(rgb[y0:y1], (w * ss, (y1 - y0) * ss), interpolation=cv2.INTER_LANCZOS4)
        Lu = to_lab(up); a = np.empty(Lu.shape[:2], np.int16)
        for i in range(0, Lu.shape[0], 200):
            X = Lu[i:i + 200]; a[i:i + 200] = ((X[..., None, :] - pal_lab) ** 2).sum(-1).argmin(-1)
        cnt = np.stack([(a == i).reshape(y1 - y0, ss, w, ss).sum(axis=(1, 3)) for i in range(len(pal_lab))], -1)
        out[y:y + 250] = cnt.argmax(-1)[y - y0:y - y0 + min(250, h - y)]
    return out

def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def underbase(top, choke=2, thin_choke=1):   # top: basilan ust renklerin birlesimi (bool)
    t = top.astype(np.uint8)
    thick = cv2.morphologyEx(t, cv2.MORPH_OPEN, disk(choke)).astype(bool)
    thin = top & ~cv2.dilate(thick.astype(np.uint8), disk(1)).astype(bool)
    return cv2.erode(t, disk(choke)).astype(bool) | (cv2.erode(t, disk(thin_choke)).astype(bool) & thin)

def spread(mask, s=2): return cv2.dilate(mask.astype(np.uint8), disk(s)).astype(bool)

def save_1bit(ink, path, ppi=508):           # film mantigi: basilacak yer SIYAH (0)
    im = Image.fromarray(~ink.astype(bool)); assert im.mode == "1"
    im.save(path, dpi=(ppi, ppi), optimize=True)
```

2. **Ön Analiz Karar Eşikleri (`tools/analiz.py`):**
   - `duz_oran > 0.85` ve `doku_oran < 0.08`: Düz renkli grafik → Tramsız spot renk ayrımı (`tools/tramsiz.py`).
   - `doku_oran > 0.25` veya `kenar_oran > 0.25`: Dokulu/boyama → AM tram + çakışmasız karışım (45 LPI, 508 PPI; `tools/bolge_am.py`).
   - Aksi durumda: Karma/hibrit → Düz alanlar tramsız, geçişli alanlar tramlı (`tools/hibrit2.py`).
   - `gercek_renk_tahmini(>=1%) > 8`: Makine kafa sayısı kontrol edilir, palet birleştirme talep edilir.
   - Kağıt/zemin beyaz noktası `< 235`: Zemin kirli/gri; beyaza normalize edilmelidir.
   - Kenar zemin `std > 1.2`: Zeminde leke/fırça dokusu var; basılmayacaksa maskelenir.
   - İnce detay kalınlığı `< tram_hucre * 0.6`: 45 LPI hücrenin %60'ından dar çizgiler elekte kopabilir uyarısı üretilir.
   - JPEG blokluluk oranı `> 1.08`: 8x8 DCT blok artefaktları temizlenmeli veya netleştirilmelidir.

---

## L. Photoshop JSX Otomasyon Araçları

1. **Tek Adımda Master / Multichannel PSD Üretimi (`psd_olustur_tek_adim.jsx` & `master_psd_olustur.jsx`):**
   - `kanallar_hibrit.json` içindeki her kanalın `rol` (`kumas`, `fon`, `ust_beyaz`, `lap`, `tram`, `surekli_ton`) ve `solidity` (`100` veya `16`) bilgisini okur.
   - Kumaş rengi spot kanalını (`Solidity = 100%`, tam tuval dolu) otomatik ekler; `fon` için `100%`, tüm üst renkler ve üst `beyaz` için **`16%` Solidity** atar.
   - `master_psd_olustur.jsx` isteğe bağlı olarak **2 Katlı Master PSD** (üstte kumaş + 8-bit sürekli ton çalışma kanalları, altta 2. kumaş ayırıcısı + numaralı 1-bit baskı kanalları) üretir ve büyük dosyaları PSB destekli kaydeder.
2. **Kanal Birleştirme ve Renk Atama:**
   - Script: `tools/photoshop/film_klasoru_multichannel.jsx` + `film_renk_ata.jsx`
   - İşlev: Set klasöründeki filmleri tek Multichannel belgede toplar, ardından renkleri `kanallar_hibrit.json`'dan Spot Color olarak atar.
3. **Renk Katmanı ve Spot Kanal Ayırıcı (açık PSD üzerinde):**
   - Script: `tools/photoshop/kedihane_renk_ve_emprime_ayirici.jsx` (v1.2, Photoshop CS6 ve CC 2018–2026)
   - İşlev: Açık belgeyi Color Range ile renk katmanlarına ayırır ve emprime serigrafi için spot kanallar üretir (`choke = 2 px` varsayılan, `fon = 100%`, üst renkler `16%` Solidity).

---

## M. Sınama Sonuçları (Sentetik 2000×2000 px, 508 PPI)

| Konu | Sonuç |
|---|---|
| Spot atama | Tam kapsama: 4.000.000 pikselde boşluk 0, çakışma 0 |
| Düz kenar (20°) raggedness | 0,271 px = 0,0136 mm; 1-bit'te teorik alt sınır `1/√12·cos(açı)`. Üst örnekleme kenarı düzeltmez |
| Yanlış renk saçağı | 4× Lanczos üst örnekle + çoğunluk oyu: 2849 → 1033 px (%63,7 azalma) |
| İnce çizgi (1 / 0,5 / 0,25 pt) | 7 / 4 / 2 px genişlikte, kopuksuz |
| Choke 3 / 5 / 7 px | Taşma 0; kayıp alan %7,05 / %10,16 / %12,65 |
| 1 pt çizginin alt baskısı | Choke 3'te %14 kalır; choke ≥5'te tamamen kaybolur. 0,5 pt choke 3'te bile kaybolur |
| İnce detaya choke 1 veya spread 2 | 1 pt'nin %71'i, 0,5 pt'nin %50'si korunur. 0,25 pt hiçbir ayarda korunmaz |
| Dither 508'de doğrudan | Noktaların %80,9'u tekil piksel (0,05 mm), %94,4'ü 2 px'ten dar |
| Dither 254 dpi + ×2 büyütme | Tekil nokta %0 (en küçük 0,10 mm); ton doğruluğu aynı (ΔE 5,50 / 5,51) |
| Dither 169,33 dpi + ×3 | Tekil nokta %0 (0,15 mm); ΔE 5,54 |
| AM halftone %5 nokta | Mesh 230 (51,1 LPI): çap 0,126 mm; mesh 305 (67,8 LPI): 0,098 mm, %2 noktaların %15'i hiç oluşmuyor |
| Moiré (kaba sayısal) | 0° her iki elekte en kötü; 305 mesh/22,5°'de 6,4 mm'lik vuru görüldü |

---

## N. Terimler Sözlüğü

| Terim | Anlamı |
|---|---|
| Elek / kalıp / şablon | Serigrafi kalıbı (MYK standardı "elek şeklindeki kalıp" der) |
| Mesh | Elek sıklığı. MYK: 1 cm'deki iplik sayısı (T/cm). Mesh/inç = T/cm × 2,54 |
| Tram | Görüntüyü oluşturan küçük noktalar. Tramsız tek tonlu filme MYK "tire film" der |
| LPI | Tramın inç başına çizgi sayısı (Lines Per Inch) |
| Binme / kayıt | Renklerin üst üste denk gelmesi (registration) |
| Spot renk | Her mürekkep için tek film; logo ve keskin grafik |
| Simulated process | Spot mürekkeplerle fotoğrafik görüntü; koyu tişörtte yaygın |
| Index color | Eşit boy kare noktalar, diffusion dither (stochastic benzeri) |
| Alt baskı (underbase) | Koyu kumaşta renklerin canlı çıkması için önce basılan beyaz |
| Choke / spread (trap) | Choke: alt baskıyı içeri daraltma. Spread: üst rengi şişirme |
| Lap (atölye jargonu) | Tramsız, `%100` dolu katı (solid) renk kalıbı — bölge alt-tabanı (ör. yaprak altı krem, gül altı açık kırmızı) veya bağımsız katı yazı/grafik (`700.psd`, `701.psd` ile doğrulandı) |
| Solidity (spot kanal opaklığı) | Photoshop spot kanalında mürekkebin ekran örtücülüğü: kumaş ve `fon` `%100`, üst renkler ve üst `beyaz` `%16` (`699–701.psd` standardı) |
| Üst beyaz (Highlight White) | Koyu kumaşta `fon` (`2 px choke`) üzerine yalnızca saf beyaz kalacak alanlar için basılan ikinci beyaz kat (`0 px choke`, `Solidity %16`; `699.psd`) |
| Flash cure | Renkler arası ara kurutma (yüzey ≈93–110 °C, 3–8 sn; tek kaynak) |

*"Emprime" teriminin kaynaklı bir tanımı bulunamadı.*
*508 PPI piksel yoğunluğu metrik 200 dpcm karşılığıdır ve 651, 654, 699, 700, 701, 702 üretim dosyalarının tamamında değişmez atölye standardıdır.*

---

## O. Tramsız Düz Renk Ayrımı

1. **Vektörel ve Keskin Grafikler (`tools/tramsiz.py`):**
   ```bash
   python tools/tramsiz.py girdi.jpg cikti_klasoru --renk 4 --choke 2 --ust-beyaz
   ```
2. **Özellikler:**
   - **508 PPI** standardında tram içermeyen katı spot renkler (otomatik Lab k-means veya `--palet` ile özel renkler).
   - `TRAP_PX = 2–3` (`~0,10–0,15 mm @ 508 PPI`): Açık renklerin koyu rengin altına kontrollü taşması sağlanır.
   - Koyu kumaşta `fon` (`2 px` choke, ince detayda `1 px` korunur) ve isteğe bağlı `--ust-beyaz` (Highlight White) + `--kumas` rengi üretilir.

---

## P. Emprime Boya Kimyası ve Kumaş Eşleşmesi

Atölye pratiğinde **Şeffaf**, **Pat** ve **Beyaz** olmak üzere 3 temel boya cinsi kullanılır [usta, 2026-11-19]:

| Boya Cinsi | Karışım | Kullanım Alanı | Avantajları | Risk / Sınırları |
|---|---|---|---|---|
| **Beyazlı Boya** | Beyaz boya + kök boyalar + çatlama önleyici az **şeffaf** | Koyu zeminli, canlı ve `fon` gerektiren baskılar | Mükemmel parlaklık, yüksek örtücülük ve doygunluk | Kıvamı koyudur; baskıda sululuk azaldıkça **elekte tıkanmaya meyillidir**; tuşe daha **serttir** |
| **Pat Boya** | Pat boya + kök boyalar | Açık zeminler ve CMYK (**"fotoğraf baskı"**) | Suluboya gibidir; kalıpta uzun süre **tıkanmaz**, tuşe **yumuşacıktır**, yüksek adetli baskıya uygundur | Kumaş boyayı su gibi çeker; **koyu kumaşta rengi hiç göstermez (kaybolur)** |
| **Şeffaf** | Beyazlı boyaya katkı olarak | Esneklik / çatlama önleme | Beyazlı boyanın kuruyup çatlamasını önler | Tek başına örtücü değildir |

> **Baskı Kararına Etkisi:** Koyu zeminli beyazlı boya baskılarında tıkanmayı önlemek için `fon` mutlaka `%100` LAP yapılır ve spot tram **`45 LPI`** seçilir; açık zeminli pat boya CMYK ("fotoğraf baskı") işlerinde tıkanma riski düşük olduğundan **`55 LPI`** tercih edilir.
