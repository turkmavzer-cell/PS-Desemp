# Görsel analiz ve karar rehberi

Amaç: Yeni bir görsel (JPEG/PNG) geldiğinde **neye bakıp hangi kararı vereceğimizi** baştan sabitlemek; böylece aynı hatalar (düz lekeler, beyaz noktalar, zıt renk benekleri, yanlış PPI, atölye kapasitesini aşan renk sayısı) tekrar etmez.

Kaynak etiketleri: **[kaynak]** web kaynağından doğrulandı (bağlantı en altta), **[atölye]** atölyeden gelen yorum/dosya, **[deney]** bu projede gözlendi, **[yorum]** benim çıkarımım, doğrulanmadı.

Tarih: 2026-10-14 · Güncelleme: 2026-11-19 (onaylı üretim dosyaları 699–702 ölçüm bulgularıyla tekilleştirildi).

---

## 0. Zorunlu akış (her yeni iş)

1. `python3 tools/analiz.py girdi.png` çalıştır, raporu oku (bölüm 1).
2. Kullanıcıya sor: **tek aşamalı mı iki aşamalı mı** (bkz. `docs/is-akisi-tek-ve-iki-asama.md`), **tişört rengi** (açık/koyu/ikisi), **kaç mürekkep** (atölyenin kafa sayısı), **zemin/arka plan baskıda olsun mu**. Kullanıcı bu üçünü "iş istediğinde bana sor" diye belirtti, önceden sorma, iş istenince sor.
3. Yöntemi seç (bölüm 2) ve parametreleri bu rehberden al.
4. Ayır, sonra `python3 tools/kalite_kontrol.py cikti_klasoru kaynak.png` çalıştır (bölüm 7); uyarıları çöz.
5. Önizleme + %100 kesitle gözle doğrula (özellikle: açık tonlu sıcak alanlar, ince çizgiler, kenarlar).
6. İstenirse `tools/film_etiket_ve_kros.py` ile saf 1-bit `+` kayıt krosu + 4 alanlı film etiketi ekle; PSD'yi **508 PPI** ve doğru **Solidity (`100%` kumaş+fon / `16%` üst renkler)** ile oluştur (`tools/photoshop/master_psd_olustur.jsx` veya `psd_olustur_tek_adim.jsx`).
7. Yeni öğrenileni `docs/` içine, kodu `tools/` içine işle, commit + push.

---

## 1. Neye bakarak karar veririz (analiz ölçütleri)

| Ölçüt | Nasıl ölçülür | Karar |
|---|---|---|
| Çözünürlük ve baskı boyutu | piksel boyutu ÷ PPI = cm | Baskı boyutunda 300 PPI'nın altındaysa sorun [kaynak: printablepress]. Bu projede **508 PPI** ile çalışılır [atölye: 651, 654, 699–702]. |
| Düz alan oranı (flat) | Komşu piksel farkı küçük, geniş birleşik bileşen yüzdesi | Yüksek: tramsız spot (`tools/tramsiz.py`). Düşük, sürekli geçişli: LAP + dot-on-dot AM (`tools/lap_overprint_am.py`). |
| Degrade/doku oranı | Yerel kontrast ve gradyan dağılımı | Suluboya/boyama dokusu varsa tramsız spot **olmaz**, tram gerekir [deney]. |
| Gerçek mürekkep sayısı | Palet adayı (k-means / sayma), küçük alanlı renkler ayrı | Tasarımdaki gerçek renk sayısı belirler, üst sınır değildir. Atölye kapasitesini aşıyorsa azalt. |
| Zemin | Köşe/kenar renk ve doku | Kâğıt dokulu zemin ve fırça darbeleri baskıya girmez, kullanıcı seçer. Kâğıt beyaz noktası görselin kendi beyazına normalize edilir [deney]. Şeffaf PNG'de alfa dışı pikseller maskelenir. |
| En ince çizgi/detay | Çizgi kalınlığı (distance transform) px → mm | Tramlı ekranda ince çizgi, hücre boyutundan küçükse kopar [deney]. |
| JPEG artefaktı / hale | Kenarlarda halka, mozaik 8x8 desen, upscaler renk halesi | Hale gerçek mürekkep sanılabilir: morfolojik temizle [deney: lacivert hale]. |
| Beyaz noktası | Figür dışı kâğıt renginin yüzdelik değeri | Gri-krem kâğıt beyaza çekilir, yoksa tüm sayfa mürekkep sanılır. |
| Küçük kapsamlı renk | Palet renginin toplam piksel yüzdesi | %1 altı = ya birleştir ya atölyeye sor (siyah kanal, bej kanal örneği) [deney]. |

---

## 2. Yöntem seçimi

Kaynaklardan [Impressions: Color Separations 101, Printable Press] ve 699–702 üretim dosyalarından [atölye]:
- **Spot renk (tramsız)**: logo, çizgi film, sert kenarlı veya yıpratılmış tek/az renkli grafik (`702.psd`).
- **LAP + Dot-on-Dot Yığılmış AM (Atölye Ana Standardı)**: fotogerçekçi/gölgeli illüstrasyonlar, koyu ve renkli kumaşlar dahil (`700.psd`, `701.psd`); farklı renk aileleri ayrık, aynı ailede bölge alt-taban LAP'ı + üstüne aynı fazda `45 LPI, 15°` yuvarlak AM tram.
- **Karma (Düz Spot + Yerel Tram + Çift Beyaz)**: keskin vektörel/tipografik tasarım içinde yalnızca yerel ışık geçişi tramlı (`699.psd`).
- **İndeks / Çakışmasız AM**: piksel başına tek mürekkep istenen işler (`tools/bolge_am.py`).
- **CMYK proses**: beyaz/açık tişörtte fotogerçekçi iş (`ÖÖRNEKK`); yüksek elek, kayıt ve nokta büyümesi denetimi şart.

Bu projedeki karşılıkları:

| Görsel | Önerilen yöntem | Not |
|---|---|---|
| Düz renkli logo/yazı/rozet/mandala/yıpratılmış tek renk (`702`) | Tramsız spot (`tools/tramsiz.py`, 508 PPI) | ince kontur için sık elek, choke `2 px` [atölye] |
| Gölgeli/tonlu illüstrasyon, çok katlı barok/çiçekli desen (`700`, `701`) | **LAP + Dot-on-Dot Yığılmış AM (`tools/lap_overprint_am.py`, 45 LPI, 15°, yuvarlak)** | Bölge saflığı + aynı ailede dot-on-dot üst üste baskı; sürekli ton tavanlı (`%75–85` renk / `%85–90` siyah) [atölye] |
| Karma: düz grafik + yerel ışık geçişi + koyu kumaş (`699`) | **Karma (Düz Spot + Yerel 45 LPI 15° Tram + Çift Beyaz)** | `fon` (`2 px` choke, `100%`) + `beyaz` (`0 px`, `16%`) + yerel tram kanalı [atölye] |
| Suluboya/dokulu illüstrasyon — çakışmasız istenirse | İndeks/çakışmasız AM (`tools/bolge_am.py`), 45 LPI, 15° | piksel başına tek mürekkep, zıt renk yasağı |
| Atölye örneği "Casa Blanca" | CMY+K 45 lpi + altlık "fon" | [atölye] ayrı açı kullanılmamış, hepsi 45° |

[atölye] 651, 654, 699, 700, 701 (**5/5 üretim dosyası**) ortak standardı: **508 PPI, 45 LPI, 15°, yuvarlak nokta, aynı faz (`dot-on-dot`), Solidity `100%` (kumaş+fon) / `16%` (üst renkler), `2–3 px` choke, `+` kayıt krosu ve 4 alanlı film etiketi.** Ayrıntı: `docs/onayli-dosyalar-699-702-analizi.md`.

---

## 3. Tram parametreleri

| Konu | Bilgi |
|---|---|
| lpi | Elek sayısı ÷ 5 (güvenli) veya ÷ 4 [kaynak: screenprinting.com]. Spot emprimede **`45 LPI`** [atölye, usta]; CMYK'da (baskı dilinde **"fotoğraf baskı"**, açık zemin pat boya) **`55 LPI`** daha iyi sonuç verir [usta]. |
| Nokta aralığı | Saati'ye göre 45 lpi basılabilir ton aralığı 196 PW'de %6–%86, 230 PW'de %4–%90 [kaynak]. Sürekli ton alt kesimi `MIN_DOT=0.04` (opsiyonel); üst tavan altta LAP yoksa renk `%75–85`, altta açık LAP varsa üst koyu tram **`%50–60`**, siyah `%85–90` [atölye: 700/701, usta]. |
| Kalıp incelmesi & Nokta büyümesi (Dot Gain) | Film→kalıpta (emülsiyon/ışık/yıkama) tram **`~%5` incelir**, baskıda boya kumaşa geçerken genişleyip yayılır [usta]. Bunu dengelemek için çalışma orijinalden **`%5` daha açık** (layer parlaklık + siyah azaltma ile toplam **`%5–10`**, abartılmadan) hazırlanır [usta] ve sürekli ton üst tavanlanıp 1:1 doğrusal tramlanır [atölye: 700/701]. |
| Açı | Spot ayrımlarda tüm kanallar **aynı açıda** olur; `0°`/`90°` yasaktır. CMYK değilse açı atölye/kişisel tercihidir: 651–701 atölyesi **`15°`**, bazı firmalar **`45°`**, Londra ustası **`72°`** kullanır [atölye, usta]. |
| Nokta şekli | **Yuvarlak (Round)** [atölye: 5/5 dosya]. |
| Film PPI | **508 PPI** (tüm atölye dosyaları ve kullanıcı standardı). 45 lpi → hücre ≈ 11,3–11,47 px. |
| Faz (Dot-on-Dot vs İndeks) | **Yığılmış ton / LAP modunda (`lap_overprint_am.py`):** aynı aile içi ve siyah tramlar **aynı açı + aynı fazda (`dot-on-dot`, %90–99,6)** üst üste basılır, rozet/moiré üretmez [atölye: 700/701]. **İndeks modunda (`bolge_am.py`):** her piksel tek mürekkep (çakışma 0). |

---

## 4. Kaç mürekkep (renk sayısı)

- Her mürekkep ayrı elek ve ayrı kafa. Renk sayısı makinenin basabildiğini aşmamalı [kaynak: Impressions].
- [atölye] Makinede çoğunlukla en çok ~6 kafa; kurutmalar iptal edilirse 7–8 (`699`: 6r arka + 1r ön; `700`: 7r; `701`: 6r; `702`: 1r). Açık beyaz zeminde fon yapılmaz.
- [usta] **Profesyonel ayrım önce, renk azaltma kullanıcıda:** Baştan 6 kafaya sıkmak yerine her ana renk ailesi için **2 ton** (Açık LAP + %50–60 Koyu Tram; çok geniş/detaylı yüzeylerde **3 tona kadar**) + tek `fon` + tek `siyah` (ör. 8 kafa) üretilip renk düşürme işlemi kullanıcıya bırakılır.
- Kural: **hedef sayıyı kullanıcı söyler** (ya da atölyeden öğrenir); analiz raporundaki "gerçek renk sayısı" yalnızca bilgi içindir.

---

## 5. Karışım, LAP ve palet kuralları

1. **Zıt renk karıştırma yasak (Bölge Saflığı):** Farklı renk aileleri birbirine karışmaz (`700`: yeşil ∩ mavi `%0,95`; `701`: kırmızı ∩ sarı `%0,56`). Paletteki iki kromatik mürekkebin Lab ton açısı farkı `> 110°` ise (C ≥ 5, en koyu hariç) karışım yasaktır.
2. **Aile içi "Açık LAP + Üst %50–60 Koyu Tram" kuralı [atölye: 700/701, usta]:** Canlı ve dolu baskı için her renk ailesinin altına açık ton `%100` düz LAP (veya LAP'a çok yakın yoğunluk) basılıp üstüne koyu ton **`%50–60` ortalama tavanlı** tram olarak bindirilir; ancak **sıcak LAP asla soğuk/zıt ailenin altına konmaz** (mavi çiçekler doğrudan beyaz `fon` üstüne basılır). Siyah tram için **CMYK `K`** kanalı yeterlidir [usta].
3. **İki renkli degradede "%85 alt bindirme" (underlap) kuralı [usta]:** Komşu degrade geçişlerinde (ör. sarı↔kırmızı) ortada beyaz fonun parlayıp geçişi soldurmaması için açık/taban renk, koyu rengin **en az `%85` dolu olduğu alana kadar altına bindirilerek** başlatılır.
4. **Aynı ailede dot-on-dot yığılma [atölye: 700/701]:** Aynı renk ailesi içinde koyu ton açık tonun üstüne (`kahve ⊂ sarı %99,4`) ve siyah gölge/kontur diğer tüm renklerin üstüne aynı fazda basılır.
5. **Orijinal görsel katmanı korunur [usta]:** Master PSD'nin `Layers` panelindeki orijinal görsel asla silinmez.
4. **Çok küçük ikincil bileşen at:** İndeks modda en koyu hariç `< %10` bileşen atılır (`MINC`); yığılmali modda sürekli ton `< %3–4` alt kesim (`MIN_DOT`) ile temizlenir.
5. **Koyu ince detay için ayrı en koyu mürekkep** tutulur; k-means tek başına koyu küçük alanı kaçırır [deney].
6. **Kullanılmayan mürekkep (`< %0,5` kapsama):** Ya birleştir ya atılacak kafa olarak işaretle.
7. **Kâğıt (tişört) beyaz kabul edilir** (açık sette); krem/gri tonlu kâğıt beyaza normalize edilir.

---

## 6. Kaynak dosya kalitesi, choke ve üst beyaz

- Raster kaynakta **netleştirilmiş / upscale edilmiş kaynak** istenir (`701.psd` XMP: Topaz Gigapixel 1.4.0 → 508 PPI baskı boyutu).
- Baskı boyutunda en az 300 PPI [kaynak: Printable Press]; hedefimiz **508 PPI**.
- Upscaler hale/blur izi bırakabilir; kenarda morfolojik açma ile temizle.
- **Koyu tişörtte underbase (`fon`) choke:** `508 PPI`'da **`2 px`** (`0,10 mm`, ince detay/genel) – **`3 px`** (`0,15 mm`, geniş lap) [atölye: 699/700/701 ölçümü `1,91–2,87 px`]. İnce `≤ 1 pt` detaylarda `1 px` choke korunur.
- **Siyah altı fon:** Tramlı siyah gölge/konturun altı **oyulmaz** (`701`: `%95,75` fonlu); yalnızca geniş düz koyu LAP bloklar kumaş rengine göre opsiyonel oyulur (`699`: antrasit fonsuz; `700`: çift fon `1 fon` + `1 fon copy`).
- **Üst Beyaz (Highlight White):** Koyu kumaşta saf beyaz alanlara `fon` (`2 px` choke, `Solidity 100%`) üstüne ayrı `beyaz` (`0 px` choke, `Solidity 16%`) basılır (`699`).
- **Solidity & Kumaş Kanalı:** Kumaş kanalı ve `fon` = `%100`, tüm üst mürekkepler ve üst `beyaz` = `%16`.

---

## 7. Çıktı doğrulama (otomatik + gözle)

`tools/kalite_kontrol.py` (`mod: yigilmali | indeks | tramsiz`) ve `tools/psd_incele.py` şunları denetler:

| Kontrol | Eşik |
|---|---|
| Kullanılmayan/çok az kapsamlı mürekkep | `< %0,5` uyarı |
| Zıt renk beneği / zıt aile çakışması | Kaynakla ton farkı `> 90°` (`> %1,5` renk / `> %3,0` koyu); yığılmali modda zıt renk ailesi (`> 110°`, C ≥ 5) çakışması `> %1,5` uyarı |
| Çok küçük izole nokta (`≤ 3 px`) | Düz/LAP `> %1` uyarı; renkli tram `> %10` ve `> 2000` uyarı; siyah gölge tramında bilgi |
| Mürekkepler arası çakışma | `indeks`: `> %0,1` uyarı · `yigilmali`: aynı aile ve siyah-üstü-renk yığılması serbest, zıt aile yasak · `tramsiz`: trap payıyla yorumlanır |
| PPI meta, boyut ve choke | `508 PPI` (±1); `psd_incele.py` ile `fon` choke (`2–3 px`), Solidity (`100%/16%`), FFT (`45 LPI, 15°`) ve dot-on-dot fazı ölçülür |

Sonra yine **gözle**: %100 kesit (yüz/saç, ince dal, kenar) ve kumaş rengi üzerinde `Solidity 16%` overprint simülasyonu.

---

## 8. Kalan atölye teyitleri

1. 45 LPI tram kanalları hangi elekte (77T / 90T / 120T) ve hangi mürekkeple (su bazlı / plastisol / pat) basılıyor?
2. Siyah gibi çok küçük kapsamlı (`< %0,5`) kanal ayrı mı tutulur, birleştirilir mi?
3. CMYK proses (`ÖÖRNEKK`, 45°) yalnızca açık zemin suluboya işlerde mi kullanılıyor?

---

## Kaynaklar

- Screenprinting.com, A Crash Course in Halftones for Screen Printing: https://screenprinting.com/blogs/news/a-crash-course-in-halftones-for-screen-printing
- T-Biz Network, Halftone Dots Made Easy: https://t-biznetwork.com/articles/screenprinting/halftone-dots-made-easy/
- Impressions, Color Separations 101: https://impressionsmagazine.com/process-technique/color-separations-101/16708/
- Printable Press, Index Color Screen Printing Explained: https://www.printablepress.com/index-color-screen-printing-explained/
- Printable Press, How to Prepare Artwork Files for Screen Printing: https://www.printablepress.com/how-to-prepare-artwork-files-for-screen-printing/
- İç kaynaklar: `docs/arastirma/serigrafi-renk-ayrimi-raporu.md`, `docs/ogrenilenler.md`, `docs/onayli-dosyalar-699-702-analizi.md`, atölye dosyaları (651, 654, ÖÖRNEKK, 699, 700, 701, 702) ve WhatsApp yorumları.
