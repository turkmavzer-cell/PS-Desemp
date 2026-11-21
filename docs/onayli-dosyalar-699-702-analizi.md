# Onaylı Üretim Dosyaları 699–702 — Derin Teknik Analiz ve Çalışma Stili (2026-11-19)

Atölyeden gelen 4 üretim PSD dosyası (`699.psd`, `700.psd`, `701.psd`, `702.psd`; toplam ~1,28 GB) tam çözünürlüklü ham RLE kanal çözümlemesi, DisplayInfo (renk uzayı + Solidity), 2D FFT tram frekans/açı ölçümü, `distanceTransform` kenar choke/spread ölçümü, 24×24 blok sürekli-ton→tram ton transferi ve 32×32 blok nokta-faz (`dot-on-dot`) örtüşmesi ile incelendi. Marka/müşteri adları gizlilik gereği yazılmadı.

---

## 1. Dosya Bazlı Ölçümler

### `699.psd` — Pusula & Koordinat Grafiği (Karma: Düz + Yerel Tram + Çift Beyaz)
- **Belge:** RGB 8-bit, `10314×11842 px`, **508 PPI** (`51,57 × 59,21 cm`), Photoshop 27.11, 8 üst katman (Smart Object'ler + ön/arka gruplar), 3 RGB + **9 spot kanal**.
- **Yerleşim:** Ön göğüs (`1r`) ve arka sırt (`6r`) aynı PSD içinde; her ikisinin altında `+` kayıt krosu ve 4 alanlı film bilgi etiketi var.
- **Kanallar ve Solidity:**
  - `Green copy` (`rgb(101,22,24)` koyu bordo, `Solidity=100%`, tüm tuval dolu): **Kumaş rengi kanalı**.
  - `fon` (beyaz, `Solidity=100%`, `%3,56`): Tüm beyaz + sarı + gri alanların altında, taşma `%0,00` (`inter_pct_of_a=100%`), dış kenar **choke medyan `1,91 px`** (`p90=2,32 px` ≈ **2 px = 0,10 mm**).
  - `beyaz` (beyaz, **`Solidity=16%`**, `%2,99`): **Üst Beyaz (Highlight White)** — `fon` ile `%78,45` örtüşür, 0 px choke ile yalnızca saf beyaz kalacak alanlara ikinci kat basılır.
  - `sarı tram` (8-bit sürekli ton, `uniq=144`, `Solidity=16%`) → `sarı tram copy` (**AM `44,3 LPI @ 15,6°`**, periyot `11,47 px` = **45 LPI, 15°, yuvarlak**, 1:1 doğrusal ton transferi `diff=±0,1%`, `≤3 px` parça `%4,41`).
  - `sarı` (düz, `Solidity=16%`, `%90,3` fon üstü), `gri` (düz, `Solidity=16%`, `%98,23` fon üstü).
  - `antrasit` (`rgb(23,23,23)`, düz, `Solidity=16%`): **Fon altında YOK (`antrasit ∩ fon = %2,08`)** — koyu bordo kumaşa doğrudan basılıyor.

### `700.psd` — Büyük Harf & Çiçekli/Kuşlu İllüstrasyon (2 Katlı Kanal Paneli + LAP + Dot-on-Dot)
- **Belge:** CMYK 8-bit (`U.S. Web Coated SWOP v2`), `7100×14154 px`, **508 PPI** (`35,50 × 70,77 cm`), Photoshop CC 2018, 10 üst katman, 4 CMYK + 1 Transparency + **19 spot kanal**.
- **Yerleşim:** Arka sırt + ön göğüs tek tuvalde (`7r`), her ikisinin altında `+` kayıt krosu ve 4 alanlı etiket.
- **2 Katlı Kanal Paneli Yapısı:**
  - *Üst set (sürekli ton çalışma):* `Yellow copy` (gri kumaş `rgb(107,107,107)`, `Solidity=100%`) + 4 adet 8-bit sürekli ton (`kahve tram`, `yeşil tram`, `mavi tram`, `siyah tram`, tümü `Solidity=16%`).
  - *Alt set (nihai numaralı baskı):* `Yellow copy 3` (2. kumaş ayırıcı kanalı, `Solidity=100%`) +
    1. `1 fon` (`Solidity=100%`, `%31,84`): Tüm tasarım altı, **choke medyan `2,32 px`** (koyu kahve lap altında `2,87 px`).
    2. `1 fon copy` (`Solidity=100%`, `%21,55`): Büyük harf/yazı (`6 koyu kahve lap`) **oyulmuş varyant fon** (`inter=%2,79`), çiçeklerde **choke `1,91 px`**.
    3. `2 krem` (`Solidity=16%`, `%13,31`): **Bölge Alt-Taban LAP (%100 düz)** — `4 yeşil tram`'ın `%98,61`'i ve `3 kahve tram`'ın `%96,93`'ü bunun üstünde; **`5 mavi tram`'ın altında YOK (`%1,33`)** (sıcak krem maviyi kirletmesin diye mavi doğrudan beyaz `1 fon` üstüne basılmış).
    4. `3 kahve tram copy`, `4 yeşil tram copy`, `5 mavi tram copy`, `7 siyah tram copy`: Hepsi `Solidity=16%`, **AM `44,3 LPI @ 15,6°` (45 LPI, 15°, yuvarlak)**.
    5. `6 koyu kahve lap` (`rgb(52,31,13)`, `Solidity=16%`, `%10,87`): **Bağımsız Katı Grafik/Yazı LAP (%100 düz)**.
- **Bölge Saflığı & Nokta-Faz (`dot-on-dot`):**
  - Farklı aileler arası: `4 yeşil ∩ 5 mavi = %0,95` (sıfır zıt karışım).
  - Aynı faz üst üste (`dot-on-dot`): `4 yeşil` ↔ `7 siyah` **`%94,4`**, `3 kahve` ↔ `7 siyah` **`%91,2`**, `5 mavi` ↔ `7 siyah` **`%89,7`**.
- **Dot Gain Tavanı:** Sürekli ton renk kanalları ~`%70–75` yoğunlukta tavanlanmış (`80–100%` bloğu yok), siyah `~%82` tavanlı; ardından 1:1 doğrusal (`diff=±0,2%`) 45 LPI 15° trama çevrilmiş.

### `701.psd` — Altın Barok Arma & İki Kırmızı Gül (Yığılmış Ton + Gül LAP)
- **Belge:** CMYK 8-bit (`U.S. Web Coated SWOP v2`), `9152×13283 px`, **508 PPI** (`45,76 × 66,41 cm`), XMP: **Topaz Gigapixel 1.4.0** → Photoshop, 3 katman, 4 CMYK + 1 Transparency + **12 spot kanal**.
- **Yerleşim:** Arka sırt + ön göğüs tek tuvalde (`6r`), her ikisinin altında `+` kayıt krosu ve 4 alanlı etiket.
- **2 Katlı Kanal Paneli Yapısı:**
  - *Üst set (sürekli ton):* `Green copy` (`rgb(79,96,97)` yeşil-gri kumaş, `Solidity=100%`) + 4 sürekli ton (`koyu kırmızı` `min=32` [max `%87`], `sarı` `min=41` [max `%84`], `kahve` `min=53` [max `%79`], `siyah` `min=28` [max `%89`], hepsi `Solidity=16%`).
  - *Alt set (nihai baskı):* `Green copy 2` (`Solidity=100%`) + `fon` (`Solidity=100%`, **choke `1,91 px`**) + `açık kırmızı` (**Gül LAP**, `%100` düz, `Solidity=16%`) + `koyu kırmızı copy`, `sarı copy`, `kahve copy`, `siyah copy` (hepsi `Solidity=16%`, **AM `44,3 LPI @ 15,6°`**).
- **Örtüşme ve Nokta-Faz (`dot-on-dot`):**
  - Gül: `koyu kırmızı tram` **%98,74** oranında düz `açık kırmızı` LAP içinde; altın barokla çakışma yalnız `%0,56`.
  - Altın barok yığılma: `kahve ⊂ sarı` **%99,44** (**`%99,6 DOT-ON-DOT`**), `siyah ∩ sarı` **`%98,4 DOT-ON-DOT`**, `siyah ∩ koyu kırmızı` **`%96,7 DOT-ON-DOT`**, `siyah ∩ kahve` **`%90,7 DOT-ON-DOT`**.
  - Siyah altı fon: **Tramlı siyahın `%95,75`'inin altında `fon` VAR** (tramlı siyah kontur/gölge fondan oyulmamış).

### `702.psd` — Yıpratılmış Dairesel Amblem (Tek Renk Tramsız Spot)
- **Belge:** RGB 8-bit, `8944×11256 px`, **508 PPI** (`44,72 × 56,28 cm`), Photoshop 27.11, 3 katman (arka + ön Smart Object), 3 RGB + **1 Spot (`Spot_1`, `Solidity=100%`, `%11,50`)**.
- **Yıpratma (Distress) Dokusu:** `≤3 px` bileşen oranı yalnızca **`%0,47`**; p10 bileşen alanı `57 px` (`≈0,43 mm` eşdeğer çap), p50 `462 px` — elekte güvenle tutunur.

---

## 2. Kesinleşen 10 Atölye Yasası

1. **Tek tram standardı (5/5 dosya — 651, 654, 699, 700, 701):** `508 PPI`, `45 LPI`, `15°`, **yuvarlak (Round)** nokta, tüm kanallarda aynı açı ve aynı faz.
2. **Bölge saflığı + aynı ailede `dot-on-dot` yığılma:** Farklı renk aileleri karışmaz (`< %1`); aynı aile içinde koyu ton ve siyah açık tonun üstüne `%90–99,6` aynı fazda biner.
3. **"LAP" = %100 dolu tramsız katı dolgu:** Bölge alt-taban lapı (`2 krem`, `açık kırmızı`) veya bağımsız katı yazı/harf (`6 koyu kahve lap`). Sıcak lap soğuk/zıt ailenin altına konmaz.
4. **Solidity standardı:** Kumaş kanalı = `%100`, `fon` = `%100`, tüm üst renkler ve üst `beyaz` = **`%16`**.
5. **Kumaş kanalı ve 2 katlı master PSD:** Üstte kumaş + 8-bit sürekli ton çalışma kanalları; altta 2. kumaş ayırıcısı + numaralı 1-bit tramlı/LAP baskı kanalları.
6. **Choke standardı:** `508 PPI`'da **`2 px`** (`0,10 mm`, ince detay/genel) – **`3 px`** (`0,15 mm`, geniş lap alan).
7. **Üst Beyaz (Highlight White):** Koyu kumaşta saf beyaz alanlara `fon` (`2 px choke`, `%100`) üstüne ikinci kat `beyaz` (`0 px choke`, `%16`) verilir.
8. **Siyah altı fon ayrımı:** Tramlı siyah gölge/konturun altı **oyulmaz**; yalnızca geniş düz koyu LAP bloklar kumaşa göre opsiyonel oyulur (veya çift fon: `1 fon` + `1 fon copy`).
9. **Dot gain kompanzasyonu sürekli ton kanalında tavanla yapılır:** Renk kanalları max `%75–85`, siyah max `%85–90` tavanlanır; sonra 1:1 doğrusal `45 LPI 15°` tram uygulanır.
10. **Üretim zarfı:** Ön + arka baskı tek tuvalde, her baskı altında `+` kayıt krosu ve 4 alanlı film bilgi etiketi (`beden/yüz | kanal adı | Xr iş_no ad | tarih`). Etiket 1-bit (anti-aliasing kapalı) basılarak ara gri kenar artığı sıfırlanır.
