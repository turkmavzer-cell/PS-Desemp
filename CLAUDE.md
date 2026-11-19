# CLAUDE.md

Bu repo, tekstil baskı (emprime/serigrafi) için görsel renk ayrımı ve katman ayırma çalışmasıdır. Kullanıcı Türkçe konuşur; direkt ve kısa iletişim tercih eder.

## Her oturumun başında
Önce oku: `README.md`, `docs/ogrenilenler.md`, `docs/onayli-dosyalar-699-702-analizi.md`, `docs/emprime-kanal-ayirma.md`, `docs/katman-ayirma.md`, `docs/yontem-gunlugu.md`, `docs/acik-sorular.md`, `docs/arastirma/gorsel-analiz-karar-rehberi.md`, `docs/is-akisi-tek-ve-iki-asama.md`, `docs/bozuk-gorselden-desen-uretim-promptu.md`. Orada yazanı yeniden keşfetme.

## Ana kurallar (özet — 699–702 üretim dosyaları ve 2026-11-19 usta değerlendirmesiyle doğrulandı)
- Düz renkli tasarım **tramsız spot** (`tools/tramsiz.py`, **508 PPI**) ayrılır; tram yalnızca degrade/gölge/fotoğraf. CMYK Split Channels emprime spot ayrımı değildir; CMYK baskı dilinde **"fotoğraf baskı"**dır (açık zeminde pat boyayla **`55 LPI`**).
- **Tram standardı:** **508 PPI**, spot emprimede **`45 LPI`**, **yuvarlak (Round)** nokta, tüm kanallarda aynı açı ve aynı faz (`dot-on-dot`). Açı spot ayrımında atölye/kişisel tercihine göre seçilir (`15°`, `45°`, **`72°`**; `0°/90°` yasak — 699–701 atölyesi `15°`, Londra ustası `72°` kullanıyor).
- **Aile İçi "Açık LAP + Üst %50–60 Koyu Tram" & Degrade "%85 Alt Bindirme":** Canlı ve dolu baskı için her renk ailesinde altta açık ton **`%100` LAP** (veya LAP'a çok yakın yoğunluk), üstünde koyu ton **`%50–60` ortalama tavanlı tram** basılır; siyah tram için **CMYK `K`** kanalı yeterlidir. İki renkli degradelerde (ör. sarı↔kırmızı) ortadan beyaz fon parlamaması için açık renk, koyu rengin **en az `%85` dolu olduğu alana kadar altına bindirilir**.
- **Profesyonel Ayrım Önce, Renk Azaltma Kullanıcıda:** Baştan 6 kafaya sıkmak yerine her ana renk ailesine **2 ton** (geniş/detaylı yüzeyde **3 tona kadar**) + tek `fon` + tek `siyah` (ör. 8 kafa) üretmek, renk düşürmeyi kullanıcıya bırakmak profesyonel standarttır.
- **Orijinal Katman, Fon, Choke, Üst Beyaz & %5–10 Baskı Açma:** Master PSD'nin `Layers` panelinde **orijinal görsel katmanı asla silinmez**. `fon` neredeyse her zaman beyaz ve `%100` LAP'tır (beyaz boyanın tıkanmaması ve üst renklerin zayıf kalmaması için); choke **`2 px`** (genel/ince) – **`3 px`** (geniş lap). Koyu kumaşta saf beyaz alanlara `fon` (`100%`) üstüne ayrı **Üst Beyaz (`beyaz`, `0 px`, `16%`)** eklenir. Tramlı siyah altı **fondan oyulmaz**. Kalıp `%5` incelmesi ve baskı boya yayılmasını dengelemek için çalışma orijinalden **`%5–10` daha açık** hazırlanır (abartılmadan).
- **Solidity & Kumaş Kanalı:** PSD'de kumaş kanalı (`100%`, tam dolu — koyu zemin önizlemesi ve kanal ayırıcısı) + `fon` (`100%`) + tüm üst renkler ve üst `beyaz` **`16%` Solidity**; 2 katlı master PSD'de üstte 8-bit sürekli ton çalışma seti, altta 2. kumaş ayırıcısı + 1-bit baskı seti yer alır.
- **Boya Kimyası (3 Cins):** Koyu zemin ve fonlu işlerde **beyazlı boya** (beyaz + kök boya + çatlama önleyici az şeffaf — parlak/doyurucu ama tıkanmaya meyilli, sert tuşe); açık zemin ve CMYK'da **pat boya** (pat + kök boya — suluboya gibi, tıkanmaz, yumuşak tuşe, koyu kumaşta kaybolur).
- Dosya yeri ya da ayar belirsizse **kullanıcıya sor**, tahmin etme (tişört rengi, baskı boyutu, kaç kafa/mürekkep, elek/mürekkep/film türü).
- Sonuçları önizleme + %100 kenar kesitiyle gözle doğrula.

## Zorunlu iş akışı (her yeni görsel; ayrıntı: `docs/arastirma/gorsel-analiz-karar-rehberi.md`)
1. Önce `docs/arastirma/gorsel-analiz-karar-rehberi.md` oku. `python3 tools/analiz.py girdi.png [maske.png]` çalıştır, raporu kullan.
2. Kullanıcı iş istediğinde **tek aşamalı mı iki aşamalı mı** yapılacağını (kısa açıklamasıyla), **tişört rengini, kaç mürekkep (kafa sayısı) istediğini ve zemin/arka planın basılıp basılmayacağını sor**; önceden ya da tahminle karar verme. Cevap gelince başla. Yöntemlerin açıklaması: `docs/is-akisi-tek-ve-iki-asama.md`.
   - Tek aşamalı: program renkleri seçer, doğrudan sürekli ton + 1-bit filmlere ayırır.
   - İki aşamalı: (A) 2 katlı master PSD'de 8-bit sürekli ton çalışma kanalları Photoshop'ta düzeltilip tramlanır; veya (B) tramlı indexed önizleme (`ASAMA=onizleme`) onaylanıp `tools/onizlemeden_ayir.py` ile filmlere ayrılır. Onay gelmeden 2. aşamaya geçme.
3. Ayır:
   - **Düz renkli / tek renk / yıpratılmış grafik:** `tools/tramsiz.py` (`508 PPI`, otomatik/özel palet, `2–3 px` choke, `--ust-beyaz`).
   - **Tonlu/gölgeli illüstrasyon (Önerilen Atölye Standardı):** `tools/lap_overprint_am.py` (bölge saflığı + LAP + dot-on-dot `45 LPI 15°` + sürekli ton tavanı + çift fon / üst beyaz).
   - **Çakışmasız (indeks) AM istenirse:** `tools/bolge_am.py` (zıt renk karıştırma yasak, en koyu hariç `< %10` bileşen at, `≤ 3 px` mikro parça sil).
4. İstenirse `tools/film_etiket_ve_kros.py` ile saf 1-bit `+` kayıt krosu, 4 alanlı film bilgi etiketi ve ön+arka yerleşim ekle.
5. `python3 tools/kalite_kontrol.py cikti_klasoru kaynak.png` (ve PSD üretildiyse `python3 tools/psd_incele.py cikti.psd`) çalıştır; uyarıları çöz ya da kullanıcıya bildir.
6. %100 kesit + önizleme ile gözle doğrula. Kullanıcı yan yana karşılaştırma istemez; yalnızca bitmiş, doğrulanmış iş ver.
7. Film/PSD **508 PPI**. Yazdığını repoda doğrula (git/dosya), sonra "yazdım" de.
8. Düz (tramsız) bölge (rozet, logo, yazı) kontrolü: film rengi kaynaktan `dE ≤ 12`, kenar pürüzlülüğü `≤ 0,03` (`kalite_kontrol.py` ölçer).
9. Atölye yorumları karar kuralıdır: makine ~6 kafa (7–8 kurutma iptaliyle), açık beyaz zeminde fon yok, kalın yazılı desen kaba kalır.

## İstek kapsamı
- "Repoda güncelle / düzenle" = yalnızca kod ve doküman değişikliği + commit/push. Ayrım çalıştırma, kullanıcının bilgisayarına dosya yazma, Photoshop açma **yapılmaz**; kullanıcı ayrıca "işlem yap" demeden iş üretilmez.

## Katmanlara ayırma işi (renk ayrımı dışı)
Görseli katmanlara ayır istenirse (`docs/katman-ayirma.md`): önce analiz et (`tools/katman/zemin_desen_katman.py`, `tools/katman/uc_renk_katman.py`, `tools/katman/bolge_murekkep_katman.py`), sonra **kaç katman istendiğini sor** ve seçenekleri görselin yapısına göre kendin belirleyip sun (ör. 2: Zemin+Desen; 3: Zemin+Desen opak+Geçiş saydam; illüstrasyon/zırh için 6-10 yapısal nesne). Katmanlar üst üste binince orijinal çıkmalı (ölç). Katman adlarında Türkçe karakter kullanma.
- **Kritik Kurallar (SOP: `docs/katman-ayirma-rehberi.md`):**
  1. Asla dikdörtgen koordinat kutusu (`bounding box`) ile nesne kesme; altta dikdörtgen oyuk bırakır.
  2. Kavisli zırh/organik parçalarda düz poligon kesimi yapma; dikiş/birleşim kavislerini izle.
  3. Kafatası/gövde gibi anatomik bütünlerin içini delik deşik etme (göz çukuru/ağız boşluğu delinmez, ışık ayrı üst katmandır).
  4. 1-bit sert jilet kenar yasaktır; kenarlara 1-1.5 px alt-piksel yumuşatma uygula.

## Bozuk / giyilmiş görselden desen üretimi (Nano Banana Pro prompt işi)
Kullanıcı kullanımı imkansız, bozuk, eksik veya model üzerinde giyilmiş tişört/elbise/kumaş görseli yükleyip düzeltilmesi / yeniden üretilmesi için prompt istediğinde:
- Doğrudan `docs/bozuk-gorselden-desen-uretim-promptu.md` içindeki **MeChi 3.3 & Desen Mühendisliği** protokolünü uygula.
- Görseli 7 maddelik kontrol listesine göre mikro düzeyde analiz et (hiçbir ayrıntıyı, nokta/çizgi yapısını, renk paletini ve tipografiyi atlama).
- Global iskelet + görsele uygun değişken modülleri (Modül A: Tam Tuval Metraj / Tekil Baskı, Modül B: Zemin Rengi, Modül C: Tipografi) birleştirerek **Nano Banana Pro** için tek görsel üretecek **İngilizce** promptu oluştur.

## Çalışma şekli
- Ağır hesap: `tools/lap_overprint_am.py`, `tools/tramsiz.py`, `tools/bolge_am.py`, `tools/film_etiket_ve_kros.py`, `tools/psd_incele.py` (Python, numpy/opencv/Pillow). Photoshop aktarımı: `tools/photoshop/*.jsx` (`master_psd_olustur.jsx`, `psd_olustur_tek_adim.jsx`, `kedihane_renk_ve_emprime_ayirici.jsx`).
- Yeni öğrenilen her şeyi, her denemeyi ilgili `docs/*.md` dosyasına işle; kod değişikliklerini commit + push et (kullanıcı bu repo için onay verdi).
- Orijinal görseller ve üretilen filmler repoya **konmaz** (`.gitignore`).
- Ortam sınırları ve Photoshop/computer use dersleri: `docs/photoshop-ve-ortam-notlari.md`.

Proje talimatı (Claude Project Instructions) metni: `docs/proje-talimatlari.md`.
