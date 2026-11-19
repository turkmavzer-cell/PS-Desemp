# Açık sorular ve yapılacaklar

## Atölyeye / ustaya sorulacaklar
(Araştırma raporundan uyarlanmıştır.)
1. Elek: renkler ve underbase için hangi T numarası ve iplik kalınlığı?
2. Film mi basılıyor, doğrudan kalıba (CTS/CTP) mı pozlanıyor? Film ise hangi yazıcı, Dmax yeterli mi?
3. ~~Mürekkep: plastisol, su bazlı, pigment, discharge?~~ **Çözüldü (2026-11-19, [usta]):** 3 temel boya cinsi kullanılır: **Beyazlı boya** (beyaz + kök boya + çatlama önleyici az şeffaf → koyu zemin & fonlu canlı işler; koyu kıvamlı, tıkanmaya meyilli, sert tuşe) ve **Pat boya** (pat + kök boya → açık zemin & CMYK "fotoğraf baskı"; suluboya kıvamlı, tıkanmaz, yumuşak tuşe, koyu kumaşta kaybolur). Ayrıntı: `docs/emprime-kanal-ayirma.md` §P.
4. Tişört rengi? Geniş koyu LAP bloklar basılmayıp kumaş rengi mi kullanılsın, yoksa çift fon (`1_fon` + `1_fon_koyusuz`) mı verilsin?
5. Pres manuel mi otomatik mi? Butt registration mı trap mı? Trap payı kaç mm?
6. Dosya biçimi: 2 katlı master PSD (sürekli ton + 1-bit tram), ayrı 1-bit film PNG/TIFF, vektör (AI/PDF)?
7. En ince çizgi ve en küçük yazı için minimum kalınlık (ör. 0,3 mm / 8 pt)?
8. Renk sırası ve ara kurutma?
9. Kayıt işareti biçimi, pozlama testi, ilk provayı onaya gönderme? *(Not: "Lap" terimi `700.psd`/`701.psd` ile çözüldü: %100 dolu tramsız katı renk.)*

## Geliştirme listesi
- [x] Palet otomatik tespiti `tramsiz.py`'ye eklensin (`--renk N`, `--palet`, 508 PPI, choke 2–3 px, üst beyaz). *(2026-11-19)*
- [ ] Turuncu/kırmızı ve krem/beyaz ayrımı için palet genişletme seçeneği.
- [ ] Anahattı açık kapalı şekiller (örn. "E" harfi): kapatma/doldurma seçeneği veya uyarı.
- [ ] Vektöre çevirme (trace) yolu: ne zaman raster yeterli, ne zaman şart?
- [x] Kayıt işaretleri (`+`) ve 4 alanlı film kenar bilgisi (beden/yüz, kanal adı, `Xr`, iş no, tarih) eklensin (`tools/film_etiket_ve_kros.py`, saf 1-bit). *(2026-11-19)*
- [ ] Film çıkışı: 1-bit PNG yeterli mi, TIFF/PDF mi isteniyor?
- [x] Belgeyi PSB destekli + 2 katlı master PSD (kumaş kanalı `100%`, `fon 100%`, üst renkler `16%` Solidity) kaydetme adımı (`tools/photoshop/master_psd_olustur.jsx`). *(2026-11-19)*
- [x] Birleşik JSX betiğinde tek adımda renk + Solidity (`100%/16%`) ataması (`psd_olustur_tek_adim.jsx` v2 / `master_psd_olustur.jsx`). *(2026-11-19)*
- [x] Gerçek degradeli / yığılmış tonlu tasarımlar için **LAP + Dot-on-Dot Yığılmış AM** motoru (`tools/lap_overprint_am.py`, sürekli ton tavanı ile dot gain kompanzasyonu, çift fon, üst beyaz). *(2026-11-19)*
- [ ] Başka test görselleri (logo, yazı ağırlıklı, degradeli) ile yöntemi sına.
- [x] Underbase seçenekleri: `2–3 px` choke, tramlı siyah altı oyulmaz, geniş koyu lap için `--cift-fon`, koyu kumaşta `--ust-beyaz`. *(2026-11-19)*

## Karar bekleyenler
- Usta ve atölye onayı: yeni `lap_overprint_am.py` ve v3/v10 filmleri gerçek fiziksel baskıda denenmedi.
- Koyu tişörtte lacivert-siyah anahatların görünmemesi tasarım değişikliği gerektirebilir (anahat rengi).


## Eklenenler (13 Ekim)
- [ ] Atölye: 0,127 mm (3 px) ve 0,085 mm (2 px) dagitik tram noktasi hangi elekte basilabilir? Ayicik icin FM mi AM mi tercih?
- [ ] Ayicik koyu tisort seti (underbase) istenirse eklenecek.
- [ ] hibrit.py: palet ve esikler otomatik/parametrik olsun; `--koyu` secenegi.
- [ ] Mandala v4: lacivert cizgi kanali, krem/sari kontur, potrace tabanli arac `tools/` altina alinacak.
- [ ] Ayicik: `GAM=1 SHARP=0.25` ile tum gorseli yeniden ayir, cene/göz/yanak kontrol et, filmleri ve PSD'yi guncelle.
- [ ] Koyu bolgelerdeki dagnik tek beyaz noktalar (parlama maskesi kalintisi) temizlensin.

- [x] Koyu alanlardaki beyaz benekler SPEC_MIN ile temizlendi (v5.3).
- [ ] Atölye (WhatsApp grubu) görüşü bekleniyor: 3 px (0,127 mm) nokta baskıda tutar mı; siyah kanal (%0,17) ayrı film olarak mantıklı mı; elek/mürekkep/film türü.
- [x] Ayıcık AM tram denemesi yapıldı (45 lpi, 15°, yuvarlak nokta, tek açı, 9 film; hibrit2.py AM=1). *(Açı/nokta standardı 699–701 dosyalarıyla %100 kesinleşti: 45 LPI, 15°, yuvarlak, tek açı + dot-on-dot.)*
- [ ] Atölyenin makine kafa sayısı (6? 8?) ve kurutma durumu → hedef mürekkep sayısı; ayıcık için 6/7/8 renkli azaltılmış set (siyah, 3 kahve, lacivert, +beyaz/fon). Beyaz tişörtte fon yok, Be Gentle içi kâğıt rengi; koyu tişörtte beyaz mürekkep gerekir.
- [ ] Atölyeye sor (karar rehberi bölüm 8): 45 LPI için elek sayısı, en küçük basılabilir nokta/detay, siyah gibi `<%0,5` kanalın birleştirilmesi.
- [ ] Kız kişisi: PSD'yi yeniden oluştur (düzeltilmiş filmler `Emprime_Kiz_AM_508`'de; Photoshop'ta bilgisayar erişimi gerekir). Bej kanal %0,21 → atılıp 5 renge inilebilir.
- [x] analiz.py (şeffaf PNG alfa maskeleme + yöntem önerisi) ve kalite_kontrol.py (`yigilmali`/`indeks`/`tramsiz` profilleri + kaynak ölçekleme) güncellendi. *(2026-11-19)*
- [x] **Bölge saflığı planını uygula** (v8, `tools/bolge_am.py`; segmentasyon yerine ton ailesi + yumuşatma kullanıldı) (`docs/plan-bolge-safligi.md`).
- [ ] bolge_am: aile içinde koyu ton gerekirse (koyu turkuaz kontur) aileye koyu mürekkep ayır; elbise beyazı/kâğıt ayrımı için daha iyi figür maskesi.
- [ ] bolge_am'ı ayıda dene (geriye dönük kontrol).
- [ ] İki aşamalı yöntemi gerçek görselle dene (hem 2 katlı sürekli ton PSD hem indexed PNG akışı).

- [ ] Kız kişisi v9.1: önizleme (İndirilenler\Emprime_Kiz_Onizleme) kullanıcı onayı bekleniyor; onaydan sonra `onizlemeden_ayir.py ... acik onizleme_palet.json`, kalite kontrol, PSD.

- [x] "Lap" tam olarak ne demek? **`700.psd` (`6 koyu kahve lap`, `2 krem`) ve `701.psd` (`açık kırmızı`) ile %100 kesinleşti:** LAP = tramsız, `%100` dolu katı (solid) renk kalıbı (bölge alt-tabanı veya bağımsız katı grafik/yazı). *(2026-11-19)*
- [x] LAP + üst üste (`dot-on-dot`) baskı desteği: `tools/lap_overprint_am.py` ve `tools/kalite_kontrol.py` (`mod: yigilmali`). *(2026-11-19)*

## Eklenenler (20 Ekim) — bölge/mürekkep katmanları
- [x] `japon_katmanli.psd` kullanıcı tarafından açılıp kontrol edildi: "başarılı, katmanlar başarılı ayrıştırılmış" (2026-10-20).
- [ ] Teslim edilen PSD'nin adları Türkçe karakterli; kullanıcı sonucu başarılı buldu, ASCII sürüm istemedi. Kural (ASCII ad) geçerli kalır; bu dosya için yeniden üretim yapılmadı, sonraki işlerde baştan ASCII ver.
- [ ] Fuji'nin alt koyu kütlesi Dağ grubuna verildi; kullanıcı başka istiyor mu?
- [ ] Siyah katmanlardaki yarı saydam kâğıt dokusu kabul mü, eşikle sıfırlansın mı (daha temiz, kenar yumuşaklığı azalır)?
- [ ] Bölge maskesini otomatik çıkaran genel yol yok (örnek el çizimi çokgenlerle kılavuzlu); başka görselde yeniden kurulacak.
