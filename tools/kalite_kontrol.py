#!/usr/bin/env python3
"""Film ciktisi icin otomatik kalite kontrol (v2 — yigilmali / indeks / tramsiz profilleri).
Karar rehberi bolum 7: docs/arastirma/gorsel-analiz-karar-rehberi.md

Kullanim:
  python3 tools/kalite_kontrol.py cikti_klasoru [kaynak.png] [PPI=508]
Ortam degiskenleri:
  PPI=508   SET=<SetAdi>   MOD=yigilmali|indeks|tramsiz
Cikis kodu 1 = uyari var, 0 = temiz. Gozle dogrulamanin yerini TUTMAZ.
"""
import sys, os, json, math
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Resmi SAATI / EU-US ipek katalog fizik degerleri ve Rule-of-N (N=5.0) sinirlari
SAATI_MESH_CATALOG = {
    "45": {"tpi": 115, "thread_um": 80, "open_area_pct": 40.7, "max_lpi": 25.0, "min_line_mm": 0.55},
    "55": {"tpi": 140, "thread_um": 70, "open_area_pct": 37.8, "max_lpi": 32.0, "min_line_mm": 0.48},
    "77": {"tpi": 195, "thread_um": 55, "open_area_pct": 33.4, "max_lpi": 45.0, "min_line_mm": 0.40},
    "90": {"tpi": 230, "thread_um": 48, "open_area_pct": 32.0, "max_lpi": 50.0, "min_line_mm": 0.32},
    "120": {"tpi": 305, "thread_um": 35, "open_area_pct": 33.5, "max_lpi": 65.0, "min_line_mm": 0.25},
}


def analyze_min_thickness_mm(mask_bool, ppi=508, min_mm=0.40):
    """Morfolojik acma + medial eksen mesafe donusumuyle en ince cizgi/nokta kalinligini (mm) olcer."""
    m = np.asarray(mask_bool, dtype=bool)
    total_ink = int(m.sum())
    if total_ink == 0:
        return {"min_detected_mm": 0.0, "thin_ratio_pct": 0.0, "thin_pixels": 0, "total_pixels": 0, "ok": True}

    px_mm = 25.4 / float(max(72, int(ppi)))
    m_u8 = m.astype(np.uint8)
    min_diam_px = max(3, int(round(float(min_mm) / px_mm)))
    k_sz = min_diam_px if (min_diam_px % 2 == 1) else (min_diam_px + 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))

    # Morfolojik acma: min_mm altindaki ince cizgi ve noktalar elenir, kalin govdeler korunur
    opened = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kern) > 0
    thin_px = int((m & (~opened)).sum())
    thin_ratio_pct = (thin_px / float(total_ink)) * 100.0

    # Medial eksen (yerel tepe mesafesi) uzerinden gercek minimum cizgi kalinligi
    dist_px = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    dil_dist = cv2.dilate(dist_px, np.ones((3, 3), np.uint8))
    medial = m & (dist_px >= (dil_dist - 1e-4)) & (dist_px > 0)
    vals = (2.0 * dist_px[medial] * px_mm) if np.any(medial) else (2.0 * dist_px[m] * px_mm)
    min_detected = float(np.percentile(vals, 2.0)) if vals.size > 0 else 0.0

    return {
        "min_detected_mm": round(min_detected, 3),
        "thin_ratio_pct": round(thin_ratio_pct, 2),
        "thin_pixels": thin_px,
        "total_pixels": total_ink,
        "ok": bool(thin_ratio_pct <= 25.0),
    }


def parse_args():
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python3 tools/kalite_kontrol.py cikti_klasoru [kaynak.png] [PPI=508]")
    d = sys.argv[1]
    srcp = None
    ppi_arg = None
    for a in sys.argv[2:]:
        if a.isdigit():
            ppi_arg = int(a)
        elif srcp is None:
            srcp = a
    ppi_exp = ppi_arg or int(os.environ.get('PPI', 508))
    return d, srcp, ppi_exp


def rgb_to_lab_single(rgb):
    return cv2.cvtColor(np.array([[rgb]], np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0, 0]


def normalize_channel_entry(entry):
    """Hem eski [nm, fn, rgb] hem yeni {'ad','dosya','rgb','rol','solidity','aile'} formatini destekler."""
    if isinstance(entry, dict):
        return {
            'ad': entry.get('ad', entry.get('dosya', '')),
            'dosya': entry['dosya'],
            'rgb': entry['rgb'],
            'rol': entry.get('rol', 'tram'),
            'solidity': entry.get('solidity', None),
            'aile': entry.get('aile', None),
        }
    nm, fn, rgb = entry[0], entry[1], entry[2]
    low = nm.lower()
    rol = 'fon' if ('fon' in low or 'underbase' in low or 'beyaz zemin' in low) else 'tram'
    return {'ad': nm, 'dosya': fn, 'rgb': rgb, 'rol': rol, 'solidity': None, 'aile': None}


def is_underbase_or_white(ch):
    low = ch['ad'].lower()
    return ch['rol'] in ('fon', 'fon_koyusuz', 'ust_beyaz', 'kumas') or ('fon' in low) or ('underbase' in low) or ('beyaz zemin' in low)


def main():
    d, srcp, PPI_EXP = parse_args()
    meta_file = os.path.join(d, 'kanallar_hibrit.json')
    if not os.path.exists(meta_file):
        meta_file = os.path.join(d, 'kanallar_tramsiz.json')
    with open(meta_file, encoding='utf-8') as fp:
        J = json.load(fp)

    mod = os.environ.get('MOD') or J.get('mod') or ('tramsiz' if 'tramsiz' in os.path.basename(meta_file) else ('yigilmali' if J.get('lap_altlik') else 'indeks'))
    set_names = [os.environ['SET']] if os.environ.get('SET') else list(J['sets'].keys())

    warn = []
    info = [f'Mod: {mod} | Beklenen PPI: {PPI_EXP} | Setler: {", ".join(set_names)}']

    for setn in set_names:
        info.append(f'--- Set: {setn} ---')
        raw_sets = J['sets'][setn]
        films = []
        for entry in raw_sets:
            ch = normalize_channel_entry(entry)
            fpath = os.path.join(d, setn, ch['dosya'])
            im = Image.open(fpath)
            if im.mode != '1':
                warn.append(f"[{setn}] {ch['dosya']}: 1-bit degil ({im.mode})")
            dp = im.info.get('dpi', (None,))[0]
            if dp is None or abs(dp - PPI_EXP) > 1:
                warn.append(f"[{setn}] {ch['dosya']}: PPI meta {dp}, beklenen {PPI_EXP}")
            a = ~np.array(im.convert('1'))  # True = murekkep
            ch['mask'] = a
            ch['lab'] = rgb_to_lab_single(ch['rgb'])
            films.append(ch)

        if not films:
            warn.append(f'[{setn}] hic film yok')
            continue

        H, W = films[0]['mask'].shape
        for ch in films:
            if ch['mask'].shape != (H, W):
                warn.append(f"[{setn}] {ch['dosya']}: boyut farkli {ch['mask'].shape}")
        info.append(f'boyut {W}x{H} px ({W/PPI_EXP*2.54:.1f}x{H/PPI_EXP*2.54:.1f} cm), {len(films)} film')

        # 1) Kapsama ve Solidity kontrolu
        tot = H * W
        for ch in films:
            c = ch['mask'].sum() / tot * 100.0
            sol_s = f" sol={ch['solidity']}%" if ch['solidity'] is not None else ""
            info.append(f"  {ch['ad']:<22} rol={ch['rol']:<10} kapsama %{c:.2f}{sol_s}")
            if c < 0.5 and not is_underbase_or_white(ch):
                warn.append(f"[{setn}] {ch['ad']}: kapsama %{c:.2f} <0,5 — neredeyse kullanilmayan murekkep")
            if ch['solidity'] is not None:
                if ch['rol'] in ('fon', 'fon_koyusuz', 'kumas') and ch['solidity'] != 100:
                    warn.append(f"[{setn}] {ch['ad']}: fon/kumas Solidity %{ch['solidity']} (beklenen %100)")
                elif ch['rol'] in ('tram', 'lap', 'ust_beyaz') and ch['solidity'] == 100 and mod == 'yigilmali':
                    info.append(f"    (not: {ch['ad']} Solidity %100; atolye overprint onizlemesi icin %16 onerilir)")

        # 2) Cakisma kontrolu (mode gore)
        # Eger alt 3 cm etiket+kros seridi varsa (kros_etiket=true veya boy farki),
        # kanallar arasi cakisma kontrolunu desen alaninda (alt kros+etiket haric) yap:
        strip_px = int(round(30 * (PPI_EXP / 25.4))) if J.get('kros_etiket') else 0
        H_art_ov = H - strip_px if (strip_px > 0 and H > strip_px * 2) else H
        for i in range(len(films)):
            for j in range(i + 1, len(films)):
                ci, cj = films[i], films[j]
                if is_underbase_or_white(ci) or is_underbase_or_white(cj):
                    continue
                mi = ci['mask'][:H_art_ov, :]
                mj = cj['mask'][:H_art_ov, :]
                min_area = min(int(mi.sum()), int(mj.sum()))
                if min_area == 0:
                    continue
                ov = int((mi & mj).sum())
                ov_pct = ov / min_area * 100.0

                if mod == 'indeks':
                    if ov_pct > 0.1:
                        warn.append(f"[{setn}] {ci['ad']} ile {cj['ad']} cakisiyor: {ov} px (%{ov_pct:.2f})")
                elif mod == 'yigilmali':
                    # Ayni aile icinde veya siyah/notr golge ustunde yigilma serbest (700/701 dot-on-dot)
                    li, lj = ci['lab'], cj['lab']
                    chi_c, chj_c = float(math.hypot(li[1], li[2])), float(math.hypot(lj[1], lj[2]))
                    if li[0] < 22 or lj[0] < 22 or chi_c < 5 or chj_c < 5:
                        continue  # siyah/koyu notr ust uste serbest
                    if ci['aile'] is not None and cj['aile'] is not None and ci['aile'] == cj['aile']:
                        continue  # ayni aile ici lap+tram veya acik+koyu yigilma serbest
                    # Zit kromatik aileler arasi cakisma YASAK (Lab ton farki > 110 derece)
                    hi_deg = math.degrees(math.atan2(li[2], li[1]))
                    hj_deg = math.degrees(math.atan2(lj[2], lj[1]))
                    dh = abs(((hi_deg - hj_deg + 180.0) % 360.0) - 180.0)
                    if dh > 110.0 and ov_pct > 1.5:
                        warn.append(f"[{setn}] ZIT AILE CAKISMASI ({dh:.0f}°): {ci['ad']} ile {cj['ad']} %{ov_pct:.2f} ({ov} px)")
                elif mod == 'tramsiz':
                    # Trap payindan buyuk anormal cakisma kontrolu
                    if ov_pct > 15.0:
                        warn.append(f"[{setn}] tramsiz asiri cakisma: {ci['ad']} ile {cj['ad']} %{ov_pct:.2f}")

        # 3) Izole kucuk noktalar (<= 3-4 px)
        cell = PPI_EXP / 45.0
        tiny_area = max(3, int(0.03 * cell * cell))
        for ch in films:
            nlab, _, st, _ = cv2.connectedComponentsWithStats(ch['mask'].astype(np.uint8), connectivity=8)
            areas = st[1:, 4] if nlab > 1 else np.array([], dtype=np.int32)
            tn = int((areas <= tiny_area).sum())
            tot_n = len(areas)
            frac = tn / max(tot_n, 1)
            info.append(f"  {ch['ad']:<22} bilesen {tot_n}, <= {tiny_area}px: {tn} (%{frac*100:.1f})")
            is_black_tram = (ch['rol'] == 'tram' and ch['lab'][0] < 20)
            if ch['rol'] in ('fon', 'fon_koyusuz', 'lap', 'ust_beyaz'):
                if frac > 0.02 and tn > 50:
                    warn.append(f"[{setn}] {ch['ad']} (duz/lap): %{frac*100:.1f} ({tn}) mikro parca <= {tiny_area}px")
            elif not is_black_tram:
                if frac > 0.10 and tn > 2000:
                    warn.append(f"[{setn}] {ch['ad']}: %{frac*100:.0f} bilesen <= {tiny_area}px (elekte tutunmayacak mikro nokta)")

            # 3B) Duz / lap kanallarda 0.40 mm minimum pozitif cizgi kalinligi denetimi
            if ch['rol'] in ('lap', 'fon', 'ust_beyaz') or mod == 'tramsiz':
                th_res = analyze_min_thickness_mm(ch['mask'], ppi=PPI_EXP, min_mm=0.40)
                if th_res['total_pixels'] > 0:
                    info.append(
                        f"    kalinlik (%2 pctl): {th_res['min_detected_mm']:.2f} mm, <0.40mm payi %{th_res['thin_ratio_pct']:.1f}"
                    )
                    if not th_res['ok'] and ch['rol'] not in ('fon', 'fon_koyusuz'):
                        warn.append(
                            f"[{setn}] {ch['ad']}: %{th_res['thin_ratio_pct']:.1f} alan < 0.40 mm (kalip yikamada patlama riski)"
                        )

        # 4) Duz bolgeler kontrolu
        for dz in J.get('duz_bolgeler', []):
            fa = [c for c in films if c['dosya'] == dz.get('film')]
            if not fa or not dz.get('bbox'):
                continue
            ch = fa[0]
            bx, by, bw, bh = dz['bbox']
            m = ch['mask'][by:by + bh, bx:bx + bw].astype(np.uint8)
            mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cs, _ = cv2.findContours(mo, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            per = max(sum(len(c) for c in cs), 1)
            rough = float((m.sum() - mo.sum()) / per)
            sl = rgb_to_lab_single(dz['kaynak_rgb'])
            de = float(np.linalg.norm(sl - ch['lab']))
            info.append(f"  duz bolge {dz['ad']:<10} film {ch['ad']}: kenar puruzlulugu {rough:.3f}, renk dE {de:.1f}")
            if rough > 0.03:
                warn.append(f"[{setn}] duz bolge {dz['ad']} ({ch['ad']}): puruzluluk {rough:.3f} > 0,03")
            if de > 12:
                warn.append(f"[{setn}] duz bolge {dz['ad']}: renk dE {de:.1f} > 12")

        # 5) Zit renk benegi (kaynak verilmisse — boyut farkliysa otomatik olcekler, RGBA alfa maskeler)
        if srcp:
            im_s = Image.open(srcp)
            f4 = 4
            h4, w4 = max(1, H // f4), max(1, W // f4)
            if im_s.mode in ('RGBA', 'LA') or ('transparency' in im_s.info):
                rgba_s = np.array(im_s.convert('RGBA'))
                af_s = rgba_s[..., 3:4].astype(np.float32) / 255.0
                src = np.clip(rgba_s[..., :3] * af_s + 255.0 * (1.0 - af_s), 0, 255).astype(np.uint8)
                src_alpha = rgba_s[..., 3]
            else:
                src = np.array(im_s.convert('RGB'))
                src_alpha = None
            # Eger filmlerde alt 3 cm etiket+kros seridi varsa (film boyu kaynak orantisindan ~600 px uzunsa),
            # kaynak ton uyumsuzlugu kontrolunu yalnizca desen yuksekliginde (H_art) yap:
            H_art = int(round(im_s.height * (W / float(im_s.width))))
            if H > H_art and (H - H_art) <= int(round(45 * (PPI_EXP / 25.4))):
                H_chk = H_art
            else:
                H_chk = H
            h4, w4 = max(1, H_chk // f4), max(1, W // f4)
            if src_alpha is not None:
                valid_s = cv2.resize(src_alpha, (w4, h4), interpolation=cv2.INTER_AREA) >= 128
            else:
                valid_s = np.ones((h4, w4), dtype=bool)
            src_s = cv2.resize(src, (w4, h4), interpolation=cv2.INTER_AREA)
            sl = cv2.cvtColor(cv2.GaussianBlur(src_s, (0, 0), 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
            sch = np.hypot(sl[..., 1], sl[..., 2])
            shu = np.degrees(np.arctan2(sl[..., 2], sl[..., 1]))
            del src, src_s
            for ch in films:
                if is_underbase_or_white(ch):
                    continue
                il = ch['lab']
                ch_c = float(math.hypot(il[1], il[2]))
                hu = float(math.degrees(math.atan2(il[2], il[1])))
                if ch_c <= 5.0:
                    continue
                cov = cv2.resize(ch['mask'][:H_chk, :].astype(np.float32), (w4, h4), interpolation=cv2.INTER_AREA)
                dh = np.abs(((shu - hu + 180.0) % 360.0) - 180.0)
                bad = valid_s & (sch > 8.0) & (dh > 90.0)
                if il[0] < 35.0:
                    bad &= (sl[..., 0] > 55.0)
                frac = float((cov * bad).sum() / max(cov.sum(), 1e-6))
                info.append(f"  {ch['ad']:<22} kaynak tonuyla uyumsuz (>90°) payi %{frac*100:.2f}")
                esik = 0.03 if il[0] < 35.0 else 0.015
                if frac > esik:
                    warn.append(f"[{setn}] {ch['ad']}: %{frac*100:.1f} tonu uyumsuz kaynak alanda (zit renk benegi)")

    for w_ in J.get('uyarilar', []):
        warn.append(w_)

    print('\n'.join(info))
    print()
    if warn:
        print('UYARILAR:')
        for w in warn:
            print(' -', w)
        sys.exit(1)
    print('Otomatik kontrol temiz. Yine de %100 kesit ve onizlemeyle gozle dogrula.')


if __name__ == '__main__':
    main()
