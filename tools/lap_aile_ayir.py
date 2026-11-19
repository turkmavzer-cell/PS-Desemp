#!/usr/bin/env python3
"""
Kedihane — v3 Karma Emprime Renk Ayirici (Duz LAP + Keskin Stroke + Yalnizca Degradede Tram + %85 Alt Bindirme).

v3 Ozellikleri:
  1. Piksel Turu Haritasi (compute_flat_stroke_gradient_maps):
     - stroke_mask: Siyah kontur ve duz siyah bloklar -> Siyah kanalinda %100 kati (TRAMSIZ).
     - flat_mask: Duz boyali alanlar -> Ilgili spot renginde %100 kati LAP (TRAMSIZ, solma yok).
     - grad_mask: En az 6-10 px surekli renk degisimi gosteren gercek degrade / yumusak golgeler.
  2. Degrade Bolgesinde Surekli Renk Karisimi (solve_gradient_weights):
     - argmax sert kesimi yapilmaz; en yakin 2 renk ailesi arasinda dogrusal projeksiyon + %85 alt bindirme.
  3. Karma Kanal Tramlama (screen_channel_hybrid):
     - >= 0.96 yogunluk -> %100 kati 1-bit (TRAMSIZ); [min_dot..0.96) -> 45/55 LPI yuvarlak AM tram.
"""
import argparse
import datetime
import io
import json
import os
import re
import struct
import sys
import zlib

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw
from psd_tools import PSDImage
from psd_tools.compression.rle import encode as encode_rle
from psd_tools.constants import Compression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lap_overprint_am as L
from film_etiket_ve_kros import stamp_single_film


# ---------------------------------------------------------------------------
# 1) OTOMATIK TOPOLOJIK MASKE & KAPALI GOKYUZU CEBI TEMIZLEYICI (SIFIR SABIT KOORDINAT)
# ---------------------------------------------------------------------------

def expand_connected_from_seed(seed_mask, candidate_mask):
    """candidate_mask icinde seed_mask'e 4-bagli (bitisik) tum bilesenleri dondurur."""
    union = (seed_mask | candidate_mask).astype(np.uint8)
    _, lbl = cv2.connectedComponents(union, connectivity=4)
    seed_lbls = np.unique(lbl[seed_mask])
    seed_lbls = seed_lbls[seed_lbls > 0]
    if len(seed_lbls) == 0:
        return np.zeros_like(candidate_mask, dtype=bool)
    return candidate_mask & np.isin(lbl, seed_lbls)


def _border_uniform_bg_ratio(rgb):
    """Kenar seridindeki piksellerin medyan arka plan rengiyle ne oranda eslestigini (0..1) dondurur."""
    h, w = rgb.shape[:2]
    bw = max(4, min(h, w) // 50)
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3),
        rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3),
        rgb[:, -bw:].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(border, axis=0).astype(np.uint8)
    bg_lab = L.rgb_tuple_to_lab(tuple(int(x) for x in bg_rgb))
    border_lab = L.rgb_to_lab_cv(border.reshape(-1, 1, 3)).reshape(-1, 3)
    de_border = np.sqrt(((border_lab - bg_lab[None, :]) ** 2).sum(axis=-1))
    return float((de_border < 10.0).mean())


def build_auto_clean_figure_mask(im_pil, mask_path=None, zemin_maske="oto"):
    """
    Sifir sabit koordinatli, tamamen topolojik ve renk-yapi tabanli figur + kapali gokyuzu cebi temizleyici.
    --zemin-maske:
      - 'tumu': Tum tuval baskiya girer (fig_clean = True), hicbir gokyuzu cebi oyulmaz.
      - 'maskeli': Kenara bagli dis arka plan cikarilir + kapali siyah halat kafesi gokyuzu cepleri temizlenir.
      - 'oto' (varsayilan): Kenar seridinde duz arka plan orani < %35 ise (tam tuval metraj desen) otomatik
        'tumu' moduna gecer; >= %35 ise 'maskeli' calisir (ancak cerceveli/metraj iceride %82+ doluysa delmez).
    Donus: (rgb_full, fig_clean)
    """
    has_alpha = (
        (im_pil.mode in ("RGBA", "LA") or ("transparency" in im_pil.info))
        and (np.array(im_pil.convert("RGBA"))[..., 3] < 250).mean() > 0.01
    )

    if zemin_maske == "tumu" and not mask_path:
        if has_alpha:
            rgba = np.array(im_pil.convert("RGBA"))
            af = rgba[..., 3:4].astype(np.float32) / 255.0
            rgb = np.clip(rgba[..., :3] * af + 255.0 * (1.0 - af), 0, 255).astype(np.uint8)
        else:
            rgb = np.array(im_pil.convert("RGB"))
        fig_full = np.ones(rgb.shape[:2], dtype=bool)
        print("[auto_mask] Zemin modu=tumu: Tum tuval baskiya dahil (%100.00)")
        return rgb, fig_full

    if zemin_maske == "oto" and not mask_path and not has_alpha:
        rgb_chk = np.array(im_pil.convert("RGB"))
        border_ratio = _border_uniform_bg_ratio(rgb_chk)
        if border_ratio < 0.35:
            fig_full = np.ones(rgb_chk.shape[:2], dtype=bool)
            print(f"[auto_mask] Zemin modu=oto -> metraj/tam tuval algilandi (kenar={border_ratio*100:.1f}% < 35%), tum tuval korunuyor.")
            return rgb_chk, fig_full

    rgb, fig = L.extract_figure_mask(im_pil, mask_path)
    h, w = fig.shape

    # Cerceveli veya nerdeyse tam tuval desenlerde (fig > %82) ic yaprak/cicek dokusunu asla delme
    if zemin_maske == "oto" and not mask_path and fig.mean() > 0.82:
        print(f"[auto_mask] Zemin modu=oto -> ic alan dolu ({fig.mean()*100:.1f}% > 82%), gokyuzu cebi oyucu devre disi.")
        return rgb, fig

    s = 2 if max(h, w) > 3500 else 1
    sh, sw = h // s, w // s
    rgb_s = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    fig_s = cv2.resize(fig.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    lab_s = L.rgb_to_lab_cv(rgb_s)
    L_s, a_s, b_s = lab_s[..., 0], lab_s[..., 1], lab_s[..., 2]
    C_s = np.hypot(a_s, b_s)
    de_w = np.sqrt((100.0 - L_s) ** 2 + a_s ** 2 + b_s ** 2)

    # A) Otomatik Kirmizi / Kromatik Odak Nesnesi Konveks Zarf Kilidi (yalnizca ic yazi/far korumasi icin)
    red_core = fig_s & (a_s > 16.0) & (C_s > 24.0)
    red_closed = cv2.morphologyEx(red_core.astype(np.uint8), cv2.MORPH_CLOSE, L.disk(20))
    n_rc, _, st_rc, _ = cv2.connectedComponentsWithStats(red_closed, connectivity=8)
    focal_lock = np.zeros((sh, sw), dtype=bool)

    if n_rc > 1:
        areas = st_rc[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max())
        if max_area > int(0.01 * sh * sw):
            main_id = int(np.argmax(areas)) + 1
            bx, by, bw, bh = st_rc[main_id, :4]
            pad = int(round(0.01 * max(sh, sw)))
            y0, y1 = max(0, by - pad), min(sh, by + bh + pad)
            x0, x1 = max(0, bx - pad), min(sw, bx + bw + pad)
            sub_red = np.zeros((sh, sw), dtype=np.uint8)
            sub_red[y0:y1, x0:x1] = (a_s[y0:y1, x0:x1] > 14.0).astype(np.uint8)
            sub_closed = cv2.morphologyEx(sub_red, cv2.MORPH_CLOSE, L.disk(20))
            cnts, _ = cv2.findContours(sub_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts_list = [c for c in cnts if cv2.contourArea(c) > 200]
            if pts_list:
                hull_img = np.zeros((sh, sw), dtype=np.uint8)
                hull = cv2.convexHull(np.vstack(pts_list))
                cv2.drawContours(hull_img, [hull], -1, 1, thickness=-1)
                focal_lock = cv2.dilate(hull_img, L.disk(5)) > 0

    # B) Otomatik Sicak Altin/Bej Mimari Ic Bosluk Kilidi
    warm_arch = fig_s & (b_s > 11.0) & (C_s > 14.0) & (L_s >= 25.0) & (L_s <= 86.0) & (a_s < 16.0)
    warm_closed = cv2.morphologyEx(warm_arch.astype(np.uint8), cv2.MORPH_CLOSE, L.disk(28))
    inv_w = (~(warm_closed > 0)).astype(np.uint8)
    nw, lw, sw_st, _ = cv2.connectedComponentsWithStats(inv_w, connectivity=4)
    warm_interior_lock = np.zeros((sh, sw), dtype=bool)
    for i in range(1, nw):
        bx, by, bw, bh, ar = sw_st[i]
        if bx > 0 and by > 0 and (bx + bw < sw) and (by + bh < sh) and ar < int(0.004 * sh * sw):
            warm_interior_lock |= (lw == i)

    yy_s, xx_s = np.ogrid[:sh, :sw]
    water_band = (yy_s >= int(0.735 * sh))
    safe_lock = focal_lock | warm_interior_lock | water_band

    # C) Asama 1: Koyu siyah halat/kiris kafesiyle (medyan L < 18) sarili saf beyaz gokyuzu pencereleri
    pure_white = fig_s & (~safe_lock) & (L_s > 92.0) & (C_s < 5.0)
    n_pw, lbl_pw, st_pw, _ = cv2.connectedComponentsWithStats(pure_white.astype(np.uint8), connectivity=4)
    min_area_pw = int(160 * (sw / 2048.0) ** 2)

    sky_holes = np.zeros((sh, sw), dtype=bool)
    for i in range(1, n_pw):
        area = st_pw[i, cv2.CC_STAT_AREA]
        if area < min_area_pw:
            continue
        cm = (lbl_pw == i)
        bx, by, bw, bh = st_pw[i, :4]
        y0, y1 = max(0, by - 7), min(sh, by + bh + 7)
        x0, x1 = max(0, bx - 7), min(sw, bx + bw + 7)
        sub_c = cm[y0:y1, x0:x1].astype(np.uint8)
        ring = (cv2.dilate(sub_c, L.disk(6)) > 0) & (sub_c == 0) & fig_s[y0:y1, x0:x1]
        if not ring.any():
            continue
        ra = a_s[y0:y1, x0:x1][ring]
        rb = b_s[y0:y1, x0:x1][ring]
        rL = L_s[y0:y1, x0:x1][ring]
        if (ra > 14.0).mean() > 0.12:
            continue
        if (rb > 10.0).mean() > 0.22:
            continue
        if np.median(rL) < 18.0 and (rL < 38.0).mean() > 0.70:
            sky_holes |= cm

    # D) Asama 2: Halat kafesi zarfi icindeki ufuk gokyuzu bosluklari ve asma kablo arasi gokyuzu cepleri
    if sky_holes.any():
        hy, hx = np.where(sky_holes)
        cables_x0 = max(0, int(hx.min()) - int(0.10 * sw))
        cables_x1 = min(sw, int(hx.max()) + int(0.08 * sw))
        cables_y0 = max(0, int(hy.min()) - int(0.09 * sh))
        cables_y1 = min(sh, int(hy.max()) + int(0.075 * sh))
        cable_zone = np.zeros((sh, sw), dtype=bool)
        cable_zone[cables_y0:cables_y1, cables_x0:cables_x1] = True
        cable_zone &= (~safe_lock)

        hy_med = int(np.percentile(hy, 45))
        low_cable_zone = cable_zone & (yy_s >= hy_med)
        cand_low = low_cable_zone & fig_s & (~sky_holes) & (de_w < 12.2) & (L_s > 89.0) & (a_s < 8.0)
        add_low = expand_connected_from_seed((~fig_s) | sky_holes, cand_low)
        rem_low = cand_low & (~add_low) & (yy_s <= int(hy.max()) + int(0.045 * sh))
        nl, ll, sl, _ = cv2.connectedComponentsWithStats(rem_low.astype(np.uint8), connectivity=4)
        for i in range(1, nl):
            cm = (ll == i)
            if sl[i, cv2.CC_STAT_AREA] >= 45:
                bx, by, bw, bh = sl[i, :4]
                y0, y1 = max(0, by - 5), min(sh, by + bh + 5)
                x0, x1 = max(0, bx - 5), min(sw, bx + bw + 5)
                ring = (cv2.dilate(cm[y0:y1, x0:x1].astype(np.uint8), L.disk(4)) > 0) & (~cm[y0:y1, x0:x1]) & fig_s[y0:y1, x0:x1]
                if ring.any() and (b_s[y0:y1, x0:x1][ring] > 11.0).mean() < 0.25:
                    add_low |= cm
        sky_holes |= add_low

        hx_mid = int(np.percentile(hx, 50))
        upper_cable_zone = cable_zone & (xx_s >= hx_mid) & (yy_s <= hy_med + int(0.04 * sh))
        cand_tri = upper_cable_zone & fig_s & (~sky_holes) & (L_s > 81.5) & (C_s < 9.5) & (b_s < 5.5) & (a_s < 4.0)
        nt, lt, st_t, _ = cv2.connectedComponentsWithStats(cand_tri.astype(np.uint8), connectivity=4)
        add_up = np.zeros((sh, sw), dtype=bool)
        for i in range(1, nt):
            if st_t[i, cv2.CC_STAT_AREA] >= 55:
                cm = (lt == i)
                bx, by, bw, bh = st_t[i, :4]
                y0, y1 = max(0, by - 5), min(sh, by + bh + 5)
                x0, x1 = max(0, bx - 5), min(sw, bx + bw + 5)
                ring = (cv2.dilate(cm[y0:y1, x0:x1].astype(np.uint8), L.disk(4)) > 0) & (~cm[y0:y1, x0:x1]) & fig_s[y0:y1, x0:x1]
                if ring.any() and (L_s[y0:y1, x0:x1][ring] < 68.0).mean() > 0.15:
                    add_up |= cm

        mid_cable_zone = upper_cable_zone & (yy_s >= max(0, int(hy.min()) - int(0.04 * sh)))
        cand_mid = mid_cable_zone & fig_s & (~sky_holes) & (L_s > 84.0) & (C_s < 6.5) & (a_s < 8.0)
        add_up |= expand_connected_from_seed((~fig_s) | sky_holes | add_up, cand_mid)

        roof_band = mid_cable_zone & (yy_s >= int(hy.min()) - int(0.02 * sh))
        rem_mid = cand_mid & (~add_up)
        nr, lr, sr, _ = cv2.connectedComponentsWithStats(rem_mid.astype(np.uint8), connectivity=4)
        for i in range(1, nr):
            ar = sr[i, cv2.CC_STAT_AREA]
            cm = (lr == i)
            if ar >= 350 or (ar >= 25 and (cm & roof_band).any()):
                add_up |= cm

        sky_holes |= add_up

        # E) Asama 3: Dis arka plana (~fig_s) dogrudan bitisik ufuk kenari gokyuzu girintileri
        horizon_y0 = int(np.percentile(hy, 45))
        horizon_y1 = int(np.percentile(hy, 98))
        edge_band = (yy_s >= horizon_y0) & (yy_s <= horizon_y1) & ((xx_s < int(0.12 * sw)) | (xx_s > int(0.88 * sw))) & (~safe_lock)
        cand_edge = edge_band & fig_s & (~sky_holes) & (de_w < 14.5) & (L_s > 86.0) & (C_s < 6.5)
        sky_holes |= expand_connected_from_seed(~fig_s, cand_edge)

        # F) 1 px yumusak beyaz halo temizligi
        sky_dil = (cv2.dilate(sky_holes.astype(np.uint8), L.disk(1)) > 0) & fig_s & (L_s > 84.0) & (C_s < 7.5) & (~safe_lock)
        sky_holes |= sky_dil

        # G) Oyulan gokyuzu cepleri icinde yuzen mikro toz adaciklari (< 600 px @ 2800)
        sky_holes[safe_lock] = False
        fig_s_clean = fig_s & (~sky_holes)
        nf, lf, sf, _ = cv2.connectedComponentsWithStats(fig_s_clean.astype(np.uint8), connectivity=8)
        for i in range(1, nf):
            if sf[i, cv2.CC_STAT_AREA] < 600:
                cm = (lf == i)
                if (cm & cable_zone).all():
                    fig_s_clean[cm] = False
                    sky_holes[cm] = True

        not_sky = (~sky_holes).astype(np.uint8)
        ns_n, ns_l, ns_s, _ = cv2.connectedComponentsWithStats(not_sky, connectivity=8)
        for i in range(1, ns_n):
            if ns_s[i, cv2.CC_STAT_AREA] < 600:
                cm = (ns_l == i)
                if (cm & cable_zone).all():
                    fig_s_clean[cm] = False
                    sky_holes[cm] = True

        all_bg = (~fig_s_clean).astype(np.uint8)
        all_bg_closed = cv2.morphologyEx(all_bg, cv2.MORPH_CLOSE, L.disk(1))
        not_all_bg = (all_bg_closed == 0).astype(np.uint8)
        nab_n, nab_l, nab_s, _ = cv2.connectedComponentsWithStats(not_all_bg, connectivity=8)
        for i in range(1, nab_n):
            if nab_s[i, cv2.CC_STAT_AREA] < 500:
                cm = (nab_l == i) & fig_s_clean
                if cm.any() and (cm & cable_zone).all():
                    cm_d = (cv2.dilate((nab_l == i).astype(np.uint8), L.disk(1)) > 0) & fig_s_clean & cable_zone
                    fig_s_clean[cm_d] = False
                    sky_holes[cm_d] = True

    sky_holes[safe_lock] = False
    sky_full = cv2.resize(sky_holes.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    fig_clean = fig & (~sky_full)

    print(f"[auto_mask] Kapali gokyuzu cebi temizlendi: {sky_holes.mean()*100:.2f}% | Net figur: {fig_clean.mean()*100:.2f}%")
    return rgb, fig_clean


def ink_projection_density(rgb_f32, ink_rgb, max_ink):
    """Beyaz zemin (255,255,255) ustunde ink_rgb murekkebinin dogrusal RGB izdusum yogunlugu (0..max_ink)."""
    v = 255.0 - np.array(ink_rgb, dtype=np.float32)
    denom = float(np.dot(v, v)) + 1e-6
    p = 255.0 - rgb_f32
    t = (p[..., 0] * v[0] + p[..., 1] * v[1] + p[..., 2] * v[2]) / denom
    return np.clip(t, 0.0, 1.0) * max_ink


def auto_emprime_channel_name(rgb, is_dual_light=False, is_dual_dark=False, suffix="tram"):
    """
    Herhangi bir gorselden cikarilan spot murekkep RGB degerine gore anlasilir ASCII Turkce
    emprime kanal adi uretir. Sicak altin/bej-kahve cift kanalli ailede acik kanala Altin_Bej_tram,
    koyu kanala Kahve_tram adini otomatik verir; diger tum renklerde (Yesil, Mor, Pembe, Turkuaz,
    Sari, Mavi, Kirmizi vb.) L.auto_color_name kullanir.
    """
    L_val, a_val, b_val = L.rgb_tuple_to_lab(rgb)
    C_val = float(np.hypot(a_val, b_val))
    hue_deg = float(np.degrees(np.arctan2(b_val, a_val)) % 360.0)
    if is_dual_light and (50.0 <= hue_deg <= 110.0) and (60.0 <= L_val <= 88.0) and (12.0 <= C_val <= 45.0):
        return f"Altin_Bej_{suffix}" if suffix else "Altin_Bej"
    if is_dual_dark and (45.0 <= hue_deg <= 110.0) and (20.0 <= L_val <= 58.0):
        return f"Kahve_{suffix}" if suffix else "Kahve"
    raw = L.auto_color_name(rgb, suffix)
    if raw.startswith("Turuncu_") and (L_val >= 65.0) and (C_val < 36.0):
        return f"Altin_Bej_{suffix}" if suffix else "Altin_Bej"
    if raw.startswith("Turuncu_") and (L_val < 52.0) and (C_val < 32.0):
        return f"Kahve_{suffix}" if suffix else "Kahve"
    if raw.startswith("Turkuaz_") and (hue_deg >= 215.0 or C_val < 22.0):
        return f"Mavi_{suffix}" if suffix else "Mavi"
    return raw


def apply_baski_aciklama(rgb_u8, amount=0.05):
    """
    Kalip %5 incelmesi ve boya yayilmasina karsi ayrima giren RGB gorseli kontrollu acar.
    Beyaz (255) ve saf siyah (0) korunur, orta ve 3/4 tonlar ~%5-7 acilir.
    """
    if amount <= 0.0:
        return rgb_u8.copy()
    p = float(np.clip(amount, 0.0, 0.25))
    gamma = max(0.65, 1.0 - p * 1.15)
    norm = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    lifted = np.power(norm, gamma) * 255.0
    return np.clip(np.round(lifted), 0, 255).astype(np.uint8)


def extract_cmyk_k_density(rgb_u8):
    """
    Pillow ImageCms (sRGB -> CoatedFOGRA39 / GRACoL / Euroscale) uzerinden gercek CMYK K kanali (0..1) uretir.
    Sistemde ICC profili bulunamazsa analitik GCR siyah egrisi yedegini kullanir.
    """
    icc_candidates = [
        r"C:\Windows\System32\spool\drivers\color\CoatedFOGRA39.icc",
        r"C:\Windows\System32\spool\drivers\color\CoatedGRACoL2006.icc",
        r"C:\Windows\System32\spool\drivers\color\EuroscaleCoated.icc",
        "/usr/share/color/icc/colord/FOGRA39L_coated.icc",
    ]
    for icc_path in icc_candidates:
        if os.path.exists(icc_path):
            try:
                srgb = ImageCms.createProfile("sRGB")
                cmyk_prof = ImageCms.getOpenProfile(icc_path)
                xform = ImageCms.buildTransformFromOpenProfiles(
                    srgb, cmyk_prof, "RGB", "CMYK",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                )
                cmyk = np.array(ImageCms.applyTransform(Image.fromarray(rgb_u8, "RGB"), xform))
                return cmyk[..., 3].astype(np.float32) / 255.0
            except Exception:
                pass

    k_raw = 1.0 - rgb_u8.max(axis=-1).astype(np.float32) / 255.0
    return np.power(np.clip((k_raw - 0.30) / 0.70, 0.0, 1.0), 1.35).astype(np.float32)


def _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, l_lo, l_hi, p_lo, p_hi, min_c=12.0):
    sel = fmask & (L_s >= l_lo) & (L_s <= l_hi) & (C_s >= min_c)
    if sel.sum() < 50:
        q_lo, q_hi = np.percentile(f_L, [p_lo, p_hi])
        sel = fmask & (L_s >= q_lo) & (L_s <= q_hi)
    pts = rgb_sm[sel if sel.any() else fmask]
    return tuple(int(round(x)) for x in np.median(pts, axis=0))


def detect_and_assign_families_dynamic(rgb_sm, lab_s, fig_s, num_top_colors=None, profil="profesyonel"):
    """
    Herhangi bir gorseldeki kromatik renk ailelerini tum tuvalde kesintisiz tespit eder,
    pikselleri tek bir aileye atar (Bolge Safligi: zit aile karisimi = 0) ve
    --profil (profesyonel / ekonomik) + --renk butcesine gore aile basina 1, 2 veya 3 ton uretir.
    """
    ws_h, ws_w = fig_s.shape
    L_s, a_s, b_s = lab_s[..., 0], lab_s[..., 1], lab_s[..., 2]
    C_s = np.hypot(a_s, b_s)
    hue_s = (np.degrees(np.arctan2(b_s, a_s)) % 360.0).astype(np.float32)

    if num_top_colors is None:
        max_fams = 4 if profil == "profesyonel" else 3
    else:
        chrom_budget = max(1, num_top_colors - 1)
        max_fams = min(max(1, chrom_budget), 4)

    families = L.detect_color_families(
        lab_s, fig_s, max_families=max_fams, fam_c=8.0, min_mass=0.025, min_sep_deg=32.0
    )

    if num_top_colors is not None:
        chrom_budget = max(1, num_top_colors - 1)
        if len(families) >= chrom_budget and chrom_budget >= 3:
            families = families[:chrom_budget - 1]
    else:
        if profil != "profesyonel":
            chrom_budget = 4
            if len(families) >= chrom_budget:
                families = families[:chrom_budget - 1]

    if not families:
        families = [(75.0, 1.0)]

    n_f = len(families)
    scores = np.zeros((n_f, ws_h, ws_w), dtype=np.float32)
    chrom = fig_s & (C_s >= 5.5)
    soft_neutral = fig_s & (~chrom) & (L_s < 84.5)

    warm_idx = None
    cool_idx = None
    best_warm_d = 999.0
    best_cool_d = 999.0

    for fi, (fh, _) in enumerate(families):
        d_w = L.hue_diff_deg(fh, 78.0)
        d_c = L.hue_diff_deg(fh, 235.0)
        if d_w < best_warm_d:
            best_warm_d, warm_idx = d_w, fi
        if d_c < best_cool_d:
            best_cool_d, cool_idx = d_c, fi

    for fi, (fh, _) in enumerate(families):
        dh = np.abs(((hue_s - fh + 180.0) % 360.0) - 180.0)
        min_neighbor = min([L.hue_diff_deg(fh, fh2) for fj, (fh2, _) in enumerate(families) if fj != fi] or [90.0])
        half_win = float(np.clip(min_neighbor * 0.95, 34.0, 75.0))

        is_red_fam = (fh <= 48.0 or fh >= 335.0) and (min_neighbor < 48.0)
        if is_red_fam:
            m_f = chrom & (a_s > 10.0) & ((hue_s < 54.0) | (hue_s > 335.0))
            sc_raw = np.where(m_f, np.clip(1.0 - dh / 35.0, 0.2, 1.0) * C_s * 1.35, 0.0)
        else:
            m_f = chrom & (dh <= half_win)
            sc_raw = np.where(m_f, np.clip(1.0 - dh / half_win, 0.15, 1.0) * C_s, 0.0)

        if fi == warm_idx and best_warm_d <= 45.0 and warm_idx != cool_idx:
            sc_raw = np.where(soft_neutral & (b_s > 0.8), 2.5, sc_raw)
        elif fi == cool_idx and best_cool_d <= 45.0:
            sc_raw = np.where(soft_neutral & (b_s <= 0.8), 2.5, sc_raw)

        scores[fi] = cv2.GaussianBlur(sc_raw.astype(np.float32), (0, 0), 1.5)

    best_sc = scores.max(axis=0)
    best_id = scores.argmax(axis=0).astype(np.int16)
    fam_map = np.where(fig_s & (best_sc > 1.2), best_id, -1)

    if n_f > 1:
        fam_u8 = (fam_map + 1).astype(np.uint8)
        fam_u8 = cv2.medianBlur(fam_u8, 3)
        fam_map = fam_u8.astype(np.int16) - 1
        fam_map[~fig_s] = -1

    # Kanal butcesini ailelere dagit
    fmasks_tmp = []
    for fi, (fh, _) in enumerate(families):
        fm = (fam_map == fi)
        dh = np.abs(((hue_s - fh + 180.0) % 360.0) - 180.0)
        fm = fm & (~((C_s > 7.5) & (dh > 78.0)))
        fmasks_tmp.append(fm)

    order = sorted(range(n_f), key=lambda i: families[i][1], reverse=True)
    if num_top_colors is None and profil == "profesyonel":
        # Profesyonel tam ayrim: her baskin aileye 2 ton (en fazla 8 kromatik + 1 siyah = 9 ust renk)
        fam_alloc = [2] * n_f
        while sum(fam_alloc) > 8:
            # En kucuk aileleri 1 tona indir
            for fi in reversed(order):
                if fam_alloc[fi] > 1 and sum(fam_alloc) > 8:
                    fam_alloc[fi] -= 1
        # Eger <=2 aile varsa ve genis L araligina sahip buyuk bir aile varsa 3. tonu ver
        if sum(fam_alloc) < 8:
            for fi in order:
                if sum(fam_alloc) >= 8:
                    break
                fm = fmasks_tmp[fi]
                if fm.sum() >= 50 and families[fi][1] >= 0.38:
                    p10, p90 = np.percentile(L_s[fm], [10, 90])
                    if (p90 - p10) >= 28.0 and fam_alloc[fi] < 3:
                        fam_alloc[fi] += 1
    else:
        eff_renk = num_top_colors if num_top_colors is not None else 5
        chrom_budget = max(1, eff_renk - 1)
        fam_alloc = [1] * n_f
        extra = chrom_budget - n_f
        max_per_fam = 3 if profil == "profesyonel" else 2
        while extra > 0:
            progressed = False
            for fi in order:
                if extra <= 0:
                    break
                if fam_alloc[fi] < max_per_fam:
                    fam_alloc[fi] += 1
                    extra -= 1
                    progressed = True
            if not progressed:
                break

    fam_specs = []
    for fi, (fh, fmass) in enumerate(families):
        fmask = fmasks_tmp[fi]
        is_red_fam = (fh <= 48.0 or fh >= 335.0)
        if fmask.sum() < 20:
            continue

        f_L = L_s[fmask]
        n_ch = fam_alloc[fi]
        base_sfx = "baz" if (profil == "profesyonel" and n_ch >= 2) else "tram"

        if n_ch >= 3:
            rgb_hi = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, 62.0, 84.0, 50, 86, min_c=10.0)
            rgb_mid = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, 42.0, 64.0, 32, 60, min_c=12.0)
            rgb_lo = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, 20.0, 44.0, 8, 32, min_c=12.0)
            name_hi = auto_emprime_channel_name(rgb_hi, is_dual_light=True, suffix=base_sfx)
            name_mid = auto_emprime_channel_name(rgb_mid, suffix="tram")
            name_lo = auto_emprime_channel_name(rgb_lo, is_dual_dark=True, suffix="tram")
            if name_mid == name_hi:
                name_mid = f"Orta_{name_mid}"
            if name_lo in (name_hi, name_mid):
                name_lo = f"Koyu_{name_lo}"
            tones = [
                {"tier": "light", "name": name_hi, "rgb": rgb_hi, "is_lap": (profil == "profesyonel")},
                {"tier": "mid", "name": name_mid, "rgb": rgb_mid, "is_lap": False},
                {"tier": "dark", "name": name_lo, "rgb": rgb_lo, "is_lap": False},
            ]
        elif n_ch == 2:
            rgb_hi = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, 64.0, 88.0, 58, 92, min_c=12.0)
            rgb_lo = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, 24.0, 50.0, 10, 44, min_c=12.0)
            name_hi = auto_emprime_channel_name(rgb_hi, is_dual_light=True, suffix=base_sfx)
            name_lo = auto_emprime_channel_name(rgb_lo, is_dual_dark=True, suffix="tram")
            if name_lo == name_hi:
                name_lo = f"Koyu_{name_lo}"
            tones = [
                {"tier": "light", "name": name_hi, "rgb": rgb_hi, "is_lap": (profil == "profesyonel")},
                {"tier": "dark", "name": name_lo, "rgb": rgb_lo, "is_lap": False},
            ]
        else:
            l_lo, l_hi = (32.0, 60.0) if is_red_fam else (36.0, 66.0)
            min_c_1 = 30.0 if is_red_fam else 12.0
            rgb_main = _sample_tone_median(rgb_sm, fmask, L_s, C_s, f_L, l_lo, l_hi, 25, 75, min_c=min_c_1)
            name_main = auto_emprime_channel_name(rgb_main, suffix="tram")
            tones = [
                {"tier": "single", "name": name_main, "rgb": rgb_main, "is_lap": False},
            ]

        fam_specs.append({
            "aile_id": fi,
            "hue_deg": fh,
            "fmask": fmask,
            "is_red": is_red_fam,
            "tones": tones,
        })

    return fam_specs


def compute_flat_stroke_gradient_maps(rgb_full, fig_full):
    """
    v3 Karma Piksel Turu Haritasi (Tam Cozunurluk, Olcek-Bagimsiz Gradyan + Anti-Aliasing Korumali):
      - stroke_mask: Siyah kontur ve duz siyah bloklar (L < 26.0, C < 14.0 + 1px koyu AA kenari) -> %100 kati Siyah (tramsiz).
      - flat_mask: Duz boyali alanlar ve iki duz renk arasi 1-4 px sert kenar gecisleri -> %100 kati LAP (tramsiz).
      - grad_mask: En az ~12+ px boyunca surekli renk degisimi gosteren gercek degrade ve yumusak golge
        bolgeleri -> surekli karisim + %85 alt bindirme + 45/55 LPI AM tram.
    """
    lab_full = cv2.cvtColor(rgb_full[..., ::-1], cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_full[..., 0] = lab_full[..., 0] * (100.0 / 255.0)
    lab_full[..., 1] = lab_full[..., 1] - 128.0
    lab_full[..., 2] = lab_full[..., 2] - 128.0

    L_f = lab_full[..., 0]
    a_f = lab_full[..., 1]
    b_f = lab_full[..., 2]
    C_f = np.hypot(a_f, b_f)

    # 1) Siyah Kontur & Duz Siyah Bloklar (+ 1px koyu anti-aliasing kenari)
    stroke_core = fig_full & (L_f < 26.0) & (C_f < 14.0)
    if stroke_core.any():
        stroke_dil = cv2.dilate(stroke_core.astype(np.uint8), L.disk(1)) > 0
        stroke_mask = stroke_core | (fig_full & stroke_dil & (L_f < 40.0) & (C_f < 20.0))
    else:
        stroke_mask = stroke_core

    # 2) Olcek-bagimsiz gercek degrade tespiti (Lab Sobel gradyan buyuklugu)
    # sigma=5.0 ile 8-bit quantization basamaklarini asip gercek yumusak gecisleri kesintisiz yakala
    lab_blur = cv2.GaussianBlur(lab_full, (0, 0), 5.0)
    gx_L = cv2.Sobel(lab_blur[..., 0], cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy_L = cv2.Sobel(lab_blur[..., 0], cv2.CV_32F, 0, 1, ksize=3) * 0.125
    gx_a = cv2.Sobel(lab_blur[..., 1], cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy_a = cv2.Sobel(lab_blur[..., 1], cv2.CV_32F, 0, 1, ksize=3) * 0.125
    gx_b = cv2.Sobel(lab_blur[..., 2], cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy_b = cv2.Sobel(lab_blur[..., 2], cv2.CV_32F, 0, 1, ksize=3) * 0.125
    grad_mag = np.sqrt(gx_L**2 + gy_L**2 + gx_a**2 + gy_a**2 + gx_b**2 + gy_b**2)

    # Keskin iki-renk kenarlarinin +/-5 px bandini elemek icin stroke_mask cevresini de koru
    stroke_pad = cv2.dilate(stroke_mask.astype(np.uint8), L.disk(5)) > 0 if stroke_mask.any() else stroke_mask
    grad_cand = fig_full & (~stroke_pad) & (grad_mag >= 0.035)

    # Sert basamak kenarlari (kalinligi <= 10 px) erozyonda yok olur; genis degradeler (>= 12 px) sag kalir
    if grad_cand.any():
        k_erode = L.disk(5)
        grad_seed = cv2.erode(grad_cand.astype(np.uint8), k_erode) > 0
        if grad_seed.any():
            k_dil = L.disk(6)
            grad_mask = (cv2.dilate(grad_seed.astype(np.uint8), k_dil) > 0) & fig_full & (~stroke_mask) & (grad_mag >= 0.015)
            # 8-bit bantlasma kaynakli ince delikleri kapat
            grad_mask = (cv2.morphologyEx(grad_mask.astype(np.uint8), cv2.MORPH_CLOSE, L.disk(5)) > 0) & fig_full & (~stroke_mask)
        else:
            grad_mask = np.zeros_like(fig_full, dtype=bool)
    else:
        grad_mask = np.zeros_like(fig_full, dtype=bool)

    flat_mask = fig_full & (~stroke_mask) & (~grad_mask)
    return stroke_mask, flat_mask, grad_mask, lab_full


def collect_missing_flat_tones(rgb_s, lab_s, flat_s, stroke_s, existing_tones, max_total_tones=9, min_de=20.0):
    """
    Duz boyali alanlarda (or. sicak krem yazi veya kucuk duz renk bolgeleri) mevcut aile tonlarina
    uzak (dE > min_de) kalan belirgin kumeleri (>= %0.25 alan) otomatik ek duz spot ton olarak yakalar.
    """
    if len(existing_tones) >= max_total_tones or not flat_s.any():
        return []

    cand = flat_s & (~stroke_s)
    if cand.sum() < 30:
        return []

    L_s = lab_s[..., 0]
    C_s = np.hypot(lab_s[..., 1], lab_s[..., 2])
    # Saf beyaz disindaki duz alanlari (krem veya renkli duz alanlar) hedefle
    cand = cand & (~((L_s >= 89.0) & (C_s < 5.0)))
    if cand.sum() < 30:
        return []

    pts_lab = lab_s[cand]
    pts_rgb = rgb_s[cand]
    fig_area = max(int(flat_s.sum() + stroke_s.sum()), 1)

    extra_tones = []
    cur_labs = [np.array(L.rgb_tuple_to_lab(t["rgb"]), dtype=np.float32) for t in existing_tones]
    de_cut = max(8.0, float(min_de))

    for _ in range(max_total_tones - len(existing_tones)):
        if not cur_labs:
            break
        labs_arr = np.stack(cur_labs, axis=0)
        dists = np.linalg.norm(pts_lab[:, None, :] - labs_arr[None, :, :], axis=2).min(axis=1)
        far_mask = dists > de_cut
        if far_mask.sum() < max(25, int(0.0025 * fig_area)):
            break

        far_labs = pts_lab[far_mask]
        far_rgbs = pts_rgb[far_mask]
        # En yogun uzak kumenin merkezini bul (en uzak yerine en buyuk bagli/benzer kume)
        seed_idx = int(np.argmax(dists[far_mask]))
        seed_lab = far_labs[seed_idx]
        cluster = np.linalg.norm(far_labs - seed_lab[None, :], axis=1) <= 14.0
        if cluster.sum() < max(20, int(0.002 * fig_area)):
            # Kume kucukse bu noktayi atla
            pts_lab = pts_lab[~far_mask]
            pts_rgb = pts_rgb[~far_mask]
            break

        med_rgb = tuple(int(round(x)) for x in np.median(far_rgbs[cluster], axis=0))
        med_lab = np.array(L.rgb_tuple_to_lab(med_rgb), dtype=np.float32)
        t_name = auto_emprime_channel_name(med_rgb, is_dual_light=(med_lab[0] >= 78.0), suffix="lap")
        if med_lab[0] >= 80.0 and np.hypot(med_lab[1], med_lab[2]) < 24.0:
            t_name = "Krem_lap"
        extra_tones.append({
            "tier": "flat_extra",
            "name": t_name,
            "rgb": med_rgb,
            "is_lap": True,
            "aile_id": -2,
        })
        cur_labs.append(med_lab)

    return extra_tones


def merge_close_tones(tone_list, de_thresh):
    """
    --renk-birlestirme yuzdesinden gelen Lab dE esigi (de_thresh) icinde kalan yakin tonlari
    tek spot renk (tek kafa) olarak birlestirir.
    """
    if de_thresh <= 0.0 or len(tone_list) <= 1:
        return tone_list

    labs = [np.array(L.rgb_tuple_to_lab(t["rgb"]), dtype=np.float32) for t in tone_list]
    kept = []
    kept_labs = []
    # Oncelik: light / single / flat_extra -> mid -> dark
    prio = {"light": 0, "single": 0, "flat_extra": 0, "mid": 1, "dark": 2}
    ordered = sorted(range(len(tone_list)), key=lambda i: prio.get(tone_list[i].get("tier", "single"), 1))

    for idx in ordered:
        t = tone_list[idx]
        t_lab = labs[idx]
        merged = False
        for k_lab in kept_labs:
            if float(np.linalg.norm(t_lab - k_lab)) <= de_thresh:
                merged = True
                break
        if not merged:
            kept.append(t)
            kept_labs.append(t_lab)
    return kept


def solve_gradient_weights(lab_s, grad_s, tone_list, degrade_bindirme=0.85, flat_s=None):
    """
    grad_mask icindeki piksellerde sert argmax kesimi yerine en yakin iki spot ton arasinda
    dogrusal Lab projeksiyonu (w_i, w_j) ve %85 alt bindirme uygular.
    Her bagli degrade bolgesinde yalnizca o bolgenin cevre duz tonlari + ic uc tonlari kullanilir,
    boylece Sari->Mavi gecisinin ortasina alakasiz Ten/Kahve tonlari karismaz.
    """
    ws_h, ws_w = grad_s.shape
    n_tones = len(tone_list)
    weights = np.zeros((n_tones, ws_h, ws_w), dtype=np.float32)
    if n_tones == 0 or not grad_s.any():
        return weights

    tone_labs = np.array([L.rgb_tuple_to_lab(t["rgb"]) for t in tone_list], dtype=np.float32)  # (M, 3)

    if n_tones == 1:
        ys, xs = np.where(grad_s)
        pts = lab_s[ys, xs]
        L_t = float(tone_labs[0, 0])
        w = np.clip((98.0 - pts[:, 0]) / max(98.0 - L_t, 10.0), 0.0, 0.95)
        weights[0, ys, xs] = w.astype(np.float32)
        return weights

    # Her bagli degrade bolgesi icin izinli yerel ton kumesini belirle
    n_cc, cc_lbl = cv2.connectedComponents(grad_s.astype(np.uint8), connectivity=8)
    all_tone_ids = list(range(n_tones))
    fig_area_s = max(int(grad_s.sum() + (flat_s.sum() if flat_s is not None else 0)), 1)

    for cid in range(1, n_cc):
        comp_mask = (cc_lbl == cid)
        ys, xs = np.where(comp_mask)
        if len(ys) == 0:
            continue
        pts = lab_s[ys, xs]

        allowed_ids = all_tone_ids
        is_large_comp = comp_mask.sum() >= 0.25 * fig_area_s
        if is_large_comp:
            non_extra = [ti for ti in all_tone_ids if tone_list[ti].get("tier") != "flat_extra"]
            if len(non_extra) >= 2:
                allowed_ids = non_extra
        elif flat_s is not None and flat_s.any():
            dil = cv2.dilate(comp_mask.astype(np.uint8), L.disk(6)) > 0
            neigh_flat = dil & flat_s
            if neigh_flat.sum() >= 10:
                n_pts = lab_s[neigh_flat]
                # Bellek dostu ornekleme
                if len(n_pts) > 4000:
                    step = len(n_pts) // 4000
                    n_pts = n_pts[::step]
                n_best = np.argmin(np.linalg.norm(n_pts[:, None, :] - tone_labs[None, :, :], axis=2), axis=1)
                counts = np.bincount(n_best, minlength=n_tones)
                local_set = set(int(ti) for ti in range(n_tones) if counts[ti] >= max(3, int(0.02 * len(n_pts))))
                # Yalnizca tek komsu duz ton varsa (or. Ten ustu yanak alligi), ic uc noktayi ekle;
                # kucuk/orta bagli degrade bolgelerinde (>= 2 komsu duz ton) en baskin 2 komsu tonu sec
                # ki gecis ortasinda 3. bir tona sicrama (yatay kesik) olmasin!
                if len(local_set) == 1:
                    neigh_med = np.median(n_pts, axis=0)
                    far_idx = int(np.argmax(np.linalg.norm(pts - neigh_med[None, :], axis=1)))
                    far_tone = int(np.argmin(np.linalg.norm(tone_labs - pts[far_idx][None, :], axis=1)))
                    local_set.add(far_tone)
                elif len(local_set) > 2 and comp_mask.sum() < 0.25 * max(int(grad_s.sum() + flat_s.sum()), 1):
                    top2 = sorted(local_set, key=lambda ti: int(counts[ti]), reverse=True)[:2]
                    local_set = set(top2)
                if len(local_set) >= 2:
                    allowed_ids = sorted(local_set)

        sub_labs = tone_labs[allowed_ids]
        diff = pts[:, None, :] - sub_labs[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        order = np.argsort(dist, axis=1)
        i1_sub = order[:, 0]
        i2_sub = order[:, 1]
        idx_n = np.arange(len(pts))
        d1 = dist[idx_n, i1_sub]
        d2 = dist[idx_n, i2_sub]

        allowed_arr = np.array(allowed_ids, dtype=np.int32)
        i1 = allowed_arr[i1_sub]
        i2 = allowed_arr[i2_sub]

        lab1 = tone_labs[i1]
        lab2 = tone_labs[i2]
        v = lab2 - lab1
        v2 = np.sum(v * v, axis=1) + 1e-6
        t = np.clip(np.sum((pts - lab1) * v, axis=1) / v2, 0.0, 1.0)
        w1 = 1.0 - t
        w2 = t

        # Buyuk dokulu bolgelerde (>= %25 figur) zit aile (> 95 derece) karisimini engelle (Bolge Safligi)
        single_tone_only = np.zeros(len(pts), dtype=bool)
        if is_large_comp:
            h1 = np.degrees(np.arctan2(lab1[:, 2], lab1[:, 1])) % 360.0
            h2 = np.degrees(np.arctan2(lab2[:, 2], lab2[:, 1])) % 360.0
            c1 = np.hypot(lab1[:, 1], lab1[:, 2])
            c2 = np.hypot(lab2[:, 1], lab2[:, 2])
            dh12 = np.abs(((h1 - h2 + 180.0) % 360.0) - 180.0)
            opp_mask = (dh12 > 95.0) & (c1 >= 5.0) & (c2 >= 5.0)
            if opp_mask.any():
                w1[opp_mask] = np.clip((98.0 - pts[opp_mask, 0]) / np.maximum(98.0 - lab1[opp_mask, 0], 10.0), 0.0, 0.95)
                w2[opp_mask] = 0.0
                single_tone_only[opp_mask] = True

        # Tek renk acik zemine/beyaza dogru aciliyorsa ikinci tona karistirma
        white_fade = (pts[:, 0] > lab1[:, 0] + 8.0) & (d2 > d1 * 2.2) & (pts[:, 0] > 78.0)
        if white_fade.any():
            w_wf = np.clip((98.0 - pts[white_fade, 0]) / np.maximum(98.0 - lab1[white_fade, 0], 10.0), 0.0, 0.95)
            w1[white_fade] = w_wf
            w2[white_fade] = 0.0
            single_tone_only[white_fade] = True

        L1 = lab1[:, 0]
        L2 = lab2[:, 0]
        i1_is_base = L1 >= L2
        base_idx = np.where(i1_is_base, i1, i2)
        top_idx = np.where(i1_is_base, i2, i1)
        w_base_raw = np.where(i1_is_base, w1, w2)
        w_top = np.where(i1_is_base, w2, w1)

        if degrade_bindirme > 0.0:
            db = float(degrade_bindirme)
            fade = np.clip((1.0 - w_top) / max(1.0 - db, 0.01), 0.0, 1.0)
            w_base = np.where(w_top <= db, 1.0, np.clip(w_base_raw + w_top * fade, 0.0, 1.0))
        else:
            w_base = w_base_raw

        w_top_tram = np.clip(w_top, 0.0, 0.95)
        w_base_out = np.where(w_base >= 0.96, 1.0, w_base)

        # Tek ton modunda (white_fade veya opp_mask) i2 asla alt bindirme ile 1.0 basilmaz, yalniz i1 basilir
        two_tone = ~single_tone_only
        if two_tone.any():
            weights[base_idx[two_tone], ys[two_tone], xs[two_tone]] = np.maximum(
                weights[base_idx[two_tone], ys[two_tone], xs[two_tone]], w_base_out[two_tone].astype(np.float32)
            )
            weights[top_idx[two_tone], ys[two_tone], xs[two_tone]] = np.maximum(
                weights[top_idx[two_tone], ys[two_tone], xs[two_tone]], w_top_tram[two_tone].astype(np.float32)
            )
        if single_tone_only.any():
            weights[i1[single_tone_only], ys[single_tone_only], xs[single_tone_only]] = np.maximum(
                weights[i1[single_tone_only], ys[single_tone_only], xs[single_tone_only]],
                np.clip(w1[single_tone_only], 0.0, 0.95).astype(np.float32),
            )

    return weights


def assign_flat_tones_chunked(lab_full, flat_mask, tone_list, chunk_rows=512):
    """
    Tam cozunurlukte flat_mask piksellerini en yakin spot tona %100 kati (1.0) olarak atar.
    Bellek dostu olmasi icin satir bloklari halinde calisir.
    """
    H, W = flat_mask.shape
    n_tones = len(tone_list)
    out = [np.zeros((H, W), dtype=np.float32) for _ in range(n_tones)]
    if n_tones == 0 or not flat_mask.any():
        return out

    tone_labs = np.array([L.rgb_tuple_to_lab(t["rgb"]) for t in tone_list], dtype=np.float32)
    for y0 in range(0, H, chunk_rows):
        y1 = min(H, y0 + chunk_rows)
        fm = flat_mask[y0:y1]
        if not fm.any():
            continue
        pts = lab_full[y0:y1][fm]
        dists = np.linalg.norm(pts[:, None, :] - tone_labs[None, :, :], axis=2)
        best = np.argmin(dists, axis=1)
        ys_local, xs_local = np.where(fm)
        for ti in range(n_tones):
            sel = best == ti
            if sel.any():
                out[ti][y0 + ys_local[sel], xs_local[sel]] = 1.0
    return out


def screen_channel_hybrid(cont_f32, fig_full, screen_params, min_dot=0.03, min_px=4):
    """
    v3 Karma Kanal Tramlama:
      - cont_f32 >= 0.96 -> dogrudan %100 kati (ink = True, TRAMSIZ).
      - min_dot <= cont_f32 < 0.96 -> 45/55 LPI yuvarlak AM tram (L.apply_am_screen_tiled).
    """
    cont_f32 = cont_f32.copy()
    cont_f32[~fig_full] = 0.0
    cont_f32[cont_f32 < min_dot] = 0.0

    solid_mask = fig_full & (cont_f32 >= 0.96)
    tram_zone = fig_full & (cont_f32 >= min_dot) & (cont_f32 < 0.96)

    if tram_zone.any():
        tram_density = np.where(tram_zone, cont_f32, 0.0).astype(np.float32)
        ink_tram = L.apply_am_screen_tiled(tram_density, screen_params, min_dot=min_dot)
    else:
        ink_tram = np.zeros_like(fig_full, dtype=bool)

    if min_px > 0 and ink_tram.any():
        n_cc, cc_lbl, cc_st, _ = cv2.connectedComponentsWithStats(ink_tram.astype(np.uint8), connectivity=8)
        if n_cc > 1:
            small_ids = np.where(cc_st[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
            if len(small_ids) > 0:
                ink_tram[np.isin(cc_lbl, small_ids)] = False

    ink = solid_mask | ink_tram
    if min_px > 0 and ink.any():
        n_cc, cc_lbl, cc_st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        if n_cc > 1:
            small_ids = np.where(cc_st[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
            if len(small_ids) > 0:
                ink[np.isin(cc_lbl, small_ids)] = False
    return cont_f32, ink


# ---------------------------------------------------------------------------
# 2) GERCEK 2-LAYER + SPOT KANALLI MASTER PSD & TIFF YAZICI
# ---------------------------------------------------------------------------

def build_8bim_block(rid, payload):
    hdr = b"8BIM" + struct.pack(">H", rid) + b"\x00\x00" + struct.pack(">I", len(payload))
    pad = b"\x00" if (len(payload) % 2 == 1) else b""
    return hdr + payload + pad


def build_photoshop_resources(spot_channels, ppi=508):
    res_fixed = int(round(ppi * 65536))
    r1005 = struct.pack(">IHHIHH", res_fixed, 1, 2, res_fixed, 1, 2)
    r1006_parts = []
    r1045_parts = []
    r1077_parts = [struct.pack(">I", 1)]

    for ch in spot_channels:
        name = ch["name"]
        p_bytes = name.encode("latin1", errors="replace")[:255]
        r1006_parts.append(struct.pack(">B", len(p_bytes)) + p_bytes)
        u_chars = name + "\x00"
        r1045_parts.append(struct.pack(">I", len(u_chars)) + u_chars.encode("utf-16-be"))
        r, g, b = [int(x) for x in ch["rgb"]]
        sol = int(ch["solidity"])
        r1077_parts.append(struct.pack(">HHHHHHB", 0, r * 257, g * 257, b * 257, 0, sol, 2))

    return (
        build_8bim_block(1005, r1005)
        + build_8bim_block(1006, b"".join(r1006_parts))
        + build_8bim_block(1045, b"".join(r1045_parts))
        + build_8bim_block(1077, b"".join(r1077_parts))
    )


def build_psd_layer_section(layers_list):
    """
    layers_list: [(layer_name_ascii, rgb_uint8_array), ...] (alttan uste dogru)
    psd-tools ile RLE sikistirmali Layer and Mask Information blogunu uretir.
    """
    if not layers_list:
        return struct.pack(">I", 0)
    h, w, _ = layers_list[0][1].shape
    psd_tmp = PSDImage.new("RGB", (w, h))
    for lname, larr in layers_list:
        psd_tmp.create_pixel_layer(
            Image.fromarray(larr, "RGB"),
            name=lname,
            top=0,
            left=0,
            compression=Compression.RLE,
        )
    buf = io.BytesIO()
    psd_tmp.save(buf)
    del psd_tmp
    raw = buf.getvalue()
    buf.close()
    pos = 26
    cm_len = struct.unpack(">I", raw[pos:pos + 4])[0]
    pos += 4 + cm_len
    ir_len = struct.unpack(">I", raw[pos:pos + 4])[0]
    pos += 4 + ir_len
    lm_len = struct.unpack(">I", raw[pos:pos + 4])[0]
    res = bytes(raw[pos:pos + 4 + lm_len])
    del raw
    return res


def write_spot_psd(filepath, rgb_base, spot_channels, ppi=508, layers_list=None, lm_section=None):
    """
    Hem Layer and Mask Information (orn. Orjinal_Gorsel, Acik_Tisort_Seti ve Koyu_Tisort_Seti)
    hem de 8BIM Spot Kanallari (1005, 1006, 1045, 1077 Solidity %100/%16) barindiran
    RLE sikistirmali Photoshop PSD dosyasi yazar (dusuk RAM icin dosyaya akisli yazar).
    """
    h, w, _ = rgb_base.shape
    n_total = 3 + len(spot_channels)
    resources = build_photoshop_resources(spot_channels, ppi=ppi)
    if lm_section is None:
        lm_section = build_psd_layer_section(layers_list)

    all_planes = [rgb_base[..., 0], rgb_base[..., 1], rgb_base[..., 2]]
    for ch in spot_channels:
        all_planes.append(ch["psd_plane"])

    n_lay = len(layers_list) if layers_list else (1 if lm_section and len(lm_section) > 4 else 0)
    print(f"[PSD] {os.path.basename(filepath)} yaziliyor ({n_lay} katman, {n_total} kanal, {w}x{h} @ {ppi} PPI)...")

    row_counts = []
    with open(filepath, "wb") as f:
        f.write(struct.pack(">4sH6sHIIHH", b"8BPS", 1, b"\x00" * 6, n_total, h, w, 8, 3))
        f.write(struct.pack(">I", 0))
        f.write(struct.pack(">I", len(resources)))
        f.write(resources)
        f.write(lm_section)
        f.write(struct.pack(">H", 1))
        rc_pos = f.tell()
        f.write(b"\x00" * (2 * n_total * h))
        for plane in all_planes:
            plane_c = np.ascontiguousarray(plane, dtype=np.uint8)
            for r in range(h):
                enc = encode_rle(plane_c[r].tobytes())
                row_counts.append(len(enc))
                f.write(enc)
        f.seek(rc_pos)
        f.write(struct.pack(f">{len(row_counts)}H", *row_counts))

    mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[PSD TAMAM] {filepath} ({mb:.2f} MB)")


def write_spot_tiff(filepath, rgb_base, spot_channels, ppi=508):
    """Photoshop'ta dogrudan Spot Kanallariyla acilan cok kanalli Deflate TIFF yazar."""
    h, w, _ = rgb_base.shape
    n_spot = len(spot_channels)
    n_total = 3 + n_spot
    resources = build_photoshop_resources(spot_channels, ppi=ppi)

    all_planes = [rgb_base[..., 0], rgb_base[..., 1], rgb_base[..., 2]]
    for ch in spot_channels:
        all_planes.append(ch["psd_plane"])

    rows_per_strip = 64
    strips_per_plane = (h + rows_per_strip - 1) // rows_per_strip
    total_strips = n_total * strips_per_plane

    compressed_strips = []
    for plane in all_planes:
        plane_c = np.ascontiguousarray(plane, dtype=np.uint8)
        for s_idx in range(strips_per_plane):
            r0 = s_idx * rows_per_strip
            r1 = min(h, r0 + rows_per_strip)
            compressed_strips.append(zlib.compress(plane_c[r0:r1].tobytes(), level=4))

    tags = []
    def add_tag(tag_id, dtype, count, val_bytes):
        tags.append((tag_id, dtype, count, val_bytes))

    n_tags = 15
    aux_start = 8 + 2 + n_tags * 12 + 4
    aux_buf = bytearray()

    def alloc_aux(data_bytes):
        if len(aux_buf) % 2 == 1:
            aux_buf.append(0)
        off = aux_start + len(aux_buf)
        aux_buf.extend(data_bytes)
        return off

    bps_off = alloc_aux(struct.pack(f">{n_total}H", *([8] * n_total)))
    es_off = alloc_aux(struct.pack(f">{n_spot}H", *([0] * n_spot)))
    xres_off = alloc_aux(struct.pack(">II", int(ppi), 1))
    yres_off = alloc_aux(struct.pack(">II", int(ppi), 1))
    ps_off = alloc_aux(resources)
    so_off = alloc_aux(b"\x00" * (4 * total_strips))
    sbc_off = alloc_aux(b"\x00" * (4 * total_strips))

    strips_data_start = aux_start + len(aux_buf)
    strip_offsets, strip_counts = [], []
    cur_off = strips_data_start
    for cb in compressed_strips:
        strip_offsets.append(cur_off)
        strip_counts.append(len(cb))
        cur_off += len(cb)

    so_rel = so_off - aux_start
    sbc_rel = sbc_off - aux_start
    aux_buf[so_rel:so_rel + 4 * total_strips] = struct.pack(f">{total_strips}I", *strip_offsets)
    aux_buf[sbc_rel:sbc_rel + 4 * total_strips] = struct.pack(f">{total_strips}I", *strip_counts)

    add_tag(256, 4, 1, struct.pack(">I", w))
    add_tag(257, 4, 1, struct.pack(">I", h))
    add_tag(258, 3, n_total, struct.pack(">I", bps_off))
    add_tag(259, 3, 1, struct.pack(">HH", 8, 0))
    add_tag(262, 3, 1, struct.pack(">HH", 2, 0))
    add_tag(273, 4, total_strips, struct.pack(">I", so_off))
    add_tag(277, 3, 1, struct.pack(">HH", n_total, 0))
    add_tag(278, 4, 1, struct.pack(">I", rows_per_strip))
    add_tag(279, 4, total_strips, struct.pack(">I", sbc_off))
    add_tag(282, 5, 1, struct.pack(">I", xres_off))
    add_tag(283, 5, 1, struct.pack(">I", yres_off))
    add_tag(284, 3, 1, struct.pack(">HH", 2, 0))
    add_tag(296, 3, 1, struct.pack(">HH", 2, 0))
    add_tag(338, 3, n_spot, struct.pack(">I", es_off) if n_spot > 2 else struct.pack(f">{n_spot}H", *([0] * n_spot)).ljust(4, b"\x00"))
    add_tag(34377, 1, len(resources), struct.pack(">I", ps_off))

    with open(filepath, "wb") as f:
        f.write(b"MM\x00\x2a\x00\x00\x00\x08")
        f.write(struct.pack(">H", len(tags)))
        for tid, dtype, cnt, vb in tags:
            f.write(struct.pack(">HHI", tid, dtype, cnt) + vb)
        f.write(struct.pack(">I", 0))
        f.write(aux_buf)
        for cb in compressed_strips:
            f.write(cb)

    mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[TIFF TAMAM] {filepath} ({mb:.2f} MB)")


def render_full_overprint_layer(channels_list, fabric_rgb, set_label_txt=None, art_h=None, ppi=508, beden="s-xl", yuz="on baski", is_no="703", is_ad="london", tarih=""):
    """
    Tam cozunurlukte (orn. 5600x6200 px) gercekci Solidity (%100 fon, %16 ust renk)
    overprint birlesik RGB katman goruntusu uretir.
    Eger set_label_txt ve art_h verilmisse, alt 3 cm etiket+kros seridinde kanallarin
    ust uste binmis yazisi yerine okunakli net set etiketini ve + kayit krosunu gosterir.
    """
    any_m = channels_list[0]["ink_1bit"]
    h, w = any_m.shape
    canvas = np.full((h, w, 3), np.array(fabric_rgb, dtype=np.float32), dtype=np.float32)
    h_draw = art_h if (art_h is not None and art_h < h) else h
    for ch in channels_list:
        if ch["rol"] == "kumas":
            continue
        m = ch["ink_1bit"][:h_draw, :]
        rgb = np.array(ch["rgb"], dtype=np.float32)
        alpha = 0.96 if ch["rol"] in ("fon", "fon_koyusuz", "lap", "ust_beyaz") else 0.85
        sub = canvas[:h_draw, :]
        sub[m] = sub[m] * (1.0 - alpha) + rgb * alpha

    if art_h is not None and art_h < h and set_label_txt:
        dummy = np.zeros((art_h, w), dtype=bool)
        stamped = stamp_single_film(dummy, set_label_txt, len(channels_list), beden, yuz, is_no, is_ad, tarih, ppi=ppi)
        strip_m = stamped[art_h:, :]
        # Koyu kumasta beyaz (235,235,235), acik kumasta siyah (20,20,20) net etiket+kros
        is_dark = (sum(fabric_rgb) / 3.0) < 128.0
        ink_col = np.array([235, 235, 235] if is_dark else [20, 20, 20], dtype=np.float32)
        sub_s = canvas[art_h:, :]
        sub_s[strip_m] = ink_col

    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3) ANA AYRIM MOTORU (KOYU + ACIK TISORT SETLERI, KROS+ETIKET, 2-LAYER PSD/TIF)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Kedihane Hibrit LAP + Dot-on-Dot Emprime Renk Ayirici v2 (Tek Dosya / Tam Paket, 3-Layer + Spot Kanal)"
    )
    ap.add_argument("girdi", help="Kaynak gorsel yolu (PNG/JPG/JFIF/TIF)")
    ap.add_argument("cikti", help="Cikti klasoru yolu")
    ap.add_argument("--kumas", choices=["koyu", "acik", "ikisi"], default="ikisi", help="Uretilecek kumas seti (varsayilan: ikisi)")
    ap.add_argument("--profil", choices=["profesyonel", "ekonomik"], default="profesyonel", help="Ayrim profili (varsayilan: profesyonel)")
    ap.add_argument("--renk", type=int, default=None, help="Ust renk sayisi (fon haric; verilmezse profil otomatik secer)")
    ap.add_argument("--genislik-cm", type=float, default=28.0, help="Hedef baski genisligi cm (varsayilan 28.0)")
    ap.add_argument("--ppi", type=int, default=508, help="Cozunurluk PPI (varsayilan 508)")
    ap.add_argument("--lpi", type=float, default=45.0, help="AM tram frekansi LPI (varsayilan 45.0; CMYK fotograf 55.0)")
    ap.add_argument("--aci", type=float, default=15.0, help="AM tram acisi derece (varsayilan 15.0; 45/72 destekli, 0/90 yasak)")
    ap.add_argument("--choke", type=int, default=2, help="Koyu tisort beyaz fon choke px (varsayilan 2)")
    ap.add_argument("--zemin-maske", choices=["oto", "maskeli", "tumu"], default="oto", help="Dis zemin maskeleme modu (varsayilan: oto)")
    ap.add_argument("--siyah-mod", choices=["cmyk", "lab"], default="cmyk", help="Siyah_tram kaynak yontemi (varsayilan: cmyk)")
    ap.add_argument("--baski-aciklama", type=float, default=0.05, help="Kalip incelmesine karsi parlaklik acma orani (varsayilan 0.05 = %%5, 0.0=kapali)")
    ap.add_argument("--degrade-bindirme", type=float, default=0.85, help="Komsu degrade gecislerinde alt renk bindirme tavani (varsayilan 0.85 = %%85, 0.0=kapali)")
    ap.add_argument("--renk-birlestirme", type=float, default=5.0, help="Yakin renk birlestirme yuzdesi (0..30; varsayilan 5.0 -> Lab dE=8.0)")
    ap.add_argument("--acik-lap-yogunluk", type=float, default=0.98, help="Profesyonel profilde acik baz LAP hedef yogunlugu (varsayilan 0.98)")
    ap.add_argument("--ust-tram-tavan", type=float, default=0.58, help="Profesyonel profilde ust koyu tram tavani (varsayilan 0.58)")
    ap.add_argument("--max-ink-renk", type=float, default=0.84, help="Ekonomik/tekil renk surekli ton tavani (varsayilan 0.84)")
    ap.add_argument("--max-ink-siyah", type=float, default=0.88, help="Dot gain siyah surekli ton tavani (varsayilan 0.88)")
    ap.add_argument("--min-dot", type=float, default=0.03, help="Minimum nokta esigi (varsayilan 0.03)")
    ap.add_argument("--min-px", type=int, default=4, help="Minimum izole bilesen px (varsayilan 4)")
    ap.add_argument("--maske", default=None, help="Istege bagli harici figur maskesi PNG")
    ap.add_argument("--tek-dosya", dest="tek_dosya", action="store_true", default=True, help="Diskte klasor kalabaligi olmadan tek dosya uret (varsayilan)")
    ap.add_argument("--tam-paket", dest="tek_dosya", action="store_false", help="Eski coklu klasor + 1-bit film + onizleme + JSON paketini uret")
    ap.add_argument("--format", choices=["psd", "tif"], default="psd", help="Tek dosya cikti bicimi (varsayilan: psd)")
    ap.add_argument("--cikti-dosya", default=None, help="Tek dosya modunda ozel hedef dosya yolu")
    ap.add_argument("--kros-etiket", dest="kros_etiket", action="store_true", default=True, help="Alt 3 cm bantta 4 alanli etiket ve + kayit krosu ekle (varsayilan: acik)")
    ap.add_argument("--kros-yok", dest="kros_etiket", action="store_false", help="Kros ve film bilgi etiketini kapat")
    ap.add_argument("--beden", default="s-xl", help="Etiket beden/kalip (varsayilan: 's-xl')")
    ap.add_argument("--yuz", default="on baski", help="Etiket baski yuzu (varsayilan: 'on baski')")
    ap.add_argument("--is-no", default="703", help="Etiket is numarasi (varsayilan: '703')")
    ap.add_argument("--is-ad", default=None, help="Etiket kisa is adi (varsayilan: dosya adindan otomatik)")
    ap.add_argument("--tarih", default=None, help="Etiket tarih (varsayilan: bugun GG.AA.YYYY)")
    args = ap.parse_args()

    if abs(args.aci % 90.0) < 1.0:
        raise SystemExit("HATA: 0° / 90° tram acisi yasaktir. 15°, 45° veya 72° kullanin.")

    tarih = args.tarih or datetime.date.today().strftime("%d.%m.%Y")
    if not args.is_ad:
        stem_raw = os.path.splitext(os.path.basename(args.girdi))[0].lower()
        stem_clean = re.sub(r"(calisma|çalışma|tasarim|tasarım|baski|baskı|desen)", "", stem_raw).strip(" _-")
        stem_ascii = stem_clean.translate(str.maketrans("çğıöşü", "cgiosu"))
        args.is_ad = re.sub(r"[^a-z0-9]+", "_", stem_ascii).strip("_") or "desen"

    if args.tek_dosya:
        if args.cikti_dosya:
            out_single_path = os.path.abspath(args.cikti_dosya)
            os.makedirs(os.path.dirname(out_single_path), exist_ok=True)
        else:
            os.makedirs(args.cikti, exist_ok=True)
            out_single_path = os.path.join(args.cikti, f"{args.is_ad}_emprime.{args.format}")
    else:
        os.makedirs(args.cikti, exist_ok=True)
        cont_dir = os.path.join(args.cikti, "Surekli_Ton")
        kontrol_dir = os.path.join(args.cikti, "_kontrol")
        os.makedirs(cont_dir, exist_ok=True)
        os.makedirs(kontrol_dir, exist_ok=True)

    im = Image.open(args.girdi)
    if args.genislik_cm > 0:
        target_w = int(round(args.genislik_cm / 2.54 * args.ppi))
        target_h = int(round(im.height * (target_w / float(im.width))))
        im = im.resize((target_w, target_h), Image.LANCZOS)

    rgb_orig_full, fig_full = build_auto_clean_figure_mask(im, args.maske, zemin_maske=args.zemin_maske)
    H, W = rgb_orig_full.shape[:2]
    rgb_full = apply_baski_aciklama(rgb_orig_full, args.baski_aciklama)
    print(f"[lap_aile_ayir] Desen: {W}x{H} @ {args.ppi} PPI ({W/args.ppi*2.54:.1f}x{H/args.ppi*2.54:.1f} cm) | profil={args.profil} siyah={args.siyah_mod} aciklama={args.baski_aciklama*100:.1f}%")

    ws = 2 if max(H, W) > 3500 else 1
    ws_h, ws_w = H // ws, W // ws
    rgb_s = cv2.resize(rgb_full, (ws_w, ws_h), interpolation=cv2.INTER_AREA)
    fig_s = cv2.resize(fig_full.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_NEAREST) > 0

    stroke_mask, flat_mask, grad_mask, lab_full = compute_flat_stroke_gradient_maps(rgb_full, fig_full)
    lab_full_L = lab_full[..., 0].copy()
    if ws > 1:
        stroke_s = cv2.resize(stroke_mask.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_NEAREST) > 0
        flat_s = cv2.resize(flat_mask.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_NEAREST) > 0
        grad_s = cv2.resize(grad_mask.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        stroke_s, flat_s, grad_s = stroke_mask, flat_mask, grad_mask
    print(f"  [v3 Karma Harita] Stroke: {stroke_mask.mean()*100:.2f}% | Flat (LAP): {flat_mask.mean()*100:.2f}% | Degrade (Tram): {grad_mask.mean()*100:.2f}%")

    rgb_sm = cv2.bilateralFilter(rgb_s, 5, 10, 3)
    lab_s = L.rgb_to_lab_cv(rgb_sm)
    L_s, a_s, b_s = lab_s[..., 0], lab_s[..., 1], lab_s[..., 2]
    C_s = np.hypot(a_s, b_s)
    hue_s = (np.degrees(np.arctan2(b_s, a_s)) % 360.0).astype(np.float32)

    fam_specs = detect_and_assign_families_dynamic(
        rgb_sm, lab_s, fig_s, num_top_colors=args.renk, profil=args.profil
    )

    all_tones = []
    for sp in fam_specs:
        for t in sp["tones"]:
            t_copy = dict(t)
            t_copy["aile_id"] = sp["aile_id"]
            all_tones.append(t_copy)

    max_chrom_budget = max(1, (args.renk - 1)) if args.renk is not None else 9
    merge_pct = float(np.clip(args.renk_birlestirme, 0.0, 30.0))
    de_thresh = merge_pct * 1.6
    extra_flat_tones = collect_missing_flat_tones(
        rgb_sm, lab_s, flat_s, stroke_s, all_tones, max_total_tones=max_chrom_budget, min_de=max(8.0, de_thresh)
    )
    if extra_flat_tones:
        all_tones.extend(extra_flat_tones)
        print(f"  [v3 Ek Duz Ton] {len(extra_flat_tones)} duz alan tonu eklendi: " + ", ".join(f"{t['name']}{t['rgb']}" for t in extra_flat_tones))

    before_cnt = len(all_tones)
    all_tones = merge_close_tones(all_tones, de_thresh)
    if len(all_tones) < before_cnt:
        print(f"  [Yakin Renk Birlestirme %{merge_pct:.1f} (dE<={de_thresh:.1f})] {before_cnt} -> {len(all_tones)} spot ton")

    print("  Dinamik Tespit Edilen Spot Murekkepler:")
    for sp in fam_specs:
        area_pct = sp["fmask"].mean() * 100.0
        t_desc = " + ".join(f"{t['name']}({t['tier']}){t['rgb']}" for t in sp["tones"])
        print(f"    Aile {sp['aile_id']} ({sp['hue_deg']:.1f}°, alan={area_pct:.2f}%): {t_desc}")

    screen_params = L.build_am_threshold_tile(lpi=args.lpi, angle_deg=args.aci, ppi=float(args.ppi))

    def screen_channel(cont_f32):
        return screen_channel_hybrid(
            cont_f32, fig_full, screen_params, min_dot=args.min_dot, min_px=args.min_px
        )

    # 1) Duz Boyali Alanlar (flat_mask): Tam cozunurlukte en yakin spot tona %100 kati (1.0) LAP
    flat_d_list = assign_flat_tones_chunked(lab_full, flat_mask, all_tones)
    del lab_full

    # 2) Degrade / Yumusak Golge Alanlari (grad_mask): Surekli Lab projeksiyon + %85 alt bindirme
    grad_w_s = solve_gradient_weights(lab_s, grad_s, all_tones, degrade_bindirme=args.degrade_bindirme, flat_s=flat_s)

    light_or_single_k = []
    dark_dual_k = []
    light_or_single_a = []
    dark_dual_a = []
    dual_pairs_for_dod = []

    used_names = {}
    def unique_ch_name(nm):
        cnt = used_names.get(nm, 0) + 1
        used_names[nm] = cnt
        return nm if cnt == 1 else f"{nm}_{cnt}"

    need_koyu = args.kumas in ("koyu", "ikisi") or (not args.tek_dosya)
    need_acik = args.kumas in ("acik", "ikisi") or (not args.tek_dosya)

    base_masks_by_aile = {}
    for ti, t_info in enumerate(all_tones):
        aile_id = t_info["aile_id"]
        ch_name = unique_ch_name(t_info["name"])
        is_base_tier = t_info["tier"] in ("light", "single", "flat_extra")
        flat_px = int((flat_d_list[ti] >= 0.96).sum())

        if grad_s.any() and grad_w_s[ti].any():
            if ws > 1:
                flat_s_ti = cv2.resize(flat_d_list[ti], (ws_w, ws_h), interpolation=cv2.INTER_NEAREST)
                combined_s = np.maximum(flat_s_ti, grad_w_s[ti])
                grad_full = cv2.resize(combined_s, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                grad_full = grad_w_s[ti]
            grad_only = np.where(grad_mask, grad_full, 0.0).astype(np.float32)
        else:
            grad_only = np.zeros((H, W), dtype=np.float32)

        d_full_k = np.maximum(flat_d_list[ti], grad_only)
        d_full_a = np.maximum(flat_d_list[ti], np.clip(grad_only * 1.03, 0.0, 0.95))

        if (d_full_k >= args.min_dot).sum() < args.min_px and (d_full_a >= args.min_dot).sum() < args.min_px:
            continue

        c_k, m_k = screen_channel(d_full_k) if need_koyu else (None, None)
        c_a, m_a = screen_channel(d_full_a) if need_acik else (None, None)

        cov_check = m_k.mean() if m_k is not None else m_a.mean()
        if cov_check < 0.0045 and t_info["tier"] != "flat_extra":
            continue

        grad_px = int((grad_only >= args.min_dot).sum())
        base_rol = "lap" if (t_info.get("is_lap") and flat_px >= grad_px) else "tram"

        if need_koyu:
            entry_k = {
                "ad": ch_name, "rgb": list(t_info["rgb"]), "rol": base_rol,
                "solidity": 16, "aile": aile_id, "is_lap": False, "ink_1bit": m_k, "cont_full": c_k,
            }
            if is_base_tier:
                light_or_single_k.append(entry_k)
                base_masks_by_aile[aile_id] = (ch_name, m_k)
            else:
                dark_dual_k.append(entry_k)
                if aile_id in base_masks_by_aile:
                    b_name, b_mask = base_masks_by_aile[aile_id]
                    dual_pairs_for_dod.append((ch_name, m_k, b_name, b_mask))

        if need_acik:
            entry_a = {
                "ad": ch_name, "rgb": list(t_info["rgb"]), "rol": base_rol if base_rol == "lap" else "tram",
                "solidity": 16, "aile": aile_id, "is_lap": False, "ink_1bit": m_a, "cont_full": c_a,
            }
            if is_base_tier:
                light_or_single_a.append(entry_a)
            else:
                dark_dual_a.append(entry_a)

    # Siyah Kanali: stroke_mask -> %100 kati (1.0, TRAMSIZ); grad_mask -> yumusak golge (tramli); flat_mask -> 0.0
    if args.siyah_mod == "cmyk":
        k_cmyk = extract_cmyk_k_density(rgb_full)
        siyah_scale = args.max_ink_siyah * (1.0 - 0.5 * args.baski_aciklama)
        d_siyah_grad = np.where(
            grad_mask,
            np.clip((k_cmyk - 0.035) / 0.965, 0.0, 1.0) * siyah_scale,
            0.0,
        ).astype(np.float32)
    else:
        C_full_s = cv2.resize(C_s, (W, H), interpolation=cv2.INTER_LINEAR)
        L_start_k = np.where(C_full_s < 8.0, 50.0, 42.0).astype(np.float32)
        siyah_scale = args.max_ink_siyah * (1.0 - 0.5 * args.baski_aciklama)
        d_siyah_grad = np.where(
            grad_mask,
            np.clip((L_start_k - lab_full_L) / (L_start_k - 10.0), 0.0, 1.0) * siyah_scale,
            0.0,
        ).astype(np.float32)

    d_siyah_grad[d_siyah_grad < args.min_dot] = 0.0
    d_siyah_full_k = np.where(stroke_mask, 1.0, d_siyah_grad).astype(np.float32)
    d_siyah_full_a = np.where(stroke_mask, 1.0, np.clip(d_siyah_grad * 1.03, 0.0, 0.90)).astype(np.float32)

    koyu_raw = []
    acik_raw = []
    if need_koyu:
        c_bk_k, m_bk_k = screen_channel(d_siyah_full_k)
        fon_mask = L.build_underbase_mask(fig_full, choke_px=args.choke, thin_choke_px=1, min_px=args.min_px)
        # Figur icindeki duz acik krem/beyaz yazilarin choke ile incelmesini onle
        eroded_fig = cv2.erode(fig_full.astype(np.uint8), L.disk(max(1, args.choke))) > 0
        fon_mask |= (eroded_fig & flat_mask & (lab_full_L >= 84.0))
        koyu_raw = (
            [{"ad": "fon", "rgb": [255, 255, 255], "rol": "fon", "solidity": 100, "aile": -10, "is_lap": True, "ink_1bit": fon_mask, "cont_full": None}]
            + light_or_single_k
            + dark_dual_k
            + [{"ad": "Siyah_tram", "rgb": [0, 0, 0], "rol": "tram", "solidity": 16, "aile": -1, "is_lap": False, "ink_1bit": m_bk_k, "cont_full": c_bk_k}]
        )
    if need_acik:
        c_bk_a, m_bk_a = screen_channel(d_siyah_full_a)
        acik_raw = (
            light_or_single_a
            + dark_dual_a
            + [{"ad": "Siyah_tram", "rgb": [0, 0, 0], "rol": "tram", "solidity": 16, "aile": -1, "is_lap": False, "ink_1bit": m_bk_a, "cont_full": c_bk_a}]
        )

    # Kayit Krosu (+) ve 4 Alanli Film Bilgi Etiketi Ekleme
    def apply_stamping_to_set(ch_list):
        if not ch_list:
            return ch_list
        n_kafa = len(ch_list)
        for ch in ch_list:
            if args.kros_etiket:
                label_txt = ch["ad"].replace("_", " ")
                ch["ink_1bit"] = stamp_single_film(
                    ch["ink_1bit"], label_txt, n_kafa,
                    args.beden, args.yuz, args.is_no, args.is_ad, tarih, ppi=args.ppi,
                )
                if ch["cont_full"] is not None:
                    new_h, new_w = ch["ink_1bit"].shape
                    old_h = ch["cont_full"].shape[0]
                    c_pad = np.zeros((new_h, new_w), dtype=np.float32)
                    c_pad[:old_h, :] = ch["cont_full"]
                    c_pad[old_h:, :] = ch["ink_1bit"][old_h:, :].astype(np.float32)
                    ch["cont_full"] = c_pad
        return ch_list

    koyu_list = apply_stamping_to_set(koyu_raw)
    acik_list = apply_stamping_to_set(acik_raw)

    ref_list = koyu_list if koyu_list else acik_list
    H_final, W_final = ref_list[0]["ink_1bit"].shape
    print(f"  Final Film Tuvali: {W_final}x{H_final} px ({W_final/args.ppi*2.54:.1f}x{H_final/args.ppi*2.54:.1f} cm)")

    # En altta Orjinal_Gorsel katmani (asla silinmez) + secilen tisort katmanlari
    layer_orjinal_rgb = np.full((H_final, W_final, 3), 255, dtype=np.uint8)
    layer_orjinal_rgb[:H, :] = rgb_orig_full

    layer_acik_rgb = None
    if acik_list:
        layer_acik_rgb = render_full_overprint_layer(
            acik_list, (255, 255, 255),
            set_label_txt="Acik Tisort Seti", art_h=H, ppi=args.ppi,
            beden=args.beden, yuz=args.yuz, is_no=args.is_no, is_ad=args.is_ad, tarih=tarih,
        )

    layer_koyu_rgb = None
    if koyu_list:
        layer_koyu_rgb = render_full_overprint_layer(
            koyu_list, (30, 30, 30),
            set_label_txt="Koyu Tisort Seti", art_h=H, ppi=args.ppi,
            beden=args.beden, yuz=args.yuz, is_no=args.is_no, is_ad=args.is_ad, tarih=tarih,
        )

    kumas_full_plane = np.zeros((H_final, W_final), dtype=np.uint8)

    # ------------------------------------------------------------------
    # TEK DOSYA MODU (Varsayilan): Diskte alt klasor/log/json birakmaz
    # ------------------------------------------------------------------
    if args.tek_dosya:
        master_layers = [("Orjinal_Gorsel", layer_orjinal_rgb)]
        master_spots = []

        if args.kumas in ("acik", "ikisi") and layer_acik_rgb is not None:
            master_layers.append(("Acik_Tisort_Seti", layer_acik_rgb))
        if args.kumas in ("koyu", "ikisi") and layer_koyu_rgb is not None:
            master_layers.append(("Koyu_Tisort_Seti", layer_koyu_rgb))

        if args.kumas in ("koyu", "ikisi") and koyu_list:
            master_spots.append({"name": "Kumas_Koyu", "rgb": [30, 30, 30], "solidity": 100, "psd_plane": kumas_full_plane})
            for idx, ch in enumerate(koyu_list, start=1):
                master_spots.append({
                    "name": f"K_{idx}_{ch['ad']}", "rgb": ch["rgb"],
                    "solidity": ch["solidity"], "psd_plane": np.where(ch["ink_1bit"], 0, 255).astype(np.uint8),
                })

        if args.kumas in ("acik", "ikisi") and acik_list:
            master_spots.append({"name": "Kumas_Acik", "rgb": [255, 255, 255], "solidity": 100, "psd_plane": kumas_full_plane})
            for idx, ch in enumerate(acik_list, start=1):
                master_spots.append({
                    "name": f"A_{idx}_{ch['ad']}", "rgb": ch["rgb"],
                    "solidity": ch["solidity"], "psd_plane": np.where(ch["ink_1bit"], 0, 255).astype(np.uint8),
                })

        composite_rgb = layer_koyu_rgb if (args.kumas in ("koyu", "ikisi") and layer_koyu_rgb is not None) else layer_acik_rgb
        if args.format == "tif":
            write_spot_tiff(out_single_path, composite_rgb, master_spots, ppi=args.ppi)
        else:
            write_spot_psd(out_single_path, composite_rgb, master_spots, ppi=args.ppi, layers_list=master_layers)

        for name_lo, m_lo_k, name_hi, m_hi_k in dual_pairs_for_dod:
            ov_d = (m_lo_k & m_hi_k).sum() / max(m_lo_k.sum(), 1) * 100.0
            print(f"  [dot-on-dot] {name_lo} -> {name_hi} faz eslesmesi: %{ov_d:.2f}")
        print(f"[TAMAM] Tek dosya hazir ({len(master_layers)} katman, {len(master_spots)} spot kanal): {out_single_path}")
        return

    # ------------------------------------------------------------------
    # TAM PAKET MODU (--tam-paket): 1-bit film klasorleri + onizleme + JSON + Master PSD/TIF
    # ------------------------------------------------------------------
    def export_set_files(set_name, ch_list, fabric_rgb, save_cont=False):
        png_dir = os.path.join(args.cikti, set_name)
        tif_dir = os.path.join(args.cikti, f"{set_name}_TIF")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(tif_dir, exist_ok=True)
        for d in (png_dir, tif_dir):
            for fn in os.listdir(d):
                if fn.lower().endswith((".png", ".tif")):
                    os.remove(os.path.join(d, fn))

        json_chs = []
        full_masks = {}
        baski_spots = []
        st_spots = []

        for idx, ch in enumerate(ch_list, start=1):
            fname = f"{idx:02d}_{ch['ad']}.png"
            fpath = os.path.join(png_dir, fname)
            L.save_1bit_png(ch["ink_1bit"], fpath, ppi=args.ppi)
            full_masks[fname] = ch["ink_1bit"]

            tif_path = os.path.join(tif_dir, f"{idx:02d}_{ch['ad']}.tif")
            Image.open(fpath).save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(args.ppi, args.ppi))

            cont_fname = None
            if ch["cont_full"] is not None and not ch["is_lap"]:
                prefix = "" if save_cont else f"{set_name[:4]}_"
                cont_fname = f"{prefix}{idx:02d}_{ch['ad']}_ton.png"
                L.save_cont_png(ch["cont_full"], os.path.join(cont_dir, cont_fname), ppi=args.ppi)
                st_u8 = np.clip(np.round((1.0 - np.clip(ch["cont_full"], 0.0, 1.0)) * 255.0), 0, 255).astype(np.uint8)
                st_spots.append({
                    "name": f"ST_{idx}_{ch['ad'].replace('_tram', '')}",
                    "rgb": ch["rgb"], "solidity": ch["solidity"], "psd_plane": st_u8,
                })

            plane_u8 = np.where(ch["ink_1bit"], 0, 255).astype(np.uint8)
            baski_spots.append({
                "name": f"{idx} {ch['ad']}", "rgb": ch["rgb"],
                "solidity": ch["solidity"], "psd_plane": plane_u8,
            })

            cov = round(float(ch["ink_1bit"].mean() * 100.0), 3)
            print(f"  [{set_name} {idx:02d}] {fname:24s} | rol={ch['rol']:4s} sol={ch['solidity']:3d}% rgb={ch['rgb']} kapsama={cov:.2f}%")
            json_chs.append({
                "ad": ch["ad"], "dosya": fname, "surekli_ton_dosya": cont_fname,
                "rgb": ch["rgb"], "rol": ch["rol"], "solidity": ch["solidity"],
                "aile": ch["aile"], "kapsama_pct": cov,
            })

        prev_img = L.simulate_overprint_preview(json_chs, full_masks, fabric_rgb, scale_max=1600)
        prev_path = os.path.join(args.cikti, f"onizleme_{set_name.lower()}_yigilmali.png")
        Image.fromarray(prev_img).save(prev_path)
        return json_chs, baski_spots, st_spots

    sets_meta = {}
    koyu_json, koyu_baski_spots, koyu_st_spots = export_set_files("Koyu_Tisort", koyu_list, (30, 30, 30), save_cont=True)
    acik_json, acik_baski_spots, acik_st_spots = export_set_files("Acik_Tisort", acik_list, (255, 255, 255), save_cont=False)

    if args.kumas in ("koyu", "ikisi"):
        sets_meta["Koyu_Tisort"] = koyu_json
    if args.kumas in ("acik", "ikisi"):
        sets_meta["Acik_Tisort"] = acik_json

    sc_prev = min(1.0, 1600.0 / float(max(H_final, W_final)))
    pw, ph = max(1, int(round(W_final * sc_prev))), max(1, int(round(H_final * sc_prev)))
    Image.fromarray(cv2.resize(layer_koyu_rgb, (pw, ph), interpolation=cv2.INTER_AREA)).save(
        os.path.join(args.cikti, "onizleme_koyu_tisort_yigilmali.png")
    )
    Image.fromarray(cv2.resize(layer_acik_rgb, (pw, ph), interpolation=cv2.INTER_AREA)).save(
        os.path.join(args.cikti, "onizleme_acik_tisort_yigilmali.png")
    )

    three_layers_koyu_top = [
        ("Orjinal_Gorsel", layer_orjinal_rgb),
        ("Acik_Tisort_Seti", layer_acik_rgb),
        ("Koyu_Tisort_Seti", layer_koyu_rgb),
    ]
    three_layers_acik_top = [
        ("Orjinal_Gorsel", layer_orjinal_rgb),
        ("Koyu_Tisort_Seti", layer_koyu_rgb),
        ("Acik_Tisort_Seti", layer_acik_rgb),
    ]

    out_prefix = os.path.basename(os.path.abspath(args.cikti))

    k_calisma = {"name": "Kumas_Calisma", "rgb": [30, 30, 30], "solidity": 100, "psd_plane": kumas_full_plane}
    k_baski = {"name": "Kumas_Baski", "rgb": [30, 30, 30], "solidity": 100, "psd_plane": kumas_full_plane}
    koyu_master_spots = [k_calisma] + koyu_st_spots + [k_baski] + koyu_baski_spots
    koyu_only_baski_spots = [k_baski] + koyu_baski_spots

    koyu_master_psd = os.path.join(args.cikti, f"{out_prefix}_Koyu_Tisort_Master.psd")
    koyu_master_tif = os.path.join(args.cikti, f"{out_prefix}_Koyu_Tisort_Master.tif")
    koyu_baski_tif = os.path.join(args.cikti, f"{out_prefix}_Koyu_Tisort_Baski.tif")

    lm_koyu_top = build_psd_layer_section(three_layers_koyu_top)
    write_spot_psd(koyu_master_psd, layer_koyu_rgb, koyu_master_spots, ppi=args.ppi, layers_list=three_layers_koyu_top, lm_section=lm_koyu_top)
    write_spot_tiff(koyu_master_tif, layer_koyu_rgb, koyu_master_spots, ppi=args.ppi)
    write_spot_tiff(koyu_baski_tif, layer_koyu_rgb, koyu_only_baski_spots, ppi=args.ppi)

    a_calisma = {"name": "Kumas_Calisma", "rgb": [255, 255, 255], "solidity": 100, "psd_plane": kumas_full_plane}
    a_baski = {"name": "Kumas_Baski", "rgb": [255, 255, 255], "solidity": 100, "psd_plane": kumas_full_plane}
    acik_master_spots = [a_calisma] + acik_st_spots + [a_baski] + acik_baski_spots

    acik_master_psd = os.path.join(args.cikti, f"{out_prefix}_Acik_Tisort_Master.psd")
    acik_master_tif = os.path.join(args.cikti, f"{out_prefix}_Acik_Tisort_Master.tif")

    write_spot_psd(acik_master_psd, layer_acik_rgb, acik_master_spots, ppi=args.ppi, layers_list=three_layers_acik_top)
    write_spot_tiff(acik_master_tif, layer_acik_rgb, acik_master_spots, ppi=args.ppi)

    birlesik_spots = [{"name": "Kumas_Koyu", "rgb": [30, 30, 30], "solidity": 100, "psd_plane": kumas_full_plane}]
    for idx, ch in enumerate(koyu_list, start=1):
        birlesik_spots.append({
            "name": f"K_{idx}_{ch['ad']}", "rgb": ch["rgb"], "solidity": ch["solidity"],
            "psd_plane": np.where(ch["ink_1bit"], 0, 255).astype(np.uint8),
        })
    birlesik_spots.append({"name": "Kumas_Acik", "rgb": [255, 255, 255], "solidity": 100, "psd_plane": kumas_full_plane})
    for idx, ch in enumerate(acik_list, start=1):
        birlesik_spots.append({
            "name": f"A_{idx}_{ch['ad']}", "rgb": ch["rgb"], "solidity": ch["solidity"],
            "psd_plane": np.where(ch["ink_1bit"], 0, 255).astype(np.uint8),
        })

    birlesik_psd = os.path.join(args.cikti, f"{out_prefix}_Birlesik_Koyu_Acik_Master.psd")
    birlesik_tif = os.path.join(args.cikti, f"{out_prefix}_Birlesik_Koyu_Acik_Master.tif")
    write_spot_psd(birlesik_psd, layer_koyu_rgb, birlesik_spots, ppi=args.ppi, layers_list=three_layers_koyu_top, lm_section=lm_koyu_top)
    write_spot_tiff(birlesik_tif, layer_koyu_rgb, birlesik_spots, ppi=args.ppi)

    prev_k = Image.open(os.path.join(args.cikti, "onizleme_koyu_tisort_yigilmali.png"))
    prev_a = Image.open(os.path.join(args.cikti, "onizleme_acik_tisort_yigilmali.png"))
    side_by_side = Image.new("RGB", (prev_k.width + prev_a.width + 24, max(prev_k.height, prev_a.height)), (180, 180, 180))
    side_by_side.paste(prev_k, (0, 0))
    side_by_side.paste(prev_a, (prev_k.width + 24, 0))
    side_by_side.save(os.path.join(args.cikti, "onizleme_koyu_ve_acik_birlesik.png"))

    meta = {
        "mod": "yigilmali",
        "profil": args.profil,
        "siyah_mod": args.siyah_mod,
        "zemin_maske": args.zemin_maske,
        "baski_aciklama": args.baski_aciklama,
        "degrade_bindirme": args.degrade_bindirme,
        "ppi": args.ppi,
        "lpi": args.lpi,
        "aci": args.aci,
        "choke_px": args.choke,
        "max_ink_renk": args.max_ink_renk,
        "max_ink_siyah": args.max_ink_siyah,
        "kumas_rgb": [30, 30, 30],
        "kros_etiket": args.kros_etiket,
        "etiket_bilgi": {
            "beden": args.beden, "yuz": args.yuz,
            "is_no": args.is_no, "is_ad": args.is_ad, "tarih": tarih,
        },
        "surekli_ton_klasor": "Surekli_Ton",
        "master_dosyalar": {
            "birlesik_psd": os.path.basename(birlesik_psd),
            "birlesik_tif": os.path.basename(birlesik_tif),
            "koyu_psd": os.path.basename(koyu_master_psd),
            "koyu_tif": os.path.basename(koyu_master_tif),
            "acik_psd": os.path.basename(acik_master_psd),
            "acik_tif": os.path.basename(acik_master_tif),
        },
        "sets": sets_meta,
    }
    meta_path = os.path.join(args.cikti, "kanallar_hibrit.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # _kontrol/ %100 kesit (merkez + etiket/kros) ve film temas kagidi
    cw, ch_h = min(900, W), min(900, H)
    cx0 = max(0, (W - cw) // 2)
    cy0 = max(0, (H - ch_h) // 2)
    Image.fromarray(layer_koyu_rgb[cy0:cy0 + ch_h, cx0:cx0 + cw]).save(os.path.join(kontrol_dir, "desen_merkez_koyu_100pct.png"))
    Image.fromarray(layer_acik_rgb[cy0:cy0 + ch_h, cx0:cx0 + cw]).save(os.path.join(kontrol_dir, "desen_merkez_acik_100pct.png"))

    if args.kros_etiket and H_final > H:
        Image.fromarray(layer_koyu_rgb[H:, :]).save(os.path.join(kontrol_dir, "etiket_ve_kros_koyu_100pct.png"))
        Image.fromarray(layer_acik_rgb[H:, :]).save(os.path.join(kontrol_dir, "etiket_ve_kros_acik_100pct.png"))

    tw, th_t, pad, header = 520, 560, 20, 52
    cols = min(3, max(1, len(koyu_list)))
    rows = (len(koyu_list) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th_t + header) + (rows + 1) * pad), (242, 242, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, ch in enumerate(koyu_list):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th_t + header + pad)
        small = cv2.resize((~ch["ink_1bit"]).astype(np.uint8) * 255, (tw, th_t), interpolation=cv2.INTER_AREA)
        sheet.paste(Image.fromarray(small, mode="L").convert("RGB"), (x, y + header))
        draw.rectangle([x - 1, y + header - 1, x + tw, y + header + th_t], outline=(120, 120, 120), width=1)
        draw.rectangle([x, y + 6, x + 36, y + 42], fill=tuple(ch["rgb"]), outline=(40, 40, 40), width=1)
        cov = ch["ink_1bit"].mean() * 100.0
        draw.text((x + 44, y + 6), f"{idx+1:02d}_{ch['ad']} ({ch['rol']}, sol={ch['solidity']}%)", fill=(20, 20, 20))
        draw.text((x + 44, y + 26), f"RGB={ch['rgb']} | Kapsama: %{cov:.2f}", fill=(70, 70, 70))
    sheet.save(os.path.join(kontrol_dir, "filmler_ozet.png"))

    for name_lo, m_lo_k, name_hi, m_hi_k in dual_pairs_for_dod:
        ov_d = (m_lo_k & m_hi_k).sum() / max(m_lo_k.sum(), 1) * 100.0
        print(f"  [dot-on-dot] {name_lo} -> {name_hi} faz eslesmesi: %{ov_d:.2f}")
    print(f"[TAMAM] Koyu ({len(koyu_list)} film) + Acik ({len(acik_list)} film) + 3-Layer Spot Master PSD/TIF hazir: {args.cikti}")


if __name__ == "__main__":
    main()
