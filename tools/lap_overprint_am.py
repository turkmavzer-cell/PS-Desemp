#!/usr/bin/env python3
"""
Atolye standardi LAP + Dot-on-Dot Yigilmis Ton AM Renk Ayrimi (508 PPI, 45 LPI, 15° Yuvarlak).

699, 700, 701, 702 uretim PSD dosyalarindan cikarilan gercek baski mantigini uygular:
  1. Farkli renk aileleri arasi sifir karisim (Bolge Safligi).
  2. Ayni aile icinde: istege bagli %100 duz bolge alt-taban LAP'i (sicak aile altina,
     soguk/zit aile altina konmaz) + uzerine 8-bit surekli tonlu koyu golge kanallari.
  3. Dot Gain kompanzasyonu: tram oncesi 8-bit surekli ton kanallarinda ust yogunluk tavani
     (renkler varsayilan %80, siyah %88) + 1:1 dogrusal 45 LPI 15° yuvarlak AM tramlama.
  4. Pixel-locked tek izgaradan Dot-on-Dot (%90-100 faz eslesmeli) 1-bit film uretimi.
  5. Akilli alt baski (fon): 2 px (genel/ince) - 3 px (genis lap) choke; tramli siyah alti
     OYULMAZ; genis duz koyu bloklar icin opsiyonel --cift-fon; koyu kumasta --ust-beyaz.
  6. 2 katli Master PSD icin hem 8-bit surekli ton calisma PNG'leri hem 1-bit baski filmleri
     ve Solidity (kumas=%100, fon=%100, ust renkler=%16) metadata'li kanallar_hibrit.json.

Kullanim:
  python tools/lap_overprint_am.py girdi.png cikti_klasoru [secenekler]
"""
import argparse
import colorsys
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image


def parse_hex_rgb(s):
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Gecersiz hex renk: {s}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def rgb_to_lab_cv(rgb_u8):
    """cv2 BGR->Lab donusumu -> gercek CIE Lab (L:0..100, a:-128..127, b:-128..127)."""
    bgr = rgb_u8[..., ::-1]
    lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab_cv[..., 0] * (100.0 / 255.0)
    a = lab_cv[..., 1] - 128.0
    b = lab_cv[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)


def rgb_tuple_to_lab(rgb):
    arr = np.array([[rgb]], dtype=np.uint8)
    return rgb_to_lab_cv(arr)[0, 0]


def hue_diff_deg(h1, h2):
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def disk(r):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def auto_color_name(rgb, suffix=""):
    """RGB degerinden gercek HSV + Lab parlaklik ile ASCII Turkce emprime kanal adi uret."""
    L, a, b = rgb_tuple_to_lab(rgb)
    C = math.hypot(a, b)
    h01, s01, v01 = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    hue = h01 * 360.0
    if C < 7.0 or s01 < 0.08:
        if L > 90:
            base = "Beyaz"
        elif L > 75:
            base = "Acik_Gri"
        elif L > 45:
            base = "Gri"
        elif L > 22:
            base = "Antrasit"
        else:
            base = "Siyah"
    else:
        if L > 82 and s01 < 0.25 and (25 <= hue <= 65):
            base = "Krem"
        elif hue < 15 or hue >= 340:
            base = "Koyu_Kirmizi" if L < 38 else ("Acik_Kirmizi" if L > 60 else "Kirmizi")
        elif hue < 42:
            base = "Koyu_Kahve" if L < 34 else ("Kahve" if L < 58 else "Turuncu")
        elif hue < 70:
            base = "Hardal" if L < 62 else "Sari"
        elif hue < 165:
            base = "Koyu_Yesil" if L < 42 else "Yesil"
        elif hue < 198:
            base = "Turkuaz"
        elif hue < 252:
            base = "Lacivert" if L < 36 else "Mavi"
        elif hue < 295:
            base = "Mor"
        else:
            base = "Pembe"
    return f"{base}_{suffix}" if suffix else base


MESH_PROFILES = {
    "45": {
        "tpi": 115,
        "rec_lpi": 25.0,
        "max_lpi": 25.0,
        "min_dot": 0.10,
        "max_dot": 0.80,
        "shape": "round",
        "boya_notu": "Kalın beyaz fon / düz LAP; tram önerilmez",
    },
    "55": {
        "tpi": 140,
        "rec_lpi": 32.0,
        "max_lpi": 32.0,
        "min_dot": 0.08,
        "max_dot": 0.82,
        "shape": "round",
        "boya_notu": "Koyu zemin beyaz fon ve düz spot için ideal",
    },
    "77": {
        "tpi": 195,
        "rec_lpi": 45.0,
        "max_lpi": 45.0,
        "min_dot": 0.05,
        "max_dot": 0.85,
        "shape": "round",
        "boya_notu": "Su bazlı / beyazlı boyada 45 LPI spot tram eşiği",
    },
    "90": {
        "tpi": 230,
        "rec_lpi": 45.0,
        "max_lpi": 50.0,
        "min_dot": 0.04,
        "max_dot": 0.88,
        "shape": "elliptical",
        "boya_notu": "Simulated Process & CMYK+Spot optimumu",
    },
    "120": {
        "tpi": 305,
        "rec_lpi": 55.0,
        "max_lpi": 65.0,
        "min_dot": 0.03,
        "max_dot": 0.90,
        "shape": "elliptical",
        "boya_notu": "CMYK fotoğraf baskı (pat/şeffaf boya) ve ince tram",
    },
}


def build_am_threshold_tile(lpi=45.0, angle_deg=15.0, ppi=508.0, shape="round", phase_u=0.0, phase_v=0.0):
    """
    Alan-dogrusal (CDF kalibreli) yuvarlak veya eliptik (chain) kosinus AM tram esik fonksiyonu.
    Tum kanallar ayni piksel koordinatlarindan uretildigi icin %100 dot-on-dot faz garantiler;
    phase_u / phase_v ile istenirse kontrollu faz kaydirma uygulanabilir.
    """
    shape_norm = "elliptical" if str(shape).lower() in ("elliptical", "eliptik", "chain") else "round"
    period = ppi / lpi  # ~11.29 px @ 508 PPI / 45 LPI
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    # 128x128 ornek hucrede CDF kalibrasyonu (alan dogrusalligi)
    u = np.linspace(-math.pi, math.pi, 128, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, u)
    if shape_norm == "elliptical":
        spot = 0.5 * (1.0 + 0.62 * np.cos(uu) + 0.38 * np.cos(vv))
    else:
        # Yuvarlak nokta spot fonksiyonu: merkezde 1 (en koyu), kenarda 0
        spot = 0.5 * (1.0 + 0.5 * (np.cos(uu) + np.cos(vv)))
    vals = np.sort(spot.ravel())
    # CDF LUT: spot degeri -> [0, 1] esik (kucuk esik = nokta merkezi, buyuk esik = kenar)
    # Yogunluk d >= threshold -> murekkep (1)
    # Merkez (spot en buyuk) en dusuk esik olmali: rank_desc = 1 - cdf
    lut_x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    cdf = np.searchsorted(vals, lut_x, side="right").astype(np.float32) / float(vals.size)
    thresh_lut = 1.0 - cdf  # merkezde 0.0, disarda 1.0
    return {
        "freq": (2.0 * math.pi) / period,
        "ca": ca,
        "sa": sa,
        "lut": thresh_lut,
        "shape": shape_norm,
        "phase_u": float(phase_u),
        "phase_v": float(phase_v),
    }


def apply_am_screen_tiled(density_f32, screen_params, min_dot=0.03, max_dot=1.0, tile_h=512):
    """
    8-bit surekli ton yogunluk haritasini (0..1) tek fazli 45/55 LPI yuvarlak veya eliptik AM trama cevirir.
    Donus: bool mask (True = murekkep / siyah film).
    """
    h, w = density_f32.shape
    out = np.zeros((h, w), dtype=bool)
    freq = screen_params["freq"]
    ca = screen_params["ca"]
    sa = screen_params["sa"]
    lut = screen_params["lut"]
    phase_u = float(screen_params.get("phase_u", 0.0))
    phase_v = float(screen_params.get("phase_v", 0.0))
    is_elliptical = screen_params.get("shape", "round") == "elliptical"
    apply_cap = float(max_dot) < 0.999
    cap_val = float(max_dot)
    xs = np.arange(w, dtype=np.float32)
    for y0 in range(0, h, tile_h):
        y1 = min(h, y0 + tile_h)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        xx = xs[None, :]
        u = (xx * ca + ys * sa) * freq + phase_u
        v = (-xx * sa + ys * ca) * freq + phase_v
        if is_elliptical:
            spot = 0.5 * (1.0 + 0.62 * np.cos(u) + 0.38 * np.cos(v))
        else:
            spot = 0.5 * (1.0 + 0.5 * (np.cos(u) + np.cos(v)))
        idx = np.clip((spot * 1023.0).astype(np.int32), 0, 1023)
        thr = lut[idx]
        d = density_f32[y0:y1]
        if apply_cap:
            d = np.minimum(d, cap_val)
        out[y0:y1] = (d >= min_dot) & (d > thr)
    return out


def save_1bit_png(ink_bool, path, ppi=508):
    """1-bit PNG: basilacak murekkep SIYAH (0), bosluk BEYAZ (1)."""
    im = Image.fromarray((~ink_bool).astype(bool))
    assert im.mode == "1"
    im.save(path, dpi=(ppi, ppi), optimize=True)


def save_cont_png(density_f32, path, ppi=508):
    """8-bit Grayscale surekli ton calisma PNG: 0 = %100 murekkep, 255 = bos."""
    u8 = np.clip(np.round((1.0 - np.clip(density_f32, 0.0, 1.0)) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path, dpi=(ppi, ppi), optimize=True)


def extract_figure_mask(img_pil, mask_path=None):
    """RGBA alfa, harici maske veya kenar-bagli zemin tespitiyle figur maskesi cikarir."""
    w, h = img_pil.size
    if mask_path:
        m = np.array(Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST)) > 127
        rgb = np.array(img_pil.convert("RGB"))
        return rgb, m

    if img_pil.mode in ("RGBA", "LA") or ("transparency" in img_pil.info):
        rgba = np.array(img_pil.convert("RGBA"))
        alpha = rgba[..., 3]
        if (alpha < 250).mean() > 0.01:
            # Alfa uzerinden temiz beyaz zemin kompoziti + kesin maske
            af = alpha.astype(np.float32)[..., None] / 255.0
            rgb = np.clip(rgba[..., :3] * af + 255.0 * (1.0 - af), 0, 255).astype(np.uint8)
            return rgb, (alpha >= 64)

    rgb = np.array(img_pil.convert("RGB"))
    # Kenar seridinden arka plan rengi kestir
    bw = max(4, min(h, w) // 50)
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3),
        rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3),
        rgb[:, -bw:].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(border, axis=0).astype(np.uint8)
    bg_lab = rgb_tuple_to_lab(tuple(int(x) for x in bg_rgb))
    # 4x kucultulmus floodfill ile kenara bagli arka plani bul
    s = max(1, max(h, w) // 1600)
    small = rgb[::s, ::s]
    lab_s = rgb_to_lab_cv(small)
    de = np.sqrt(((lab_s - bg_lab[None, None, :]) ** 2).sum(axis=-1))
    bg_cand = (de < 10.0).astype(np.uint8)
    n_lbl, lbl = cv2.connectedComponents(bg_cand, connectivity=4)
    edge_lbls = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :])) | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
    edge_lbls.discard(0)
    if edge_lbls:
        bg_mask_s = np.isin(lbl, list(edge_lbls))
        fig_s = (~bg_mask_s).astype(np.uint8) * 255
        fig = cv2.resize(fig_s, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    else:
        fig = np.ones((h, w), dtype=bool)
    return rgb, fig


def detect_color_families(lab_s, fig_s, max_families=3, fam_c=8.0, min_mass=0.05, min_sep_deg=50.0):
    """
    Kroma agirlikli 72-bin ton acisi histogramindan baskin renk ailelerini tespit eder.
    Donus: [(peak_hue_deg, mass_ratio), ...]
    """
    a = lab_s[..., 1]
    b = lab_s[..., 2]
    C = np.hypot(a, b)
    valid = fig_s & (C >= fam_c)
    if valid.sum() < 100:
        return []
    hues = (np.degrees(np.arctan2(b[valid], a[valid])) % 360.0).astype(np.float32)
    weights = C[valid].astype(np.float32)
    hist, _ = np.histogram(hues, bins=72, range=(0.0, 360.0), weights=weights)
    # Dairesel yumusatma (3-tap)
    hist_pad = np.concatenate([hist[-2:], hist, hist[:2]])
    hist_sm = np.convolve(hist_pad, [0.15, 0.2, 0.3, 0.2, 0.15], mode="valid")
    total_w = float(hist_sm.sum())
    if total_w <= 0:
        return []
    peaks = []
    for i in range(72):
        prev_v = hist_sm[(i - 1) % 72]
        next_v = hist_sm[(i + 1) % 72]
        if hist_sm[i] >= prev_v and hist_sm[i] >= next_v and hist_sm[i] > 0:
            # Yerel 25 derece icindeki kutle
            mass = sum(hist_sm[(i + k) % 72] for k in range(-4, 5)) / total_w
            peaks.append((i * 5.0 + 2.5, mass, hist_sm[i]))
    peaks.sort(key=lambda x: x[2], reverse=True)
    selected = []
    for h_deg, mass, _ in peaks:
        if mass < min_mass and len(selected) >= 1:
            continue
        if all(hue_diff_deg(h_deg, sh) >= min_sep_deg for sh, _ in selected):
            selected.append((h_deg, mass))
        if len(selected) >= max_families:
            break
    return selected


def assign_family_map(lab_s, fig_s, families, fam_c=6.0):
    """
    Her piksele yalnizca TEK renk ailesi atar (Bolge Safligi - aileler arasi karisim = 0).
    -1: akromatik / notr / disari.
    """
    h, w = fig_s.shape
    if not families:
        return np.full((h, w), -1, dtype=np.int16)
    a = lab_s[..., 1]
    b = lab_s[..., 2]
    C = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) % 360.0).astype(np.float32)

    n_f = len(families)
    scores = np.zeros((n_f, h, w), dtype=np.float32)
    chrom = fig_s & (C >= fam_c)
    for fi, (fh, _) in enumerate(families):
        dh = np.abs(((hue - fh + 180.0) % 360.0) - 180.0)
        # 55 dereceye kadar yumusak destek, otesinde 0
        sc = np.where(chrom & (dh < 65.0), np.clip(1.0 - dh / 65.0, 0.0, 1.0) * C, 0.0)
        scores[fi] = cv2.GaussianBlur(sc, (0, 0), 2.0)

    best_score = scores.max(axis=0)
    best_idx = scores.argmax(axis=0).astype(np.int16)
    fam_map = np.where(fig_s & (best_score > 1.5), best_idx, -1)
    # Kucuk izole yanlis aile beneklerini medyanla temizle
    if n_f > 1:
        fam_u8 = (fam_map + 1).astype(np.uint8)
        fam_u8 = cv2.medianBlur(fam_u8, 5)
        fam_map = fam_u8.astype(np.int16) - 1
        fam_map[~fig_s] = -1
    return fam_map


def build_underbase_mask(top_union, choke_px=2, thin_choke_px=1, min_px=4):
    """
    Atolye standardi choke: genel/ince 2 px, genis lap 3 px;
    ince (<= 2*choke+1 px) detaylarda thin_choke_px (1 px) korunur, <min_px kenar artigi temizlenir.
    """
    if choke_px <= 0:
        res = top_union.copy()
    else:
        t = top_union.astype(np.uint8)
        thick = cv2.morphologyEx(t, cv2.MORPH_OPEN, disk(choke_px)).astype(bool)
        thin = top_union & ~cv2.dilate(thick.astype(np.uint8), disk(1)).astype(bool)
        eroded_thick = cv2.erode(t, disk(choke_px)).astype(bool)
        if thin_choke_px > 0:
            eroded_thin = cv2.erode(t, disk(thin_choke_px)).astype(bool) & thin
        else:
            eroded_thin = thin
        res = eroded_thick | eroded_thin
    if min_px > 0:
        n_cc, cc_lbl, cc_st, _ = cv2.connectedComponentsWithStats(res.astype(np.uint8), connectivity=8)
        if n_cc > 1:
            small_ids = np.where(cc_st[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
            if len(small_ids) > 0:
                res[np.isin(cc_lbl, small_ids)] = False
    return res


def simulate_overprint_preview(channels_info, masks_or_densities, canvas_rgb, scale_max=1400):
    """
    Photoshop Solidity (%100 kumas+fon, %16 ust renkler) benzeri gercekci overprint onizlemesi.
    """
    any_arr = next(iter(masks_or_densities.values()))
    h, w = any_arr.shape
    sc = min(1.0, scale_max / float(max(h, w)))
    tw, th = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    canvas = np.full((th, tw, 3), np.array(canvas_rgb, dtype=np.float32), dtype=np.float32)

    for ch in channels_info:
        if ch["rol"] == "kumas":
            continue
        arr = masks_or_densities[ch["dosya"]]
        if arr.dtype == bool:
            f = arr.astype(np.float32)
        else:
            f = arr.astype(np.float32)
        small = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = np.array(ch["rgb"], dtype=np.float32)
        # Fon ve LAP tam ortucu; tram kanallari yari-opatik ust uste biner
        if ch["rol"] in ("fon", "fon_koyusuz", "lap", "ust_beyaz"):
            alpha = np.clip(small * 0.96, 0.0, 1.0)
        else:
            alpha = np.clip(small * 0.85, 0.0, 1.0)
        canvas = canvas * (1.0 - alpha[..., None]) + rgb[None, None, :] * alpha[..., None]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Atolye standardi LAP + Dot-on-Dot Yigilmis Ton AM Renk Ayrimi (508 PPI, 45 LPI, 15°)"
    )
    ap.add_argument("girdi", help="Kaynak gorsel (PNG/JPG/TIF)")
    ap.add_argument("cikti", help="Cikti klasoru")
    ap.add_argument("--renk", type=int, default=5, help="Ust renk sayisi (fon haric, varsayilan 5 -> koyuda 6 kafa)")
    ap.add_argument("--kumas", default="koyu", help="'acik', 'koyu' (#1e1e1e) veya hex '#4f6061'")
    ap.add_argument("--maske", default=None, help="Istege bagli figur maskesi PNG")
    ap.add_argument("--ppi", type=int, default=508, help="Film cozunurlugu (varsayilan 508)")
    ap.add_argument("--lpi", type=float, default=45.0, help="AM tram frekansi (varsayilan 45)")
    ap.add_argument("--aci", type=float, default=15.0, help="AM tram acisi derece (varsayilan 15; 0 yasak)")
    ap.add_argument("--choke", type=int, default=2, help="Underbase (fon) choke px (varsayilan 2; genis lap 3)")
    ap.add_argument("--max-ink-renk", type=float, default=0.80, help="Dot gain renk surekli ton tavani (varsayilan 0.80)")
    ap.add_argument("--max-ink-siyah", type=float, default=0.88, help="Dot gain siyah surekli ton tavani (varsayilan 0.88)")
    ap.add_argument("--min-dot", type=float, default=0.03, help="Alt kesim yogunlugu (varsayilan 0.03)")
    ap.add_argument("--lap-aileler", choices=["oto", "yok", "hepsi"], default="oto", help="Bolge alt-taban LAP modu")
    ap.add_argument("--ust-beyaz", action="store_true", help="Koyu kumasta saf beyaz alanlara ayri Ust Beyaz (Highlight White)")
    ap.add_argument("--cift-fon", action="store_true", help="Genis koyu LAP varsa koyu oyulmus 2. fon varyanti da uret")
    ap.add_argument("--genislik-cm", type=float, default=0.0, help="Hedef baski genisligi cm (0 = kaynak boyutu koru)")
    ap.add_argument("--min-px", type=int, default=4, help="Temizlenecek mikro parca alt siniri (varsayilan 4, 0=kapali)")
    args = ap.parse_args()

    if abs(args.aci % 90.0) < 1.0:
        raise SystemExit("HATA: 0° / 90° tram acisi yasaktir (maksimum moire). 15° kullanin.")

    # Kumas rengi
    is_light_fabric = (args.kumas.lower() == "acik")
    if is_light_fabric:
        fabric_rgb = (255, 255, 255)
        set_name = "Acik_Tisort"
    elif args.kumas.lower() == "koyu":
        fabric_rgb = (30, 30, 30)
        set_name = "Koyu_Tisort"
    else:
        fabric_rgb = parse_hex_rgb(args.kumas)
        L_fab = rgb_tuple_to_lab(fabric_rgb)[0]
        is_light_fabric = (L_fab > 88.0)
        set_name = "Acik_Tisort" if is_light_fabric else "Koyu_Tisort"

    os.makedirs(args.cikti, exist_ok=True)
    film_dir = os.path.join(args.cikti, set_name)
    cont_dir = os.path.join(args.cikti, "Surekli_Ton")
    os.makedirs(film_dir, exist_ok=True)
    os.makedirs(cont_dir, exist_ok=True)

    im = Image.open(args.girdi)
    if args.genislik_cm > 0:
        target_w = int(round(args.genislik_cm / 2.54 * args.ppi))
        target_h = int(round(im.height * (target_w / float(im.width))))
        im = im.resize((target_w, target_h), Image.LANCZOS)

    rgb_full, fig_full = extract_figure_mask(im, args.maske)
    H, W = rgb_full.shape[:2]
    print(f"[lap_overprint_am] Boyut: {W}x{H} @ {args.ppi} PPI ({W/args.ppi*2.54:.1f}x{H/args.ppi*2.54:.1f} cm) | Set: {set_name}")

    # Hesaplama hizi icin 2x kucultulmus calisma izgarasi (ton haritasi) -> tam cozunurluge bilinear
    work_scale = 2 if max(H, W) > 3500 else 1
    ws_h, ws_w = H // work_scale, W // work_scale
    rgb_s = cv2.resize(rgb_full, (ws_w, ws_h), interpolation=cv2.INTER_AREA)
    fig_s = cv2.resize(fig_full.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_NEAREST) > 0

    # Hafif gurultu yumusatma (kenar korumali)
    rgb_sm = cv2.bilateralFilter(rgb_s, 7, 14, 5)
    lab_s = rgb_to_lab_cv(rgb_sm)
    L_s = lab_s[..., 0]
    a_s = lab_s[..., 1]
    b_s = lab_s[..., 2]
    C_s = np.hypot(a_s, b_s)

    # 1) Genis duz koyu blok (koyu LAP aday) tespiti (orn. 700.psd '6 koyu kahve lap')
    local_mean = cv2.blur(L_s, (5, 5))
    local_sq = cv2.blur(L_s * L_s, (5, 5))
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean * local_mean))
    dark_flat_cand = fig_s & (L_s < 26.0) & (C_s < 18.0) & (local_std < 3.5)
    dark_flat_clean = cv2.morphologyEx(dark_flat_cand.astype(np.uint8), cv2.MORPH_OPEN, disk(5)) > 0
    has_dark_lap = (dark_flat_clean.sum() / max(fig_s.sum(), 1)) >= 0.06 and args.renk >= 4

    # 2) Renk ailelerini tespit et (koyu LAP disindaki kromatik alanlardan)
    work_for_fam = fig_s & (~dark_flat_clean if has_dark_lap else True)
    # Kafa butcesi: 1 siyah + (1 koyu LAP varsa) + kalan kromatik kanallar
    rem_colors = max(1, args.renk - 1 - (1 if has_dark_lap else 0))
    max_fam = min(3, rem_colors)
    families = detect_color_families(lab_s, work_for_fam, max_families=max_fam)
    fam_map = assign_family_map(lab_s, work_for_fam, families)

    print(f"  Tespit edilen kromatik aile sayisi: {len(families)} | Genis koyu LAP: {has_dark_lap}")
    for fi, (fh, fm) in enumerate(families):
        print(f"    Aile {fi}: ton={fh:.1f}°, kutle={fm*100:.1f}%")

    # Ailelere kanal butcesi dagit (en buyuk kutleli aileye 2 kanal -> acik/LAP + koyu golge)
    fam_alloc = [1] * len(families)
    extra_budget = rem_colors - len(families)
    if families and extra_budget > 0:
        order = sorted(range(len(families)), key=lambda i: families[i][1], reverse=True)
        for fi in order:
            if extra_budget <= 0:
                break
            fam_alloc[fi] += 1
            extra_budget -= 1

    # 3) Kanal listesi ve surekli ton yogunluk haritalari (work_scale'de)
    # Her oge: dict(name, rgb, rol, solidity, aile, density_s, is_lap)
    generated = []

    for fi, (fh, _mass) in enumerate(families):
        fmask = (fam_map == fi)
        if fmask.sum() < 50:
            continue
        f_L = L_s[fmask]
        f_rgb = rgb_sm[fmask]
        p95_L = float(np.percentile(f_L, 95))
        p20_L = float(np.percentile(f_L, 20))
        p60_L = float(np.percentile(f_L, 60))
        n_ch = fam_alloc[fi]

        if n_ch >= 2:
            # Acik/taban renk + koyu golge (yigilmis)
            hi_sel = (f_L >= np.percentile(f_L, 60)) & (f_L <= np.percentile(f_L, 92))
            lo_sel = (f_L >= np.percentile(f_L, 12)) & (f_L <= np.percentile(f_L, 45))
            rgb_light = tuple(int(round(x)) for x in np.median(f_rgb[hi_sel] if hi_sel.any() else f_rgb, axis=0))
            rgb_dark = tuple(int(round(x)) for x in np.median(f_rgb[lo_sel] if lo_sel.any() else f_rgb, axis=0))

            # Sicak/yogun parlaklik azsa (orn. kirmizi gul veya koyu yaprak) -> bolge alt-taban LAP
            # Parlak isik gecisi varsa (orn. altin barok) -> surekli ton acik + yigilmis koyu tram
            use_lap = (args.lap_aileler == "hepsi") or (args.lap_aileler == "oto" and p95_L < 82.0)

            if use_lap:
                d_light = fmask.astype(np.float32)
                generated.append({
                    "name": auto_color_name(rgb_light, "lap"),
                    "rgb": rgb_light,
                    "rol": "lap",
                    "solidity": 16,
                    "aile": fi,
                    "density_s": d_light,
                    "is_lap": True,
                })
            else:
                # Surekli ton ana renk (beyaz isikta 0, orta/koyuda max_ink_renk)
                L_hi = max(p95_L, 88.0)
                L_lo = min(p20_L, 40.0)
                d_light = np.where(
                    fmask,
                    np.clip((L_hi - L_s) / max(L_hi - L_lo, 15.0), 0.0, 1.0) * args.max_ink_renk,
                    0.0,
                ).astype(np.float32)
                generated.append({
                    "name": auto_color_name(rgb_light, "tram"),
                    "rgb": rgb_light,
                    "rol": "tram",
                    "solidity": 16,
                    "aile": fi,
                    "density_s": d_light,
                    "is_lap": False,
                })

            # Koyu golge kanali: acik kanalin alt kumesi (701: kahve <= sari -> %99+ dot-on-dot!)
            L_mid = max(p60_L, 48.0)
            L_deep = min(p20_L, 22.0)
            d_dark = np.where(
                fmask,
                np.clip((L_mid - L_s) / max(L_mid - L_deep, 15.0), 0.0, 1.0) * args.max_ink_renk,
                0.0,
            ).astype(np.float32)
            if not use_lap:
                d_dark = np.minimum(d_dark, d_light)
            generated.append({
                "name": auto_color_name(rgb_dark, "tram"),
                "rgb": rgb_dark,
                "rol": "tram",
                "solidity": 16,
                "aile": fi,
                "density_s": d_dark,
                "is_lap": False,
            })
        else:
            # Tek renkli aile (orn. 700: mavi cicek -> dogrudan beyaz fon ustune tram)
            mid_sel = (f_L >= np.percentile(f_L, 25)) & (f_L <= np.percentile(f_L, 75))
            rgb_main = tuple(int(round(x)) for x in np.median(f_rgb[mid_sel] if mid_sel.any() else f_rgb, axis=0))
            L_hi = max(p95_L, 86.0)
            L_lo = min(p20_L, 30.0)
            d_main = np.where(
                fmask,
                np.clip((L_hi - L_s) / max(L_hi - L_lo, 18.0), 0.0, 1.0) * args.max_ink_renk,
                0.0,
            ).astype(np.float32)
            generated.append({
                "name": auto_color_name(rgb_main, "tram"),
                "rgb": rgb_main,
                "rol": "tram",
                "solidity": 16,
                "aile": fi,
                "density_s": d_main,
                "is_lap": False,
            })

    # 4) Genis duz koyu LAP kanali (varsa)
    dark_lap_full = None
    if has_dark_lap:
        rgb_dlap = tuple(int(round(x)) for x in np.median(rgb_sm[dark_flat_clean], axis=0))
        generated.append({
            "name": auto_color_name(rgb_dlap, "lap"),
            "rgb": rgb_dlap,
            "rol": "lap",
            "solidity": 16,
            "aile": -2,
            "density_s": dark_flat_clean.astype(np.float32),
            "is_lap": True,
        })

    # 5) Ortak Siyah Golge / Kontur Tram Kanali (dot-on-dot ust uste)
    siyah_region = fig_s & (~dark_flat_clean if has_dark_lap else True)
    d_siyah = np.where(
        siyah_region,
        np.clip((44.0 - L_s) / 38.0, 0.0, 1.0) * args.max_ink_siyah,
        0.0,
    ).astype(np.float32)
    if (d_siyah >= args.min_dot).sum() > 50:
        generated.append({
            "name": "Siyah_tram",
            "rgb": (0, 0, 0),
            "rol": "tram",
            "solidity": 16,
            "aile": -1,
            "density_s": d_siyah,
            "is_lap": False,
        })

    # Baski sirasina gore sirala: LAP tabanlar -> aciktan koyuya tram -> koyu LAP -> Siyah
    def sort_key(item):
        L_val = rgb_tuple_to_lab(item["rgb"])[0]
        if item["aile"] == -1:
            return (3, -L_val)
        if item["aile"] == -2:
            return (2, -L_val)
        if item["is_lap"]:
            return (0, -L_val)
        return (1, -L_val)

    generated.sort(key=sort_key)

    # 6) Tam cozunurluge acma + Tek Fazli (Pixel-Locked) 45 LPI 15° AM Tramlama
    screen_params = build_am_threshold_tile(lpi=args.lpi, angle_deg=args.aci, ppi=float(args.ppi))

    full_channels = []
    full_masks = {}
    used_names = {}

    for item in generated:
        base_n = item["name"]
        cnt = used_names.get(base_n, 0) + 1
        used_names[base_n] = cnt
        uniq_name = base_n if cnt == 1 else f"{base_n}_{cnt}"

        if item["is_lap"]:
            d_full = cv2.resize(item["density_s"], (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            ink_1bit = d_full & fig_full
            cont_full = ink_1bit.astype(np.float32)
            if item["aile"] == -2:
                dark_lap_full = ink_1bit
        else:
            cont_full = cv2.resize(item["density_s"], (W, H), interpolation=cv2.INTER_LINEAR)
            cont_full[~fig_full] = 0.0
            cont_full[cont_full < args.min_dot] = 0.0
            ink_1bit = apply_am_screen_tiled(cont_full, screen_params, min_dot=args.min_dot)
            if args.min_px > 0:
                n_cc, cc_lbl, cc_st, _ = cv2.connectedComponentsWithStats(ink_1bit.astype(np.uint8), connectivity=8)
                if n_cc > 1:
                    small_ids = np.where(cc_st[1:, cv2.CC_STAT_AREA] < args.min_px)[0] + 1
                    if len(small_ids) > 0:
                        ink_1bit[np.isin(cc_lbl, small_ids)] = False

        full_channels.append({
            "ad": uniq_name,
            "rgb": list(item["rgb"]),
            "rol": item["rol"],
            "solidity": item["solidity"],
            "aile": item["aile"],
            "is_lap": item["is_lap"],
            "ink_1bit": ink_1bit,
            "cont_full": cont_full,
        })

    # 7) Koyu / Renkli kumasta Underbase (fon), Cift Fon ve Ust Beyaz (Highlight White)
    final_list = []
    if not is_light_fabric:
        # Fon altligi: figur maskesi (tramli siyah alti OYULMAZ - 701.psd kurali!)
        fon_mask = build_underbase_mask(fig_full, choke_px=args.choke, thin_choke_px=1)
        final_list.append({
            "ad": "fon",
            "rgb": [255, 255, 255],
            "rol": "fon",
            "solidity": 100,
            "aile": -10,
            "is_lap": True,
            "ink_1bit": fon_mask,
            "cont_full": None,
        })
        if args.cift_fon and dark_lap_full is not None:
            fon_no_dark = build_underbase_mask(fig_full & (~dark_lap_full), choke_px=args.choke, thin_choke_px=1)
            final_list.append({
                "ad": "fon_koyusuz",
                "rgb": [255, 255, 255],
                "rol": "fon_koyusuz",
                "solidity": 100,
                "aile": -10,
                "is_lap": True,
                "ink_1bit": fon_no_dark,
                "cont_full": None,
            })
        if args.ust_beyaz:
            # Saf beyaz alanlar: L > 88 ve C < 7 (0 px choke, Solidity 16%)
            white_s = fig_s & (L_s > 88.0) & (C_s < 7.0)
            white_full = (cv2.resize(white_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0) & fig_full
            if white_full.mean() > 0.001:
                final_list.append({
                    "ad": "beyaz_ust",
                    "rgb": [255, 255, 255],
                    "rol": "ust_beyaz",
                    "solidity": 16,
                    "aile": -11,
                    "is_lap": True,
                    "ink_1bit": white_full,
                    "cont_full": None,
                })

    final_list.extend(full_channels)

    # 8) Dosyalari yaz (1-bit filmler + 8-bit surekli ton calisma kanallari)
    json_channels = []
    for idx, ch in enumerate(final_list, start=1):
        fname = f"{idx:02d}_{ch['ad']}.png"
        fpath = os.path.join(film_dir, fname)
        save_1bit_png(ch["ink_1bit"], fpath, ppi=args.ppi)
        full_masks[fname] = ch["ink_1bit"]

        cont_fname = None
        if ch["cont_full"] is not None and not ch["is_lap"]:
            cont_fname = f"{idx:02d}_{ch['ad']}_ton.png"
            save_cont_png(ch["cont_full"], os.path.join(cont_dir, cont_fname), ppi=args.ppi)

        cov = round(float(ch["ink_1bit"].mean() * 100.0), 3)
        print(f"  [{idx:02d}] {fname:28s} | rol={ch['rol']:11s} sol={ch['solidity']:3d}% rgb={ch['rgb']} kapsama={cov:.2f}%")
        json_channels.append({
            "ad": ch["ad"],
            "dosya": fname,
            "surekli_ton_dosya": cont_fname,
            "rgb": ch["rgb"],
            "rol": ch["rol"],
            "solidity": ch["solidity"],
            "aile": ch["aile"],
            "kapsama_pct": cov,
        })

    # 9) Kumas ustu gercekci overprint onizleme
    prev_img = simulate_overprint_preview(json_channels, full_masks, fabric_rgb)
    prev_path = os.path.join(args.cikti, f"onizleme_{set_name.lower()}_yigilmali.png")
    Image.fromarray(prev_img).save(prev_path)

    # 10) Zengin kanallar_hibrit.json yaz (mevcut JSX + kalite_kontrol v2 + master_psd_olustur uyumlu)
    meta_path = os.path.join(args.cikti, "kanallar_hibrit.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fp:
                meta = json.load(fp)
        except Exception:
            meta = {}
    meta.setdefault("sets", {})
    meta["mod"] = "yigilmali"
    meta["ppi"] = args.ppi
    meta["lpi"] = args.lpi
    meta["aci"] = args.aci
    meta["choke_px"] = args.choke
    meta["max_ink_renk"] = args.max_ink_renk
    meta["max_ink_siyah"] = args.max_ink_siyah
    meta["kumas_rgb"] = list(fabric_rgb)
    meta["surekli_ton_klasor"] = "Surekli_Ton"
    meta["sets"][set_name] = json_channels

    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    print(f"[TAMAM] {len(json_channels)} film + onizleme + {meta_path}")


if __name__ == "__main__":
    main()
