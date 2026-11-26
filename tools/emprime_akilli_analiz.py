#!/usr/bin/env python3
"""
MeChi / Kedihane — Akilli Emprime Analiz & Minimum Renk Optimizasyon Motoru.

Ozellikler:
  1. CMYK (CoatedFOGRA39 / GCR) Yeterlilik & Gamut Disi Analizi + Yardimci Spot Onerisi
  2. "Ana/Koyu Renk Sec + Aciklari Beyaz/Zeminle Tramla" Fiziksel Karisim Optimizasyonu
     (K=3..10 arasi kalite egrisi ve Dirsek / Minimum Renk Optimumu)
  3. 3 Analiz Derinligi: hizli (~5-10 sn), dengeli (~30-60 sn), derin (2-5 dk)
  4. --stop-flag destegi ile kullanici istedigi an o ana kadarki en iyi sonucla durabilir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageCms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lap_overprint_am as L
import lap_aile_ayir as LA


# ---------------------------------------------------------------------------
# 1) VEKTORIZE RENK UZAYI VE FIZIKSEL KARISIM MODELI (SAF NUMPY)
# ---------------------------------------------------------------------------

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_D65_REF = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

_ICC_CANDIDATES = [
    r"C:\Windows\System32\spool\drivers\color\CoatedFOGRA39.icc",
    r"C:\Windows\System32\spool\drivers\color\CoatedGRACoL2006.icc",
    r"C:\Windows\System32\spool\drivers\color\EuroscaleCoated.icc",
    "/usr/share/color/icc/colord/FOGRA39L_coated.icc",
]


def rgb_f32_to_lab(rgb_f32: np.ndarray) -> np.ndarray:
    """[0..255] sekilli (..., 3) RGB dizisini exact D65 CIE Lab'a cevirir."""
    srgb = np.clip(rgb_f32, 0.0, 255.0) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))
    xyz = np.matmul(lin, _SRGB_TO_XYZ.T) / _D65_REF
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), (kappa * xyz + 16.0) / 116.0)
    L_val = 116.0 * f[..., 1] - 16.0
    a_val = 500.0 * (f[..., 0] - f[..., 1])
    b_val = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L_val, a_val, b_val], axis=-1).astype(np.float32)


def rgb_to_hex(rgb: Tuple[int, int, int] | List[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(np.clip(rgb[0], 0, 255)),
        int(np.clip(rgb[1], 0, 255)),
        int(np.clip(rgb[2], 0, 255)),
    )


def find_cmyk_icc_path() -> Optional[str]:
    for p in _ICC_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def extract_cmyk_4ch_and_roundtrip(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    sRGB -> CMYK (4 kanal, 0..1) -> sRGB (gamut sinirli onizleme) donusumu.
    ICC profili varsa gercek FOGRA39/GRACoL donusumu, yoksa analitik gamut + GCR yedegi.
    """
    icc_path = find_cmyk_icc_path()
    if icc_path:
        try:
            srgb_prof = ImageCms.createProfile("sRGB")
            cmyk_prof = ImageCms.getOpenProfile(icc_path)
            to_cmyk = ImageCms.buildTransformFromOpenProfiles(
                srgb_prof, cmyk_prof, "RGB", "CMYK",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
            to_rgb = ImageCms.buildTransformFromOpenProfiles(
                cmyk_prof, srgb_prof, "CMYK", "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
            im_in = Image.fromarray(rgb_u8, "RGB")
            im_cmyk = ImageCms.applyTransform(im_in, to_cmyk)
            im_back = ImageCms.applyTransform(im_cmyk, to_rgb)
            cmyk_f = np.array(im_cmyk, dtype=np.float32) / 255.0
            rgb_back = np.array(im_back, dtype=np.uint8)
            return cmyk_f, rgb_back, os.path.basename(icc_path)
        except Exception:
            pass

    # Analitik Yedek: GCR + gercekci CMYK gamut kisitlamasi (parlak pembe/yesil/cyan tasmasi)
    rf = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    c_raw = 1.0 - rf[..., 0]
    m_raw = 1.0 - rf[..., 1]
    y_raw = 1.0 - rf[..., 2]
    k_min = np.minimum(np.minimum(c_raw, m_raw), y_raw)
    k_gcr = np.power(np.clip((k_min - 0.25) / 0.75, 0.0, 1.0), 1.30)
    denom = np.maximum(1.0 - k_gcr, 1e-4)
    c = np.clip((c_raw - k_gcr) / denom, 0.0, 1.0)
    m = np.clip((m_raw - k_gcr) / denom, 0.0, 1.0)
    y = np.clip((y_raw - k_gcr) / denom, 0.0, 1.0)
    cmyk_f = np.stack([c, m, y, k_gcr], axis=-1).astype(np.float32)

    # Analitik CMYK baski gamut simulasyonu (saf floresan/RGB doygunlugu kisar)
    lab = rgb_f32_to_lab(rgb_u8.astype(np.float32))
    C = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    # CMYK gamut siniri yesil/mavi/parlak pembe bolgelerde daha dardir
    lim = np.where((hue >= 110) & (hue <= 270), 54.0, np.where((hue >= 315) | (hue <= 25), 62.0, 68.0))
    scale = np.where(C > lim, lim / np.maximum(C, 1e-4), 1.0)
    lab_sim = lab.copy()
    lab_sim[..., 1] *= scale
    lab_sim[..., 2] *= scale
    # Basit Lab -> uint8 RGB cevrimi (OpenCV Lab olcegi)
    l_u8 = np.clip(np.round(lab_sim[..., 0] * 2.55), 0, 255).astype(np.uint8)
    a_u8 = np.clip(np.round(lab_sim[..., 1] + 128.0), 0, 255).astype(np.uint8)
    b_u8 = np.clip(np.round(lab_sim[..., 2] + 128.0), 0, 255).astype(np.uint8)
    rgb_back = cv2.cvtColor(np.stack([l_u8, a_u8, b_u8], axis=-1), cv2.COLOR_Lab2RGB)
    return cmyk_f, rgb_back, "Analitik-CMYK-GCR"


def _compute_quality_score(mean_de: float, p95_de: float) -> float:
    raw = 100.0 * math.exp(-0.042 * float(mean_de) - 0.014 * max(0.0, float(p95_de) - 4.0))
    return round(float(np.clip(raw, 0.0, 99.8)), 1)


def _classify_ink_role(rgb: Tuple[int, int, int]) -> str:
    L_val, a_val, b_val = L.rgb_tuple_to_lab(rgb)
    C_val = math.hypot(a_val, b_val)
    if L_val < 22.0 and C_val < 12.0:
        return "siyah"
    if L_val >= 91.0 and C_val < 6.0:
        return "ust_beyaz"
    return "tram"


def _evaluate_palette_pixels(
    palette_rgb: np.ndarray,
    sample_rgb: np.ndarray,
    sample_lab: np.ndarray,
    sample_chroma: np.ndarray,
    max_dot: float = 0.96,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verilen M murekkep (M, 3) icin "Ana/Koyu Sec + Beyazla Tramla" ileri karisim modeli.
    Her piksel icin:
      1) En iyi tek murekkep + beyaz tram (alpha in [0..max_dot])
      2) Uyumlu iki murekkep + beyaz overprint (subtractive + semi-opaque karisim)
    Donus: (pixel_de (N,), best_pred_rgb (N, 3))
    """
    M = len(palette_rgb)
    N = len(sample_rgb)
    if M == 0:
        err = np.linalg.norm(sample_lab - rgb_f32_to_lab(np.full((N, 3), 255.0, dtype=np.float32)), axis=1)
        return err, np.full((N, 3), 255.0, dtype=np.float32)

    pal_f = palette_rgb.astype(np.float32)  # (M, 3)
    pal_lab = rgb_f32_to_lab(pal_f)
    pal_C = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    pal_hue = np.degrees(np.arctan2(pal_lab[:, 2], pal_lab[:, 1])) % 360.0

    V = 255.0 - pal_f  # (M, 3) absorption
    V_norm2 = np.sum(V * V, axis=1) + 1e-6  # (M,)
    U = V / 255.0  # (M, 3) normalized absorption 0..1

    P = 255.0 - sample_rgb  # (N, 3)
    # 1) Tek murekkep + beyaz projeksiyonu: (N, M)
    # Doygun/koyu kati alanlarda (alpha >= 0.90) tram yerine kati 1.0 basima izin ver (siyah yazilar solmasin)
    alpha_1_raw = np.matmul(P, V.T) / V_norm2[None, :]
    alpha_1 = np.where(alpha_1_raw >= 0.90, np.clip(alpha_1_raw, 0.0, 1.0), np.clip(alpha_1_raw, 0.0, max_dot))
    pred_1 = 255.0 - alpha_1[:, :, None] * V[None, :, :]  # (N, M, 3)
    pred_1_lab = rgb_f32_to_lab(pred_1)
    de_1 = np.linalg.norm(pred_1_lab - sample_lab[:, None, :], axis=2)  # (N, M)

    best_idx1 = np.argmin(de_1, axis=1)
    best_de = de_1[np.arange(N), best_idx1].copy()
    best_rgb = pred_1[np.arange(N), best_idx1].copy()

    # 2) Iki uyumlu murekkep + beyaz overprint karisimi
    if M >= 2:
        # Karsi zit renk kirlilik korumasi icin ciftleri filtrele
        pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                # Iki koyu achromatic notr birlesimi gereksizse de zarar vermez
                dh = L.hue_diff_deg(float(pal_hue[i]), float(pal_hue[j]))
                both_chrom = (pal_C[i] > 22.0) and (pal_C[j] > 22.0)
                is_opposite = both_chrom and (dh > 125.0)
                # Mavi (~210-285) + Sari (~75-110) -> yesil uretimi fiziksel olarak gecerlidir
                is_blue_yellow = (
                    ((200.0 <= pal_hue[i] <= 290.0) and (68.0 <= pal_hue[j] <= 115.0))
                    or ((200.0 <= pal_hue[j] <= 290.0) and (68.0 <= pal_hue[i] <= 115.0))
                )
                pairs.append((i, j, is_opposite and not is_blue_yellow))

        # Sadece en yakin tekil adaylardan birini iceren veya top ciftler uzerinde hizli degerlendir
        for i, j, block_low_chroma in pairs:
            vi = V[i]
            vj = V[j]
            ui = U[i]
            uj = U[j]
            g11 = float(np.dot(vi, vi)) + 1e-4
            g22 = float(np.dot(vj, vj)) + 1e-4
            g12 = float(np.dot(vi, vj))
            det = g11 * g22 - g12 * g12
            if det < 1e-3:
                continue

            rhs1 = np.matmul(P, vi)
            rhs2 = np.matmul(P, vj)
            a1 = np.clip((g22 * rhs1 - g12 * rhs2) / det, 0.0, max_dot)
            a2 = np.clip((g11 * rhs2 - g12 * rhs1) / det, 0.0, max_dot)

            # Subtractive + yari-opak tekstil overprint modeli
            sub_r = (1.0 - a1[:, None] * ui[None, :]) * (1.0 - a2[:, None] * uj[None, :]) * 255.0
            add_r = 255.0 - np.clip(a1[:, None] * vi[None, :] + a2[:, None] * vj[None, :], 0.0, 255.0)
            pred_pair = 0.60 * sub_r + 0.40 * add_r

            if block_low_chroma:
                # Zit iki parlak murekkep dusuk kromali bej/gri bolgede kirli benek yapmasin
                mask_valid = sample_chroma >= 26.0
                if not mask_valid.any():
                    continue
                pred_lab = rgb_f32_to_lab(pred_pair[mask_valid])
                de_pair = np.linalg.norm(pred_lab - sample_lab[mask_valid], axis=1)
                idxs = np.where(mask_valid)[0]
                imp = de_pair < best_de[idxs]
                if imp.any():
                    upd = idxs[imp]
                    best_de[upd] = de_pair[imp]
                    best_rgb[upd] = pred_pair[upd]
            else:
                pred_lab = rgb_f32_to_lab(pred_pair)
                de_pair = np.linalg.norm(pred_lab - sample_lab, axis=1)
                imp = de_pair < best_de
                if imp.any():
                    best_de[imp] = de_pair[imp]
                    best_rgb[imp] = pred_pair[imp]

    return best_de, best_rgb


def _render_simulation_image(
    rgb_prev: np.ndarray,
    fig_prev: np.ndarray,
    palette_rgb: np.ndarray,
    fabric_rgb: Tuple[int, int, int],
    max_dot: float = 0.96,
) -> np.ndarray:
    """Kullanici onizlemesi icin secili paletle tam tuval baski simulasyonu cizer."""
    h, w = fig_prev.shape
    out = np.full((h, w, 3), np.array(fabric_rgb, dtype=np.uint8), dtype=np.uint8)
    ys, xs = np.where(fig_prev)
    if len(ys) == 0:
        return out
    pts_rgb = rgb_prev[ys, xs].astype(np.float32)
    pts_lab = rgb_f32_to_lab(pts_rgb)
    pts_C = np.hypot(pts_lab[:, 1], pts_lab[:, 2])
    _, pred_rgb = _evaluate_palette_pixels(palette_rgb, pts_rgb, pts_lab, pts_C, max_dot=max_dot)
    out[ys, xs] = np.clip(np.round(pred_rgb), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 2) ORNEKLEME VE ADAY HAVUZU OLUSTURMA (NADIR GOZ/VURGU RENKLERI KORUMALI)
# ---------------------------------------------------------------------------

def _build_weighted_samples_and_candidates(
    rgb_sm: np.ndarray,
    fig_s: np.ndarray,
    rng: np.random.Generator,
    n_uniform: int = 1000,
    n_salient: int = 350,
) -> Dict[str, Any]:
    """
    Optimizasyon icin uniform + nadir/yuksek-kroma (orn. goz renkleri) agirlikli ornekler
    ve koyu/doygun ana renklerden olusan aday murekkep havuzu uretir.
    """
    pts_rgb = rgb_sm[fig_s].astype(np.float32)
    pts_lab = rgb_f32_to_lab(pts_rgb)
    pts_C = np.hypot(pts_lab[:, 1], pts_lab[:, 2])
    total = len(pts_rgb)
    if total == 0:
        raise ValueError("Gorselde basilacak figur pikseli bulunamadi.")

    # Uniform orneklem (gercek dE ve kalite olcumu icin)
    u_cnt = min(n_uniform, total)
    u_idx = rng.choice(total, size=u_cnt, replace=False) if total > u_cnt else np.arange(total)
    u_rgb = pts_rgb[u_idx]
    u_lab = pts_lab[u_idx]
    u_C = pts_C[u_idx]

    # Nadir yuksek kromali alanlar (orn. 3 kedide kucuk mavi/sari/yesil gozler)
    bins = (
        np.clip((pts_lab[:, 0] // 14.0).astype(np.int32), 0, 7) * 400
        + np.clip(((pts_lab[:, 1] + 128.0) // 16.0).astype(np.int32), 0, 15) * 20
        + np.clip(((pts_lab[:, 2] + 128.0) // 16.0).astype(np.int32), 0, 15)
    )
    uniq_b, inv_b, cnt_b = np.unique(bins, return_inverse=True, return_counts=True)
    px_bin_cnt = cnt_b[inv_b]
    # Kucuk/orta nadir ve renkli kutular
    rare_mask = (pts_C >= 20.0) & (px_bin_cnt >= max(4, int(0.0004 * total))) & (px_bin_cnt <= int(0.045 * total))
    if rare_mask.sum() < 40:
        rare_mask = pts_C >= 28.0

    rare_indices = np.where(rare_mask)[0]
    if len(rare_indices) > 0:
        s_cnt = min(n_salient, len(rare_indices))
        s_idx = rng.choice(rare_indices, size=s_cnt, replace=False) if len(rare_indices) > s_cnt else rare_indices
        opt_rgb = np.concatenate([u_rgb, pts_rgb[s_idx]], axis=0)
        opt_lab = np.concatenate([u_lab, pts_lab[s_idx]], axis=0)
        opt_C = np.concatenate([u_C, pts_C[s_idx]], axis=0)
        opt_w = np.concatenate([np.ones(len(u_rgb), dtype=np.float32), np.full(len(s_idx), 2.2, dtype=np.float32)])
    else:
        opt_rgb, opt_lab, opt_C = u_rgb, u_lab, u_C
        opt_w = np.ones(len(u_rgb), dtype=np.float32)
    opt_w = opt_w / np.sum(opt_w)

    # Aday Havuzu (K-means genel + K-means yuksek kroma + koyulastirilmis ana temsilciler + notrler)
    cand_list: List[Tuple[int, int, int]] = []
    sub_idx = rng.choice(total, size=min(30000, total), replace=False)
    sub_lab = pts_lab[sub_idx]
    sub_rgb = pts_rgb[sub_idx]
    sub_C = pts_C[sub_idx]

    def _run_kmeans_rgb(src_lab: np.ndarray, k_req: int) -> None:
        if len(src_lab) < 10:
            return
        k_use = min(k_req, max(2, len(src_lab) // 4))
        cv2.setRNGSeed(0)
        _, lbl, cen = cv2.kmeans(
            src_lab.astype(np.float32),
            k_use,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.8),
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        # Her kume icin merkez + en doygun/koyu temsilci (beyazla tram mantigi icin!)
        for ci in range(k_use):
            m_c = lbl.ravel() == ci
            if not m_c.any():
                continue
            c_rgbs = sub_rgb[m_c] if len(src_lab) == len(sub_lab) else src_lab[m_c]
            med = np.median(c_rgbs, axis=0) if len(src_lab) == len(sub_lab) else None
            if med is not None:
                cand_list.append(tuple(int(np.clip(round(x), 0, 255)) for x in med))
                # Kume icindeki en doygun/koyu %20 temsilci (aciklar beyaz tramdan gelir)
                c_labs = sub_lab[m_c]
                sat_dark_score = np.hypot(c_labs[:, 1], c_labs[:, 2]) * 0.7 - c_labs[:, 0] * 0.45
                top_q = np.percentile(sat_dark_score, 85)
                rich_pts = c_rgbs[sat_dark_score >= top_q]
                if len(rich_pts) > 0:
                    rich_med = np.median(rich_pts, axis=0)
                    cand_list.append(tuple(int(np.clip(round(x), 0, 255)) for x in rich_med))

    _run_kmeans_rgb(sub_lab, 18)

    # Yuksek kromali ve nadir renkli bolgelerden ek K-means (goz renkleri vb.)
    chrom_mask = pts_C >= 22.0
    if chrom_mask.sum() >= 20:
        c_pts_rgb = pts_rgb[chrom_mask]
        c_pts_lab = pts_lab[chrom_mask]
        if len(c_pts_lab) > 15000:
            pick = rng.choice(len(c_pts_lab), size=15000, replace=False)
            c_pts_rgb = c_pts_rgb[pick]
            c_pts_lab = c_pts_lab[pick]
        k_ch = min(14, max(2, len(c_pts_lab) // 5))
        cv2.setRNGSeed(1)
        _, clbl, _ = cv2.kmeans(
            c_pts_lab.astype(np.float32),
            k_ch,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.8),
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        for ci in range(k_ch):
            mc = clbl.ravel() == ci
            if mc.any():
                med = np.median(c_pts_rgb[mc], axis=0)
                cand_list.append(tuple(int(np.clip(round(x), 0, 255)) for x in med))

    # Notr koyu / siyah / koyu gri kontrol adaylari
    dark_mask = (pts_lab[:, 0] < 24.0) & (pts_C < 14.0)
    if dark_mask.sum() >= 10:
        med_dk = np.median(pts_rgb[dark_mask], axis=0)
        cand_list.append(tuple(int(np.clip(round(x), 0, 255)) for x in med_dk))
        if float(np.min(pts_lab[dark_mask, 0])) < 10.0:
            cand_list.append((4, 4, 4))
        cand_list.append((18, 18, 20))
    gray_mid = (pts_lab[:, 0] >= 25.0) & (pts_lab[:, 0] <= 55.0) & (pts_C < 12.0)
    if gray_mid.sum() >= 15:
        med_gr = np.median(pts_rgb[gray_mid], axis=0)
        cand_list.append(tuple(int(np.clip(round(x), 0, 255)) for x in med_gr))

    # Saf beyaza cok yakin adaylari havuzdan cikar (beyaz zemin/fondan gelir; gerekirse ust_beyaz eklenir)
    filtered: List[Tuple[int, int, int]] = []
    filtered_labs: List[np.ndarray] = []
    for rgb_c in cand_list:
        lab_c = rgb_f32_to_lab(np.array(rgb_c, dtype=np.float32))
        C_c = float(math.hypot(float(lab_c[1]), float(lab_c[2])))
        if lab_c[0] > 91.0 and C_c < 7.0:
            continue
        too_close = False
        for prev_lab in filtered_labs:
            if float(np.linalg.norm(lab_c - prev_lab)) < 5.5:
                too_close = True
                break
        if not too_close:
            filtered.append(rgb_c)
            filtered_labs.append(lab_c)

    if not filtered:
        filtered = [(24, 24, 26), (180, 50, 50), (45, 110, 175)]

    return {
        "u_rgb": u_rgb,
        "u_lab": u_lab,
        "u_C": u_C,
        "opt_rgb": opt_rgb,
        "opt_lab": opt_lab,
        "opt_C": opt_C,
        "opt_w": opt_w,
        "candidates": np.array(filtered, dtype=np.uint8),
    }


# ---------------------------------------------------------------------------
# 3) CMYK + YARDIMCI RENK ANALIZI
# ---------------------------------------------------------------------------

def _analyze_cmyk_and_helpers(
    rgb_sm: np.ndarray,
    fig_s: np.ndarray,
    u_rgb: np.ndarray,
    u_lab: np.ndarray,
    u_C: np.ndarray,
    max_dot: float,
    preview_dir: Optional[str],
    fabric_rgb: Tuple[int, int, int],
) -> Dict[str, Any]:
    _, rgb_cmyk_back, icc_name = extract_cmyk_4ch_and_roundtrip(rgb_sm)
    # Uniform figür indekslerini piksel koordinatından almak yerine tüm figür üzerinde ölçelim
    fig_orig_rgb = rgb_sm[fig_s].astype(np.float32)
    fig_back_rgb = rgb_cmyk_back[fig_s].astype(np.float32)
    if len(fig_orig_rgb) > 25000:
        step = max(1, len(fig_orig_rgb) // 25000)
        fig_orig_rgb = fig_orig_rgb[::step]
        fig_back_rgb = fig_back_rgb[::step]

    lab_o = rgb_f32_to_lab(fig_orig_rgb)
    lab_b = rgb_f32_to_lab(fig_back_rgb)
    cmyk_de = np.linalg.norm(lab_o - lab_b, axis=1)
    C_o = np.hypot(lab_o[:, 1], lab_o[:, 2])

    mean_de = float(np.mean(cmyk_de))
    p95_de = float(np.percentile(cmyk_de, 95))
    oog_mask = (cmyk_de > 6.0) & (C_o > 18.0)
    oog_pct = float(np.mean(cmyk_de > 6.0) * 100.0)
    kalite_cmyk = _compute_quality_score(mean_de, p95_de)

    cmyk_prev_path = ""
    cmyk_spot_prev_path = ""
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        prev_c = np.full_like(rgb_sm, np.array(fabric_rgb, dtype=np.uint8))
        prev_c[fig_s] = rgb_cmyk_back[fig_s]
        cmyk_prev_path = os.path.join(preview_dir, "onizleme_cmyk.png")
        Image.fromarray(prev_c).save(cmyk_prev_path)

    # Gamut disi bolgelerden en fazla 3 yardimci spot renk adayi cikar
    helpers: List[Dict[str, Any]] = []
    if oog_mask.sum() >= 15:
        oog_rgb = fig_orig_rgb[oog_mask]
        oog_lab = lab_o[oog_mask]
        k_h = min(3, max(1, len(oog_lab) // 25))
        cv2.setRNGSeed(2)
        _, hlbl, _ = cv2.kmeans(
            oog_lab.astype(np.float32),
            k_h,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.8),
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        # Kapsamaya gore sirala ve her yardimci rengin CMYK hatasini ne kadar dusurdugunu olc
        counts = np.bincount(hlbl.ravel(), minlength=k_h)
        order = np.argsort(-counts)

        # CMYK + Yardimci simülasyon: mevcut CMYK tahminini yardimci spotlarla iyilestir
        cur_rgb_sim = fig_back_rgb.copy()
        cur_de = cmyk_de.copy()

        for ci in order:
            mc = hlbl.ravel() == ci
            if mc.sum() < 5:
                continue
            med = tuple(int(np.clip(round(x), 0, 255)) for x in np.median(oog_rgb[mc], axis=0))
            spot_lab = rgb_f32_to_lab(np.array(med, dtype=np.float32))
            spot_C = math.hypot(float(spot_lab[1]), float(spot_lab[2]))
            if spot_C < 18.0:
                continue
            # Ayni tonda ikinci yardimci ekleme
            if any(np.linalg.norm(spot_lab - rgb_f32_to_lab(np.array(h["rgb"], dtype=np.float32))) < 18.0 for h in helpers):
                continue

            # Bu yardimci spot ile beyaz karisimi hangi pikselleri CMYK'dan daha iyi verir?
            s_de, s_pred = _evaluate_palette_pixels(
                np.array([med], dtype=np.uint8), fig_orig_rgb, lab_o, C_o, max_dot=max_dot
            )
            better = s_de < cur_de
            if better.mean() < 0.008 and len(helpers) > 0:
                continue
            new_de = np.minimum(cur_de, s_de)
            gain_de = float(cur_de.mean() - new_de.mean())
            if gain_de < 0.25 and len(helpers) > 0:
                continue

            cur_de = new_de
            cur_rgb_sim[better] = s_pred[better]
            new_mean = float(cur_de.mean())
            new_p95 = float(np.percentile(cur_de, 95))
            helpers.append({
                "hex": rgb_to_hex(med),
                "rgb": list(med),
                "ad": LA.auto_emprime_channel_name(med, suffix="spot"),
                "rol": "tram",
                "yeni_mean_de": round(new_mean, 2),
                "yeni_p95_de": round(new_p95, 2),
                "yeni_kalite_skoru": _compute_quality_score(new_mean, new_p95),
            })

    if preview_dir and helpers:
        helper_arr = np.array([h["rgb"] for h in helpers], dtype=np.uint8)
        full_fig_o = rgb_sm[fig_s].astype(np.float32)
        full_fig_b = rgb_cmyk_back[fig_s].astype(np.float32)
        full_lab_o = rgb_f32_to_lab(full_fig_o)
        full_lab_b = rgb_f32_to_lab(full_fig_b)
        de_b = np.linalg.norm(full_lab_o - full_lab_b, axis=1)
        C_full = np.hypot(full_lab_o[:, 1], full_lab_o[:, 2])
        de_sp, pred_sp = _evaluate_palette_pixels(helper_arr, full_fig_o, full_lab_o, C_full, max_dot=max_dot)
        better_full = de_sp < de_b
        merged_fig = full_fig_b.copy()
        merged_fig[better_full] = pred_sp[better_full]
        prev_cs = np.full_like(rgb_sm, np.array(fabric_rgb, dtype=np.uint8))
        prev_cs[fig_s] = np.clip(np.round(merged_fig), 0, 255).astype(np.uint8)
        cmyk_spot_prev_path = os.path.join(preview_dir, "onizleme_cmyk_yardimci.png")
        Image.fromarray(prev_cs).save(cmyk_spot_prev_path)

    if oog_pct < 3.5 and p95_de <= 7.5:
        karar_kodu = "cmyk_yeterli"
        yeterli_mi = True
        karar_metni = (
            f"Sadece CMYK yeterli (gamut dışı alan %{oog_pct:.1f}, kalite %{kalite_cmyk:.0f}). "
            "120T ipek / 55 LPI pat boyayla açık tişörtte doğrudan basılabilir."
        )
    elif helpers:
        karar_kodu = "cmyk_yardimci"
        yeterli_mi = False
        best_h = helpers[-1]
        karar_metni = (
            f"Sadece CMYK gamut dışı alanda (%{oog_pct:.1f}) solma yapıyor. "
            f"CMYK + {len(helpers)} yardımcı spot ile kalite %{kalite_cmyk:.0f} → %{best_h['yeni_kalite_skoru']:.0f} yükselir."
        )
    else:
        karar_kodu = "spot_onerilir"
        yeterli_mi = False
        karar_metni = (
            f"CMYK canlılık kaybı yüksek (%{oog_pct:.1f}, ort ΔE={mean_de:.1f}). "
            "Koyu/canlı spot renkler + beyaz tram (Simulated Process) önerilir."
        )

    return {
        "icc_profil": icc_name,
        "yeterli_mi": yeterli_mi,
        "karar_kodu": karar_kodu,
        "karar_metni": karar_metni,
        "mean_de": round(mean_de, 2),
        "p95_de": round(p95_de, 2),
        "oog_pct": round(oog_pct, 2),
        "kalite_skoru": kalite_cmyk,
        "onizleme_png": cmyk_prev_path,
        "yardimci_oneriler": helpers,
        "cmyk_spot_onizleme_png": cmyk_spot_prev_path,
    }


# ---------------------------------------------------------------------------
# 4) MINIMUM RENK OPTIMIZATORU (GREEDY + SWAP + SA + REFINEMENT)
# ---------------------------------------------------------------------------

def _score_subset(
    indices: List[int],
    cand_rgb: np.ndarray,
    opt_rgb: np.ndarray,
    opt_lab: np.ndarray,
    opt_C: np.ndarray,
    opt_w: np.ndarray,
    max_dot: float,
) -> float:
    pal = cand_rgb[indices]
    de, _ = _evaluate_palette_pixels(pal, opt_rgb, opt_lab, opt_C, max_dot=max_dot)
    w_mean = float(np.sum(de * opt_w))
    p95_soft = float(np.percentile(de, 95))
    return w_mean + 0.18 * p95_soft


def _refine_palette_rgb(
    pal_init: np.ndarray,
    opt_rgb: np.ndarray,
    opt_lab: np.ndarray,
    opt_C: np.ndarray,
    opt_w: np.ndarray,
    max_dot: float,
    steps: int = 2,
    stop_fn: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Secilen palet renklerinin RGB degerlerini kucuk adimlarla iyilestirir."""
    cur = pal_init.astype(np.int32).copy()
    K = len(cur)
    best_obj = _score_subset(list(range(K)), cur.astype(np.uint8), opt_rgb, opt_lab, opt_C, opt_w, max_dot)
    deltas = (16, -16, 8, -8) if steps >= 2 else (12, -12)

    for _ in range(steps):
        improved_any = False
        for i in range(K):
            for ch in range(3):
                for d in deltas:
                    if stop_fn and stop_fn():
                        return cur.astype(np.uint8)
                    val = int(cur[i, ch]) + d
                    if val < 0 or val > 255:
                        continue
                    trial = cur.copy()
                    trial[i, ch] = val
                    obj = _score_subset(
                        list(range(K)), trial.astype(np.uint8), opt_rgb, opt_lab, opt_C, opt_w, max_dot
                    )
                    if obj + 1e-4 < best_obj:
                        best_obj = obj
                        cur = trial
                        improved_any = True
                        break
        if not improved_any:
            break
    return cur.astype(np.uint8)


def _optimize_for_k(
    k: int,
    cand_rgb: np.ndarray,
    data: Dict[str, Any],
    derinlik: str,
    max_dot: float,
    deadline: float,
    stop_fn: Callable[[], bool],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float, float]:
    opt_rgb = data["opt_rgb"]
    opt_lab = data["opt_lab"]
    opt_C = data["opt_C"]
    opt_w = data["opt_w"]
    M = len(cand_rgb)
    k_eff = min(k, M)

    # 1) Greedy baslangic
    chosen: List[int] = []
    remaining = list(range(M))
    while len(chosen) < k_eff:
        if stop_fn():
            break
        best_c = remaining[0]
        best_sc = 1e18
        for c in remaining:
            sc = _score_subset(chosen + [c], cand_rgb, opt_rgb, opt_lab, opt_C, opt_w, max_dot)
            if sc < best_sc:
                best_sc = sc
                best_c = c
        chosen.append(best_c)
        remaining.remove(best_c)

    if len(chosen) < k_eff:
        chosen.extend(remaining[: k_eff - len(chosen)])

    cur_sc = _score_subset(chosen, cand_rgb, opt_rgb, opt_lab, opt_C, opt_w, max_dot)

    # 2) Swap iyilestirme
    max_swaps = {"hizli": 1, "dengeli": 3, "derin": 6}.get(derinlik, 2)
    for _ in range(max_swaps):
        if stop_fn() or time.monotonic() >= deadline:
            break
        swapped = False
        rem_set = [c for c in range(M) if c not in chosen]
        for pos in range(len(chosen)):
            if stop_fn() or time.monotonic() >= deadline:
                break
            base_other = [chosen[j] for j in range(len(chosen)) if j != pos]
            best_rep = chosen[pos]
            for c in rem_set:
                sc = _score_subset(base_other + [c], cand_rgb, opt_rgb, opt_lab, opt_C, opt_w, max_dot)
                if sc + 1e-4 < cur_sc:
                    cur_sc = sc
                    best_rep = c
                    swapped = True
            if best_rep != chosen[pos]:
                rem_set.append(chosen[pos])
                rem_set.remove(best_rep)
                chosen[pos] = best_rep
        if not swapped:
            break

    best_pal = cand_rgb[chosen].copy()

    # 3) Derin modda rastgele coklu yeniden baslatma + Simulated Annealing
    if derinlik == "derin" and M > k_eff and time.monotonic() < deadline and not stop_fn():
        n_sa_iters = 45
        temp = 0.8
        sa_chosen = list(chosen)
        sa_sc = cur_sc
        for step_sa in range(n_sa_iters):
            if stop_fn() or time.monotonic() >= deadline:
                break
            rem = [c for c in range(M) if c not in sa_chosen]
            if not rem:
                break
            pos = int(rng.integers(0, len(sa_chosen)))
            cand_new = int(rem[int(rng.integers(0, len(rem)))])
            trial_list = list(sa_chosen)
            trial_list[pos] = cand_new
            trial_sc = _score_subset(trial_list, cand_rgb, opt_rgb, opt_lab, opt_C, opt_w, max_dot)
            if trial_sc < sa_sc or rng.random() < math.exp((sa_sc - trial_sc) / max(temp, 0.05)):
                sa_chosen = trial_list
                sa_sc = trial_sc
                if trial_sc + 1e-4 < cur_sc:
                    cur_sc = trial_sc
                    best_pal = cand_rgb[trial_list].copy()
            temp *= 0.94

    # 4) Dengeli ve derin modda RGB koordinat rafinasyonu
    if derinlik in ("dengeli", "derin") and time.monotonic() < deadline and not stop_fn():
        ref_steps = 2 if derinlik == "derin" else 1
        best_pal = _refine_palette_rgb(
            best_pal, opt_rgb, opt_lab, opt_C, opt_w, max_dot, steps=ref_steps, stop_fn=stop_fn
        )

    # 5) Gercek uniform orneklem uzerinde durust hata metrikleri
    u_de, _ = _evaluate_palette_pixels(best_pal, data["u_rgb"], data["u_lab"], data["u_C"], max_dot=max_dot)
    mean_de = float(np.mean(u_de))
    p95_de = float(np.percentile(u_de, 95))
    kalite = _compute_quality_score(mean_de, p95_de)
    return best_pal, mean_de, p95_de, kalite


# ---------------------------------------------------------------------------
# 5) ANA ANALIZ API'SI VE CLI
# ---------------------------------------------------------------------------

def analyze_emprime_image(
    girdi_yolu: str,
    kumas: str = "ikisi",
    derinlik: str = "hizli",
    ipek: str = "ozel",
    zemin_maske: str = "oto",
    min_k: int = 3,
    max_k: int = 9,
    onizleme_klasor: Optional[str] = None,
    stop_flag_path: Optional[str] = None,
    max_sure_sn: Optional[float] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Gorseli analiz edip CMYK yeterliligi, yardimci spot onerileri ve K=min_k..max_k icin
    "en az renk ile en yuksek kalite" spot palet egrisini uretir.
    """
    t0 = time.monotonic()
    derinlik = derinlik if derinlik in ("hizli", "dengeli", "derin") else "hizli"
    default_budgets = {"hizli": 15.0, "dengeli": 60.0, "derin": 240.0}
    budget_sn = float(max_sure_sn) if (max_sure_sn is not None and max_sure_sn > 0) else default_budgets[derinlik]
    deadline = t0 + budget_sn

    def _emit(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        print(f"PROGRESS {pct} {msg}", flush=True)

    def _should_stop() -> bool:
        return bool(stop_flag_path and os.path.exists(stop_flag_path))

    _emit(5, "Görsel yükleniyor ve figür maskesi çıkarılıyor...")
    im = Image.open(girdi_yolu)
    rgb_orig, fig_full = LA.build_auto_clean_figure_mask(im, None, zemin_maske=zemin_maske)
    H, W = rgb_orig.shape[:2]

    work_max = {"hizli": 360, "dengeli": 460, "derin": 560}[derinlik]
    scale = min(1.0, float(work_max) / float(max(H, W)))
    sw = max(32, int(round(W * scale)))
    sh = max(32, int(round(H * scale)))
    rgb_sm = cv2.resize(rgb_orig, (sw, sh), interpolation=cv2.INTER_AREA)
    fig_s = cv2.resize(fig_full.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    mesh_prof = L.MESH_PROFILES.get(str(ipek), {})
    max_dot = float(mesh_prof.get("max_dot", 0.96))
    rec_lpi = float(mesh_prof.get("rec_lpi", 45.0))
    fabric_rgb = (255, 255, 255) if kumas == "acik" else (28, 30, 34)

    # Doku / duz oranlari
    lab_cv = L.rgb_to_lab_cv(rgb_sm)
    mu = cv2.blur(lab_cv, (5, 5))
    mu2 = cv2.blur(lab_cv * lab_cv, (5, 5))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0).sum(axis=2))
    n_fig = max(int(fig_s.sum()), 1)
    duz_oran = float(((sd < 2.2) & fig_s).sum() / n_fig)
    doku_oran = float(((sd >= 2.2) & fig_s).sum() / n_fig)

    rng = np.random.default_rng(42)
    _emit(14, "Renk uzayı örnekleniyor ve aday mürekkep havuzu kuruluyor...")
    data = _build_weighted_samples_and_candidates(rgb_sm, fig_s, rng)

    _emit(24, "CMYK yeterlilik ve gamut dışı analizi yapılıyor...")
    cmyk_analiz = _analyze_cmyk_and_helpers(
        rgb_sm, fig_s, data["u_rgb"], data["u_lab"], data["u_C"], max_dot, onizleme_klasor, fabric_rgb
    )

    k_lo = max(2, int(min_k))
    k_hi = max(k_lo, min(12, int(max_k)))
    k_values = list(range(k_lo, k_hi + 1))

    spot_egri: List[Dict[str, Any]] = []
    erken_durduruldu = False
    cand_rgb = data["candidates"]

    for idx_k, k in enumerate(k_values):
        if _should_stop() or (idx_k > 0 and time.monotonic() >= deadline):
            erken_durduruldu = _should_stop()
            break
        pct = 30 + int(62 * (idx_k / max(1, len(k_values))))
        _emit(pct, f"{k} renk için en yüksek kalite kombinasyonu hesaplanıyor...")

        pal_k, m_de, p95_de, kal = _optimize_for_k(
            k, cand_rgb, data, derinlik, max_dot, deadline, _should_stop, rng
        )
        # Parlaklik/ton siralamasi (aciktan koyuya dogru baski sirasi)
        pal_labs = rgb_f32_to_lab(pal_k.astype(np.float32))
        order_l = np.argsort(-pal_labs[:, 0])
        pal_k = pal_k[order_l]

        renk_list = []
        used_nm: Dict[str, int] = {}
        for rgb_item in pal_k:
            t_rgb = (int(rgb_item[0]), int(rgb_item[1]), int(rgb_item[2]))
            rol = _classify_ink_role(t_rgb)
            base_nm = LA.auto_emprime_channel_name(t_rgb, suffix=("" if rol == "siyah" else "tram"))
            cnt = used_nm.get(base_nm, 0) + 1
            used_nm[base_nm] = cnt
            nm = base_nm if cnt == 1 else f"{base_nm}_{cnt}"
            renk_list.append({
                "hex": rgb_to_hex(t_rgb),
                "rgb": list(t_rgb),
                "ad": nm,
                "rol": rol,
            })

        prev_path = ""
        if onizleme_klasor:
            os.makedirs(onizleme_klasor, exist_ok=True)
            prev_img = _render_simulation_image(rgb_sm, fig_s, pal_k, fabric_rgb, max_dot=max_dot)
            prev_path = os.path.join(onizleme_klasor, f"onizleme_spot_{k}renk.png")
            Image.fromarray(prev_img).save(prev_path)

        spot_egri.append({
            "k": k,
            "kafa_acik": k,
            "kafa_koyu": k + 1,  # +1 beyaz fon
            "mean_de": round(m_de, 2),
            "p95_de": round(p95_de, 2),
            "kalite_skoru": kal,
            "onerilen_mi": False,
            "renkler": renk_list,
            "onizleme_png": prev_path,
        })

        if _should_stop():
            erken_durduruldu = True
            break

    # Dirsek (Elbow) Optimumu: en az renk ile >= %85 kalite veya doyuma ulasan ilk K
    opt_idx = 0
    if spot_egri:
        opt_idx = len(spot_egri) - 1
        for i, row in enumerate(spot_egri):
            if row["kalite_skoru"] >= 86.0 and row["mean_de"] <= 3.6:
                opt_idx = i
                break
            if i > 0:
                gain = row["kalite_skoru"] - spot_egri[i - 1]["kalite_skoru"]
                if row["kalite_skoru"] >= 80.0 and gain < 2.2:
                    opt_idx = i - 1
                    break
        spot_egri[opt_idx]["onerilen_mi"] = True

    # Genel en iyi strateji onerisi
    opt_row = spot_egri[opt_idx] if spot_egri else None
    if cmyk_analiz["yeterli_mi"] and kumas == "acik":
        en_iyi = {
            "mod": "cmyk",
            "neden": f"Açık tişörtte sadece CMYK (4 kafa) yeterli kaliteyi (%{cmyk_analiz['kalite_skoru']:.0f}) veriyor.",
            "k": 4,
            "renkler": [],
            "onizleme_png": cmyk_analiz["onizleme_png"],
        }
    elif (
        cmyk_analiz["karar_kodu"] == "cmyk_yardimci"
        and cmyk_analiz["yardimci_oneriler"]
        and (opt_row is None or opt_row["k"] >= 6)
        and kumas == "acik"
    ):
        last_h = cmyk_analiz["yardimci_oneriler"][-1]
        en_iyi = {
            "mod": "cmyk_spot",
            "neden": (
                f"CMYK + {len(cmyk_analiz['yardimci_oneriler'])} yardımcı spot renk ile "
                f"kalite %{last_h['yeni_kalite_skoru']:.0f} seviyesine çıkar."
            ),
            "k": 4 + len(cmyk_analiz["yardimci_oneriler"]),
            "renkler": [
                {"hex": h["hex"], "rgb": h["rgb"], "ad": h["ad"], "rol": h["rol"]}
                for h in cmyk_analiz["yardimci_oneriler"]
            ],
            "onizleme_png": cmyk_analiz["cmyk_spot_onizleme_png"],
        }
    elif opt_row is not None:
        en_iyi = {
            "mod": "ozel_renk",
            "neden": (
                f"En az {opt_row['k']} seçili renk (+ açık tonlar beyaz/zemin tramı) ile "
                f"%{opt_row['kalite_skoru']:.0f} kalite (ort ΔE={opt_row['mean_de']}) elde edilir."
            ),
            "k": opt_row["k"],
            "renkler": opt_row["renkler"],
            "onizleme_png": opt_row["onizleme_png"],
        }
    else:
        en_iyi = {"mod": "oto", "neden": "Otomatik ayrım önerilir.", "k": 6, "renkler": [], "onizleme_png": ""}

    # Onerilen ipek / LPI rehberi
    if duz_oran > 0.80:
        ipek_oneri = "77T (Düz ağırlıklı işler; 45 LPI veya tramsız)"
    elif en_iyi["mod"] in ("cmyk", "cmyk_spot"):
        ipek_oneri = "120T (CMYK/Yardımcı için 55 LPI önerilir; koyu fon 77T-90T)"
    else:
        ipek_oneri = "90T (Simulated Process 45-50 LPI eliptik nokta optimumu)"

    sure_sn = round(time.monotonic() - t0, 2)
    _emit(100, f"Analiz tamamlandı ({sure_sn:.1f} sn).")

    return {
        "girdi": os.path.basename(girdi_yolu),
        "derinlik": derinlik,
        "kumas": kumas,
        "ipek": ipek,
        "onerilen_lpi": rec_lpi,
        "gorsel_ozet": {
            "boyut_px": [W, H],
            "calisma_px": [sw, sh],
            "duz_oran": round(duz_oran, 3),
            "doku_oran": round(doku_oran, 3),
            "ipek_oneri": ipek_oneri,
        },
        "cmyk_analiz": cmyk_analiz,
        "spot_egri": spot_egri,
        "en_iyi_oneri": en_iyi,
        "erken_durduruldu": erken_durduruldu,
        "sure_sn": sure_sn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MeChi Akilli Emprime Analiz & Minimum Renk Optimizatoru")
    ap.add_argument("girdi", help="Kaynak gorsel yolu")
    ap.add_argument("--cikti-json", required=True, help="Analiz rapor JSON cikti yolu")
    ap.add_argument("--onizleme-klasor", default=None, help="Simulasyon onizleme PNG klasoru")
    ap.add_argument("--kumas", choices=["koyu", "acik", "ikisi"], default="ikisi")
    ap.add_argument("--derinlik", choices=["hizli", "dengeli", "derin"], default="hizli")
    ap.add_argument("--ipek", choices=["ozel", "45", "55", "77", "90", "120"], default="ozel")
    ap.add_argument("--zemin-maske", choices=["oto", "maskeli", "tumu"], default="oto")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=9)
    ap.add_argument("--stop-flag", default=None, help="Bu dosya olusursa analizi en iyi mevcut sonucla durdur")
    ap.add_argument("--max-sure-sn", type=float, default=None, help="Maksimum sure siniri (saniye)")
    args = ap.parse_args()

    rapor = analyze_emprime_image(
        args.girdi,
        kumas=args.kumas,
        derinlik=args.derinlik,
        ipek=args.ipek,
        zemin_maske=args.zemin_maske,
        min_k=args.min_k,
        max_k=args.max_k,
        onizleme_klasor=args.onizleme_klasor,
        stop_flag_path=args.stop_flag,
        max_sure_sn=args.max_sure_sn,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.cikti_json)), exist_ok=True)
    with open(args.cikti_json, "w", encoding="utf-8") as fp:
        json.dump(rapor, fp, ensure_ascii=False, indent=2)
    print(f"[ANALIZ TAMAM] {args.cikti_json} ({rapor['sure_sn']} sn)", flush=True)


if __name__ == "__main__":
    main()
