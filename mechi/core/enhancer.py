"""Detay korumali, pastellesmesiz 4K yeniden-cizim netlestirme ve PSD katman temizleme motoru.

Mimari Boru Hatti:
1. Yapi Tensoru &Cok Olcekli Detay Kalkani (_compute_detail_shield):
   Ince tuy taramalarini, dikisleri, tisort cizgilerini ve suluboya/duman gecislerini %100 korur.
2. Izole Tek-Piksel Leke Onarimi & Self-Guided Mikro-Gurultu Temizligi:
   RECURS_FILTER veya medianBlur(5) KULLANMAZ; pastellesme ve doku erimesi yaratmaz.
3. Kademeli (2x -> 4x) Iterative Back-Projection (IBP) + Yapi Tensoru Tegetsel Kenar Akisi:
   Buyuturken orijinal piksel enerjisini geri yansitir, merdiven kiriklarini cizgiye paralel duzeltir
   ve kontur gecislerini halesiz (halo-free) sikilastirir.
4. Cok Olcekli Guided Detail & Murekkep Cizgisi Rekonstruksiyonu:
   Ince cizgi ve mikro-dokulari hale olusturmadan yeniden cizilmis netligine kavusturur.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from mechi.core.worker import TaskCancelled

PRESETS: Dict[str, dict] = {
    "hafif": {
        "label": "Hafif (Ham Doku & Suluboya)",
        "denoise": 10,
        "sharpen": 45,
        "vector_clean": 0,
        "edge_crisp": 30,
        "deblock": False,
        "despeckle": False,
    },
    "dengeli": {
        "label": "Detay Korumalı 4K Netleştirme",
        "denoise": 18,
        "sharpen": 60,
        "vector_clean": 15,
        "edge_crisp": 55,
        "deblock": True,
        "despeckle": True,
    },
    "guclu": {
        "label": "Bozuk / JPEG Piksel Onarım",
        "denoise": 42,
        "sharpen": 68,
        "vector_clean": 40,
        "edge_crisp": 70,
        "deblock": True,
        "despeckle": True,
    },
    "vektor_yakin": {
        "label": "Yeniden Çizim & Vektör Netliği (Önerilen)",
        "denoise": 22,
        "sharpen": 72,
        "vector_clean": 35,
        "edge_crisp": 78,
        "deblock": True,
        "despeckle": True,
    },
    "vektor_keskin": {
        "label": "Düz Grafik / Spot Baskı",
        "denoise": 35,
        "sharpen": 80,
        "vector_clean": 65,
        "edge_crisp": 90,
        "deblock": True,
        "despeckle": True,
    },
}


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TaskCancelled()


def _bleed_rgb_into_transparent(rgb: np.ndarray, alpha: np.ndarray, radius: int = 6) -> np.ndarray:
    """Seffaf piksel kenarlarinda siyah/beyaz halo olusmasini onlemek icin kenar rengini yayar."""
    opaque = alpha >= 16
    if opaque.all() or not opaque.any():
        return rgb
    out = rgb.copy()
    known = opaque.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(radius):
        dilated_mask = cv2.dilate(known.astype(np.uint8), kernel, iterations=1) > 0
        ring = dilated_mask & (~known)
        if not ring.any():
            break
        sum_rgb = cv2.blur(np.where(known[..., None], out, 0).astype(np.float32), (3, 3))
        cnt = cv2.blur(known.astype(np.float32), (3, 3))
        valid = ring & (cnt > 1e-4)
        if valid.any():
            fill = np.clip(np.round(sum_rgb / np.maximum(cnt[..., None], 1e-4)), 0, 255).astype(np.uint8)
            out[valid] = fill[valid]
            known[valid] = True
    return out


def _guided_filter_mono(img_01: np.ndarray, guide_01: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Kutu filtre tabanli hizli O(N) Edge-Preserving Guided Filter (pastellesme ve hale yaratmaz)."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide_01, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(img_01, cv2.CV_32F, ksize)
    mean_ip = cv2.boxFilter(guide_01 * img_01, cv2.CV_32F, ksize)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide_01 * guide_01, cv2.CV_32F, ksize)
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide_01 + mean_b


def _compute_detail_shield(rgb: np.ndarray) -> np.ndarray:
    """Yapi Tensoru tutarliligi (coherence) ile gercek cizgi/tuy/suluboya detayini rastgele gurultuden ayiran [0..1] kalkan."""
    rgb_f = rgb.astype(np.float32)
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    luma_s = cv2.GaussianBlur(luma, (3, 3), 0.75)

    # 1) Yapi Tensoru (Structure Tensor) yonlu tutarliligi (Coherence, entegrasyon olcegi 9x9):
    # Gercek tuy, cizgi ve konturlarda coherence yuksektir (~0.45..1.0); rastgele kum gurultusunde dusuktur (<0.20)
    gx_l = cv2.Scharr(luma_s, cv2.CV_32F, 1, 0) / 16.0
    gy_l = cv2.Scharr(luma_s, cv2.CV_32F, 0, 1) / 16.0
    j_xx = cv2.GaussianBlur(gx_l * gx_l, (9, 9), 2.0)
    j_xy = cv2.GaussianBlur(gx_l * gy_l, (9, 9), 2.0)
    j_yy = cv2.GaussianBlur(gy_l * gy_l, (9, 9), 2.0)
    coh = np.sqrt((j_xx - j_yy) ** 2 + 4.0 * (j_xy ** 2)) / (j_xx + j_yy + 1e-4)
    coh_gate = np.clip((coh - 0.25) / 0.35, 0.0, 1.0)

    # 2) Koyu murekkep/tuy cizgileri (Black-Hat) ve parlak dikis/detaylar (Top-Hat)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bh = np.maximum(
        cv2.morphologyEx(luma, cv2.MORPH_BLACKHAT, k3),
        0.65 * cv2.morphologyEx(luma, cv2.MORPH_BLACKHAT, k5),
    )
    th = np.maximum(
        cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, k3),
        0.65 * cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, k5),
    )

    # 3) Ham ve makro (gurultusu bastirilmis 7x7) RGB gradyanlari (beyaz uzeri turkuaz/suluboya lekeleri yakalar)
    smooth_rgb = cv2.GaussianBlur(rgb_f, (7, 7), 1.6)
    grad_raw = np.zeros_like(luma)
    grad_macro = np.zeros_like(luma)
    for ch in range(3):
        c_raw = rgb_f[..., ch]
        grad_raw = np.maximum(
            grad_raw,
            cv2.magnitude(cv2.Sobel(c_raw, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c_raw, cv2.CV_32F, 0, 1, ksize=3)),
        )
        c_mac = smooth_rgb[..., ch]
        grad_macro = np.maximum(
            grad_macro,
            cv2.magnitude(cv2.Sobel(c_mac, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c_mac, cv2.CV_32F, 0, 1, ksize=3)),
        )

    coherent_line_energy = (grad_raw * 0.45 + bh * 1.55 + th * 1.40) * coh_gate
    macro_edge_energy = np.maximum(0.0, grad_macro - 18.0) * 0.70

    energy = np.maximum(coherent_line_energy, macro_edge_energy)
    shield = np.clip((energy - 5.0) / 14.0, 0.0, 1.0)
    return np.maximum(shield, cv2.GaussianBlur(shield, (3, 3), 0.6))


def _despeckle_preserving_edges(rgb: np.ndarray, strength: int) -> np.ndarray:
    """Yalnizca 8 komsusunun disina tasan izole 1-piksellik kirik/nokta hatalarini onarir; cizgi silmez."""
    if strength <= 0:
        return rgb
    s = max(0.0, min(100.0, float(strength)))
    ring_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb_min = cv2.erode(rgb, ring_kernel, iterations=1).astype(np.int16)
    nb_max = cv2.dilate(rgb, ring_kernel, iterations=1).astype(np.int16)

    rgb_i = rgb.astype(np.int16)
    tol = max(3, int(round(18.0 - (s / 100.0) * 12.0)))

    # Bir piksel tum 8 komsusundan da kopuksa (izole hot/dead piksel veya tekil JPEG sicramasi)
    isolated_low = rgb_i < (nb_min - tol)
    isolated_high = rgb_i > (nb_max + tol)
    isolated_mask = np.any(isolated_low | isolated_high, axis=2)

    if not isolated_mask.any():
        return rgb

    med3 = cv2.medianBlur(rgb, 3)
    out = rgb.copy()
    out[isolated_mask] = med3[isolated_mask]
    return out


def _denoise_and_clean_flat_surfaces(
    rgb: np.ndarray,
    denoise_strength: int,
    vector_clean: int,
    deblock: bool,
) -> np.ndarray:
    """Detay kalkanini kullanarak ince tuy/cizgileri bozmadan duz alanlardaki gurultu ve JPEG bloklarini temizler."""
    if denoise_strength <= 0 and vector_clean <= 0 and not deblock:
        return rgb

    dn = max(0.0, min(100.0, float(denoise_strength)))
    vc = max(0.0, min(100.0, float(vector_clean)))
    shield = _compute_detail_shield(rgb)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_01 = lab[..., 0] / 255.0
    a_01 = lab[..., 1] / 255.0
    b_01 = lab[..., 2] / 255.0

    # 1) Chroma (a, b) JPEG renk dalgalanmalarini Luma rehberliginde temizle (kenar tasirmaz)
    if deblock or dn > 0 or vc > 0:
        chroma_eps = (0.018 + (max(dn, vc) / 100.0) * 0.045) ** 2
        a_clean = _guided_filter_mono(a_01, l_01, radius=4, eps=chroma_eps)
        b_clean = _guided_filter_mono(b_01, l_01, radius=4, eps=chroma_eps)
        chroma_w = np.clip(0.90 - shield * 0.60, 0.20, 0.90)
        a_01 = a_01 * (1.0 - chroma_w) + a_clean * chroma_w
        b_01 = b_01 * (1.0 - chroma_w) + b_clean * chroma_w

    # 2) Luma (L) mikro-gurultu temizligi: Detay kalkaninin 1.0 oldugu tuy ve cizgilerde %95+ orijinal korunur
    if dn > 0 or deblock:
        eps_fine = (0.014 + (dn / 100.0) * 0.055) ** 2
        r_den = 3 if dn >= 35 else 2
        l_guided = _guided_filter_mono(l_01, l_01, radius=r_den, eps=eps_fine)
        # Duz zeminde gurultuyu guclu temizle, detay/tuy bolgesinde orijinal pikseli koru
        preserve = np.clip(shield * 0.96, 0.0, 0.97)
        denoise_mix = min(0.92, 0.35 + (dn / 100.0) * 0.58) * (1.0 - preserve)
        l_01 = l_01 * (1.0 - denoise_mix) + l_guided * denoise_mix

    # 3) Duz Alan Puruzsuzlestirme (vector_clean): YALNIZCA gercek duz alanlarda (shield < 0.28) devreye girer
    if vc > 0:
        flat_zone = np.clip(1.0 - (shield / 0.28), 0.0, 1.0) ** 2
        eps_flat = (0.022 + (vc / 100.0) * 0.055) ** 2
        l_flat = _guided_filter_mono(l_01, l_01, radius=4, eps=eps_flat)
        a_flat = _guided_filter_mono(a_01, l_01, radius=4, eps=eps_flat)
        b_flat = _guided_filter_mono(b_01, l_01, radius=4, eps=eps_flat)

        flat_mix = flat_zone * (vc / 100.0) * 0.85
        l_01 = l_01 * (1.0 - flat_mix) + l_flat * flat_mix
        a_01 = a_01 * (1.0 - flat_mix) + a_flat * flat_mix
        b_01 = b_01 * (1.0 - flat_mix) + b_flat * flat_mix

    lab[..., 0] = np.clip(np.round(l_01 * 255.0), 0, 255)
    lab[..., 1] = np.clip(np.round(a_01 * 255.0), 0, 255)
    lab[..., 2] = np.clip(np.round(b_01 * 255.0), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _tangential_edge_flow(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Yapi Tensoru (Structure Tensor) ile kenara dik degil, YALNIZCA kenara paralel (teget) yonde
    merdiven kiriklarini (jaggies) yumusatir; cizgiyi bulandirmadan yeniden cizilmis gibi puruzsuzlestirir."""
    if strength <= 0.01:
        return rgb

    h, w = rgb.shape[:2]
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    gx = cv2.Scharr(luma, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(luma, cv2.CV_32F, 0, 1) / 16.0

    j_xx = cv2.GaussianBlur(gx * gx, (5, 5), 1.1)
    j_xy = cv2.GaussianBlur(gx * gy, (5, 5), 1.1)
    j_yy = cv2.GaussianBlur(gy * gy, (5, 5), 1.1)

    diff = j_xx - j_yy
    lambda_diff = np.sqrt(diff * diff + 4.0 * j_xy * j_xy)
    trace = j_xx + j_yy + 1e-5
    coherence = lambda_diff / trace
    grad_mag = np.sqrt(trace)

    # Dominant gradyan acisina dik olan birim teget vektoru (tx, ty)
    angle = 0.5 * np.arctan2(2.0 * j_xy, diff)
    tx = -np.sin(angle)
    ty = np.cos(angle)

    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    step = 1.15
    map_x_pos = grid_x + step * tx
    map_y_pos = grid_y + step * ty
    map_x_neg = grid_x - step * tx
    map_y_neg = grid_y - step * ty

    rgb_f = rgb.astype(np.float32)
    samp_pos = cv2.remap(rgb_f, map_x_pos, map_y_pos, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    samp_neg = cv2.remap(rgb_f, map_x_neg, map_y_neg, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    l_pos = cv2.remap(luma, map_x_pos, map_y_pos, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    l_neg = cv2.remap(luma, map_x_neg, map_y_neg, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    # Kose ve hat kesisimlerini korumak icin yalnizca luma farki kucuk olan teget ornekleri kabul et
    w_pos = np.clip(1.0 - np.abs(l_pos - luma) / 16.0, 0.0, 1.0)[..., None]
    w_neg = np.clip(1.0 - np.abs(l_neg - luma) / 16.0, 0.0, 1.0)[..., None]

    tangent_avg = (rgb_f + 0.45 * w_pos * samp_pos + 0.45 * w_neg * samp_neg) / (1.0 + 0.45 * w_pos + 0.45 * w_neg)

    # Yalnizca belirgin ve yonlu (coherent) konturlarda uygula; izotropik dokulara dokunma
    coh_mask = np.clip((coherence - 0.32) / 0.45, 0.0, 1.0)
    edge_mask = np.clip((grad_mag - 6.0) / 22.0, 0.0, 1.0)
    mix = (coh_mask * edge_mask * min(0.85, strength))[..., None]

    out = rgb_f * (1.0 - mix) + tangent_avg * mix
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _shock_tighten_contours(rgb: np.ndarray, edge_crisp: int, kernel_size: int = 3) -> np.ndarray:
    """Bulanik kenar gecis rampasini halesiz (min-max sinirli) morfolojik bukum noktasiyla sikilastirir."""
    if edge_crisp <= 0:
        return rgb

    ec = max(0.0, min(100.0, float(edge_crisp)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    l_min = cv2.erode(luma, kernel, iterations=1)
    l_max = cv2.dilate(luma, kernel, iterations=1)
    l_span = l_max - l_min
    l_mid = 0.5 * (l_min + l_max)

    c_min = cv2.erode(rgb, kernel, iterations=1).astype(np.float32)
    c_max = cv2.dilate(rgb, kernel, iterations=1).astype(np.float32)
    c_mid = 0.5 * (c_min + c_max)
    c_span = c_max - c_min

    pos = np.clip((luma - l_mid) / np.maximum(0.5 * l_span, 1.0), -1.0, 1.0)
    slope = 1.8 + (ec / 100.0) * 1.1
    shock = np.tanh(pos * slope)[..., None]
    target = c_mid + 0.5 * shock * c_span

    # Yalnizca belirgin kontur gecislerinde devreye girer; yumusak suluboya/golge gecislerini ve duz gurultuyu kesmez
    edge_mask = np.clip((l_span - 24.0) / 42.0, 0.0, 1.0)[..., None]
    alpha_step = (ec / 100.0) * 0.52

    rgb_f = rgb.astype(np.float32)
    tight = rgb_f + (target - rgb_f) * (alpha_step * edge_mask)
    return np.clip(np.round(tight), c_min, c_max).astype(np.uint8)


def _progressive_redraw_upscale(
    rgb: np.ndarray,
    target_w: int,
    target_h: int,
    edge_crisp: int,
    cancel_event: Optional[threading.Event] = None,
) -> np.ndarray:
    """Kademeli (en fazla 2x adimlarla) Iterative Back-Projection (IBP) + Tegetsel Kenar Akisi + Kontur Sikilastirma."""
    curr = rgb
    ch, cw = curr.shape[:2]
    if target_w == cw and target_h == ch:
        return curr

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flow_strength = 0.35 + (max(0.0, min(100.0, float(edge_crisp))) / 100.0) * 0.50

    while cw != target_w or ch != target_h:
        _check_cancel(cancel_event)
        ratio_w = target_w / float(cw)
        ratio_h = target_h / float(ch)
        if max(ratio_w, ratio_h) > 2.05:
            nw = min(target_w, max(1, int(round(cw * 2.0))))
            nh = min(target_h, max(1, int(round(ch * 2.0))))
        else:
            nw, nh = target_w, target_h

        # 1) Lanczos4 baslangic buyutmesi + yerel halesiz min/max zarfi
        up = cv2.resize(curr, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        src_min = cv2.erode(curr, k3, iterations=1)
        src_max = cv2.dilate(curr, k3, iterations=1)
        up_min = cv2.resize(src_min, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        up_max = cv2.resize(src_max, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        up = np.clip(up, up_min, up_max)

        # 2) Iterative Back-Projection (IBP): Orijinal ince tuy ve cizgi enerjisini birebir geri kazandirir
        curr_f = curr.astype(np.float32)
        ibp_iters = 3 if max(nw, nh) <= 2200 else 1
        for _ in range(ibp_iters):
            down = cv2.resize(up, (cw, ch), interpolation=cv2.INTER_AREA)
            err = curr_f - down
            err_up = cv2.resize(err, (nw, nh), interpolation=cv2.INTER_CUBIC)
            up = np.clip(up + 0.60 * err_up, up_min, up_max)

        stage_rgb = np.clip(np.round(up), 0, 255).astype(np.uint8)

        # 3) Yapi Tensoru Tegetsel Akis: Egik konturlardaki piksel merdivenlerini cizgi yonunde puruzsuzlestir
        if max(nw, nh) <= 2600:
            stage_rgb = _tangential_edge_flow(stage_rgb, strength=flow_strength)

        # 4) Olcek-uyumlu kontur sikilastirma: Buyutme gecis rampasini daralt
        if edge_crisp > 0:
            stage_rgb = _shock_tighten_contours(stage_rgb, edge_crisp=edge_crisp, kernel_size=3)

        curr = stage_rgb
        ch, cw = curr.shape[:2]

    return curr


def _reconstruct_fine_details_and_ink(
    rgb: np.ndarray,
    edge_crisp: int,
    sharpen_strength: int,
) -> np.ndarray:
    """Cok olcekli Guided Detail Boost ve Murekkep Kontur Rekonstruksiyonu (Sifir Hale / Halo-Free)."""
    if edge_crisp <= 0 and sharpen_strength <= 0:
        return rgb

    out = rgb
    if edge_crisp > 0:
        out = _shock_tighten_contours(out, edge_crisp=edge_crisp, kernel_size=3)

    if sharpen_strength <= 0:
        return out

    sh = max(0.0, min(100.0, float(sharpen_strength))) / 100.0
    ec = max(0.0, min(100.0, float(edge_crisp))) / 100.0

    # Yalnizca gercek detay/kontur bolgelerini netlestir; duz zemindeki artik gurultuyu buyutme
    detail_gate = np.clip(_compute_detail_shield(out) * 1.25, 0.0, 1.0)

    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_01 = lab[..., 0] / 255.0

    # Guided Filter tabanli 2 olcekli detay ayristirma (kenar asimi yapmaz, hale uretmez)
    base_fine = _guided_filter_mono(l_01, l_01, radius=2, eps=0.024 ** 2)
    base_mid = _guided_filter_mono(base_fine, base_fine, radius=5, eps=0.036 ** 2)

    detail_fine = (l_01 - base_fine) * detail_gate
    detail_mid = (base_fine - base_mid) * detail_gate

    # Koyu murekkep/tuy cizgilerini (detail < 0) daha net vurgula, parlak tarafta beyaz hale olusmasini onle
    gain_fine = sh * 1.15
    gain_mid = (sh * 0.55) + (ec * 0.35)

    boost_fine = np.where(detail_fine < 0.0, detail_fine * (gain_fine * 1.22), detail_fine * (gain_fine * 0.85))
    boost_mid = np.where(detail_mid < 0.0, detail_mid * (gain_mid * 1.25), detail_mid * (gain_mid * 0.75))

    # Yerel 5x5 min-max sinirlari icinde tutarak siyah/beyaz ringing olusumunu matematiksel olarak engelle
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    l_min = cv2.erode(l_01, k5, iterations=1)
    l_max = cv2.dilate(l_01, k5, iterations=1)

    l_new = np.clip(l_01 + boost_fine + boost_mid, l_min, l_max)
    lab[..., 0] = np.clip(np.round(l_new * 255.0), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _sharpen_lab_l(rgb: np.ndarray, sharpen_strength: int) -> np.ndarray:
    return _reconstruct_fine_details_and_ink(rgb, edge_crisp=0, sharpen_strength=sharpen_strength)


def enhance_image_array(
    img_arr: np.ndarray,
    denoise_strength: int = 22,
    sharpen_strength: int = 72,
    deblock: bool = True,
    upscale_mode: str = "none",
    vector_clean: int = 35,
    edge_crisp: int = 78,
    despeckle: bool = True,
    cancel_event: Optional[threading.Event] = None,
) -> np.ndarray:
    """RGB veya RGBA numpy dizisine detay korumali, pastellesmesiz 4K yeniden-cizim netlestirme uygular."""
    _check_cancel(cancel_event)
    has_alpha = img_arr.ndim == 3 and img_arr.shape[2] == 4
    if has_alpha:
        rgb = img_arr[..., :3].copy()
        alpha = img_arr[..., 3].copy()
        if not (alpha > 0).any():
            return img_arr.copy()
        rgb = _bleed_rgb_into_transparent(rgb, alpha, radius=6)
    else:
        rgb = img_arr[..., :3].copy() if img_arr.ndim == 3 else np.stack([img_arr] * 3, axis=-1)
        alpha = None

    _check_cancel(cancel_event)
    # 1) Izole 1-piksellik leke onarimi (ince cizgi ve tuylere dokunmaz)
    if despeckle:
        rgb = _despeckle_preserving_edges(rgb, strength=max(denoise_strength, vector_clean, 25))

    _check_cancel(cancel_event)
    # 2) Detay kalkanli Self-Guided gurultu ve duz alan temizligi (pastellesme yapmaz)
    rgb = _denoise_and_clean_flat_surfaces(
        rgb,
        denoise_strength=denoise_strength,
        vector_clean=vector_clean,
        deblock=deblock,
    )

    _check_cancel(cancel_event)
    # 3) Kademeli (2x -> 4x) IBP + Yapi Tensoru Tegetsel Kenar Akisi ile buyutme
    h, w = rgb.shape[:2]
    scale = 1.0
    if upscale_mode == "2x":
        scale = 2.0
    elif upscale_mode == "4k":
        if max(h, w) < 3840:
            scale = 3840.0 / float(max(h, w))
    elif upscale_mode == "4k_ultra":
        if max(h, w) < 4096:
            scale = 4096.0 / float(max(h, w))

    if scale > 1.01:
        max_dim = 6000.0
        if max(h, w) * scale > max_dim:
            scale = max_dim / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        rgb = _progressive_redraw_upscale(
            rgb,
            target_w=nw,
            target_h=nh,
            edge_crisp=edge_crisp,
            cancel_event=cancel_event,
        )
        if alpha is not None:
            alpha = np.clip(cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4), 0, 255).astype(np.uint8)
            if edge_crisp > 0:
                a_f = (alpha.astype(np.float32) / 255.0) - 0.5
                gain = 1.0 + (min(100.0, float(edge_crisp)) / 100.0) * 1.35
                alpha = np.clip(np.round((0.5 + a_f * gain) * 255.0), 0, 255).astype(np.uint8)
    elif edge_crisp > 20:
        # 1:1 olcekte bile kontur merdivenlerini tegetsel akisla duzelt
        rgb = _tangential_edge_flow(rgb, strength=(edge_crisp / 100.0) * 0.45)

    _check_cancel(cancel_event)
    # 4) Cok olcekli Guided Detail ve Murekkep Cizgisi Rekonstruksiyonu
    rgb = _reconstruct_fine_details_and_ink(rgb, edge_crisp=edge_crisp, sharpen_strength=sharpen_strength)

    if alpha is not None:
        return np.dstack([rgb, alpha])
    return rgb


def enhance_single_file(
    input_path: str,
    out_path: str,
    options: dict,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **_kwargs,
) -> dict:
    """Tek bir JPEG/PNG/TIF gorselini orijinaline dokunmadan detay korumali 4K netlestirip kaydeder."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    if progress_cb:
        progress_cb(15, f"Görsel okunuyor: {os.path.basename(input_path)}")

    with Image.open(input_path) as im:
        dpi = im.info.get("dpi")
        has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in im.info)
        converted = im.convert("RGBA" if has_alpha else "RGB")
        arr = np.array(converted)

    if progress_cb:
        progress_cb(40, "Detay kalkanı, kademeli IBP büyütme ve mürekkep netleştirme uygulanıyor...")

    out_arr = enhance_image_array(
        arr,
        denoise_strength=int(options.get("denoise", 22)),
        sharpen_strength=int(options.get("sharpen", 72)),
        deblock=bool(options.get("deblock", True)),
        upscale_mode=str(options.get("upscale", "none")),
        vector_clean=int(options.get("vector_clean", 35)),
        edge_crisp=int(options.get("edge_crisp", 78)),
        despeckle=bool(options.get("despeckle", True)),
        cancel_event=cancel_event,
    )

    if progress_cb:
        progress_cb(85, f"Kaydediliyor: {os.path.basename(out_path)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    mode = "RGBA" if out_arr.shape[2] == 4 else "RGB"
    out_im = Image.fromarray(out_arr, mode=mode)

    if ext in (".jpg", ".jpeg") and mode == "RGBA":
        bg = Image.new("RGB", out_im.size, (255, 255, 255))
        bg.paste(out_im, mask=out_im.split()[3])
        out_im = bg

    target_dpi = int(options.get("target_dpi", 254))
    if target_dpi > 0:
        out_dpi = target_dpi
        save_dpi = (float(target_dpi), float(target_dpi))
    elif dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2 and float(dpi[0]) > 0:
        out_dpi = int(round(float(dpi[0])))
        save_dpi = (float(dpi[0]), float(dpi[1]))
    else:
        out_dpi = 254
        save_dpi = (254.0, 254.0)

    save_kwargs: dict = {"dpi": save_dpi}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 96
        save_kwargs["subsampling"] = 0
    elif ext in (".tif", ".tiff"):
        save_kwargs["compression"] = "tiff_lzw"

    out_im.save(out_path, **save_kwargs)

    if progress_cb:
        progress_cb(100, "Netleştirme tamamlandı.")

    return {
        "mode": "single_file",
        "input_path": input_path,
        "output_path": out_path,
        "width": out_im.width,
        "height": out_im.height,
        "dpi": out_dpi,
        "width_cm": round(out_im.width / float(out_dpi) * 2.54, 1),
        "height_cm": round(out_im.height / float(out_dpi) * 2.54, 1),
    }


def enhance_psd_via_photoshop(
    ps_bridge,
    src_psd_or_none: Optional[str],
    out_psd_path: str,
    options: dict,
    expected_doc_name: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **_kwargs,
) -> dict:
    """PS COM hibrit yoluyla cok katmanli PSD'nin her piksel katmanina ayri netlestirme uygular."""
    import shutil
    import tempfile

    work_dir = tempfile.mkdtemp(prefix="mechi_psd_net_")
    try:
        if progress_cb:
            progress_cb(10, "Photoshop'tan piksel katmanları dışa aktarılıyor...")
        ok, info = ps_bridge.export_raster_layers_for_enhance(
            src_psd_or_none,
            work_dir,
            expected_doc_name=expected_doc_name,
        )
        if not ok:
            raise RuntimeError(str(info))

        layers = info.get("layers") or []
        skipped = int(info.get("skipped_vector_count", 0))
        if not layers:
            raise RuntimeError("Belgede netleştirilecek piksel katmanı bulunamadı.")

        total = len(layers)
        upscale_mode = "none"

        for idx, lyr in enumerate(layers, start=1):
            _check_cancel(cancel_event)
            pct = 15 + int(round((idx / float(total)) * 65))
            if progress_cb:
                progress_cb(pct, f"Katman netleştiriliyor ({idx}/{total}): {lyr.get('name', 'Katman')}")
            png_path = lyr["png_path"]
            with Image.open(png_path) as im:
                arr = np.array(im.convert("RGBA"))
            out_arr = enhance_image_array(
                arr,
                denoise_strength=int(options.get("denoise", 22)),
                sharpen_strength=int(options.get("sharpen", 72)),
                deblock=bool(options.get("deblock", True)),
                upscale_mode=upscale_mode,
                vector_clean=int(options.get("vector_clean", 35)),
                edge_crisp=int(options.get("edge_crisp", 78)),
                despeckle=bool(options.get("despeckle", True)),
                cancel_event=cancel_event,
            )
            Image.fromarray(out_arr, mode="RGBA").save(png_path)

        if progress_cb:
            progress_cb(85, "Temizlenen katmanlar Photoshop'a geri yerleştiriliyor...")

        ok_imp, res_msg = ps_bridge.reimport_enhanced_layers(work_dir, out_psd_path, keep_open=True)
        if not ok_imp:
            raise RuntimeError(res_msg)

        if progress_cb:
            progress_cb(100, f"PSD netleştirme tamamlandı ({total} katman).")

        return {
            "mode": "psd_layers",
            "output_path": res_msg,
            "enhanced_layers": total,
            "skipped_vector_layers": skipped,
        }
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
