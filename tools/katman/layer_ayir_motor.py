#!/usr/bin/env python3
"""MeChi Studio — 3 Modlu Profesyonel Katman (Layer) Ayirma Motoru.

Modlar:
  1) renk   : Maks. 10 katman, renk ailesi ton birlestirme + topolojik ic-detay kapsama.
  2) obje   : Kesiksiz butun nesne ayirma (Gemini Vision semantik kesif + ozel yazi maskesi + GrabCut/kontur).
  3) hibrit : Her obje ayri Photoshop Klasoru (LayerSet) + icinde renk ailesi katmanlari.

Cikti:
  - katmanlar_manifest.json (0.0 px kaymasiz kirpilmis katman koordinatlari ve grup yapisi)
  - layers/*.png (sikistirilmis RGBA PNG)
  - <ad>_katmanli.psd (psd_tools RGBA + TRANSPARENCY_MASK bagimsiz PSD)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Compression

# Renk ailesi isimleri (ASCII) ve algisal sabitler
FAMILY_ORDER = (
    "Siyah_Koyu",
    "Beyaz_Krem",
    "Gri_Gumus",
    "Kirmizi",
    "Turuncu",
    "Kahverengi",
    "Sari",
    "Yesil",
    "Turkuaz",
    "Mavi",
    "Mor",
    "Pembe_Magenta",
)


def tr_ascii(text: str, fallback: str = "Katman") -> str:
    """Photoshop katman/grup adlari icin temiz ASCII ad uretir."""
    clean = (text or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean).strip("_.")
    return clean or fallback


def _unique_name(base: str, used: set[str]) -> str:
    name = tr_ascii(base)
    if name not in used:
        used.add(name)
        return name
    idx = 2
    while f"{name}_{idx}" in used:
        idx += 1
    out = f"{name}_{idx}"
    used.add(out)
    return out


def _rgb_to_lab_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _rgb_to_hsv_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _extract_dpi(im: Image.Image) -> int:
    dpi = im.info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 1 and dpi[0]:
        val = int(round(float(dpi[0])))
        if 36 <= val <= 2400:
            return val
    return 300


def _soften_alpha(mask_bool: np.ndarray, sigma: float = 0.9) -> np.ndarray:
    """1-1.5 px alt-piksel yumusak kenar (SOP kurali: sert 1-bit jilet kenar yasak)."""
    m = mask_bool.astype(np.float32)
    if sigma <= 0.05:
        return m
    blurred = cv2.GaussianBlur(m, (0, 0), sigma)
    # Ic cekirdek tam opak kalsin, yalniz kenar gecisleri yumusasin
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    alpha = np.where(eroded, 1.0, blurred)
    alpha = np.where(alpha < 0.03, 0.0, alpha)
    return np.clip(alpha, 0.0, 1.0)


def _decontaminate_edges(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_rgb: Optional[np.ndarray],
) -> np.ndarray:
    """Yari saydam dis kenarlarda zemin rengi sizintisini azaltir."""
    if bg_rgb is None:
        return rgb
    semi = (alpha > 0.04) & (alpha < 0.92)
    if not np.any(semi):
        return rgb
    out = rgb.astype(np.float32).copy()
    bg = bg_rgb.astype(np.float32)
    a = np.maximum(alpha[..., None], 0.18)
    recovered = np.clip((out - bg[None, None, :] * (1.0 - a)) / a, 0.0, 255.0)
    out[semi] = recovered[semi]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


@dataclass
class LayerOutput:
    name: str
    rgba: np.ndarray  # H x W x 4 uint8
    group: str = ""
    kind: str = "color"
    visible: bool = True


# ---------------------------------------------------------------------------
# 1. Zemin Algilama ve Ayirma
# ---------------------------------------------------------------------------

def detect_background_mask(
    im_rgba: np.ndarray,
    zemin_mod: str = "otomatik",
    bg_tol: float = 18.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Dondurur: (fg_mask_bool, bg_rgb_or_None, bg_alpha_float).

    Dis kenarlara bagli olmayan ic beyaz parlamalar ASLA zemin diye oyulmaz.
    """
    H, W = im_rgba.shape[:2]
    rgb = im_rgba[..., :3]
    in_alpha = im_rgba[..., 3].astype(np.float32) / 255.0

    if zemin_mod == "ayirma":
        fg = np.ones((H, W), dtype=bool)
        return fg, None, np.zeros((H, W), dtype=np.float32)

    # Mevcut anlamli seffaflik varsa koru (gercek seffaf background <0.15 alfa icerir;
    # 84..255 arasi kismi kenar yumusatmalari opak zemin algisina devam eder)
    has_meaningful_alpha = bool((in_alpha < 0.15).mean() > 0.01 and (in_alpha > 0.05).any())
    if zemin_mod == "seffaf_koru" or (zemin_mod == "otomatik" and has_meaningful_alpha):
        fg = in_alpha > 0.04
        return fg, None, 1.0 - in_alpha

    # Opak gorselde dis cerceve piksellerinden zemin rengini tahmin et
    border = max(3, int(round(min(H, W) * 0.02)))
    strips = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[H - border :, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, W - border :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_rgb = np.median(strips, axis=0).astype(np.float32)

    lab = _rgb_to_lab_u8(rgb).astype(np.float32)
    bg_lab = _rgb_to_lab_u8(bg_rgb.astype(np.uint8).reshape(1, 1, 3)).astype(np.float32)[0, 0]
    de = np.sqrt(((lab - bg_lab[None, None, :]) ** 2).sum(axis=2))
    # L kanali olcegini Lab DeltaE'ye yaklastir
    cand = (de <= float(bg_tol)).astype(np.uint8)

    # Yalnizca dis kenarlara temas eden bagli bilesenleri zemin say
    n_cc, cc_labels = cv2.connectedComponents(cand, connectivity=4)
    edge_labels = set()
    edge_labels.update(np.unique(cc_labels[0, :]).tolist())
    edge_labels.update(np.unique(cc_labels[H - 1, :]).tolist())
    edge_labels.update(np.unique(cc_labels[:, 0]).tolist())
    edge_labels.update(np.unique(cc_labels[:, W - 1]).tolist())
    edge_labels.discard(0)

    if not edge_labels:
        return np.ones((H, W), dtype=bool), None, np.zeros((H, W), dtype=np.float32)

    bg_mask = np.isin(cc_labels, list(edge_labels))
    # Zemin cok kucukse (<%1.5) zemin sokme, tam tuval calis
    if float(bg_mask.mean()) < 0.015:
        return np.ones((H, W), dtype=bool), None, np.zeros((H, W), dtype=np.float32)

    fg_mask = ~bg_mask
    # 1-2 px dis kenarda anti-alias alfa tahmini
    bg_alpha = bg_mask.astype(np.float32)
    return fg_mask, bg_rgb.astype(np.uint8), bg_alpha


# ---------------------------------------------------------------------------
# 2. Renk Ailesi Siniflandirma & Topolojik Kapsama (Mod 1: Renk Ayir)
# ---------------------------------------------------------------------------

def _classify_color_families(rgb: np.ndarray, fg_mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Her piksele bir algisal renk ailesi indeksi atar.

    Ayni rengin acik/orta/koyu tonlari (ornegin acik kirmizi + kirmizi + bordo)
    ayni ailede birlesir; siniflandirma oncesi bilateral yumusatma ile JPEG/golge
    konfetisi onlenirken orijinal RGB piksel verisi korunur.
    """
    H, W = rgb.shape[:2]
    rgb_cls = cv2.bilateralFilter(rgb, 7, 35, 35)
    hsv = _rgb_to_hsv_u8(rgb_cls)
    lab = _rgb_to_lab_u8(rgb_cls).astype(np.float32)

    L_lab = lab[..., 0] * (100.0 / 255.0)
    hue_deg = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)

    # Siyah/koyu tespiti keskin orijinal RGB uzerinden yapilir (ince siyah yazilarin bilateral ile solmasini onler);
    # koyu kirmizi gibi yuksek kromali koyu tonlar L_orig>=19 iken kendi renk ailesinde kalir.
    lab_orig = _rgb_to_lab_u8(rgb).astype(np.float32)
    L_orig = lab_orig[..., 0] * (100.0 / 255.0)
    C_orig = np.sqrt((lab_orig[..., 1] - 128.0) ** 2 + (lab_orig[..., 2] - 128.0) ** 2)

    fam_idx = {name: i + 1 for i, name in enumerate(FAMILY_ORDER)}
    out = np.zeros((H, W), dtype=np.int32)

    is_dark = (L_orig < 19.0) | ((L_orig < 40.0) & (C_orig < 12.0))
    is_achromatic = (chroma <= 10.5) & (~is_dark)
    is_white = is_achromatic & (L_lab >= 86.0)
    is_gray = is_achromatic & (~is_white)

    out[fg_mask & is_dark] = fam_idx["Siyah_Koyu"]
    out[fg_mask & is_white] = fam_idx["Beyaz_Krem"]
    out[fg_mask & is_gray] = fam_idx["Gri_Gumus"]

    chrom_mask = fg_mask & (out == 0)
    if np.any(chrom_mask):
        is_red = (hue_deg >= 342.0) | (hue_deg < 16.0)
        is_brown = (hue_deg >= 16.0) & (hue_deg < 46.0) & (val < 135.0)
        is_orange = (hue_deg >= 16.0) & (hue_deg < 38.0) & (~is_brown)
        is_yellow = (hue_deg >= 38.0) & (hue_deg < 72.0) & (~is_brown)
        is_green = (hue_deg >= 72.0) & (hue_deg < 162.0)
        is_cyan = (hue_deg >= 162.0) & (hue_deg < 194.0)
        is_blue = (hue_deg >= 194.0) & (hue_deg < 258.0)
        is_purple = (hue_deg >= 258.0) & (hue_deg < 296.0)
        is_pink = (hue_deg >= 296.0) & (hue_deg < 342.0)

        pairs = (
            ("Kirmizi", is_red),
            ("Kahverengi", is_brown),
            ("Turuncu", is_orange),
            ("Sari", is_yellow),
            ("Yesil", is_green),
            ("Turkuaz", is_cyan),
            ("Mavi", is_blue),
            ("Mor", is_purple),
            ("Pembe_Magenta", is_pink),
        )
        for name, cond in pairs:
            out[chrom_mask & cond] = fam_idx[name]

    # 3x3 medyan ile tekil anti-alias kenar gurultusunu temizle
    if np.any(fg_mask):
        med = cv2.medianBlur(out.astype(np.uint8), 3).astype(np.int32)
        swap = fg_mask & (med > 0) & (out != med)
        out = np.where(swap, med, out)

        # Mekansal konfeti temizligi: cok kucuk daginik adaciklari temas eden baskin komsuya aktar
        fg_total = max(1, int(fg_mask.sum()))
        min_island = max(18, int(round(fg_total * 0.0004)))
        for fid in range(1, len(FAMILY_ORDER) + 1):
            fm = fg_mask & (out == fid)
            if not np.any(fm):
                continue
            n_cc, cc_lab, cc_st, _ = cv2.connectedComponentsWithStats(fm.astype(np.uint8), connectivity=8)
            for cid in range(1, n_cc):
                if int(cc_st[cid, cv2.CC_STAT_AREA]) >= min_island:
                    continue
                bx, by, bw, bh = [int(v) for v in cc_st[cid, :4]]
                y0, y1 = max(0, by - 2), min(H, by + bh + 2)
                x0, x1 = max(0, bx - 2), min(W, bx + bw + 2)
                sub_m = (cc_lab[y0:y1, x0:x1] == cid)
                dil = (cv2.dilate(sub_m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0) & (~sub_m)
                neigh = out[y0:y1, x0:x1][dil & fg_mask[y0:y1, x0:x1]]
                neigh = neigh[(neigh > 0) & (neigh != fid)]
                if neigh.size > 0:
                    uniq_n, cnt_n = np.unique(neigh, return_counts=True)
                    best_n = int(uniq_n[int(np.argmax(cnt_n))])
                    sub_out = out[y0:y1, x0:x1]
                    sub_out[sub_m] = best_n

    return out, list(FAMILY_ORDER)


def _merge_small_families_to_budget(
    fam_map: np.ndarray,
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    family_names: List[str],
    max_color_layers: int,
    min_area_ratio: float = 0.008,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Gurultu ailelerini, golgeden dogan komsu parcalanmalari ve fazla aileleri birlestirir."""
    fg_total = max(1, int(fg_mask.sum()))
    lab = _rgb_to_lab_u8(rgb).astype(np.float32)

    active: Dict[int, str] = {}
    counts: Dict[int, int] = {}
    centers: Dict[int, np.ndarray] = {}

    for idx, name in enumerate(family_names, start=1):
        m = fg_mask & (fam_map == idx)
        cnt = int(m.sum())
        if cnt > 0:
            active[idx] = name
            counts[idx] = cnt
            centers[idx] = lab[m].mean(axis=0)

    if len(active) <= 1:
        return fam_map, active

    work = fam_map.copy()

    def _absorb(src_id: int, dst_id: int) -> None:
        work[work == src_id] = dst_id
        m = fg_mask & (work == dst_id)
        counts[dst_id] = int(m.sum())
        if counts[dst_id] > 0:
            centers[dst_id] = lab[m].mean(axis=0)
        active.pop(src_id, None)
        counts.pop(src_id, None)
        centers.pop(src_id, None)

    def _find_best_target(src_id: int, prefer_chromatic: bool = False) -> int:
        # Komsu sinirda temas eden en baskin & Lab'de en yakin aileyi sec
        src_mask = (work == src_id).astype(np.uint8)
        dil = cv2.dilate(src_mask, np.ones((5, 5), np.uint8), iterations=1) > 0
        border_vals = work[dil & fg_mask & (work != src_id) & (work > 0)]
        cand_ids = [cid for cid in active if cid != src_id]
        if prefer_chromatic:
            chrom_cands = [cid for cid in cand_ids if active[cid] not in ("Siyah_Koyu", "Beyaz_Krem", "Gri_Gumus")]
            if chrom_cands:
                cand_ids = chrom_cands
        if border_vals.size > 0:
            uniq, u_cnt = np.unique(border_vals, return_counts=True)
            touch_map = {int(u): int(c) for u, c in zip(uniq, u_cnt) if int(u) in cand_ids}
            if touch_map:
                ranked = sorted(touch_map.keys(), key=lambda c: (-touch_map[c], np.linalg.norm(centers[c] - centers[src_id])))
                return ranked[0]
        return min(cand_ids, key=lambda c: float(np.linalg.norm(centers[c] - centers[src_id])))

    # 1) Cok kucuk (< %0.8) gecis/anti-alias/golge tozu ailelerini temizle
    min_px = max(12, int(round(fg_total * min_area_ratio)))
    while len(active) > 1:
        smallest = min(active.keys(), key=lambda k: counts[k])
        if counts[smallest] >= min_px:
            break
        dst = _find_best_target(smallest)
        _absorb(smallest, dst)

    # 2) Golge gecisinden dogan yakin aile parcalanmasini birlestir (orn. kucuk Kahverengi/Kirmizi/Sari < %4 ve Turuncu'ya bitisikse)
    shading_pairs = (("Kahverengi", "Turuncu"), ("Turuncu", "Sari"), ("Kirmizi", "Kahverengi"), ("Turkuaz", "Mavi"))
    name_to_id = {nm: fid for fid, nm in active.items()}
    for nm_a, nm_b in shading_pairs:
        id_a = name_to_id.get(nm_a)
        id_b = name_to_id.get(nm_b)
        if id_a in active and id_b in active and len(active) > 2:
            small_id, big_id = (id_a, id_b) if counts[id_a] < counts[id_b] else (id_b, id_a)
            if counts[small_id] < int(fg_total * 0.045):
                sm_m = (work == small_id).astype(np.uint8)
                dil = cv2.dilate(sm_m, np.ones((5, 5), np.uint8), iterations=1) > 0
                bvals = work[dil & fg_mask & (work != small_id) & (work > 0)]
                if bvals.size > 0 and float((bvals == big_id).mean()) >= 0.35:
                    _absorb(small_id, big_id)
                    name_to_id = {nm: fid for fid, nm in active.items()}

    # 3) Maksimum renk katmani butcesine indir (Orn. Zemin + maks 9 renk = 10)
    target_budget = max(1, int(max_color_layers))
    while len(active) > target_budget:
        smallest = min(active.keys(), key=lambda k: counts[k])
        dst = _find_best_target(smallest)
        _absorb(smallest, dst)

    return work, active


def apply_containment_rule(
    fam_map: np.ndarray,
    active_families: Dict[int, str],
    fg_mask: np.ndarray,
    kapsama_orani: float = 0.35,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Topolojik kapsama: Bir objenin/rengin kapali siniri icinde kalan farkli renkteki
    olusumlari (orn. mavi objenin icindeki sari yildiz/beyaz parlama) delik acmadan
    tasiyici ana renk katmanina kendi orijinal pikselleriyle dahil eder.
    Siyah_Koyu/Gri_Gumus ve ince dis konturlar asla kapsayici (host) olmaz."""
    if len(active_families) <= 1:
        return fam_map, active_families

    non_host_names = {"Siyah_Koyu", "Gri_Gumus"}
    work = fam_map.copy()
    ratio_limit = float(np.clip(kapsama_orani, 0.05, 0.95))
    fg_total = max(1, int(fg_mask.sum()))

    for _pass in range(2):
        # Buyuk alanlardan kucuklere dogru tara
        ordered = sorted(
            list(active_families.keys()),
            key=lambda fid: int((work == fid).sum()),
            reverse=True,
        )
        changed = False
        for fid in ordered:
            if active_families.get(fid) in non_host_names:
                continue
            f_mask = (work == fid) & fg_mask
            if int(f_mask.sum()) < 24:
                continue

            # Dis konturlari kapatarak her bagli tasiyici bilesenin icini doldur
            n_cc, cc_lab, cc_stats, _ = cv2.connectedComponentsWithStats(f_mask.astype(np.uint8), connectivity=8)
            for comp_id in range(1, n_cc):
                comp_area = int(cc_stats[comp_id, cv2.CC_STAT_AREA])
                if comp_area < 40:
                    continue
                comp_mask = (cc_lab == comp_id).astype(np.uint8)
                touch_edges = (
                    int(bool(np.any(comp_mask[0, :] > 0)))
                    + int(bool(np.any(comp_mask[-1, :] > 0)))
                    + int(bool(np.any(comp_mask[:, 0] > 0)))
                    + int(bool(np.any(comp_mask[:, -1] > 0)))
                )
                # Tuvalin >=3 dis kenarina degen bilesen dis fon/cercevedir; icteki ana objeleri yutmaz
                if touch_edges >= 3:
                    continue
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                filled = np.zeros_like(comp_mask)
                cv2.drawContours(filled, contours, -1, 1, thickness=-1)
                filled_area = max(1, int((filled > 0).sum()))
                # Ince cerceve / cizgi konturuysa (doluluk < %45) icerideki renkleri yutma
                if (comp_area / float(filled_area)) < 0.45:
                    continue

                holes = (filled > 0) & (comp_mask == 0) & fg_mask & (work != fid)
                if not np.any(holes):
                    continue

                n_h, h_lab, h_stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
                for hid in range(1, n_h):
                    h_area = int(h_stats[hid, cv2.CC_STAT_AREA])
                    if h_area > max(16, int(comp_area * ratio_limit)):
                        continue
                    # Zemin (~fg_mask) ile ayrilmis bagimsiz objeyi (orn. cerceve ici hayvan) yutma:
                    # deligin dis siniri dogrudan tasiyici bilesene temas etmeli
                    hm = (h_lab == hid).astype(np.uint8)
                    ring = (cv2.dilate(hm, np.ones((3, 3), np.uint8), iterations=1) > 0) & (hm == 0)
                    if not np.any(ring):
                        continue
                    touch_host = float((comp_mask[ring] > 0).mean())
                    touch_bg = float((~fg_mask[ring]).mean())
                    if touch_bg <= 0.08 and touch_host >= 0.72:
                        work[h_lab == hid] = fid
                        changed = True

        # Ek halka kontrolu: tamamen veya >=%78 tek bir aileyle sarili kucuk adaciklar
        for fid in list(active_families.keys()):
            f_mask = (work == fid) & fg_mask
            if not np.any(f_mask):
                continue
            n_c, c_lab, c_st, _ = cv2.connectedComponentsWithStats(f_mask.astype(np.uint8), connectivity=8)
            for cid in range(1, n_c):
                c_area = int(c_st[cid, cv2.CC_STAT_AREA])
                if c_area > int(fg_total * 0.025):
                    continue
                c_m = (c_lab == cid).astype(np.uint8)
                all_ring = (cv2.dilate(c_m, np.ones((5, 5), np.uint8), iterations=1) > 0) & (c_m == 0)
                if not np.any(all_ring) or float((~fg_mask[all_ring]).mean()) > 0.10:
                    continue
                ring_vals = work[all_ring & fg_mask]
                ring_vals = ring_vals[(ring_vals > 0) & (ring_vals != fid)]
                if ring_vals.size == 0:
                    continue
                uniq, u_cnt = np.unique(ring_vals, return_counts=True)
                best_idx = int(np.argmax(u_cnt))
                host_id = int(uniq[best_idx])
                if active_families.get(host_id) in non_host_names:
                    continue
                if float(u_cnt[best_idx]) / float(ring_vals.size) >= 0.78:
                    work[c_lab == cid] = host_id
                    changed = True

        if not changed:
            break

    # Bosalan aileleri cikar
    final_active = {fid: name for fid, name in active_families.items() if np.any((work == fid) & fg_mask)}
    return work, final_active


def separate_by_color_family(
    im_rgba: np.ndarray,
    fg_mask: np.ndarray,
    bg_rgb: Optional[np.ndarray],
    max_layers: int = 10,
    ic_detay_kapsa: bool = True,
    kapsama_orani: float = 0.35,
    include_background_layer: bool = True,
    group_name: str = "",
) -> Tuple[List[LayerOutput], float]:
    """Renk ailesi (ton birlestirmeli) + topolojik kapsama ile en fazla `max_layers` katman uretir."""
    H, W = im_rgba.shape[:2]
    rgb = im_rgba[..., :3]
    has_bg_layer = include_background_layer and bool((~fg_mask).any())
    color_budget = max(1, int(max_layers) - (1 if has_bg_layer else 0))

    fam_map, fam_names = _classify_color_families(rgb, fg_mask)
    fam_map, active = _merge_small_families_to_budget(fam_map, rgb, fg_mask, fam_names, color_budget)

    if ic_detay_kapsa and len(active) > 1:
        fam_map, active = apply_containment_rule(fam_map, active, fg_mask, kapsama_orani=kapsama_orani)

    outputs: List[LayerOutput] = []
    comp = np.zeros((H, W, 3), dtype=np.float32)

    if has_bg_layer:
        bg_mask = ~fg_mask
        bg_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        bg_rgba[..., :3] = rgb
        bg_rgba[..., 3] = np.where(bg_mask, 255, 0).astype(np.uint8)
        outputs.append(LayerOutput(name="Zemin", rgba=bg_rgba, group=group_name, kind="background", visible=True))
        comp[bg_mask] = rgb[bg_mask].astype(np.float32)
    elif bg_rgb is not None:
        comp = np.tile(bg_rgb.astype(np.float32), (H, W, 1))

    # Katmanlari alandan buyukten kucuge sira ile diz (baskın govde altta, ince detaylar ustte)
    ordered_fids = sorted(active.keys(), key=lambda fid: int((fam_map == fid).sum()), reverse=True)
    for fid in ordered_fids:
        m = (fam_map == fid) & fg_mask
        if not np.any(m):
            continue
        alpha = _soften_alpha(m, sigma=0.85)
        clean_rgb = _decontaminate_edges(rgb, alpha, bg_rgb if has_bg_layer is False else None)
        rgba = np.dstack([clean_rgb, np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)])
        outputs.append(LayerOutput(name=active[fid], rgba=rgba, group=group_name, kind="color", visible=True))
        a3 = alpha[..., None]
        comp = comp * (1.0 - a3) + clean_rgb.astype(np.float32) * a3

    if fg_mask.any():
        err = float(np.abs(comp[fg_mask] - rgb[fg_mask].astype(np.float32)).mean())
    else:
        err = 0.0
    return outputs, err


# ---------------------------------------------------------------------------
# 3. Akilli Obje Ayirma (Mod 2: Obje Ayir)
# ---------------------------------------------------------------------------

def _encode_jpeg_b64(rgb: np.ndarray, max_side: int = 1280) -> str:
    pil = Image.fromarray(rgb, mode="RGB")
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def discover_objects_vision(
    rgb: np.ndarray,
    api_key: str,
    max_objects: int = 9,
) -> List[dict]:
    """Gemini Vision (2.5-flash -> 2.0-flash) ile kesiksiz obje kutulari ve Z-sirasi alir."""
    b64 = _encode_jpeg_b64(rgb)
    prompt = (
        f"Analyze this artwork/graphic design and detect up to {max_objects} logical, COMPLETE Photoshop layers "
        "ordered from back to front (z_order 1 = furthest back foreground element).\n"
        "STRICT RULES:\n"
        "1) NEVER split a single physical object into sub-parts (e.g., keep a whole hat as ONE 'Sapka' object, a whole animal as ONE 'Hayvan' object, a whole decorative border as ONE 'Cerceve' object).\n"
        "2) Group letters belonging to the same word/banner/text block into ONE object with kind='text' (do NOT split letter by letter).\n"
        "3) Do NOT include the plain outer background.\n"
        "4) Return ONLY valid JSON with key 'objects': a list of items, each with:\n"
        '   - "name_ascii": short descriptive ASCII name without spaces (e.g. "Cerceve", "Hayvan", "Sapka", "Yazi", "Yildizlar")\n'
        '   - "kind": one of "text", "subject", "frame", "effect"\n'
        '   - "z_order": integer starting at 1 (back to front)\n'
        '   - "box_2d": [ymin, xmin, ymax, xmax] normalized 0..1000 enclosing the COMPLETE object tightly with slight margin.\n'
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }

    last_err = ""
    for mdl in ("gemini-2.5-flash", "gemini-2.0-flash"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key}"
        resp = requests.post(url, json=payload, timeout=45)
        if resp.status_code == 200:
            data = resp.json()
            cands = data.get("candidates") or []
            if not cands:
                continue
            parts = (cands[0].get("content") or {}).get("parts") or []
            txt = "\n".join(str(p.get("text", "")) for p in parts).strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            parsed = json.loads(txt)
            objs = parsed.get("objects") if isinstance(parsed, dict) else parsed
            if isinstance(objs, list) and objs:
                return objs[:max_objects]
        last_err = f"HTTP {resp.status_code}: {resp.text[:180]}"
    raise RuntimeError(f"Gemini Vision obje kesfi basarisiz: {last_err}")


def discover_objects_local(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    max_objects: int = 9,
) -> List[dict]:
    """API anahtari olmadiginda bagli bilesen ve yatay yazi-bandi kesfiyle tam obje kutulari cikarir."""
    H, W = rgb.shape[:2]
    if not np.any(fg_mask):
        return []

    fg_u8 = fg_mask.astype(np.uint8)
    n_cc, cc_lab, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    fg_area = max(1, int(fg_mask.sum()))

    # Kucuk/orta harf benzeri bilesenleri yatay olarak grupla
    small_mask = np.zeros((H, W), dtype=np.uint8)
    large_boxes: List[dict] = []

    for cid in range(1, n_cc):
        x, y, bw, bh, area = [int(v) for v in stats[cid]]
        if area < max(12, int(fg_area * 0.0005)):
            continue
        is_glyph_like = (area < fg_area * 0.04) and (bh < H * 0.20) and (bw < W * 0.25)
        if is_glyph_like:
            small_mask[cc_lab == cid] = 1
        else:
            large_boxes.append({"x": x, "y": y, "w": bw, "h": bh, "area": area, "kind": "subject"})

    objs: List[dict] = []
    # Buyuk nesneler altta
    large_boxes.sort(key=lambda b: -b["area"])
    for idx, b in enumerate(large_boxes[: max(1, max_objects - 2)], start=1):
        objs.append(
            {
                "name_ascii": f"Obje_{idx}" if len(large_boxes) > 1 else "Ana_Obje",
                "kind": "subject",
                "z_order": idx,
                "box_2d": [
                    int(round(b["y"] / float(H) * 1000)),
                    int(round(b["x"] / float(W) * 1000)),
                    int(round((b["y"] + b["h"]) / float(H) * 1000)),
                    int(round((b["x"] + b["w"]) / float(W) * 1000)),
                ],
            }
        )

    # Yatay yakin harf bilesenlerini yazi bloklarinda topla
    if np.any(small_mask):
        kx = max(7, int(round(W * 0.035)))
        ky = max(3, int(round(H * 0.012)))
        merged = cv2.dilate(small_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        nt, _, tst, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        t_idx = 0
        for tid in range(1, nt):
            tx, ty, tbw, tbh, tarea = [int(v) for v in tst[tid]]
            if tarea < 40:
                continue
            t_idx += 1
            aspect = tbw / float(max(1, tbh))
            kind = "text" if aspect >= 1.5 and tbh < H * 0.25 else "subject"
            label = f"Yazi_{t_idx}" if kind == "text" else f"Detay_{t_idx}"
            objs.append(
                {
                    "name_ascii": label,
                    "kind": kind,
                    "z_order": len(objs) + 1,
                    "box_2d": [
                        int(round(ty / float(H) * 1000)),
                        int(round(tx / float(W) * 1000)),
                        int(round((ty + tbh) / float(H) * 1000)),
                        int(round((tx + tbw) / float(W) * 1000)),
                    ],
                }
            )

    return objs[:max_objects]


def _box_norm_to_px(box_2d: List[Any], W: int, H: int, pad_ratio: float = 0.03) -> Tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
    y0 = int(round(ymin / 1000.0 * H))
    x0 = int(round(xmin / 1000.0 * W))
    y1 = int(round(ymax / 1000.0 * H))
    x1 = int(round(xmax / 1000.0 * W))
    if y1 <= y0:
        y1 = min(H, y0 + 8)
    if x1 <= x0:
        x1 = min(W, x0 + 8)
    px_pad = max(4, int(round((x1 - x0) * pad_ratio)))
    py_pad = max(4, int(round((y1 - y0) * pad_ratio)))
    return max(0, x0 - px_pad), max(0, y0 - py_pad), min(W, x1 + px_pad), min(H, y1 + py_pad)


def extract_text_layer_mask(rgb: np.ndarray, fg_mask: np.ndarray, box_px: Tuple[int, int, int, int]) -> np.ndarray:
    """Kutu icindeki yaziyi harf ic bosluklarini (O, A, B, R) bozmadan cikarir."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box_px
    out = np.zeros((H, W), dtype=bool)
    roi_rgb = rgb[y0:y1, x0:x1]
    roi_fg = fg_mask[y0:y1, x0:x1]
    rh, rw = roi_rgb.shape[:2]
    if rh < 2 or rw < 2:
        return out

    # Eger zemin ayirma kapali veya arka/tasiyici obje icinde yazi varsa cevre kontrastindan cikart
    border = max(2, int(round(min(rh, rw) * 0.06)))
    ring = np.ones((rh, rw), dtype=bool)
    if rh > 2 * border and rw > 2 * border:
        ring[border:-border, border:-border] = False
    ring_pixels = roi_rgb[ring] if np.any(ring) else roi_rgb.reshape(-1, 3)
    env_rgb = np.median(ring_pixels, axis=0).astype(np.uint8).reshape(1, 1, 3)

    roi_lab = _rgb_to_lab_u8(roi_rgb).astype(np.float32)
    env_lab = _rgb_to_lab_u8(env_rgb).astype(np.float32)[0, 0]
    de = np.sqrt(((roi_lab - env_lab[None, None, :]) ** 2).sum(axis=2))

    # Harf ic bosluklarini (O, A, B, R) korumak icin her zaman cevre kontrastini uygula
    max_de = float(de.max())
    if max_de < 12.0:
        return out
    de_u8 = np.clip(de * (255.0 / max_de), 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(de_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(18.0, min(48.0, float(otsu_t) * (max_de / 255.0) * 0.72))
    mask_local = (de > thresh_val) & roi_fg

    out[y0:y1, x0:x1] = mask_local
    return out


def extract_whole_object_mask(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    box_px: Tuple[int, int, int, int],
    fill_holes: bool = True,
) -> np.ndarray:
    """Kutu icindeki fiziksel objeyi kesiksiz ve ic deliksiz cikarir (GrabCut + kontur)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box_px
    out = np.zeros((H, W), dtype=bool)

    roi_fg = fg_mask[y0:y1, x0:x1]
    fg_frac = float(roi_fg.mean()) if roi_fg.size else 0.0
    if fg_frac < 0.01:
        return out

    # Kutu ici dis zeminle cevrelenmis ayrik bir adaciksa dogrudan tam bileseni al (en hizli & kesiksiz)
    n_cc, cc_lab, stats, _ = cv2.connectedComponentsWithStats(roi_fg.astype(np.uint8), connectivity=8)
    edge_touched = set()
    rh, rw = roi_fg.shape
    if rh > 0 and rw > 0:
        edge_touched.update(np.unique(cc_lab[0, :]).tolist())
        edge_touched.update(np.unique(cc_lab[rh - 1, :]).tolist())
        edge_touched.update(np.unique(cc_lab[:, 0]).tolist())
        edge_touched.update(np.unique(cc_lab[:, rw - 1]).tolist())
    edge_touched.discard(0)

    if not edge_touched and n_cc > 1:
        local_mask = roi_fg.copy()
    else:
        # Bitisik objelerden ayirmak icin hizli GrabCut (max 640px calisma olceginde)
        roi_rgb = rgb[y0:y1, x0:x1]
        scale = min(1.0, 640.0 / float(max(rw, rh)))
        sw = max(16, int(round(rw * scale)))
        sh = max(16, int(round(rh * scale)))
        s_rgb = cv2.resize(roi_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        s_fg = cv2.resize(roi_fg.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

        gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
        gc_mask[s_fg] = cv2.GC_PR_FGD
        # Merkez %50 cekirdek guvenli on-plan
        mx0, mx1 = max(1, int(sw * 0.22)), min(sw - 1, int(sw * 0.78))
        my0, my1 = max(1, int(sh * 0.22)), min(sh - 1, int(sh * 0.78))
        if mx1 > mx0 and my1 > my0:
            core = np.zeros((sh, sw), dtype=bool)
            core[my0:my1, mx0:mx1] = True
            gc_mask[core & s_fg] = cv2.GC_FGD
        # En dis 2 px cerceve muhtemel arka plan (disariya tasan baska obje varsa ayirsin)
        gc_mask[0:2, :] = np.where(s_fg[0:2, :], cv2.GC_PR_BGD, cv2.GC_BGD)
        gc_mask[-2:, :] = np.where(s_fg[-2:, :], cv2.GC_PR_BGD, cv2.GC_BGD)
        gc_mask[:, 0:2] = np.where(s_fg[:, 0:2], cv2.GC_PR_BGD, cv2.GC_BGD)
        gc_mask[:, -2:] = np.where(s_fg[:, -2:], cv2.GC_PR_BGD, cv2.GC_BGD)

        if np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) and np.any(
            (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)
        ):
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(s_rgb, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
                s_out = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            except Exception:
                s_out = s_fg
        else:
            s_out = s_fg

        if float(s_out.mean()) < 0.02:
            s_out = s_fg
        local_mask = (cv2.resize(s_out.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_LINEAR) > 0) & roi_fg

    if fill_holes and np.any(local_mask):
        contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(local_mask, dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
        local_mask = (filled > 0) & roi_fg

    out[y0:y1, x0:x1] = local_mask
    return out


def separate_by_objects(
    im_rgba: np.ndarray,
    fg_mask: np.ndarray,
    bg_rgb: Optional[np.ndarray],
    max_layers: int = 10,
    vision_kesif: bool = True,
    api_key: str = "",
    yazi_ayir: bool = True,
    alt_doldur: bool = False,
    custom_objects: Optional[List[dict]] = None,
    yerel_ai: bool = True,
    ozel_etiketler: str = "",
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[LayerOutput], float, List[dict]]:
    """Kesiksiz butun nesne katmanlari uretir (Obje Ayir)."""
    H, W = im_rgba.shape[:2]
    rgb = im_rgba[..., :3]
    has_bg_layer = bool((~fg_mask).any())
    obj_budget = max(1, min(9, int(max_layers) - (1 if has_bg_layer else 0)))

    objs_sorted: List[dict] = []
    raw_entries: List[Tuple[str, str, np.ndarray]] = []
    used_onnx_ai = False

    if custom_objects is None and yerel_ai:
        katman_dir = os.path.dirname(os.path.abspath(__file__))
        if katman_dir not in sys.path:
            sys.path.insert(0, katman_dir)
        try:
            import onnx_ai_segmenter

            if log_cb:
                log_cb("Yerel ONNX AI (RapidOCR + Grounding DINO + SlimSAM) baslatiliyor...")
            ai_entries = onnx_ai_segmenter.segment_semantic_groups_onnx(
                rgb,
                fg_mask,
                max_layers=obj_budget,
                yazi_ayir=yazi_ayir,
                custom_labels=ozel_etiketler,
                log_cb=log_cb,
            )
            if ai_entries:
                used_onnx_ai = True
                for name, kind, z_ord, m in ai_entries:
                    raw_entries.append((name, kind, m))
                    objs_sorted.append({"name_ascii": name, "kind": kind, "z_order": z_ord})
        except Exception as exc:
            if log_cb:
                log_cb(f"Yerel ONNX AI atlandi ({exc}); yedek kesfe geciliyor...")
            raw_entries = []
            used_onnx_ai = False

    if not used_onnx_ai:
        objs = custom_objects
        if objs is None:
            if vision_kesif and api_key.strip():
                try:
                    if log_cb:
                        log_cb("Gemini Vision ile tam obje ve yazi kesfi yapiliyor...")
                    objs = discover_objects_vision(rgb, api_key.strip(), max_objects=obj_budget)
                except Exception as exc:
                    if log_cb:
                        log_cb(f"Vision kesfi atlandi ({exc}); yerel kesfe geciliyor...")
                    objs = None
            if not objs:
                objs = discover_objects_local(rgb, fg_mask, max_objects=obj_budget)

        objs_sorted = sorted(objs, key=lambda o: int(o.get("z_order", 1)))[:obj_budget]

        # Her obje icin ham maske cikar
        for idx, obj in enumerate(objs_sorted, start=1):
            kind = str(obj.get("kind", "subject")).lower().strip()
            if not yazi_ayir and kind == "text":
                kind = "subject"
            name = tr_ascii(str(obj.get("name_ascii") or f"Obje_{idx}"))
            box = obj.get("box_2d") or [0, 0, 1000, 1000]
            box_px = _box_norm_to_px(box, W, H, pad_ratio=0.02 if kind == "text" else 0.04)

            if kind == "text":
                m = extract_text_layer_mask(rgb, fg_mask, box_px)
            else:
                m = extract_whole_object_mask(rgb, fg_mask, box_px, fill_holes=True)

            if int(m.sum()) >= 16:
                raw_entries.append((name, kind, m))

    # Kayip onleme: hicbir objeye dusmeyen anlamsiz pikselleri koru (harf icindeki zemin bosluklari haric)
    union_mask = np.zeros((H, W), dtype=bool)
    for _, _, m in raw_entries:
        union_mask |= m
    zemin_ici_delikler = np.zeros((H, W), dtype=bool)
    if bg_rgb is not None:
        lab_full = _rgb_to_lab_u8(rgb).astype(np.float32)
        bg_lab_1 = _rgb_to_lab_u8(bg_rgb.astype(np.uint8).reshape(1, 1, 3)).astype(np.float32)[0, 0]
        de_bg = np.sqrt(((lab_full - bg_lab_1[None, None, :]) ** 2).sum(axis=2))
        zemin_ici_delikler = fg_mask & (~union_mask) & (de_bg <= 20.0)

    leftover = fg_mask & (~union_mask) & (~zemin_ici_delikler)
    fg_sum = max(1, int(fg_mask.sum()))
    if used_onnx_ai and int(leftover.sum()) >= max(120, int(fg_sum * 0.02)):
        # Kalan pikselleri cerceve/cizgi ve renkli arka efekt olarak yapilandir
        lab_left = _rgb_to_lab_u8(rgb).astype(np.float32)
        L_left = lab_left[..., 0] * (100.0 / 255.0)
        C_left = np.sqrt((lab_left[..., 1] - 128.0) ** 2 + (lab_left[..., 2] - 128.0) ** 2)

        dark_stroke_m = leftover & (L_left < 38.0) & (C_left < 16.0)
        # Yalnizca semantik nesnelerden ayrilmis/kenarli siyah cerceveler veya belirgin cizgiler
        splash_m = leftover & (~dark_stroke_m) & (C_left >= 20.0)
        rest_m = leftover & (~dark_stroke_m) & (~splash_m)

        extra_parts: List[Tuple[str, str, np.ndarray]] = []
        if int(rest_m.sum()) >= max(64, int(fg_sum * 0.008)):
            extra_parts.append(("Arka_Plan_ve_Sahne", "subject", rest_m))
        else:
            # Cok az kalan piksel varsa en yakin parcaya kat
            if int(splash_m.sum()) >= int(dark_stroke_m.sum()):
                splash_m |= rest_m
            else:
                dark_stroke_m |= rest_m

        if int(dark_stroke_m.sum()) >= max(64, int(fg_sum * 0.012)):
            extra_parts.append(("Cerceve_ve_Cizgiler", "frame", dark_stroke_m))
        elif int(dark_stroke_m.sum()) > 0:
            if extra_parts:
                extra_parts[0] = (extra_parts[0][0], extra_parts[0][1], extra_parts[0][2] | dark_stroke_m)
            else:
                extra_parts.append(("Arka_Plan_ve_Sahne", "subject", dark_stroke_m))

        if int(splash_m.sum()) >= max(64, int(fg_sum * 0.015)):
            extra_parts.append(("Arka_Efekt", "effect", splash_m))
        elif int(splash_m.sum()) > 0:
            if extra_parts:
                extra_parts[0] = (extra_parts[0][0], extra_parts[0][1], extra_parts[0][2] | splash_m)
            else:
                extra_parts.append(("Arka_Plan_ve_Sahne", "subject", splash_m))

        if not extra_parts:
            extra_parts.append(("Arka_Plan_ve_Sahne", "subject", leftover))

        # Butce asilmasin diye ihtiyaca gore birlestir
        while len(raw_entries) + len(extra_parts) > obj_budget and len(extra_parts) > 1:
            last_n, last_k, last_m = extra_parts.pop()
            extra_parts[0] = ("Arka_Plan_ve_Sahne", "subject", extra_parts[0][2] | last_m)
        while len(raw_entries) >= obj_budget and len(raw_entries) > 0:
            smallest_i = min(range(len(raw_entries)), key=lambda i: int(raw_entries[i][2].sum()))
            _, _, sm_mask = raw_entries.pop(smallest_i)
            extra_parts[0] = (extra_parts[0][0], extra_parts[0][1], extra_parts[0][2] | sm_mask)
        for ep in reversed(extra_parts):
            raw_entries.insert(0, ep)
    elif int(leftover.sum()) >= max(64, int(fg_sum * 0.004)):
        leftover_name = "Arka_Plan_ve_Sahne" if used_onnx_ai else "Govde_ve_Detaylar"
        while len(raw_entries) >= obj_budget and len(raw_entries) > 0:
            # Butce asiliyorsa en kucuk katmani kalan sahneyle birlestir
            smallest_i = min(range(len(raw_entries)), key=lambda i: int(raw_entries[i][2].sum()))
            _, _, sm_mask = raw_entries.pop(smallest_i)
            leftover |= sm_mask
        raw_entries.insert(0, (leftover_name, "subject", leftover))

    # Z-sirasi cakismasi: ust obje alt katmandan dusulur; alt_doldur=True ise inpaint ile oyuk kapanir
    occupied_above = np.zeros((H, W), dtype=bool)
    resolved: List[Tuple[str, str, np.ndarray, np.ndarray]] = []  # (name, kind, mask, rgb_src)

    for name, kind, m in reversed(raw_entries):
        hole_under = m & occupied_above
        clean_m = m & (~occupied_above)
        if (
            (not alt_doldur)
            and name not in ("Zemin", "Arka_Plan_ve_Sahne", "Cerceve_ve_Cizgiler", "Arka_Efekt", "Govde_ve_Detaylar")
            and int(m.sum()) > 0
            and (int(hole_under.sum()) / float(int(m.sum()))) > 0.80
            and int(clean_m.sum()) < int(fg_sum * 0.015)
        ):
            continue
        layer_rgb = rgb.copy()
        if alt_doldur and np.any(hole_under) and np.any(clean_m):
            # Alt katmani ust objenin oyugu altinda duzgunce tamamla
            inpaint_mask = cv2.dilate(hole_under.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
            layer_rgb = cv2.inpaint(layer_rgb, inpaint_mask, 5, cv2.INPAINT_TELEA)
            full_m = m
        else:
            full_m = clean_m

        if np.any(full_m):
            resolved.append((name, kind, full_m, layer_rgb))
        occupied_above |= m

    resolved.reverse()  # Tekrar alttan uste

    outputs: List[LayerOutput] = []
    comp = np.zeros((H, W, 3), dtype=np.float32)

    if has_bg_layer:
        bg_mask = (~fg_mask) | zemin_ici_delikler
        bg_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        bg_rgba[..., :3] = rgb
        bg_rgba[..., 3] = np.where(bg_mask, 255, 0).astype(np.uint8)
        outputs.append(LayerOutput(name="Zemin", rgba=bg_rgba, kind="background", visible=True))
        comp[bg_mask] = rgb[bg_mask].astype(np.float32)
    elif bg_rgb is not None:
        comp = np.tile(bg_rgb.astype(np.float32), (H, W, 1))

    used_names: set[str] = {"Orijinal_Gorsel", "Zemin"}
    for name, kind, mask_b, layer_rgb in resolved:
        uname = _unique_name(name, used_names)
        sigma = 0.65 if kind == "text" else 0.9
        alpha = _soften_alpha(mask_b, sigma=sigma)
        clean_rgb = _decontaminate_edges(layer_rgb, alpha, bg_rgb if not has_bg_layer else None)
        rgba = np.dstack([clean_rgb, np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)])
        outputs.append(LayerOutput(name=uname, rgba=rgba, kind=kind, visible=True))
        a3 = alpha[..., None]
        comp = comp * (1.0 - a3) + clean_rgb.astype(np.float32) * a3

    err = float(np.abs(comp[fg_mask] - rgb[fg_mask].astype(np.float32)).mean()) if fg_mask.any() else 0.0
    return outputs, err, objs_sorted


# ---------------------------------------------------------------------------
# 4. Hibrit Mod (Obje Klasorleri -> Icinde Renk Ailesi Katmanlari)
# ---------------------------------------------------------------------------

def separate_hybrid(
    im_rgba: np.ndarray,
    fg_mask: np.ndarray,
    bg_rgb: Optional[np.ndarray],
    max_layers: int = 10,
    ic_detay_kapsa: bool = True,
    kapsama_orani: float = 0.35,
    vision_kesif: bool = True,
    api_key: str = "",
    yazi_ayir: bool = True,
    alt_doldur: bool = False,
    custom_objects: Optional[List[dict]] = None,
    yerel_ai: bool = True,
    ozel_etiketler: str = "",
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[LayerOutput], float, List[dict]]:
    """Her ana obje icin Photoshop klasoru (group) ve icinde renk ailesi katmanlari uretir."""
    H, W = im_rgba.shape[:2]
    rgb = im_rgba[..., :3]

    obj_layers, _, objs_meta = separate_by_objects(
        im_rgba,
        fg_mask,
        bg_rgb,
        max_layers=max_layers,
        vision_kesif=vision_kesif,
        api_key=api_key,
        yazi_ayir=yazi_ayir,
        alt_doldur=alt_doldur,
        custom_objects=custom_objects,
        yerel_ai=yerel_ai,
        ozel_etiketler=ozel_etiketler,
        log_cb=log_cb,
    )

    outputs: List[LayerOutput] = []
    comp = np.zeros((H, W, 3), dtype=np.float32)
    sub_budget = max(2, min(5, int(max_layers)))

    for lyr in obj_layers:
        if lyr.kind == "background":
            outputs.append(lyr)
            bg_m = lyr.rgba[..., 3] > 128
            comp[bg_m] = lyr.rgba[..., :3][bg_m].astype(np.float32)
            continue

        if lyr.kind == "text" and yazi_ayir:
            outputs.append(lyr)
            a3 = lyr.rgba[..., 3:4].astype(np.float32) / 255.0
            comp = comp * (1.0 - a3) + lyr.rgba[..., :3].astype(np.float32) * a3
            continue

        obj_fg = lyr.rgba[..., 3] > 12
        if int(obj_fg.sum()) < 24:
            continue

        obj_rgba = np.dstack([lyr.rgba[..., :3], np.full((H, W), 255, dtype=np.uint8)])
        sub_layers, _ = separate_by_color_family(
            obj_rgba,
            obj_fg,
            bg_rgb=None,
            max_layers=sub_budget,
            ic_detay_kapsa=ic_detay_kapsa,
            kapsama_orani=kapsama_orani,
            include_background_layer=False,
            group_name=lyr.name,
        )
        if not sub_layers:
            outputs.append(lyr)
            continue
        for sl in sub_layers:
            outputs.append(sl)
            a3 = sl.rgba[..., 3:4].astype(np.float32) / 255.0
            comp = comp * (1.0 - a3) + sl.rgba[..., :3].astype(np.float32) * a3

    err = float(np.abs(comp[fg_mask] - rgb[fg_mask].astype(np.float32)).mean()) if fg_mask.any() else 0.0
    return outputs, err, objs_meta


# ---------------------------------------------------------------------------
# 5. Disa Aktarim (Kirpilmis PNG + Manifest + Bagimsiz RGBA PSD)
# ---------------------------------------------------------------------------

def export_layers_and_psd(
    src_rgba: np.ndarray,
    layers: List[LayerOutput],
    out_dir: str,
    job_name: str,
    dpi: int,
    mode: str,
    mean_abs_err: float,
) -> Dict[str, Any]:
    """Katmanlari sikistirilmis RGBA PNG, manifest JSON ve bagimsiz PSD olarak yazar."""
    H, W = src_rgba.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    layers_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    # En altta gizli Orijinal_Gorsel referans katmani
    orig_rgba = src_rgba.copy()
    if orig_rgba[..., 3].max() == 0:
        orig_rgba[..., 3] = 255
    all_layers: List[LayerOutput] = [
        LayerOutput(name="Orijinal_Gorsel", rgba=orig_rgba, group="", kind="reference", visible=False)
    ] + list(layers)

    manifest_layers: List[dict] = []
    psd = PSDImage.new("RGBA", (W, H))
    group_cache: Dict[str, Any] = {}

    for idx, lyr in enumerate(all_layers):
        a = lyr.rgba[..., 3]
        ys, xs = np.where(a > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        if lyr.kind == "reference":
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

        crop_rgba = lyr.rgba[y0:y1, x0:x1]
        safe_name = tr_ascii(lyr.name, f"Katman_{idx}")
        safe_group = tr_ascii(lyr.group, "") if lyr.group else ""
        png_file = os.path.join(layers_dir, f"{idx:02d}_{safe_name}.png")
        pil_crop = Image.fromarray(crop_rgba, mode="RGBA")
        pil_crop.save(png_file, format="PNG", compress_level=6, dpi=(dpi, dpi))

        manifest_layers.append(
            {
                "index": len(manifest_layers),
                "name": safe_name,
                "group": safe_group,
                "kind": lyr.kind,
                "png_path": os.path.abspath(png_file),
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "visible": bool(lyr.visible),
            }
        )

        # Bagimsiz PSD (RGBA + TRANSPARENCY_MASK)
        if safe_group:
            if safe_group not in group_cache:
                group_cache[safe_group] = psd.create_group(name=safe_group, open_folder=True)
            container = group_cache[safe_group]
        else:
            container = psd
        psd_lyr = container.create_pixel_layer(
            pil_crop,
            name=safe_name,
            top=y0,
            left=x0,
            compression=Compression.ZIP,
        )
        psd_lyr.visible = bool(lyr.visible)

    psd_path = os.path.abspath(os.path.join(out_dir, f"{tr_ascii(job_name, 'desen')}_katmanli.psd"))
    psd.save(psd_path)

    manifest_path = os.path.abspath(os.path.join(out_dir, "katmanlar_manifest.json"))
    manifest = {
        "width": W,
        "height": H,
        "dpi": dpi,
        "mode": mode,
        "job_name": tr_ascii(job_name, "desen"),
        "mean_abs_err": round(float(mean_abs_err), 3),
        "psd_path": psd_path,
        "layers": manifest_layers,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    return {
        "manifest_path": manifest_path,
        "psd_path": psd_path,
        "width": W,
        "height": H,
        "dpi": dpi,
        "mean_abs_err": round(float(mean_abs_err), 3),
        "layer_count": len(manifest_layers),
        "layers": manifest_layers,
    }


def separate_layers(
    input_path: str,
    out_dir: str,
    mode: str = "renk",
    max_layers: int = 10,
    zemin_mod: str = "otomatik",
    bg_tol: float = 18.0,
    ic_detay_kapsa: bool = True,
    kapsama_orani: float = 0.35,
    vision_kesif: bool = True,
    api_key: str = "",
    yazi_ayir: bool = True,
    alt_doldur: bool = False,
    job_name: str = "desen",
    custom_objects: Optional[List[dict]] = None,
    yerel_ai: bool = True,
    ozel_etiketler: str = "",
    log_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Tek giris noktasi: gorseli acar, secili moda gore katmanlara ayirir ve export eder."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Girdi gorseli bulunamadi: {input_path}")

    with Image.open(input_path) as im:
        dpi = _extract_dpi(im)
        im_rgba = np.array(im.convert("RGBA"))

    fg_mask, bg_rgb, _ = detect_background_mask(im_rgba, zemin_mod=zemin_mod, bg_tol=bg_tol)
    norm_mode = (mode or "renk").lower().strip()

    if log_cb:
        log_cb(f"Katman motoru basladi (mod={norm_mode}, maks={max_layers}, zemin={zemin_mod})")

    if norm_mode == "renk":
        layers, err = separate_by_color_family(
            im_rgba,
            fg_mask,
            bg_rgb,
            max_layers=max_layers,
            ic_detay_kapsa=ic_detay_kapsa,
            kapsama_orani=kapsama_orani,
            include_background_layer=(zemin_mod != "ayirma"),
        )
    elif norm_mode == "obje":
        layers, err, _ = separate_by_objects(
            im_rgba,
            fg_mask,
            bg_rgb,
            max_layers=max_layers,
            vision_kesif=vision_kesif,
            api_key=api_key,
            yazi_ayir=yazi_ayir,
            alt_doldur=alt_doldur,
            custom_objects=custom_objects,
            yerel_ai=yerel_ai,
            ozel_etiketler=ozel_etiketler,
            log_cb=log_cb,
        )
    elif norm_mode == "hibrit":
        layers, err, _ = separate_hybrid(
            im_rgba,
            fg_mask,
            bg_rgb,
            max_layers=max_layers,
            ic_detay_kapsa=ic_detay_kapsa,
            kapsama_orani=kapsama_orani,
            vision_kesif=vision_kesif,
            api_key=api_key,
            yazi_ayir=yazi_ayir,
            alt_doldur=alt_doldur,
            custom_objects=custom_objects,
            yerel_ai=yerel_ai,
            ozel_etiketler=ozel_etiketler,
            log_cb=log_cb,
        )
    else:
        raise ValueError(f"Gecersiz katman ayirma modu: {mode}")

    res = export_layers_and_psd(
        im_rgba,
        layers,
        out_dir=out_dir,
        job_name=job_name,
        dpi=dpi,
        mode=norm_mode,
        mean_abs_err=err,
    )
    if log_cb:
        log_cb(f"Tamamlandi: {res['layer_count']} katman yazildi (ort. fark={res['mean_abs_err']})")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="MeChi Studio 3-Modlu Layer Ayirma Motoru")
    ap.add_argument("girdi", help="Kaynak gorsel yolu")
    ap.add_argument("cikti_klasoru", help="Cikti klasoru")
    ap.add_argument("--mod", choices=("renk", "obje", "hibrit"), default="renk")
    ap.add_argument("--maks-katman", type=int, default=10)
    ap.add_argument("--zemin", choices=("otomatik", "seffaf_koru", "ayirma"), default="otomatik")
    ap.add_argument("--bg-tol", type=float, default=18.0)
    ap.add_argument("--ic-detay-kapsa", choices=("0", "1"), default="1")
    ap.add_argument("--kapsama-orani", type=float, default=0.35)
    ap.add_argument("--vision-kesif", choices=("0", "1"), default="1")
    ap.add_argument("--yerel-ai", choices=("0", "1"), default="1")
    ap.add_argument("--ozel-etiketler", default="")
    ap.add_argument("--yazi-ayir", choices=("0", "1"), default="1")
    ap.add_argument("--alt-doldur", choices=("0", "1"), default="0")
    ap.add_argument("--is-ad", default="desen")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    res = separate_layers(
        input_path=args.girdi,
        out_dir=args.cikti_klasoru,
        mode=args.mod,
        max_layers=args.maks_katman,
        zemin_mod=args.zemin,
        bg_tol=args.bg_tol,
        ic_detay_kapsa=(args.ic_detay_kapsa == "1"),
        kapsama_orani=args.kapsama_orani,
        vision_kesif=(args.vision_kesif == "1"),
        api_key=api_key,
        yazi_ayir=(args.yazi_ayir == "1"),
        alt_doldur=(args.alt_doldur == "1"),
        job_name=args.is_ad,
        yerel_ai=(args.yerel_ai == "1"),
        ozel_etiketler=args.ozel_etiketler,
        log_cb=lambda msg: print(msg, flush=True),
    )
    print(f"MANIFEST:{res['manifest_path']}", flush=True)
    print(f"PSD:{res['psd_path']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
