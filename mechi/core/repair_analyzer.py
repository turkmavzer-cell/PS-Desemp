"""Bozuk / giyilmis gorselleri yerel OpenCV+KMeans ve/veya Gemini Vision ile analiz edip MeChi 3.3 promptu uretir."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from mechi.core.worker import TaskCancelled


MECHI33_SYSTEM_INSTRUCTION = (
    "You are a high-precision, protocol-driven 2D Textile Graphic Recreation Engine (MeChi 3.3). "
    "Operational Principle: 'Absolute Recreation on a Perfect Surface' (Stretched Canvas / Smart UV Unwrap). "
    "NEVER edit, filter, retouch, or iron the source photograph. Treat the input image strictly as a forensic "
    "structural and stylistic blueprint, and reconstruct the original pre-print 2D digital artwork from scratch "
    "at ultra-high 4K (3840px+) print-ready clarity on an idealized, perfectly taut flat digital canvas.\n"
    "Protocol 3.1 (Perfect Surface Recreation): Completely eliminate fabric weave, knit grain, seams, stitching, "
    "wrinkles, folds, creases, drapery, garment silhouettes, mannequin/human body parts, shadows, glare, reflections, "
    "and photographic noise.\n"
    "Protocol 3.2 (Orthographic Projection & Geometric Reset): Rebuild every motif, border, and letterform in strict "
    "2D orthographic projection with razor-straight baselines and upright letter axes (never trace 3D fabric-fold squishing).\n"
    "Protocol 3.3 (Seamless Continuity & Command Hierarchy): Directives starting with CRITICAL:, DO NOT, or ABSOLUTELY NO "
    "override all other instructions. Output strictly ONE standalone flat 2D graphic artwork image—no split panels, "
    "no before/after comparisons, no mockups, and no variants."
)

MECHI33_LICENSE_SAFE_SYSTEM_INSTRUCTION = (
    "You are a 2D apparel graphic print specialist. "
    "First, make minimal subtle adjustments to the graphic in the reference image so it is an original, "
    "non-infringing version while preserving the same overall visual layout, pose, drawing style, and color energy. "
    "Then produce a crystal-clear, sharp, high-resolution, completely flat 2D standalone print graphic on a solid "
    "background with straight level horizontal and vertical alignment and clean margins, removing any garment or folds."
)

SOURCE_CONDITIONS: Dict[str, dict] = {
    "worn_garment": {
        "label_tr": "Giyilmiş Tişört / Manken (3D Kıvrımlı)",
        "damage_en": (
            "3D chest and body curvature, worn-garment t-shirt/dress creases, deep fabric folds, perspective foreshortening, "
            "collar/sleeve/hair/arm occlusions, and uneven studio lighting falloff"
        ),
        "unwrap_en": (
            "Apply Smart UV Unwrap / Stretched Canvas projection to strip all 3D human body/mannequin geometry and "
            "mathematically flatten every worn garment curve onto an orthographic 2D plane."
        ),
        "prohibit_extra": "",
    },
    "flat_wrinkled": {
        "label_tr": "Buruşuk Kumaş / Masa Fotoğrafı",
        "damage_en": (
            "laid-flat fabric wrinkles, crumpled cloth waves, knit cotton weave texture, camera perspective skew, "
            "overhead flash glare, and cast fold shadows"
        ),
        "unwrap_en": (
            "Mathematically flatten all surface waves, perspective keystone skew, and table-top lighting gradients "
            "as if printed directly on an idealized flat vector-smooth artboard."
        ),
        "prohibit_extra": "",
    },
    "lowres_damaged": {
        "label_tr": "Düşük Çözünürlük / Yıpranmış",
        "damage_en": (
            "low-resolution pixelated blur, JPEG compression blocking, faded washed-out ink, print cracking/peeling, "
            "color noise, and partially cropped artwork edges"
        ),
        "unwrap_en": (
            "Rebuild every blurred edge, broken stroke, and missing fragment from scratch at crisp 4K (3840px+) "
            "pre-press sharpness while honoring the exact original artistic structure."
        ),
        "prohibit_extra": "",
    },
    "coated_textured": {
        "label_tr": "Dokulu Yüzey (Kanvas / Deri / Nakış)",
        "damage_en": (
            "coated-canvas weave, leather pebble grain, heavy knit ribbing, raised 3D embroidery relief, bumpy surface "
            "highlights, specular leather glare, and material stitching"
        ),
        "unwrap_en": (
            "Strip 100% of physical substrate grain, leather bumps, coated-canvas texture, and embossed 3D relief "
            "to recover the pure flat pre-manufacturing 2D graphic pattern."
        ),
        "prohibit_extra": (
            "- DO NOT include any leather texture, coated canvas weave, pebble grain, knit ribbing, bumps, "
            "stitching relief, or 3D embossed surface depth.\n"
        ),
    },
}

RENDER_STYLES: Dict[str, dict] = {
    "auto": {
        "label_tr": "Otomatik (Kaynak Stili)",
        "directive_en": "",
    },
    "flat_vector": {
        "label_tr": "Düz Vektörel Spot (Tramsız)",
        "directive_en": (
            "Render strictly as a razor-sharp flat vector spot-color 2D illustration with uniform solid color fills, "
            "crisp Bezier-clean boundaries, zero gradients, and zero photographic halftone noise (ready for flat spot screen printing)."
        ),
    },
    "stipple_dot": {
        "label_tr": "Noktalı / Gravür / Tramlı",
        "directive_en": (
            "Preserve the exact screen-print dot-matrix / circular stipple / engraving cross-hatch construction cleanly; "
            "maintain crisp uniform dot spacing and dot-row alignment without merging stipple dots into muddy solid bars."
        ),
    },
    "painterly_2d": {
        "label_tr": "Temiz 2D İllüstrasyon",
        "directive_en": (
            "Reconstruct as an ultra-clean 2D commercial editorial illustration with crisp inked boundaries and smooth "
            "idealized digital shading/tonal transitions completely stripped of physical paper/fabric grain or lighting falloff."
        ),
    },
}


@dataclass
class LocalMetrics:
    width: int
    height: int
    aspect_label: str
    is_low_res: bool
    blur_score: float
    blur_label_tr: str
    border_uniform_ratio: float
    bg_hex: str
    suggested_scope: str
    suggested_bg_mode: str
    palette: List[dict]
    lighting_unevenness: float = 0.0
    lighting_label_tr: str = "Dengeli Aydınlatma"
    texture_grain_score: float = 0.0
    has_fabric_grain: bool = False
    flatness_ratio: float = 0.0
    suggested_render_style: str = "flat_vector"
    edge_clipped_sides: int = 0
    tilt_deg: float = 0.0
    keystone_skew_deg: float = 0.0
    has_rect_frame: bool = False
    has_text_bands: bool = False
    text_band_tilt_deg: float = 0.0
    clipped_sides: Tuple[str, ...] = ()
    geometry_issues_tr: Tuple[str, ...] = ()
    recommended_aspect: str = "1:1"


@dataclass
class RepairPromptResult:
    analysis_mode: str  # "gemini_vision" | "local_python"
    summary_tr: str
    detected_modules: str
    palette_hexes: List[str]
    full_prompt: str
    metrics: LocalMetrics
    fallback_note: str = ""
    system_instruction: str = MECHI33_SYSTEM_INSTRUCTION
    rectified_ref_path: str = ""


def _aspect_label(w: int, h: int) -> str:
    if h <= 0 or w <= 0:
        return "1:1"
    ratio = w / float(h)
    if 0.88 <= ratio <= 1.14:
        return "1:1"
    if ratio < 0.88:
        return "3:4" if ratio >= 0.65 else "9:16"
    return "4:3" if ratio <= 1.55 else "16:9"


def _weighted_median(values_and_weights: List[Tuple[float, float]]) -> float:
    if not values_and_weights:
        return 0.0
    ordered = sorted(values_and_weights, key=lambda item: item[0])
    total_w = sum(max(1e-4, w) for _, w in ordered)
    acc = 0.0
    for val, w in ordered:
        acc += max(1e-4, w)
        if acc >= 0.5 * total_w:
            return float(val)
    return float(ordered[-1][0])


def _measure_frame_and_text_geometry(
    gray: np.ndarray,
    bg_luma: float,
    suggested_scope: str,
) -> Tuple[float, float, bool, bool, float]:
    """HoughLinesP ile cerceve/eksen egimini, perspektif yamulmayi ve morfolojik metin bantlarini olcer."""
    sh, sw = gray.shape[:2]
    edges = cv2.Canny(gray, 45, 140)
    min_len = max(18, min(sh, sw) // 7)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=24,
        minLineLength=min_len,
        maxLineGap=10,
    )

    h_lines: List[Tuple[float, float, float]] = []  # (dev_deg, weight, mid_y)
    v_lines: List[Tuple[float, float, float]] = []  # (dev_deg, weight, mid_x)
    if lines is not None and len(lines) > 0:
        for seg in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            if dx < 0.0:
                dx, dy = -dx, -dy
            ang = float(np.degrees(np.arctan2(dy, dx)))
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            if abs(ang) <= 14.0:
                # Dis cerceve/baslik kusaklarindaki yatay cizgilere 2.5x agirlik ver
                perim_w = length * (2.5 if (my <= 0.32 * sh or my >= 0.68 * sh) else 1.0)
                h_lines.append((ang, perim_w, my))
            elif abs(ang) >= 76.0:
                v_dev = (ang + 90.0) if ang < 0.0 else (ang - 90.0)
                perim_w = length * (2.5 if (mx <= 0.32 * sw or mx >= 0.68 * sw) else 0.6)
                v_lines.append((v_dev, perim_w, mx))

    has_rect_frame = len(h_lines) >= 1 and len(v_lines) >= 1 and (len(h_lines) + len(v_lines)) >= 3

    h_med = _weighted_median([(d, w) for d, w, _ in h_lines]) if h_lines else 0.0
    v_med = _weighted_median([(d, w) for d, w, _ in v_lines]) if v_lines else 0.0

    # Rijit eksen donmesi (tilt_deg) yalnizca yatay ve dikey eksenler ayni yonde tutarli donmusse vardir
    if h_lines and v_lines:
        if abs(h_med - v_med) <= 3.5:
            all_devs = [(d, w) for d, w, _ in h_lines] + [(d, w) for d, w, _ in v_lines]
            tilt_deg = round(_weighted_median(all_devs), 1)
        elif (h_med * v_med) > 0.0:
            sign = 1.0 if h_med > 0.0 else -1.0
            tilt_deg = round(sign * min(abs(h_med), abs(v_med)), 1)
        else:
            tilt_deg = 0.0
    elif len(h_lines) >= 2:
        tilt_deg = round(h_med, 1)
    else:
        tilt_deg = 0.0

    if abs(tilt_deg) < 0.5:
        tilt_deg = 0.0

    # Perspektif / trapez yamulma (ust vs alt yatay veya sol vs sag dikey veya yatay-dikey aci farki)
    skew_candidates = [0.0]
    if len(h_lines) >= 2:
        top_h = [(d, w) for d, w, my in h_lines if my < 0.5 * sh]
        bot_h = [(d, w) for d, w, my in h_lines if my >= 0.5 * sh]
        if top_h and bot_h:
            skew_candidates.append(abs(_weighted_median(top_h) - _weighted_median(bot_h)))
    if len(v_lines) >= 2:
        left_v = [(d, w) for d, w, mx in v_lines if mx < 0.5 * sw]
        right_v = [(d, w) for d, w, mx in v_lines if mx >= 0.5 * sw]
        if left_v and right_v:
            skew_candidates.append(abs(_weighted_median(left_v) - _weighted_median(right_v)))
    if h_lines and v_lines:
        skew_candidates.append(abs(h_med - v_med))
    keystone_skew_deg = round(float(max(skew_candidates)), 1)

    # Morfolojik metin bandi tespiti (ust %32 ve alt %32 kusaklarda harf dizilimi)
    has_text_bands = False
    text_band_tilts: List[float] = []
    if suggested_scope == "A2":
        fg_mask = (np.abs(gray.astype(np.int16) - int(round(bg_luma))) >= 36).astype(np.uint8) * 255
        band_h = max(16, int(round(sh * 0.32)))
        for y0, y1 in ((0, band_h), (sh - band_h, sh)):
            band = fg_mask[y0:y1, :]
            num_lbl, _lbls, stats, centroids = cv2.connectedComponentsWithStats(band, connectivity=8)
            glyphs: List[Tuple[float, float]] = []
            for idx in range(1, num_lbl):
                bw_c = int(stats[idx, cv2.CC_STAT_WIDTH])
                bh_c = int(stats[idx, cv2.CC_STAT_HEIGHT])
                area_c = int(stats[idx, cv2.CC_STAT_AREA])
                if 2 <= bw_c <= int(sw * 0.22) and 4 <= bh_c <= int(sh * 0.20) and area_c >= 8:
                    cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
                    glyphs.append((cx, cy))

            if len(glyphs) >= 3:
                glyphs.sort(key=lambda pt: pt[0])
                xs = np.array([pt[0] for pt in glyphs], dtype=np.float32)
                ys = np.array([pt[1] for pt in glyphs], dtype=np.float32)
                x_span = float(xs.max() - xs.min())
                y_std = float(ys.std())
                if x_span >= sw * 0.14 and y_std <= sh * 0.10:
                    has_text_bands = True
                    slope = float(np.polyfit(xs, ys, 1)[0])
                    ang_b = float(np.degrees(np.arctan(slope)))
                    if abs(ang_b) <= 25.0:
                        text_band_tilts.append(ang_b)

    text_band_tilt_deg = (
        round(max(text_band_tilts, key=lambda a: abs(a)), 1)
        if text_band_tilts
        else (tilt_deg if has_text_bands else 0.0)
    )
    return tilt_deg, keystone_skew_deg, has_rect_frame, has_text_bands, text_band_tilt_deg


def pre_rectify_reference_image(
    image_path: str,
    out_path: str,
    metrics: LocalMetrics,
    pad_ratio: float = 0.12,
) -> str:
    """Referans gorseldeki egimi (-tilt_deg) duzeltir ve 4 kenara guvenli duz zemin boslugu (%12-%16) ekler."""
    hx = (metrics.bg_hex or "#FFFFFF").lstrip("#")
    if len(hx) == 6:
        bg_rgb = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
    else:
        bg_rgb = (255, 255, 255)

    with Image.open(image_path) as im:
        if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
            rgba = im.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, bg_rgb)
            canvas.paste(rgba, mask=rgba.split()[3])
            arr = np.array(canvas)
        else:
            arr = np.array(im.convert("RGB"))

    h, w = arr.shape[:2]
    # 1) Eger belirgin eksen/cerceve egimi varsa (1.2° .. 15.0°), kose kesmeden deskew uygula
    if 1.2 <= abs(metrics.tilt_deg) <= 15.0:
        angle = float(metrics.tilt_deg)
        center = (w * 0.5, h * 0.5)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rad = np.radians(abs(angle))
        nw = int(round(w * np.cos(rad) + h * np.sin(rad)))
        nh = int(round(h * np.cos(rad) + w * np.sin(rad)))
        rot_mat[0, 2] += (nw * 0.5) - center[0]
        rot_mat[1, 2] += (nh * 0.5) - center[1]
        arr = cv2.warpAffine(
            arr,
            rot_mat,
            (nw, nh),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_rgb,
        )
        h, w = arr.shape[:2]

    # 2) 4 kenara temiz zemin boslugu (kirpilmis kenarlarda %16, digerlerinde pad_ratio=%12)
    clipped = set(metrics.clipped_sides or ())
    top_r = max(pad_ratio, 0.16) if "top" in clipped else pad_ratio
    bot_r = max(pad_ratio, 0.16) if "bottom" in clipped else pad_ratio
    left_r = max(pad_ratio, 0.16) if "left" in clipped else pad_ratio
    right_r = max(pad_ratio, 0.16) if "right" in clipped else pad_ratio

    pad_top = max(8, int(round(h * top_r)))
    pad_bot = max(8, int(round(h * bot_r)))
    pad_left = max(8, int(round(w * left_r)))
    pad_right = max(8, int(round(w * right_r)))

    padded = cv2.copyMakeBorder(
        arr,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_rgb,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(padded, mode="RGB").save(out_path, format="PNG")
    return out_path


def _describe_rgb(r: int, g: int, b: int) -> Tuple[str, str]:
    mx, mn = max(r, g, b), min(r, g, b)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = (mx - mn) / float(mx) if mx > 0 else 0.0
    if luma < 38 and sat < 0.35:
        return "Deep Ink Black", "Koyu Siyah"
    if luma > 224 and sat < 0.15:
        return "Pure Crisp White", "Saf Beyaz"
    if sat < 0.18:
        if luma < 110:
            return "Dark Charcoal Gray", "Koyu Grafit"
        if luma < 185:
            return "Neutral Slate Gray", "Orta Gri"
        return "Warm Ivory Cream", "Krem / Açık Gri"

    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    hue = int(hsv[0]) * 2  # 0..360
    if hue < 15 or hue >= 345:
        return ("Deep Crimson Red", "Kırmızı") if luma < 135 else ("Bright Coral Red", "Mercan Kırmızı")
    if hue < 42:
        return ("Warm Rust Terracotta", "Kiremit / Turuncu") if luma < 145 else ("Vibrant Orange", "Turuncu")
    if hue < 68:
        return ("Antique Ochre Gold", "Altın / Hardal") if luma < 165 else ("Bright Lemon Yellow", "Sarı")
    if hue < 155:
        return ("Deep Forest Green", "Koyu Yeşil") if luma < 120 else ("Emerald Green", "Yeşil")
    if hue < 195:
        return ("Aegean Teal", "Turkuaz")
    if hue < 255:
        return ("Midnight Navy Blue", "Lacivert") if luma < 105 else ("Cobalt Royal Blue", "Mavi")
    if hue < 305:
        return ("Deep Violet Purple", "Mor")
    return ("Burgundy Magenta", "Bordo / Fuşya") if luma < 130 else ("Rose Pink", "Pembe")


def analyze_image_local(image_path: str) -> LocalMetrics:
    """OpenCV + NumPy + Lab K-Means ile gorselin boyut, netlik, zemin ve spot renk paletini olcer."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    with Image.open(image_path) as im:
        orig_w, orig_h = im.size
        has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in im.info)
        rgba_full = im.convert("RGBA") if has_alpha else None
        if has_alpha and rgba_full is not None:
            alpha_chk = np.array(rgba_full)[..., 3]
            has_alpha = float((alpha_chk < 245).mean()) > 0.02

        work_im = rgba_full if has_alpha and rgba_full is not None else im.convert("RGB")
        scale = min(1.0, 320.0 / float(max(orig_w, orig_h)))
        sw = max(16, int(round(orig_w * scale)))
        sh = max(16, int(round(orig_h * scale)))
        small = work_im.resize((sw, sh), Image.LANCZOS)
        arr = np.array(small)

    if has_alpha:
        alpha = arr[..., 3]
        rgb = arr[..., :3]
        opaque_mask = alpha >= 128
        # Seffaf dis alan varsa tekil baski olarak ele al
        rgb_for_border = np.where(opaque_mask[..., None], rgb, 255).astype(np.uint8)
    else:
        rgb = arr[..., :3]
        opaque_mask = np.ones((sh, sw), dtype=bool)
        rgb_for_border = rgb

    gray = cv2.cvtColor(rgb_for_border, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 70.0:
        blur_label = "Düşük Netlik / Bulanık"
    elif blur_score < 220.0:
        blur_label = "Orta Netlik (Yumuşak Kenar)"
    else:
        blur_label = "Yüksek Kenar Keskinliği"

    # Kenar seridi homojenlik analizi -> A1 (Metraj) vs A2 (Tekil Baski)
    bw = max(4, min(sh, sw) // 20)
    named_strips = [
        ("top", rgb_for_border[:bw].reshape(-1, 3)),
        ("bottom", rgb_for_border[-bw:].reshape(-1, 3)),
        ("left", rgb_for_border[:, :bw].reshape(-1, 3)),
        ("right", rgb_for_border[:, -bw:].reshape(-1, 3)),
    ]
    strips = [st for _, st in named_strips]
    border_px = np.concatenate(strips, axis=0)
    med_rgb = np.median(border_px, axis=0).astype(np.uint8)
    med_lab = cv2.cvtColor(med_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    border_lab = cv2.cvtColor(border_px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    de = np.sqrt(((border_lab - med_lab[None, :]) ** 2).sum(axis=-1))
    border_ratio = float((de < 15.0).mean())

    suggested_scope = "A2" if (has_alpha or border_ratio >= 0.45) else "A1"

    clipped_list: List[str] = []
    for side_name, st in named_strips:
        st_lab = cv2.cvtColor(st.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        st_de = np.sqrt(((st_lab - med_lab[None, :]) ** 2).sum(axis=-1))
        if float((st_de >= 15.0).mean()) >= 0.10:
            clipped_list.append(side_name)
    clipped_sides = tuple(clipped_list) if suggested_scope == "A2" else ()
    edge_clipped_sides = len(clipped_sides)

    # 4x4 grid isik / kivrim dalgalanma analizi
    tile_meds: List[float] = []
    rh, cw = max(1, sh // 4), max(1, sw // 4)
    for r_i in range(4):
        for c_i in range(4):
            patch = gray[r_i * rh : (r_i + 1) * rh, c_i * cw : (c_i + 1) * cw]
            if patch.size > 0:
                tile_meds.append(float(np.median(patch)))
    lighting_unevenness = round(float(np.std(tile_meds)) if tile_meds else 0.0, 1)
    if lighting_unevenness >= 22.0:
        lighting_label_tr = "Yüksek Kıvrım / Işık Dalgalanması"
    elif lighting_unevenness >= 11.0:
        lighting_label_tr = "Orta Gölge / Kırışıklık"
    else:
        lighting_label_tr = "Dengeli Aydınlatma"

    # Sobel gradyan ve yuksek gecisli kumas dokusu (grain) olcumu
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    flat_mask = mag < 18.0
    flatness_ratio = round(float(flat_mask.mean()), 3)
    suggested_render_style = "flat_vector" if flatness_ratio >= 0.72 else "painterly_2d"

    gray_f = gray.astype(np.float32)
    residual = np.abs(gray_f - cv2.GaussianBlur(gray_f, (5, 5), 0))
    low_grad_mask = mag < 25.0
    if low_grad_mask.any():
        texture_grain_score = round(float(residual[low_grad_mask].mean()), 2)
    else:
        texture_grain_score = round(float(residual.mean()), 2)
    has_fabric_grain = texture_grain_score >= 2.2

    r_b, g_b, b_b = int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2])
    luma_b = 0.299 * r_b + 0.587 * g_b + 0.114 * b_b
    if has_alpha:
        suggested_bg = "B1"
        bg_hex = "#FFFFFF"
    elif luma_b >= 205:
        suggested_bg = "B1"
        bg_hex = "#FFFFFF"
    elif luma_b <= 55:
        suggested_bg = "B2"
        bg_hex = "#000000"
    else:
        suggested_bg = "B3"
        bg_hex = f"#{r_b:02X}{g_b:02X}{b_b:02X}"

    # LAB K-Means spot renk cikarimi
    fg_rgb = rgb[opaque_mask] if opaque_mask.any() else rgb.reshape(-1, 3)
    if len(fg_rgb) > 12000:
        step = max(1, len(fg_rgb) // 12000)
        fg_sample = fg_rgb[::step]
    else:
        fg_sample = fg_rgb

    fg_lab = cv2.cvtColor(fg_sample.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k = min(6, max(2, len(fg_lab)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.5)
    _compact, labels, centers_lab = cv2.kmeans(fg_lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)

    palette: List[dict] = []
    kept_labs: List[np.ndarray] = []
    total_px = float(max(1, counts.sum()))
    for idx in order:
        c_lab = centers_lab[idx]
        if any(float(np.linalg.norm(c_lab - prev)) < 18.0 for prev in kept_labs):
            continue
        c_rgb = cv2.cvtColor(np.clip(np.round(c_lab), 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)
        cr, cg, cb = int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])
        name_en, name_tr = _describe_rgb(cr, cg, cb)
        palette.append(
            {
                "hex": f"#{cr:02X}{cg:02X}{cb:02X}",
                "name_en": name_en,
                "name_tr": name_tr,
                "ratio": round(float(counts[idx]) / total_px, 3),
            }
        )
        kept_labs.append(c_lab)
        if len(palette) >= 5:
            break

    tilt_deg, keystone_skew_deg, has_rect_frame, has_text_bands, text_band_tilt_deg = _measure_frame_and_text_geometry(
        gray,
        bg_luma=luma_b,
        suggested_scope=suggested_scope,
    )

    geo_issues: List[str] = []
    if abs(tilt_deg) >= 1.2:
        geo_issues.append(f"{tilt_deg:+.1f}° çerçeve/eksen eğimi")
    if keystone_skew_deg >= 2.0:
        geo_issues.append(f"{keystone_skew_deg:.1f}° perspektif/yamulma")
    if has_rect_frame:
        geo_issues.append("Dikdörtgen çerçeve/köşebent (0°/90° kilidi)")
    if has_text_bands:
        geo_issues.append(f"Metin bandı (eğim {text_band_tilt_deg:+.1f}°)")
    if clipped_sides:
        geo_issues.append(f"Kenar kırpılması: {', '.join(clipped_sides)}")

    base_aspect = _aspect_label(orig_w, orig_h)
    if orig_h > orig_w * 1.12 or (has_text_bands and ("top" in clipped_sides or "bottom" in clipped_sides)):
        recommended_aspect = "3:4"
    else:
        recommended_aspect = base_aspect

    return LocalMetrics(
        width=orig_w,
        height=orig_h,
        aspect_label=base_aspect,
        is_low_res=max(orig_w, orig_h) < 2000,
        blur_score=round(blur_score, 1),
        blur_label_tr=blur_label,
        border_uniform_ratio=round(border_ratio, 3),
        bg_hex=bg_hex,
        suggested_scope=suggested_scope,
        suggested_bg_mode=suggested_bg,
        palette=palette,
        lighting_unevenness=lighting_unevenness,
        lighting_label_tr=lighting_label_tr,
        texture_grain_score=texture_grain_score,
        has_fabric_grain=has_fabric_grain,
        flatness_ratio=flatness_ratio,
        suggested_render_style=suggested_render_style,
        edge_clipped_sides=edge_clipped_sides,
        tilt_deg=tilt_deg,
        keystone_skew_deg=keystone_skew_deg,
        has_rect_frame=has_rect_frame,
        has_text_bands=has_text_bands,
        text_band_tilt_deg=text_band_tilt_deg,
        clipped_sides=clipped_sides,
        geometry_issues_tr=tuple(geo_issues),
        recommended_aspect=recommended_aspect,
    )


def _encode_vision_jpeg_b64(image_path: str, max_side: int = 1280) -> str:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def analyze_with_gemini_vision(
    image_path: str,
    metrics: LocalMetrics,
    api_key: str,
    overrides: dict,
    model: str = "gemini-2.5-flash",
) -> dict:
    """Gemini Vision ile 7 maddelik MeChi 3.3 adli gorsel analizi yapar (otomatik model yedekli)."""
    b64 = _encode_vision_jpeg_b64(image_path)
    measured_pal = ", ".join(f"{p['name_en']} ({p['hex']})" for p in metrics.palette)
    source_cond = str(overrides.get("source_condition") or "auto")
    render_style = str(overrides.get("render_style") or "auto")
    hint_lines = []
    if overrides.get("custom_text"):
        hint_lines.append(f"User noted text inside graphic: {overrides['custom_text']}")
    if overrides.get("extra_note"):
        hint_lines.append(f"User repair focus note: {overrides['extra_note']}")
    if overrides.get("license_safe"):
        hint_lines.append(
            "ORIGINAL REDESIGN MODE: Describe the mood, layout, and colors in generic original terms without brand or character names."
        )

    prompt = (
        "You are a forensic textile graphic analyst preparing specifications for FROM-SCRATCH 2D production recreation (MeChi 3.3 / Nano Banana Pro).\n"
        "Core Principle: DO NOT edit, retouch, or filter the photograph. Recreate the underlying artwork from scratch on an idealized stretched flat 2D digital canvas (Smart UV Unwrap).\n"
        f"Local pixel scan measured: {metrics.width}x{metrics.height} px, blur={metrics.blur_score}, "
        f"border_uniform_ratio={metrics.border_uniform_ratio}, lighting_unevenness={metrics.lighting_unevenness}, "
        f"has_fabric_grain={metrics.has_fabric_grain} (score={metrics.texture_grain_score}), flatness_ratio={metrics.flatness_ratio}, "
        f"edge_clipped_sides={metrics.edge_clipped_sides} (clipped_sides={list(metrics.clipped_sides)}), "
        f"tilt_deg={metrics.tilt_deg}, keystone_skew_deg={metrics.keystone_skew_deg}, "
        f"has_rect_frame={metrics.has_rect_frame}, has_text_bands={metrics.has_text_bands} (text_tilt={metrics.text_band_tilt_deg}), "
        f"suggested_scope={metrics.suggested_scope}, "
        f"suggested_bg={metrics.suggested_bg_mode} ({metrics.bg_hex}), measured_palette=[{measured_pal}].\n"
        f"Requested source condition='{source_cond}' (if 'auto', classify accurately as worn_garment, flat_wrinkled, lowres_damaged, or coated_textured), "
        f"requested render style='{render_style}'.\n"
        + ("\n".join(hint_lines) + "\n" if hint_lines else "")
        + "Inspect the image across all 7 forensic checklist items:\n"
        "1) Scope (A1 all-over repeating pattern vs A2 isolated garment placement graphic)\n"
        "2) Main motifs, exact symmetry, interlocking relationships, and spatial layout\n"
        "3) Technical line/texture style (flat vector, dot-matrix stipple, engraving cross-hatch, brush/painterly)\n"
        "4) Idealized unshaded solid spot-color palette with HEX codes\n"
        "5) Clean background decision (B1 #FFFFFF, B2 #000000, or B3 idealized solid background HEX)\n"
        "6) Occluded/folded/creased/cropped regions to geometrically unwrap and regenerate seamlessly\n"
        "7) Exact typography protocol: CRITICAL RULE — when hand-drawn, brush-textured, or stylized text sits on a 3D fabric fold or warped garment curve, ALWAYS select 'C4' (Hybrid Geometry Reset) instead of 'C1'!\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "source_condition": "worn_garment, flat_wrinkled, lowres_damaged, or coated_textured",\n'
        '  "scope_a": "A1 or A2",\n'
        '  "bg_b": "B1 or B2 or B3",\n'
        '  "bg_hex": "#HEXCODE of idealized flat background",\n'
        '  "bg_desc_en": "English color description of idealized background (if B3)",\n'
        '  "text_c": "C1, C2, C3, or C4",\n'
        '  "detected_text": "exact verbatim text visible on the artwork, or empty string if none/only tags",\n'
        '  "font_style_en": "English font/lettering style description if text exists",\n'
        '  "style_technique_en": "English technical rendering style (e.g., flat screen-print vector spot fills / fine stipple dot-matrix / crisp watercolor editorial 2D illustration)",\n'
        '  "general_subject_en": "concise English summary of the graphic/pattern to recreate",\n'
        '  "motif_geometry_en": "detailed English description of every motif, figure, symmetry, spacing, and relative placement",\n'
        '  "fine_details_en": "detailed English description of line weights, stipple dots, contours, cross-hatching, and textures",\n'
        '  "geometric_defects_en": "specific frame tilt, perspective keystone skew, or text-baseline curvature defects detected (or none)",\n'
        '  "damage_repair_en": "specific 3D folds, sleeves, hair, glare, blur, missing edges, or garment curves to strip and reconstruct via Smart UV Unwrap",\n'
        '  "palette_en": "clean idealized solid spot color roles with HEX codes",\n'
        '  "summary_tr": "2-3 cumle Turkce adli ozet: ne algilandi, hangi geometrik bozulmalar sifirdan yeniden yaratilacak"\n'
        "}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    models_to_try = [model] + [m for m in ("gemini-2.5-flash", "gemini-2.0-flash") if m != model]
    resp = None
    last_err = ""
    for idx, mdl in enumerate(models_to_try):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key.strip()}"
        try:
            cand_resp = requests.post(url, json=payload, timeout=60)
        except Exception as exc:
            last_err = str(exc)
            if idx < len(models_to_try) - 1:
                continue
            raise RuntimeError(f"Gemini Vision bağlantı hatası: {last_err}") from exc
        if cand_resp.status_code == 200:
            resp = cand_resp
            break
        last_err = f"HTTP {cand_resp.status_code}: {cand_resp.text[:250]}"
        if cand_resp.status_code in (404, 429, 503) and idx < len(models_to_try) - 1:
            continue
        raise RuntimeError(f"Gemini Vision {last_err}")

    if resp is None:
        raise RuntimeError(f"Gemini Vision başarısız: {last_err}")

    parts = ((resp.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    raw_txt = "".join(p.get("text", "") for p in parts).strip()
    if raw_txt.startswith("```"):
        raw_txt = re.sub(r"^```(?:json)?\s*", "", raw_txt)
        raw_txt = re.sub(r"\s*```$", "", raw_txt)
    start, end = raw_txt.find("{"), raw_txt.rfind("}")
    if start >= 0 and end > start:
        raw_txt = raw_txt[start : end + 1]
    return json.loads(raw_txt)


def build_mechi33_prompt(analysis: dict) -> str:
    """MeChi 3.3 / Nano Banana Pro 5 bloklu Sifirdan 2D Yeniden Yaratim Master Promptunu insa eder."""
    scope_a = analysis.get("scope_a", "A2")
    bg_b = analysis.get("bg_b", "B1")
    bg_hex = analysis.get("bg_hex", "#FFFFFF")
    bg_desc_en = analysis.get("bg_desc_en", "solid color")
    text_c = analysis.get("text_c", "C3")
    detected_text = (analysis.get("detected_text") or "").strip()
    font_style_en = (analysis.get("font_style_en") or "bold editorial").strip()

    source_condition = str(analysis.get("source_condition") or "worn_garment")
    cond_spec = SOURCE_CONDITIONS.get(source_condition, SOURCE_CONDITIONS["worn_garment"])

    render_style = str(analysis.get("render_style") or "auto")
    style_spec = RENDER_STYLES.get(render_style, RENDER_STYLES["auto"])
    style_dir = (analysis.get("style_technique_en") or "").strip()
    if style_spec["directive_en"]:
        style_block = style_spec["directive_en"]
    elif style_dir:
        style_block = style_dir
    else:
        style_block = RENDER_STYLES["flat_vector"]["directive_en"]

    subject_en = analysis.get("general_subject_en") or "printed textile graphic artwork"
    motif_en = analysis.get("motif_geometry_en") or (
        "Faithfully preserve every visible motif, figure, outline, proportion, and compositional relationship from the reference."
    )
    details_en = analysis.get("fine_details_en") or (
        "Restore blurred edges, interior linework, and color boundaries uniformly across the flat digital canvas."
    )
    palette_en = analysis.get("palette_en") or "Clean uniform high-contrast spot colors matching the reference artwork."
    damage_en = analysis.get("damage_repair_en") or cond_spec["damage_en"]

    strict_single = bool(analysis.get("strict_single_output", True))
    strict_zero_halluc = bool(analysis.get("strict_zero_hallucination", True))
    treat_text_as_shape = bool(analysis.get("treat_text_as_shape", False))
    license_safe = bool(analysis.get("license_safe", False))

    if bg_b == "B1":
        bg_spec = "Present the final graphic on a solid, pure, unshaded white (#FFFFFF) background with crisp, clean separation from the design elements."
    elif bg_b == "B2":
        bg_spec = "Present the final graphic on a solid, pitch-black (#000000) background with zero ambient light or gray haze."
    else:
        bg_spec = (
            f"Present the final graphic on a perfectly flat, solid, uniform background that is an idealized, "
            f"unshaded version of the original's {bg_desc_en} ({bg_hex}) color, completely stripped of any fabric grain or lighting falloff."
        )

    if scope_a == "A1":
        comp_mod = (
            "Full-Frame Edge-to-Edge Extension: The pattern must extend continuously, evenly, and seamlessly to fill the entire "
            "rectangular canvas from edge to edge (100% full bleed). DO NOT confine the pattern to the shape or silhouette of a garment "
            "or cloth piece. Every repeat unit must maintain identical scale, spacing, and geometric alignment across the entire frame."
        )
    else:
        clipped_sides = tuple(analysis.get("clipped_sides") or ())
        rec_aspect = str(analysis.get("recommended_aspect") or "1:1")
        clip_warning = ""
        if clipped_sides:
            sides_up = ", ".join(s.upper() for s in clipped_sides)
            clip_warning = (
                f" CRITICAL EDGE-CLIPPING ALERT: The reference image is tightly cropped or cut off at the [{sides_up}] edge(s). "
                f"You MUST pull the camera back, reconstruct 100% of any top/bottom headline letters or frame borders that touch the [{sides_up}] edge, "
                f"and ensure nothing touches or gets clipped by the canvas boundary."
            )
        comp_mod = (
            f"Complete Centered Composition & Safe Margin Guard (Recommended Aspect {rec_aspect}): Present the graphic as a complete, "
            f"uncropped, perfectly centered standalone 2D illustration with a generous 10–12% clean solid background breathing margin "
            f"on ALL FOUR SIDES (top, bottom, left, right).{clip_warning}"
        )

    active_text = (
        "[YOUR OWN TEXT]"
        if license_safe
        else (
            detected_text
            if detected_text
            else "all visible headline, sub-headline, slogan, and signature text from the reference"
        )
    )
    if text_c == "C1":
        typo_mod = (
            f"Graphical Text Fidelity: Meticulously recreate the exact shapes, proportions, and construction of the text '{active_text}' "
            f"as an integral graphic element. Treat each letter strictly as a precise visual shape to be drawn in its exact original "
            f"{font_style_en} style, correcting only the fabric warp while preserving the exact letterform geometry."
        )
    elif text_c == "C2":
        typo_mod = (
            f"CRITICAL Text Idealization (0.0° Horizontal Baseline Lock): COMPLETELY DISCARD the warped, curved, or slanted spatial baseline "
            f"of the original text reference. Do not trace the fabric-distorted letters. Instead, re-generate '{active_text}' using a clean, "
            f"geometrically precise {font_style_en} font, aligned on a mathematically straight 0.0° horizontal baseline with 90.0° upright vertical stems "
            f"(curved, drooping, or slanted text baselines are STRICTLY FORBIDDEN)."
        )
    elif text_c == "C4":
        typo_mod = (
            f"CRITICAL Hybrid Geometry Reset (Fold-Distorted / Slanted Text & Frame): COMPLETELY DISCARD the photo's 3D fabric-fold and camera geometry—"
            f"including horizontal foreshortening (narrowed/squished letters), curved or drooping bottom text baselines, tilted letter axes, and skewed bounding boxes. "
            f"Reconstruct '{active_text}' (both top headline and bottom sub-headline/caption lines) and any enclosing frame from scratch on a mathematically "
            f"straight 0.0° horizontal baseline, 90.0° upright vertical letter stems, uniform character widths/kerning, "
            f"and a geometrically level rectangular grid (curved, arched, or slanted baselines are STRICTLY FORBIDDEN)."
        )
    else:
        typo_mod = (
            "No Unwanted Text/Tags: DO NOT include any garment tags, size labels, watermarks, or extraneous text. "
            "The output must contain only the pure graphic artwork."
        )

    tilt_deg = float(analysis.get("tilt_deg") or 0.0)
    keystone_skew_deg = float(analysis.get("keystone_skew_deg") or 0.0)
    has_rect_frame = bool(analysis.get("has_rect_frame", False))
    has_text_bands = bool(analysis.get("has_text_bands", False))
    text_band_tilt_deg = float(analysis.get("text_band_tilt_deg") or 0.0)

    geo_lock_block = ""
    if (
        has_rect_frame
        or has_text_bands
        or abs(tilt_deg) >= 1.2
        or keystone_skew_deg >= 1.8
        or text_c in ("C2", "C4")
        or source_condition in ("worn_garment", "flat_wrinkled")
    ):
        measured_notes: List[str] = []
        if abs(tilt_deg) >= 0.5:
            measured_notes.append(f"frame/axis tilt = {tilt_deg:+.1f}°")
        if keystone_skew_deg >= 0.5:
            measured_notes.append(f"perspective keystone skew = {keystone_skew_deg:.1f}°")
        if has_text_bands:
            measured_notes.append(f"text-band baseline tilt = {text_band_tilt_deg:+.1f}°")
        measured_str = ", ".join(measured_notes) if measured_notes else "3D garment/camera perspective distortion"
        geo_lock_block = (
            "\n\n[STRICT 0°/90° ARCHITECTURAL GEOMETRY & ALIGNMENT LOCK]\n"
            f"- Forensic OpenCV geometry scan measured reference distortions to eliminate: {measured_str}.\n"
            "- ALL horizontal frame lines, corner brackets, inner picture boxes, and top/bottom text baselines MUST be mathematically level "
            "at EXACTLY 0.0° horizontal (100% parallel to the top and bottom canvas edges).\n"
            "- ALL vertical frame lines, side brackets, and upright character stems MUST be mathematically plumb "
            "at EXACTLY 90.0° vertical (100% parallel to the left and right canvas edges), forming true 90.0° right-angle corners.\n"
            "- DO NOT copy the reference photo's tilted brackets, narrow/squeezed right or left edge, trapezoidal perspective, or drooping bottom text curve—"
            "rebuild the entire frame and typography on an exact orthogonal architectural T-square grid."
        )

    extra_recon_lines = [cond_spec["unwrap_en"]]
    if strict_zero_halluc:
        extra_recon_lines.append(
            "Strict Zero-Hallucination Rule: Maintain 100% fidelity to the original motif shapes, counts, and proportions—do not hallucinate "
            "unrequested decorative elements and do not simplify intricate details."
        )
    extra_recon_lines.append(comp_mod)
    extra_recon_lines.append(typo_mod)
    if treat_text_as_shape:
        extra_recon_lines.append(
            "Treat unclear or stylized lettering strictly as geometric shapes to be drawn, NOT as text to be guessed or replaced with new words."
        )
    if license_safe:
        extra_recon_lines.append(
            "License-Safe Original Adaptation: Strip 100% of watermarks, stock-site marks, third-party brand logos, and copyrighted characters. "
            "Preserve ONLY the artistic drawing technique, color palette, and compositional balance while cleanly redesigning any trademarked "
            "character/emblem and replacing brand words with '[YOUR OWN TEXT]'."
        )

    prohibit_lines = [
        "- DO NOT include any t-shirt, dress, garment silhouette, collar, sleeves, seams, stitching, or mannequin/human body parts.",
        "- DO NOT include any fabric weave, canvas grain, knit texture, cotton fibers, or surface roughness.",
    ]
    if cond_spec["prohibit_extra"]:
        prohibit_lines.append(cond_spec["prohibit_extra"].rstrip("\n"))
    prohibit_lines.extend([
        "- DO NOT include any wrinkles, folds, creases, drapery, 3D surface curvature, or perspective distortion.",
        "- DO NOT include any shadows, studio lighting gradients, specular highlights, reflections, glare, or vignette.",
    ])
    if strict_single:
        prohibit_lines.append(
            "- ABSOLUTELY NO split panels, NO before/after comparison, NO multiple variants, NO mockups, NO separated layer sheets — "
            "output strictly ONE standalone flat 2D graphic image."
        )

    if license_safe:
        text_clause = (
            f", replacing the text with '{detected_text}'"
            if detected_text
            else ", adjusting any brand wording into clean generic lettering"
        )
        if bg_b == "B1":
            bg_natural = "solid pure white (#FFFFFF)"
        elif bg_b == "B2":
            bg_natural = "solid pure black (#000000)"
        else:
            bg_natural = f"solid flat {bg_desc_en} ({bg_hex})"

        if source_condition == "coated_textured":
            surface_clause = (
                "remove the leather or coated canvas texture, shadows, and surface grain completely "
                "so only the clean flat 2D graphic remains"
            )
        else:
            surface_clause = (
                "remove the t-shirt, fabric folds, shadows, and perspective completely "
                "so only the clean flat 2D graphic remains"
            )

        clipped_sides = tuple(analysis.get("clipped_sides") or ())
        rec_aspect = str(analysis.get("recommended_aspect") or "1:1")
        clip_natural = (
            f", leaving extra space at the {', '.join(clipped_sides)} edge so nothing is cut off"
            if clipped_sides
            else ""
        )
        if scope_a == "A1":
            scope_short = "Extend the pattern evenly edge-to-edge across the entire canvas."
        else:
            scope_short = (
                f"Center the complete uncropped graphic ({rec_aspect} framing) with clean 10-12% "
                f"background margin on all four sides{clip_natural}."
            )

        style_short_map = {
            "flat_vector": " Keep the finish as clean flat solid-color vector art with sharp edges.",
            "stipple_dot": " Preserve clean stipple dot-matrix and halftone shading details.",
            "painterly_2d": " Keep crisp 2D editorial illustration shading and clean outlines.",
        }
        style_short = style_short_map.get(render_style, "")

        geom_short = (
            " Keep all borders, brackets, and text baselines straight and level "
            "(0° horizontal, 90° vertical) with no slant or curve."
        )
        extra_note_clean = str(analysis.get("extra_note") or "").strip()
        extra_short = f" Extra adjustment: {extra_note_clean}." if extra_note_clean else ""

        return (
            "First, make minimal subtle adjustments to the graphic in this image (modifying only brand logos, words, "
            "and subtle character details just enough to be an original, non-infringing version while keeping the same "
            f"visual layout, pose, drawing style, and color energy{text_clause}).\n\n"
            "Then, recreate this adjusted graphic from scratch as a crystal-clear, sharp, 4K orthographically flat 2D "
            f"standalone print artwork on an idealized taut digital canvas with a {bg_natural} background — {surface_clause}. "
            f"{scope_short}{geom_short}{style_short} Use clean uniform colors ({palette_en}).{extra_short}"
        )

    return (
        "[SYSTEM DIRECTIVE: ABSOLUTE RECREATION ON A PERFECT SURFACE]\n"
        "DO NOT edit, filter, retouch, or iron the original photograph. Treat the input photo strictly as a forensic structural "
        "reference blueprint, and apply Stretched Canvas / Smart UV Unwrap to reconstruct the original pre-print 2D digital artwork "
        "from scratch in strict 2D orthographic projection on an idealized, perfectly taut digital surface.\n\n"
        "[MAIN TASK: ABSOLUTE 2D GRAPHIC RECREATION]\n"
        f"From the provided reference image, recreate the {subject_en} as an ultra-high-resolution (4K, 3840px+), crystal-clear, "
        "geometrically perfect, orthographically flat, full-frame 2D digital production artwork.\n\n"
        "[MICRO-DETAIL & ARCHITECTURAL SPECIFICATION]\n"
        "Meticulously analyze and faithfully replicate every single detail of the design without simplification or omission:\n"
        f"- Motif & Geometry: {motif_en}\n"
        f"- Fine Details & Render Technique: {details_en} {style_block}\n"
        f"- Idealized Color Palette: Reconstruct using clean, solid, uniform, high-contrast spot colors completely free of photographic noise or lighting shifts: {palette_en}\n"
        f"- Background Specification: {bg_spec}\n\n"
        "[FIDELITY, GEOMETRIC RESET & RECONSTRUCTION PROTOCOL]\n"
        f"Systematically strip and mathematically flatten all {damage_en}. "
        "Intelligently regenerate any missing, occluded, cropped, or folded-over sections in the exact original artistic style and geometric rhythm.\n"
        + "\n".join(extra_recon_lines)
        + geo_lock_block
        + "\n\n[CRITICAL PROHIBITION PROTOCOL - HIGHEST PRIORITY]\n"
        "CRITICAL: Present the final output strictly as a standalone, pure 2D flat graphic artwork.\n"
        + "\n".join(prohibit_lines)
        + "\nEvery color region must have uniform luminance and flat, print-ready consistency across the entire canvas."
    )


def analyze_and_build_repair_prompt(
    image_path: str,
    overrides: Optional[dict] = None,
    api_key: str = "",
    rectify_out_dir: str = "",
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **_kwargs,
) -> RepairPromptResult:
    """Tek veya cift kademeli (Yerel Python + opsiyonel Gemini Vision) analiz ile MeChi 3.3 Sifirdan Yeniden Yaratim promptu uretir."""
    overrides = overrides or {}
    if cancel_event and cancel_event.is_set():
        raise TaskCancelled()

    if progress_cb:
        progress_cb(20, "Python adli piksel geometrisi, eğim, metin bandı ve spot renk paleti ölçülüyor...")
    metrics = analyze_image_local(image_path)

    if cancel_event and cancel_event.is_set():
        raise TaskCancelled()

    rectified_ref_path = ""
    if bool(overrides.get("pre_rectify", False)) and rectify_out_dir:
        try:
            if progress_cb:
                progress_cb(35, "Referans görsel geometrik ön-düzeltmeden (Deskew + %12 kenar boşluğu) geçiriliyor...")
            stem = os.path.splitext(os.path.basename(image_path))[0]
            cand_rect = os.path.join(rectify_out_dir, f"{stem}_on_duzeltilmis_ref.png")
            rectified_ref_path = pre_rectify_reference_image(image_path, cand_rect, metrics, pad_ratio=0.12)
        except Exception:
            rectified_ref_path = ""

    raw_cond = str(overrides.get("source_condition") or "auto")
    user_locked_cond = raw_cond in SOURCE_CONDITIONS
    if user_locked_cond:
        source_condition = raw_cond
    else:
        if metrics.lighting_unevenness >= 11.0 or metrics.keystone_skew_deg >= 2.0:
            source_condition = "worn_garment"
        elif metrics.has_fabric_grain and metrics.flatness_ratio < 0.72:
            source_condition = "flat_wrinkled"
        elif metrics.is_low_res or metrics.blur_score < 120.0:
            source_condition = "lowres_damaged"
        else:
            source_condition = "worn_garment"

    render_style = str(overrides.get("render_style") or "auto")
    if render_style not in RENDER_STYLES:
        render_style = "auto"

    cond_spec = SOURCE_CONDITIONS[source_condition]
    strict_single = bool(overrides.get("strict_single_output", True))
    strict_zero_halluc = bool(overrides.get("strict_zero_hallucination", True))
    treat_text_as_shape = bool(overrides.get("treat_text_as_shape", False))
    license_safe = bool(overrides.get("license_safe", False))

    is_folded_or_tilted = (
        source_condition in ("worn_garment", "flat_wrinkled")
        or metrics.lighting_unevenness >= 11.0
        or abs(metrics.tilt_deg) >= 1.2
        or abs(metrics.text_band_tilt_deg) >= 1.2
        or metrics.keystone_skew_deg >= 1.8
    )
    auto_text_c = "C3"
    if overrides.get("custom_text") or metrics.has_text_bands:
        auto_text_c = "C4" if is_folded_or_tilted else "C2"

    analysis_mode = "local_python"
    fallback_note = ""
    pal_en = ", ".join(f"{p['name_en']} ({p['hex']})" for p in metrics.palette)
    pal_tr = ", ".join(f"{p['name_tr']} ({p['hex']})" for p in metrics.palette)
    grain_note = "Kumaş dokusu tespit edildi" if metrics.has_fabric_grain else "Düşük doku gürültüsü"
    geo_tr = " · ".join(metrics.geometry_issues_tr) if metrics.geometry_issues_tr else "Düzgün eksen"

    analysis: dict = {
        "scope_a": metrics.suggested_scope,
        "bg_b": metrics.suggested_bg_mode,
        "bg_hex": metrics.bg_hex,
        "bg_desc_en": "idealized flat tone",
        "text_c": auto_text_c,
        "detected_text": (overrides.get("custom_text") or "").strip(),
        "font_style_en": "clean geometric uppercase sans-serif / editorial display lettering",
        "style_technique_en": RENDER_STYLES[metrics.suggested_render_style]["directive_en"],
        "general_subject_en": "printed textile graphic illustration shown in the reference",
        "motif_geometry_en": (
            f"Faithfully reconstruct every motif, figure, contour, frame bracket, and geometric proportion visible in the reference "
            f"({metrics.width}x{metrics.height} px source) locked strictly to 0.0° horizontal and 90.0° vertical axes. "
            + (f"Extra structural note: {overrides['extra_note']}" if overrides.get("extra_note") else "")
        ).strip(),
        "fine_details_en": "Restore blurred edges, halftone/stipple dot clarity, and crisp linework uniformly across the canvas.",
        "damage_repair_en": cond_spec["damage_en"],
        "palette_en": pal_en,
        "summary_tr": (
            f"Yerel analiz: {metrics.width}x{metrics.height} px ({metrics.blur_label_tr}, {metrics.lighting_label_tr}, {grain_note}). "
            f"Geometri: {geo_tr} (Önerilen oran: {metrics.recommended_aspect}). "
            f"Kenar uyumu %{metrics.border_uniform_ratio * 100:.0f} → {'Tekil Baskı (A2)' if metrics.suggested_scope == 'A2' else 'Tam Tuval Metraj (A1)'}. "
            f"Algılanan palet: {pal_tr}."
        ),
    }

    if api_key and api_key.strip():
        try:
            if progress_cb:
                progress_cb(55, "Gemini Vision ile 7 maddelik MeChi 3.3 adli analizi yapılıyor...")
            vis = analyze_with_gemini_vision(image_path, metrics, api_key=api_key, overrides=overrides)
            if isinstance(vis, dict) and vis.get("general_subject_en"):
                analysis.update(vis)
                analysis_mode = "gemini_vision"
                if not user_locked_cond and vis.get("source_condition") in SOURCE_CONDITIONS:
                    source_condition = str(vis["source_condition"])
                geom_def = str(vis.get("geometric_defects_en") or "").strip()
                if geom_def and geom_def.lower() != "none":
                    analysis["damage_repair_en"] = f"{analysis['damage_repair_en']} Geometric defects to reset: {geom_def}"
                is_folded_or_tilted = (
                    source_condition in ("worn_garment", "flat_wrinkled")
                    or metrics.lighting_unevenness >= 11.0
                    or abs(metrics.tilt_deg) >= 1.2
                    or abs(metrics.text_band_tilt_deg) >= 1.2
                    or metrics.keystone_skew_deg >= 1.8
                )
        except Exception as exc:
            fallback_note = f"Gemini Vision bağlantısı sağlanamadı, yerel Python analizi kullanıldı ({exc})."

    # Kullanici secimleri veya koruma kurallari uygula
    if overrides.get("scope") in ("A1", "A2"):
        analysis["scope_a"] = overrides["scope"]
    if overrides.get("bg") in ("B1", "B2", "B3"):
        analysis["bg_b"] = overrides["bg"]
        if overrides["bg"] == "B1":
            analysis["bg_hex"] = "#FFFFFF"
        elif overrides["bg"] == "B2":
            analysis["bg_hex"] = "#000000"
    if overrides.get("text_mode") in ("C1", "C2", "C3", "C4"):
        analysis["text_c"] = overrides["text_mode"]
    else:
        if analysis.get("text_c") == "C3" and (metrics.has_text_bands or overrides.get("custom_text")):
            analysis["text_c"] = "C4" if is_folded_or_tilted else "C2"
        elif analysis.get("text_c") == "C1" and render_style != "stipple_dot" and is_folded_or_tilted:
            analysis["text_c"] = "C4"

    if overrides.get("custom_text"):
        analysis["detected_text"] = overrides["custom_text"].strip()
    if overrides.get("extra_note"):
        analysis["extra_note"] = overrides["extra_note"].strip()
        analysis["motif_geometry_en"] = f"{analysis['motif_geometry_en']} Focus: {overrides['extra_note']}"
    else:
        analysis["extra_note"] = ""

    analysis["source_condition"] = source_condition
    analysis["render_style"] = render_style if render_style != "auto" else (
        "auto" if analysis_mode == "gemini_vision" and analysis.get("style_technique_en") else metrics.suggested_render_style
    )
    analysis["strict_single_output"] = strict_single
    analysis["strict_zero_hallucination"] = strict_zero_halluc
    analysis["treat_text_as_shape"] = treat_text_as_shape
    analysis["license_safe"] = license_safe
    analysis["tilt_deg"] = metrics.tilt_deg
    analysis["keystone_skew_deg"] = metrics.keystone_skew_deg
    analysis["has_rect_frame"] = metrics.has_rect_frame
    analysis["has_text_bands"] = metrics.has_text_bands
    analysis["text_band_tilt_deg"] = metrics.text_band_tilt_deg
    analysis["clipped_sides"] = metrics.clipped_sides
    analysis["recommended_aspect"] = metrics.recommended_aspect

    if progress_cb:
        progress_cb(90, "MeChi 3.3 sıfırdan 2D yeniden yaratım promptu derleniyor...")

    full_prompt = build_mechi33_prompt(analysis)
    detected_mods = f"{analysis['scope_a']} / {analysis['bg_b']} / {analysis['text_c']}"
    if license_safe:
        detected_mods += " / TELİFSİZ"
        analysis["summary_tr"] = (
            str(analysis.get("summary_tr") or "").rstrip()
            + " | Telifsiz Özgün Dönüşüm (2 Aşamalı): Önce karakter/marka öğeleri telifsiz özgünleştirilir, ardından tişörtsüz düz 2D 4K baskı üretilir."
        )

    if progress_cb:
        progress_cb(100, "Sıfırdan 2D yeniden yaratım promptu hazır.")

    active_sys_inst = MECHI33_LICENSE_SAFE_SYSTEM_INSTRUCTION if license_safe else MECHI33_SYSTEM_INSTRUCTION

    return RepairPromptResult(
        analysis_mode=analysis_mode,
        summary_tr=str(analysis.get("summary_tr") or ""),
        detected_modules=detected_mods,
        palette_hexes=[p["hex"] for p in metrics.palette],
        full_prompt=full_prompt,
        metrics=metrics,
        fallback_note=fallback_note,
        system_instruction=active_sys_inst,
        rectified_ref_path=rectified_ref_path,
    )
