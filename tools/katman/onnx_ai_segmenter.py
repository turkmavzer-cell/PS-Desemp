"""MeChi Studio — %100 Yerel, Ucretsiz, PyTorch'suz ONNX Yapay Zeka Segmentasyon Motoru.

Bilesenler:
  1) RapidOCR (ONNX)          : Metin poligonlarini bulur (Yazilar).
  2) Grounding DINO Tiny INT8 : Acik-sozluk semantik nesne tespiti yapar.
  3) SlimSAM-77 (ONNX)        : Her tespit kutusunu piksel hassasiyetinde keser.

Tum model agirliklari yalnizca %LOCALAPPDATA%/MeChiStudio/models/ altinda saklanir.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# HuggingFace uzerinden indirilecek hafif ONNX dosyalari (~233 MB toplam)
ONNX_MODEL_SPECS: Dict[str, str] = {
    "grounding_dino/model_quantized.onnx": "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_quantized.onnx",
    "grounding_dino/tokenizer.json": "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/tokenizer.json",
    "grounding_dino/config.json": "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/config.json",
    "grounding_dino/preprocessor_config.json": "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/preprocessor_config.json",
    "slimsam/vision_encoder.onnx": "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/vision_encoder.onnx",
    "slimsam/prompt_encoder_mask_decoder.onnx": "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
    "slimsam/preprocessor_config.json": "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/preprocessor_config.json",
}

# Z-sirasi yuksek = en onde. Her grup birden fazla ornegi tek temiz layer'da birlestirir.
SEMANTIC_GROUPS: List[Tuple[str, int, Tuple[str, ...]]] = [
    ("Sapka_ve_Aksesuar", 88, ("baseball cap", "hat", "cap", "crown", "glasses", "sunglasses", "diamond", "gem", "tennis racket")),
    ("Kuslar", 86, ("white bird", "bird", "dove", "swan", "eagle", "owl", "flamingo")),
    ("Hayvanlar", 80, ("monkey", "ape", "gorilla", "animal", "mascot", "creature", "lion", "tiger", "bear", "wolf", "cat", "dog", "horse", "butterfly", "skull", "snake", "dragon")),
    ("Insan_ve_Karakter", 76, ("cartoon character", "character", "person", "figure", "statue", "bust")),
    ("Cicekler", 70, ("flower", "lotus", "rose", "blossom")),
    ("Bitkiler", 65, ("palm tree", "palm leaf", "olive branch", "leaf", "tree", "plant")),
    ("Arac_ve_Spor", 55, ("car", "airplane", "boat", "tennis court")),
    ("Mimari_ve_Sahne", 40, ("umbrella", "dome", "building", "arch", "pyramid", "sun")),
    ("Cerceve_ve_Kaide", 25, ("frame", "border", "bracket", "medallion", "emblem", "pedestal", "checkerboard")),
]

# Kullanici arayuzden Turkce ozel obje adi yazarsa Ingilizce terimlere cevirir
TR_TO_EN_HINTS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "yazi": ("Yazilar", ("text", "title", "letters")),
    "yazilar": ("Yazilar", ("text", "title", "letters")),
    "kus": ("Kuslar", ("bird", "dove", "swan", "pigeon", "eagle", "owl")),
    "kuslar": ("Kuslar", ("bird", "dove", "swan", "pigeon", "eagle", "owl")),
    "guvercin": ("Kuslar", ("dove", "bird", "pigeon")),
    "kugu": ("Kuslar", ("swan", "bird")),
    "hayvan": ("Hayvanlar", ("animal", "monkey", "lion", "tiger", "cat", "dog", "horse", "butterfly")),
    "hayvanlar": ("Hayvanlar", ("animal", "monkey", "lion", "tiger", "cat", "dog", "horse", "butterfly")),
    "maymun": ("Hayvanlar", ("monkey", "ape", "cartoon character")),
    "karakter": ("Insan_ve_Karakter", ("cartoon character", "character", "figure", "person")),
    "maskot": ("Hayvanlar", ("mascot", "monkey", "cartoon character", "animal")),
    "sicrama": ("Arka_Efekt", ("paint splash", "ink splatter", "splash")),
    "efekt": ("Arka_Efekt", ("paint splash", "splatter", "glow", "effect")),
    "cicek": ("Cicekler", ("flower", "lotus", "rose", "blossom")),
    "cicekler": ("Cicekler", ("flower", "lotus", "rose", "blossom")),
    "bitki": ("Bitkiler", ("palm tree", "palm leaf", "tree", "leaf", "foliage", "plant", "branch")),
    "bitkiler": ("Bitkiler", ("palm tree", "palm leaf", "tree", "leaf", "foliage", "plant", "branch")),
    "palmiye": ("Bitkiler", ("palm tree", "palm leaf", "tree")),
    "yaprak": ("Bitkiler", ("leaf", "palm leaf", "foliage", "branch")),
    "sapka": ("Sapka_ve_Aksesuar", ("baseball cap", "hat", "cap")),
    "elmas": ("Sapka_ve_Aksesuar", ("diamond", "gem")),
    "mucevher": ("Sapka_ve_Aksesuar", ("diamond", "gem")),
    "tac": ("Sapka_ve_Aksesuar", ("crown",)),
    "gozluk": ("Sapka_ve_Aksesuar", ("glasses", "sunglasses")),
    "raket": ("Sapka_ve_Aksesuar", ("tennis racket", "racket")),
    "insan": ("Insan_ve_Karakter", ("person", "statue", "bust")),
    "heykel": ("Insan_ve_Karakter", ("statue", "bust", "person")),
    "araba": ("Arac_ve_Spor", ("car",)),
    "ucak": ("Arac_ve_Spor", ("airplane",)),
    "gemi": ("Arac_ve_Spor", ("boat", "ship")),
    "tekne": ("Arac_ve_Spor", ("boat",)),
    "tenis": ("Arac_ve_Spor", ("tennis court", "tennis racket")),
    "kort": ("Arac_ve_Spor", ("tennis court",)),
    "bina": ("Mimari_ve_Sahne", ("building", "dome", "palace", "arch")),
    "semsiye": ("Mimari_ve_Sahne", ("umbrella", "parasol")),
    "kubbe": ("Mimari_ve_Sahne", ("dome", "building")),
    "kemer": ("Mimari_ve_Sahne", ("arch",)),
    "gunes": ("Mimari_ve_Sahne", ("sun",)),
    "amblem": ("Cerceve_ve_Kaide", ("medallion", "emblem")),
    "madalyon": ("Cerceve_ve_Kaide", ("medallion", "emblem")),
    "kaide": ("Cerceve_ve_Kaide", ("pedestal",)),
    "cerceve": ("Cerceve_ve_Kaide", ("frame", "border", "bracket")),
}

_SESSION_CACHE: Dict[str, Any] = {}


def _tr_ascii(text: str, fallback: str = "Katman") -> str:
    clean = (text or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean).strip("_.")
    return clean or fallback


def get_models_dir() -> str:
    """Model agirliklarinin repoya girmeden tutuldugu yerel AppData dizinini dondurur."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
    path = os.path.join(base, "MeChiStudio", "models")
    os.makedirs(path, exist_ok=True)
    return path


def is_local_onnx_ai_available() -> Dict[str, Any]:
    """Yerel ONNX paketlerinin ve indirilmis modellerin hazirlik durumunu dondurur."""
    pkgs = {}
    for mod in ("onnxruntime", "tokenizers", "rapidocr_onnxruntime"):
        try:
            __import__(mod)
            pkgs[mod] = True
        except Exception:
            pkgs[mod] = False

    mdir = get_models_dir()
    missing_files = []
    for rel in ONNX_MODEL_SPECS:
        p = os.path.join(mdir, rel.replace("/", os.sep))
        if not os.path.exists(p) or os.path.getsize(p) <= 0:
            missing_files.append(rel)

    packages_ok = all(pkgs.values())
    models_ok = len(missing_files) == 0
    return {
        "ready": bool(packages_ok and models_ok),
        "packages_ok": packages_ok,
        "models_ok": models_ok,
        "packages": pkgs,
        "models_dir": mdir,
        "missing_files": missing_files,
    }


def ensure_onnx_models(log_cb: Optional[Callable[[str], None]] = None) -> str:
    """Eksik ONNX dosyalarini atomic (.tmp -> rename) sekilde %LOCALAPPDATA%/MeChiStudio/models altina indirir."""
    mdir = get_models_dir()
    for rel, url in ONNX_MODEL_SPECS.items():
        dst = os.path.join(mdir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        if log_cb:
            log_cb(f"Yerel AI modeli indiriliyor (tek seferlik): {rel}")
        tmp = dst + ".tmp"
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fp:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fp.write(chunk)
        os.replace(tmp, dst)
    return mdir


def parse_custom_labels(raw_text: str) -> Tuple[List[Tuple[str, int, Tuple[str, ...]]], bool]:
    """Kullanicinin yazdigi ozel obje/grup listesini cozer.
    Dondurur: ([(group_name_ascii, z_order, (en_terms...))], include_text).
    """
    raw = (raw_text or "").strip()
    if not raw:
        return list(SEMANTIC_GROUPS), True

    tokens = [t.strip() for t in re.split(r"[,;\n]+", raw) if t.strip()]
    if not tokens:
        return list(SEMANTIC_GROUPS), True

    groups_map: Dict[str, List[str]] = {}
    z_map: Dict[str, int] = {}
    include_text = False

    # Onceden tanimli grup Z degerleri
    default_z = {name: z for name, z, _ in SEMANTIC_GROUPS}

    for idx, tok in enumerate(tokens):
        slug = _tr_ascii(tok).lower()
        if slug in ("yazi", "yazilar", "text", "metin"):
            include_text = True
            continue
        if slug in TR_TO_EN_HINTS:
            gname, en_terms = TR_TO_EN_HINTS[slug]
            groups_map.setdefault(gname, [])
            for et in en_terms:
                if et not in groups_map[gname]:
                    groups_map[gname].append(et)
            z_map.setdefault(gname, default_z.get(gname, 88 - idx))
        else:
            gname = _tr_ascii(tok.title(), f"Ozel_{idx + 1}")
            en_term = tok.strip().lower()
            groups_map.setdefault(gname, [])
            if en_term not in groups_map[gname]:
                groups_map[gname].append(en_term)
            z_map.setdefault(gname, 88 - idx)

    out = [(g, z_map[g], tuple(terms)) for g, terms in groups_map.items()]
    return out, include_text


# ---------------------------------------------------------------------------
# Oturum Yonetimi
# ---------------------------------------------------------------------------

def _get_ocr_engine():
    if "ocr" not in _SESSION_CACHE:
        from rapidocr_onnxruntime import RapidOCR

        _SESSION_CACHE["ocr"] = RapidOCR()
    return _SESSION_CACHE["ocr"]


def _get_grounding_dino(mdir: str):
    if "gdino" not in _SESSION_CACHE:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(mdir, "grounding_dino", "model_quantized.onnx")
        tok_path = os.path.join(mdir, "grounding_dino", "tokenizer.json")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(tok_path)
        _SESSION_CACHE["gdino"] = (sess, tok)
    return _SESSION_CACHE["gdino"]


def _get_slimsam(mdir: str):
    if "slimsam" not in _SESSION_CACHE:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        vis = ort.InferenceSession(
            os.path.join(mdir, "slimsam", "vision_encoder.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        dec = ort.InferenceSession(
            os.path.join(mdir, "slimsam", "prompt_encoder_mask_decoder.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _SESSION_CACHE["slimsam"] = (vis, dec)
    return _SESSION_CACHE["slimsam"]


# ---------------------------------------------------------------------------
# 1. RapidOCR Yazi Motoru
# ---------------------------------------------------------------------------

def detect_text_polygons_rapidocr(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    score_thresh: float = 0.55,
) -> np.ndarray:
    """RapidOCR ile metin poligonlarini bulur ve harf ic bosluklarini (O, A, B, R) koruyarak cikarir."""
    H, W = rgb.shape[:2]
    out_mask = np.zeros((H, W), dtype=bool)
    ocr = _get_ocr_engine()
    res, _ = ocr(rgb)
    if not res:
        return out_mask

    lab_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for item in res:
        if len(item) < 3:
            continue
        pts_raw, _txt, conf = item[0], item[1], float(item[2])
        if conf < score_thresh:
            continue
        pts = np.array(pts_raw, dtype=np.int32)
        if pts.shape != (4, 2):
            continue

        xs, ys = pts[:, 0], pts[:, 1]
        box_h = max(4, int(ys.max() - ys.min()))
        pad_x = max(3, int(round(box_h * 0.10)))
        pad_y = max(2, int(round(box_h * 0.05)))
        x0 = max(0, int(xs.min()) - pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        x1 = min(W, int(xs.max()) + pad_x)
        y1 = min(H, int(ys.max()) + pad_y)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue

        # Yalnizca poligon ici (+ hafif pad) bolgede cevre kontrastindan harfleri al
        poly_local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = pts - np.array([[x0, y0]], dtype=np.int32)
        cv2.fillPoly(poly_local, [shifted], 1)
        kx = max(3, (pad_x // 2) * 2 + 1)
        ky = max(3, (pad_y // 2) * 2 + 1)
        poly_dil = cv2.dilate(poly_local, np.ones((ky, kx), np.uint8), iterations=1) > 0

        roi_lab = lab_full[y0:y1, x0:x1]
        roi_fg = fg_mask[y0:y1, x0:x1]

        # Poligon dis halkasi cevre rengini verir (orn. pankartin krem arka plani)
        ring = poly_dil & (poly_local == 0)
        if int(ring.sum()) < 10:
            ring = ~poly_local.astype(bool)
        if not np.any(ring):
            ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
        env_lab = np.median(roi_lab[ring], axis=0)
        de = np.sqrt(((roi_lab - env_lab[None, None, :]) ** 2).sum(axis=2))
        max_de = float(de[poly_dil].max()) if np.any(poly_dil) else 0.0
        if max_de < 15.0:
            continue

        de_u8 = np.clip(de * (255.0 / max_de), 0, 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(de_u8[poly_dil], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(20.0, min(55.0, float(otsu_t) * (max_de / 255.0) * 0.72))
        letters = (de > thresh_val) & poly_dil & roi_fg

        # Poligon disina sizan alakasiz cerceve/cizgi bilesenlerini ele (>=%35'i orijinal poligonda olmali)
        if np.any(letters):
            n_lc, lc_lab, lc_st, _ = cv2.connectedComponentsWithStats(letters.astype(np.uint8), connectivity=8)
            for lcid in range(1, n_lc):
                l_area = int(lc_st[lcid, cv2.CC_STAT_AREA])
                if l_area <= 0:
                    continue
                in_poly = int(((lc_lab == lcid) & (poly_local > 0)).sum())
                if (in_poly / float(l_area)) < 0.35:
                    letters[lc_lab == lcid] = False

        out_mask[y0:y1, x0:x1] |= letters

    return out_mask


# ---------------------------------------------------------------------------
# 2. Grounding DINO ONNX Acik-Sozluk Tespiti
# ---------------------------------------------------------------------------

def _box_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _nms_group_boxes(dets: List[dict], iou_thresh: float = 0.50) -> List[dict]:
    ordered = sorted(dets, key=lambda d: -d["score"])
    kept: List[dict] = []
    for d in ordered:
        same_group_clash = any(
            (k["group"] == d["group"] and _box_iou_xyxy(k["box"], d["box"]) > iou_thresh)
            for k in kept
        )
        if not same_group_clash:
            kept.append(d)
    return kept


def _prepare_gdino_image(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """800x800 ImageNet normalize CHW float32 + [1,800,800] pixel_mask."""
    resized = np.array(Image.fromarray(rgb, mode="RGB").resize((800, 800), Image.BILINEAR), dtype=np.float32)
    norm = ((resized / 255.0) - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    chw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
    pmask = np.ones((1, 800, 800), dtype=np.int64)
    return chw, pmask


def detect_semantic_boxes_grounding_dino(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    groups: List[Tuple[str, int, Tuple[str, ...]]],
    mdir: str,
    box_threshold: float = 0.19,
    text_threshold: float = 0.14,
    log_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Grounding DINO Tiny INT8 ile semantik gruplara gore obje kutulari tespit eder."""
    H, W = rgb.shape[:2]
    fg_area = max(1, int(fg_mask.sum()))
    sess, tok = _get_grounding_dino(mdir)
    pixel_values, pixel_mask = _prepare_gdino_image(rgb)

    # Uzun sorguda token karismasini onlemek icin her seferde ~12-16 terimlik odakli partiler calistir
    batches: List[List[Tuple[str, int, Tuple[str, ...]]]] = []
    cur_batch: List[Tuple[str, int, Tuple[str, ...]]] = []
    cur_terms = 0
    for grp in groups:
        t_cnt = len(grp[2])
        if cur_batch and cur_terms + t_cnt > 18:
            batches.append(cur_batch)
            cur_batch = []
            cur_terms = 0
        cur_batch.append(grp)
        cur_terms += t_cnt
    if cur_batch:
        batches.append(cur_batch)

    raw_dets: List[dict] = []

    for b_idx, batch in enumerate(batches, start=1):
        # Her terimin karakter araligini ve bagli oldugu grubu kaydet
        term_spans: List[Tuple[int, int, str, int, str]] = []  # (c0, c1, group, z, term)
        parts: List[str] = []
        cursor = 0
        for gname, z_order, terms in batch:
            for term in terms:
                clean_t = term.strip().lower()
                if not clean_t:
                    continue
                if parts:
                    cursor += 3  # " . "
                c0 = cursor
                c1 = cursor + len(clean_t)
                parts.append(clean_t)
                term_spans.append((c0, c1, gname, z_order, clean_t))
                cursor = c1
        if not parts:
            continue

        prompt_text = " . ".join(parts) + " ."
        enc = tok.encode(prompt_text)
        seq_len = min(len(enc.ids), 255)
        input_ids = np.array([enc.ids[:seq_len]], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        attention_mask = np.array([enc.attention_mask[:seq_len]], dtype=np.int64)

        # Her terimi encoding icindeki token indekslerine esle
        term_tok_indices: List[Tuple[List[int], str, int, str]] = []
        offsets = enc.offsets[:seq_len]
        for c0, c1, gname, z_order, clean_t in term_spans:
            t_idxs = [
                ti for ti, (oc0, oc1) in enumerate(offsets)
                if oc1 > oc0 and max(c0, oc0) < min(c1, oc1)
            ]
            if t_idxs:
                term_tok_indices.append((t_idxs, gname, z_order, clean_t))

        if log_cb:
            log_cb(f"Grounding DINO parti {b_idx}/{len(batches)} taranıyor...")

        logits, pred_boxes = sess.run(
            None,
            {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "pixel_mask": pixel_mask,
            },
        )
        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[0, :, :seq_len], -40.0, 40.0)))
        boxes = pred_boxes[0]  # [900, 4] (cx, cy, w, h) 0..1

        all_term_toks = [ti for t_idxs, *_ in term_tok_indices for ti in t_idxs]
        if not all_term_toks:
            continue

        for q in range(probs.shape[0]):
            q_max = float(probs[q, all_term_toks].max())
            if q_max < box_threshold:
                continue

            # En yuksek skorlu terimi bul
            best_score = -1.0
            best_info: Optional[Tuple[str, int, str]] = None
            for t_idxs, gname, z_order, clean_t in term_tok_indices:
                sc = float(probs[q, t_idxs].mean())
                if sc > best_score:
                    best_score = sc
                    best_info = (gname, z_order, clean_t)

            if best_info is None or best_score < text_threshold:
                continue

            cx, cy, bw, bh = [float(v) for v in boxes[q]]
            x0 = max(0, int(round((cx - bw * 0.5) * W)))
            y0 = max(0, int(round((cy - bh * 0.5) * H)))
            x1 = min(W, int(round((cx + bw * 0.5) * W)))
            y1 = min(H, int(round((cy + bh * 0.5) * H)))
            if x1 - x0 < 10 or y1 - y0 < 10:
                continue

            # Dev yanlis kutu korumasi: yalnizca tum tuvali/panoyu yutan sahte kutulari ele; izole ana karakterleri koru
            box_area = (x1 - x0) * (y1 - y0)
            canvas_ratio = box_area / float(max(1, W * H))
            fg_ratio = fg_area / float(max(1, W * H))
            if canvas_ratio > 0.85 or (fg_ratio > 0.70 and box_area > int(fg_area * 0.88)):
                continue

            raw_dets.append(
                {
                    "group": best_info[0],
                    "z_order": best_info[1],
                    "label": best_info[2],
                    "score": max(q_max, best_score),
                    "box": (x0, y0, x1, y1),
                }
            )

    return _nms_group_boxes(raw_dets, iou_thresh=0.50)


# ---------------------------------------------------------------------------
# 3. SlimSAM-77 ONNX Piksel Segmentasyonu
# ---------------------------------------------------------------------------

def _prepare_slimsam_image(rgb: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """Uzun kenari 1024'e olcekler, ImageNet normalize eder ve 1024x1024 sag/alt pad uygular."""
    H, W = rgb.shape[:2]
    scale = 1024.0 / float(max(H, W))
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    resized = np.array(Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    norm = ((resized / 255.0) - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    padded = np.zeros((1024, 1024, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = norm
    chw = np.transpose(padded, (2, 0, 1))[None, ...].astype(np.float32)
    return chw, scale, new_w, new_h


def segment_boxes_slimsam(
    rgb: np.ndarray,
    boxes_px: List[Tuple[int, int, int, int]],
    fg_mask: np.ndarray,
    mdir: str,
) -> List[np.ndarray]:
    """SlimSAM ile her kutu icin kesiksiz piksel maskesi uretir (encoder goruntu basina 1 kez)."""
    if not boxes_px:
        return []

    H, W = rgb.shape[:2]
    vis, dec = _get_slimsam(mdir)
    pixel_values, scale, new_w, new_h = _prepare_slimsam_image(rgb)
    img_emb, img_pos = vis.run(None, {"pixel_values": pixel_values})

    masks_out: List[np.ndarray] = []
    for x0, y0, x1, y1 in boxes_px:
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        # SAM kutu + merkez pozitif nokta promptu: [2=sol-ust, 3=sag-alt, 1=merkez on-plan]
        # Merkez nokta kutu disindaki arka plan tasmasini (ters maskeyi) engeller.
        pts = np.array(
            [[[[x0 * scale, y0 * scale], [x1 * scale, y1 * scale], [cx * scale, cy * scale]]]],
            dtype=np.float32,
        )
        lbls = np.array([[[2, 3, 1]]], dtype=np.int64)
        iou_scores, pred_masks = dec.run(
            None,
            {
                "input_points": pts,
                "input_labels": lbls,
                "image_embeddings": img_emb,
                "image_positional_embeddings": img_pos,
            },
        )

        pad_x = max(6, int((x1 - x0) * 0.04))
        pad_y = max(6, int((y1 - y0) * 0.04))
        box_guard = np.zeros((H, W), dtype=bool)
        box_guard[max(0, y0 - pad_y) : min(H, y1 + pad_y), max(0, x0 - pad_x) : min(W, x1 + pad_x)] = True
        box_area = max(1, (x1 - x0) * (y1 - y0))

        # 3 maske cikisindan (whole / part / subpart) kutu icine en iyi oturan butun nesneyi sec
        best_mask: Optional[np.ndarray] = None
        best_score = -999.0
        for k in range(pred_masks.shape[2]):
            low_mask = pred_masks[0, 0, k]
            up_1024 = cv2.resize(low_mask.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_LINEAR)
            valid_crop = up_1024[:new_h, :new_w]
            full_logits = cv2.resize(valid_crop, (W, H), interpolation=cv2.INTER_LINEAR)
            raw_k = (full_logits > 0.0) & fg_mask
            tot_k = int(raw_k.sum())
            if tot_k < 16:
                continue
            in_ratio = float(raw_k[y0:y1, x0:x1].sum()) / float(box_area)
            leak_ratio = float((raw_k & (~box_guard)).sum()) / float(tot_k)
            if leak_ratio > 0.40:
                cand_score = -10.0 + float(iou_scores[0, 0, k])
            else:
                whole_bonus = 0.10 if k == 0 else 0.0
                cand_score = float(iou_scores[0, 0, k]) + 0.60 * min(in_ratio, 0.85) - 0.80 * leak_ratio + whole_bonus
            if cand_score > best_score:
                best_score = cand_score
                best_mask = raw_k & box_guard

        m = best_mask if best_mask is not None else np.zeros((H, W), dtype=bool)

        # Obje icindeki kucuk delikleri (< %5 ornek alani) kapat (goz/cicek ortasi delinmesin)
        obj_area = int(m.sum())
        if obj_area > 50:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(filled, contours, -1, 1, thickness=-1)
            holes = (filled > 0) & (~m) & fg_mask
            n_h, h_lab, h_st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
            for hid in range(1, n_h):
                if int(h_st[hid, cv2.CC_STAT_AREA]) <= max(24, int(obj_area * 0.05)):
                    m[h_lab == hid] = True

        masks_out.append(m)

    return masks_out


# ---------------------------------------------------------------------------
# 4. Ana Semantik Katman Gruplama Girisi
# ---------------------------------------------------------------------------

def segment_semantic_groups_onnx(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    max_layers: int = 10,
    yazi_ayir: bool = True,
    custom_labels: str = "",
    log_cb: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Dondurur (alttan uste Z-sirali): [(group_name_ascii, kind, z_order, mask_bool), ...]"""
    mdir = ensure_onnx_models(log_cb=log_cb)
    H, W = rgb.shape[:2]
    groups_cfg, custom_wants_text = parse_custom_labels(custom_labels)
    run_text = yazi_ayir and custom_wants_text

    result_masks: Dict[str, Tuple[str, int, np.ndarray]] = {}

    if run_text:
        if log_cb:
            log_cb("RapidOCR ile yazilar taranıyor...")
        text_mask = detect_text_polygons_rapidocr(rgb, fg_mask)
        if int(text_mask.sum()) >= 24:
            result_masks["Yazilar"] = ("text", 100, text_mask)

    dets = detect_semantic_boxes_grounding_dino(
        rgb, fg_mask, groups_cfg, mdir=mdir, log_cb=log_cb
    )
    if dets:
        if log_cb:
            log_cb(f"SlimSAM ile {len(dets)} nesne piksel hassasiyetinde kesiliyor...")
        boxes = [d["box"] for d in dets]
        seg_masks = segment_boxes_slimsam(rgb, boxes, fg_mask, mdir=mdir)

        for det, smask in zip(dets, seg_masks):
            if int(smask.sum()) < 40:
                continue
            gname = det["group"]
            z_ord = int(det["z_order"])
            if gname not in result_masks:
                result_masks[gname] = ("subject", z_ord, smask.copy())
            else:
                _, prev_z, prev_m = result_masks[gname]
                result_masks[gname] = ("subject", max(prev_z, z_ord), prev_m | smask)

    # Yazi katmanini en uste koru; alttan uste sirala
    entries = [(name, kind, z, mask) for name, (kind, z, mask) in result_masks.items()]
    entries.sort(key=lambda item: item[2])
    return entries
