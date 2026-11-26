"""MeChi Studio — PyTorch CUDA GPU Yapay Zeka Katman Segmentasyon Motoru.

Bilesenler:
  1) RapidOCR (ONNX)          : Metin poligonlarini bulur (Yazilar).
  2) Grounding DINO + YOLOE-PF: Acik-sozluk semantik + prompt-free nesne kutu tespiti (mobileclip yok).
  3) Ultralytics SAM 2.1 CUDA : Tespit edilen kutular ve tam-dolu posterler icin GPU piksel kesimi.

Tum model agirliklari repoya girmeden yalnizca %LOCALAPPDATA%/MeChiStudio/models/torch altinda saklanir.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

import onnx_ai_segmenter as _onnx_seg

_MODEL_CACHE: Dict[str, Any] = {}


def get_torch_models_dir() -> str:
    """PyTorch / Ultralytics agirliklarinin tutuldugu yerel AppData dizinini dondurur."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
    path = os.path.join(base, "MeChiStudio", "models", "torch")
    os.makedirs(path, exist_ok=True)
    return path


@contextlib.contextmanager
def _cwd_guard(target_dir: str) -> Iterator[None]:
    """Ultralytics model indirme/yukleme sirasinda repo kokune dosya inmesini engeller."""
    prev = os.getcwd()
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    try:
        yield
    finally:
        try:
            os.chdir(prev)
        except OSError:
            pass


def is_torch_cuda_ai_available() -> Dict[str, Any]:
    """PyTorch, CUDA GPU ve Ultralytics paketlerinin hazirlik durumunu dondurur."""
    pkgs: Dict[str, bool] = {}
    cuda_ok = False
    device_name = "CPU"
    vram_gb = 0.0

    try:
        import torch

        pkgs["torch"] = True
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            device_name = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(float(props.total_memory) / (1024.0 ** 3), 2)
    except Exception:
        pkgs["torch"] = False

    for mod in ("ultralytics", "rapidocr_onnxruntime"):
        try:
            __import__(mod)
            pkgs[mod] = True
        except Exception:
            pkgs[mod] = False

    ready = bool(pkgs.get("torch") and pkgs.get("ultralytics"))
    return {
        "ready": ready,
        "cuda": cuda_ok,
        "device": device_name,
        "vram_gb": vram_gb,
        "packages": pkgs,
        "models_dir": get_torch_models_dir(),
    }


def _resize_for_inference(rgb: np.ndarray, max_side: int = 1536) -> Tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return rgb, 1.0
    scale = float(max_side) / float(long_side)
    nw = max(64, int(round(w * scale)))
    nh = max(64, int(round(h * scale)))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale


def _fill_small_holes(mask_b: np.ndarray, fg_mask: np.ndarray, max_hole_frac: float = 0.05) -> np.ndarray:
    m = mask_b.copy()
    obj_area = int(m.sum())
    if obj_area <= 50:
        return m
    h, w = m.shape
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    holes = (filled > 0) & (~m) & fg_mask
    n_h, h_lab, h_st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    limit = max(24, int(obj_area * max_hole_frac))
    for hid in range(1, n_h):
        if int(h_st[hid, cv2.CC_STAT_AREA]) <= limit:
            m[h_lab == hid] = True
    return m


def _load_sam2_model(mdir: str, log_cb: Optional[Callable[[str], None]] = None):
    if "sam2" in _MODEL_CACHE:
        return _MODEL_CACHE["sam2"]

    from ultralytics import SAM

    with _cwd_guard(mdir):
        for fname in ("sam2.1_s.pt", "sam2.1_t.pt", "sam2_t.pt"):
            spath = os.path.join(mdir, fname)
            try:
                if log_cb and not os.path.exists(spath):
                    log_cb(f"{fname} modeli indiriliyor/yukleniyor...")
                sam = SAM(spath)
                _MODEL_CACHE["sam2"] = sam
                return sam
            except Exception:
                continue
    raise RuntimeError("SAM 2.1 segmentasyon modeli yuklenemedi.")


def _load_yoloe_prompt_free(mdir: str, log_cb: Optional[Callable[[str], None]] = None):
    """Text encoder (mobileclip) gerektirmeyen Prompt-Free YOLOE modelini opsiyonel yukler."""
    if "yoloe_pf" in _MODEL_CACHE:
        return _MODEL_CACHE["yoloe_pf"]

    from ultralytics import YOLOE

    pf_path = os.path.join(mdir, "yoloe-11s-seg-pf.pt")
    try:
        with _cwd_guard(mdir):
            if log_cb and not os.path.exists(pf_path):
                log_cb("Prompt-Free YOLOE modeli kontrol ediliyor...")
            model = YOLOE(pf_path)
            _MODEL_CACHE["yoloe_pf"] = model
            return model
    except Exception:
        _MODEL_CACHE["yoloe_pf"] = None
        return None


def _scale_mask_to_full(mask_small: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    if mask_small.shape == (full_h, full_w):
        return mask_small > 0.5
    up = cv2.resize(mask_small.astype(np.float32), (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return up > 0.5


def segment_semantic_groups_torch(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    max_layers: int = 10,
    yazi_ayir: bool = True,
    custom_labels: str = "",
    log_cb: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """PyTorch CUDA (Grounding DINO + YOLOE-PF + SAM 2.1 + RapidOCR) ile semantik katmanlar uretir.

    Dondurur (alttan uste Z-sirali): [(group_name_ascii, kind, z_order, mask_bool), ...]
    """
    import torch

    mdir = get_torch_models_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = rgb.shape[:2]
    fg_area = max(1, int(fg_mask.sum()))
    groups_cfg, custom_wants_text = _onnx_seg.parse_custom_labels(custom_labels)
    run_text = bool(yazi_ayir and custom_wants_text)

    result_masks: Dict[str, Tuple[str, int, np.ndarray]] = {}

    # 1) RapidOCR ile yazilar
    if run_text:
        try:
            if log_cb:
                log_cb("RapidOCR ile yazilar taranıyor...")
            text_mask = _onnx_seg.detect_text_polygons_rapidocr(rgb, fg_mask)
            if int(text_mask.sum()) >= 24:
                result_masks["Yazilar"] = ("text", 100, text_mask)
        except Exception as exc_ocr:
            if log_cb:
                log_cb(f"RapidOCR adimi atlandi: {exc_ocr}")

    # 2) Acik-sozluk kutu tespiti: Hazir Grounding DINO (mobileclip olmadan)
    dets: List[dict] = []
    try:
        onnx_mdir = _onnx_seg.ensure_onnx_models(log_cb=log_cb)
        dets = _onnx_seg.detect_semantic_boxes_grounding_dino(
            rgb, fg_mask, groups_cfg, mdir=onnx_mdir, log_cb=log_cb
        )
    except Exception as exc_gd:
        if log_cb:
            log_cb(f"Grounding DINO kutu kesfi atlandi ({exc_gd}); SAM 2.1 otomatik kesime geciliyor.")
        dets = []

    rgb_inf, inf_scale = _resize_for_inference(rgb, max_side=1536)
    ih, iw = rgb_inf.shape[:2]

    det_boxes_inf: List[Tuple[float, float, float, float]] = []
    det_meta: List[Tuple[str, int]] = []
    for d in dets[:24]:
        bx0, by0, bx1, by1 = d["box"]
        det_boxes_inf.append((bx0 * inf_scale, by0 * inf_scale, bx1 * inf_scale, by1 * inf_scale))
        det_meta.append((d["group"], int(d["z_order"])))

    # 3) Opsiyonel Prompt-Free YOLOE-PF (custom_labels bossa ek obje yakalama)
    pf_extra_idx = 0
    if not (custom_labels or "").strip():
        pf_model = _load_yoloe_prompt_free(mdir, log_cb=log_cb)
        if pf_model is not None:
            try:
                with torch.no_grad(), _cwd_guard(mdir):
                    pf_res = pf_model.predict(
                        source=rgb_inf,
                        conf=0.22,
                        iou=0.55,
                        device=device,
                        verbose=False,
                    )
                if pf_res and getattr(pf_res[0], "boxes", None) is not None and len(pf_res[0].boxes) > 0:
                    xyxy = pf_res[0].boxes.xyxy.detach().cpu().numpy()
                    for bi in range(min(len(xyxy), 12)):
                        x0, y0, x1, y1 = [float(v) for v in xyxy[bi]]
                        if (x1 - x0) * (y1 - y0) / float(max(1, iw * ih)) > 0.85:
                            continue
                        b_int = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
                        dup = any(
                            _onnx_seg._box_iou_xyxy(tuple(int(round(v)) for v in eb), b_int) > 0.55
                            for eb in det_boxes_inf
                        )
                        if not dup:
                            pf_extra_idx += 1
                            det_boxes_inf.append((x0, y0, x1, y1))
                            det_meta.append((f"Obje_PF_{pf_extra_idx}", 35))
            except Exception:
                pass

    # 4) PyTorch CUDA SAM 2.1 piksel hassasiyetinde kesim + Tam-dolu poster korumasi
    try:
        with torch.no_grad(), _cwd_guard(mdir):
            sam_model = _load_sam2_model(mdir, log_cb=log_cb)

            if det_boxes_inf:
                if log_cb:
                    log_cb(f"SAM 2.1 ({device}) ile {len(det_boxes_inf)} nesne kesiliyor...")
                sam_res = sam_model.predict(
                    source=rgb_inf,
                    bboxes=[list(b) for b in det_boxes_inf],
                    device=device,
                    verbose=False,
                )
                sam_masks = None
                if sam_res and getattr(sam_res[0], "masks", None) is not None:
                    sam_masks = sam_res[0].masks.data.detach().cpu().numpy()

                if sam_masks is not None:
                    for idx, (gname, z_ord) in enumerate(det_meta):
                        if idx >= len(sam_masks):
                            continue
                        cand_m = _scale_mask_to_full(sam_masks[idx], W, H) & fg_mask
                        if int(cand_m.sum()) < 48:
                            continue
                        cand_m = _fill_small_holes(cand_m, fg_mask, max_hole_frac=0.05)

                        if gname not in result_masks:
                            result_masks[gname] = ("subject", z_ord, cand_m)
                        else:
                            _, prev_z, prev_m = result_masks[gname]
                            result_masks[gname] = ("subject", max(prev_z, z_ord), prev_m | cand_m)

            # Tam-dolu poster korumasi: tespitler <%35 alandaysa SAM 2.1 otomatik bolumleme calistir
            covered = np.zeros((H, W), dtype=bool)
            for _, (_, _, m_cur) in result_masks.items():
                covered |= m_cur
            cov_ratio = float(covered.sum()) / float(fg_area)
            rem_budget = max(0, int(max_layers) - len(result_masks) - 1)

            if cov_ratio < 0.35 and rem_budget >= 2:
                if log_cb:
                    log_cb("Tam-dolu gorsel algilandi; SAM 2.1 CUDA otomatik nesne kesimi calisiyor...")
                auto_res = sam_model(source=rgb_inf, device=device, verbose=False)
                if auto_res and getattr(auto_res[0], "masks", None) is not None:
                    auto_masks = auto_res[0].masks.data.detach().cpu().numpy()
                    cands: List[Tuple[int, np.ndarray]] = []
                    for am in auto_masks:
                        full_m = _scale_mask_to_full(am, W, H) & fg_mask & (~covered)
                        a_px = int(full_m.sum())
                        if a_px < max(120, int(fg_area * 0.018)):
                            continue
                        if a_px > int(fg_area * 0.78):
                            continue
                        cands.append((a_px, full_m))
                    cands.sort(key=lambda t: -t[0])

                    added = 0
                    for _apx, cm in cands:
                        if added >= rem_budget:
                            break
                        clean_cm = cm & (~covered)
                        if int(clean_cm.sum()) < max(120, int(fg_area * 0.018)):
                            continue
                        clean_cm = _fill_small_holes(clean_cm, fg_mask, max_hole_frac=0.05)
                        added += 1
                        result_masks[f"Obje_{added}"] = ("subject", 30 + added, clean_cm)
                        covered |= clean_cm
    finally:
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    entries = [(name, kind, z, mask) for name, (kind, z, mask) in result_masks.items()]
    entries.sort(key=lambda item: item[2])
    return entries
