"""Referans klasorden Vision LLM (Gemini Flash) ile Tasarim Dili / Stil DNA ogrenme."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from typing import Callable, Dict, List, Optional

import requests
from PIL import Image

from mechi.config import PROFILES_DIR
from mechi.core.worker import TaskCancelled

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".jfif"}

ANALYSIS_PROMPT = """Analyze this t-shirt / textile graphic artwork purely for its visual design language.
CRITICAL: Strip and omit ANY trademarked brand names, logos, or copyrighted character names. Describe only generic, original, print-ready graphic design elements.
Return ONLY valid JSON with these exact keys:
{
  "style_dna": "1-2 sentence English description of the overall illustration technique, line weight, shading, and print aesthetic",
  "frame": {"label_tr": "kisa Turkce cerceve adi", "prompt": "detailed English prompt describing the border/frame geometry", "tags": ["tag1", "tag2"]},
  "ornament": {"label_tr": "kisa Turkce susleme adi", "prompt": "detailed English prompt describing foliage, clouds, ribbons, or ambient frame ornaments", "tags": ["tag1", "tag2"]},
  "subject": {"label_tr": "kisa Turkce ana obje adi", "prompt": "detailed English prompt describing the central hero subject/object", "tags": ["tag1", "tag2"]},
  "typography": {"label_tr": "kisa Turkce yazi stili adi", "prompt": "detailed English typography style prompt using placeholder [YOUR BRAND TEXT]", "tags": ["tag1", "tag2"]},
  "palette": {"label_tr": "kisa Turkce palet adi", "colors_hex": ["#111111", "#FFFFFF", "#C89B3C", "#2E5A44"], "prompt": "English color palette description"}
}
"""


def _slugify(name: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    clean = name.translate(tr).lower()
    return re.sub(r"[^a-z0-9]+", "_", clean).strip("_") or "ozel_stil"


def _encode_resized_jpeg_b64(img_path: str, max_side: int = 1024) -> str:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_block(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        clean = clean[start : end + 1]
    return json.loads(clean)


def _dedupe_items(items: List[dict], prefix: str) -> List[dict]:
    seen = set()
    out = []
    for idx, item in enumerate(items, start=1):
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        key = prompt.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        tags = item.get("tags") or ["custom"]
        if not isinstance(tags, list):
            tags = [str(tags)]
        entry = {
            "id": f"{prefix}{len(out) + 1:02d}",
            "label_tr": str(item.get("label_tr") or f"Öğe {idx}").strip(),
            "prompt": prompt,
            "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
            "weight": 1.0,
        }
        if "colors_hex" in item:
            entry["colors_hex"] = item["colors_hex"]
        out.append(entry)
    return out


def learn_style_from_folder(
    folder_path: str,
    profile_name: str,
    api_key: str,
    profiles_dir: str = PROFILES_DIR,
    model: str = "gemini-2.5-flash",
    max_images: int = 60,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **_kwargs,
) -> dict:
    """Referans klasordeki tasarimlari analiz edip yeni MeChi stil profili uretir."""
    if not api_key or not api_key.strip():
        raise ValueError(
            "Klasörden stil öğrenmek için ücretsiz Google AI Studio GEMINI_API_KEY gereklidir. "
            "Ayarlar penceresinden ekleyin."
        )
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")

    files = [
        os.path.join(folder_path, fn)
        for fn in sorted(os.listdir(folder_path))
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
    ][:max_images]
    if not files:
        raise ValueError("Seçilen klasörde desteklenen görsel bulunamadı (JPG, PNG, WEBP, TIF).")

    frames, ornaments, subjects, typographies, palettes = [], [], [], [], []
    dna_notes: List[str] = []

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"

    for idx, fpath in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            raise TaskCancelled()
        pct = int(round((idx - 1) / float(len(files)) * 90))
        if progress_cb:
            progress_cb(pct, f"Analiz ediliyor ({idx}/{len(files)}): {os.path.basename(fpath)}")

        try:
            b64 = _encode_resized_jpeg_b64(fpath)
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": ANALYSIS_PROMPT},
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
            }
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code != 200:
                if log_cb:
                    log_cb(f"[ATLA] {os.path.basename(fpath)}: HTTP {resp.status_code}")
                continue
            cand_parts = ((resp.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            raw_txt = "".join(p.get("text", "") for p in cand_parts)
            parsed = _extract_json_block(raw_txt)
            if parsed.get("style_dna"):
                dna_notes.append(str(parsed["style_dna"]).strip())
            if isinstance(parsed.get("frame"), dict):
                frames.append(parsed["frame"])
            if isinstance(parsed.get("ornament"), dict):
                ornaments.append(parsed["ornament"])
            if isinstance(parsed.get("subject"), dict):
                subjects.append(parsed["subject"])
            if isinstance(parsed.get("typography"), dict):
                typographies.append(parsed["typography"])
            if isinstance(parsed.get("palette"), dict):
                palettes.append(parsed["palette"])
        except Exception as exc:
            if log_cb:
                log_cb(f"[ATLA] {os.path.basename(fpath)}: {exc}")

    if not (frames or subjects):
        raise RuntimeError("Hiçbir görselden stil verisi çıkarılamadı. API anahtarınızı ve görselleri kontrol edin.")

    slug = _slugify(profile_name)
    out_dir = os.path.join(profiles_dir, slug)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "id": slug,
        "name": profile_name.strip() or slug,
        "description": f"{len(files)} referans görselden çıkarılan özgün stil DNA profili.",
        "style_dna": dna_notes[0] if dna_notes else "Clean flat 2D editorial textile illustration, crisp outlines.",
        "sample_count": len(files),
    }
    buckets = {
        "meta.json": meta,
        "cerceveler.json": _dedupe_items(frames, "f"),
        "suslemeler.json": _dedupe_items(ornaments, "s"),
        "objeler.json": _dedupe_items(subjects, "o"),
        "yazilar.json": _dedupe_items(typographies, "y"),
        "paletler.json": _dedupe_items(palettes, "p"),
    }
    for fname, content in buckets.items():
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fp:
            json.dump(content, fp, indent=2, ensure_ascii=False)

    if progress_cb:
        progress_cb(100, f"Stil profili kaydedildi: {meta['name']}")

    return {
        "profile_id": slug,
        "profile_dir": out_dir,
        "counts": {k: len(v) for k, v in buckets.items() if isinstance(v, list)},
    }
