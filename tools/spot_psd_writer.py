#!/usr/bin/env python3
"""Photoshop acmadan saf Multichannel (Color Mode = 7) Spot Kanalli PSD yazicisi.

Ozellikle serigrafi/emprime RIP ve film cikislarinda fazladan RGB/CMYK plakasi
uretmemek icin yalnizca baski sirasindaki Spot kanallarini (8-bit RLE, 508 PPI,
1005/1006/1045/1077 8BIM kaynaklariyla) yazar.
"""
from __future__ import annotations

import os
import re
import struct
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from psd_tools.compression.rle import encode as _psd_rle_encode
except Exception:
    _psd_rle_encode = None


def _packbits_encode(row_bytes: bytes) -> bytes:
    """psd_tools yoksa saf Python PackBits RLE kodlayici."""
    if _psd_rle_encode is not None:
        return _psd_rle_encode(row_bytes)
    n = len(row_bytes)
    if n == 0:
        return b""
    out = bytearray()
    i = 0
    while i < n:
        # Tekrar kontrolu
        run = 1
        while i + run < n and run < 128 and row_bytes[i + run] == row_bytes[i]:
            run += 1
        if run >= 2:
            out.append(257 - run)
            out.append(row_bytes[i])
            i += run
        else:
            start = i
            lit = 0
            while i < n and lit < 128:
                if i + 2 < n and row_bytes[i] == row_bytes[i + 1] == row_bytes[i + 2]:
                    break
                i += 1
                lit += 1
            out.append(lit - 1)
            out.extend(row_bytes[start:i])
    return bytes(out)


def _ascii_channel_name(raw: str, fallback: str = "Spot") -> str:
    clean = (raw or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))
    clean = re.sub(r"[^\x20-\x7e]+", "_", clean).strip()
    return (clean or fallback)[:120]


def _build_8bim_block(rid: int, payload: bytes) -> bytes:
    hdr = b"8BIM" + struct.pack(">H", rid) + b"\x00\x00" + struct.pack(">I", len(payload))
    pad = b"\x00" if (len(payload) % 2 == 1) else b""
    return hdr + payload + pad


def _build_multichannel_resources(channels: List[Dict[str, Any]], dpi: int = 508) -> bytes:
    res_fixed = int(round(max(36, int(dpi)) * 65536))
    r1005 = struct.pack(">IHHIHH", res_fixed, 1, 2, res_fixed, 1, 2)
    r1006_parts: List[bytes] = []
    r1045_parts: List[bytes] = []
    r1077_parts: List[bytes] = [struct.pack(">I", 1)]

    for idx, ch in enumerate(channels, start=1):
        name = _ascii_channel_name(str(ch.get("name") or f"Spot_{idx}"), f"Spot_{idx}")
        p_bytes = name.encode("latin1", errors="replace")[:255]
        r1006_parts.append(struct.pack(">B", len(p_bytes)) + p_bytes)

        u_chars = name + "\x00"
        r1045_parts.append(struct.pack(">I", len(u_chars)) + u_chars.encode("utf-16-be"))

        rgb = ch.get("rgb") or (0, 0, 0)
        r = max(0, min(255, int(rgb[0])))
        g = max(0, min(255, int(rgb[1])))
        b = max(0, min(255, int(rgb[2])))
        sol = max(0, min(100, int(ch.get("solidity", 16))))
        # kind = 2 (Spot color channel)
        r1077_parts.append(struct.pack(">HHHHHHB", 0, r * 257, g * 257, b * 257, 0, sol, 2))

    return (
        _build_8bim_block(1005, r1005)
        + _build_8bim_block(1006, b"".join(r1006_parts))
        + _build_8bim_block(1045, b"".join(r1045_parts))
        + _build_8bim_block(1077, b"".join(r1077_parts))
    )


def _normalize_ink_plane_for_psd(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Murekkep maskesini (True/255 = murekkep) Photoshop spot duzlemine (0 = murekkep, 255 = bos) cevirir."""
    arr = np.asarray(mask)
    if arr.shape != (height, width):
        raise ValueError(f"Kanal boyutu ({arr.shape[1]}x{arr.shape[0]}) hedef boyutla ({width}x{height}) uyusmuyor.")
    if arr.dtype == bool:
        ink_u8 = np.where(arr, 255, 0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(arr.max(initial=0.0)) <= 1.05 else 1.0
        ink_u8 = np.clip(np.round(arr * scale), 0, 255).astype(np.uint8)
    else:
        ink_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    # Photoshop Multichannel / Spot konvansiyonu: 0 = %100 murekkep, 255 = kagit/bos
    return np.ascontiguousarray(255 - ink_u8, dtype=np.uint8)


def write_multichannel_spot_psd(
    channels: Iterable[Dict[str, Any]],
    width: int,
    height: int,
    dpi: int = 508,
    out_path: str = "cikti_multichannel.psd",
) -> str:
    """Saf Multichannel (Color Mode = 7) Spot Kanalli PSD yazar ve mutlak dosya yolunu dondurur.

    Her kanal sozlugu:
      - name: str (ornegin 'fon', 'Kirmizi_Tram', 'Siyah')
      - mask: np.ndarray (H x W; bool/uint8/float, True veya 255 = murekkep)
      - rgb: Tuple[int, int, int]
      - solidity: int (0..100; fon/kumas icin 100, ust spot/tram icin 16)
    """
    ch_list = list(channels)
    if not ch_list:
        raise ValueError("En az 1 spot kanal gereklidir.")
    if len(ch_list) > 56:
        raise ValueError("Photoshop PSD en fazla 56 kanal destekler.")

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError(f"Gecersiz PSD boyutu: {w}x{h}")

    planes = [_normalize_ink_plane_for_psd(ch["mask"], w, h) for ch in ch_list]
    resources = _build_multichannel_resources(ch_list, dpi=dpi)
    n_channels = len(planes)

    abs_out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)

    row_counts: List[int] = []
    with open(abs_out, "wb") as f:
        # 1) Header: 8BPS, v1, reserved, channels, H, W, 8-bit, Color Mode = 7 (Multichannel)
        f.write(struct.pack(">4sH6sHIIHH", b"8BPS", 1, b"\x00" * 6, n_channels, h, w, 8, 7))
        # 2) Color Mode Data (Multichannel = 0)
        f.write(struct.pack(">I", 0))
        # 3) Image Resources
        f.write(struct.pack(">I", len(resources)))
        f.write(resources)
        # 4) Layer and Mask Info (pure multichannel = 0)
        f.write(struct.pack(">I", 0))
        # 5) Image Data (Compression = 1 PackBits RLE)
        f.write(struct.pack(">H", 1))
        rc_pos = f.tell()
        f.write(b"\x00" * (2 * n_channels * h))
        for plane in planes:
            for r in range(h):
                enc = _packbits_encode(plane[r].tobytes())
                row_counts.append(len(enc))
                f.write(enc)
        f.seek(rc_pos)
        f.write(struct.pack(f">{len(row_counts)}H", *row_counts))

    return abs_out
