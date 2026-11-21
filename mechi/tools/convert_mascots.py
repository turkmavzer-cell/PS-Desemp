#!/usr/bin/env python3
r"""
MeChi Studio — Maskot Video Donusturucu.

E:\ab ayçam\antigravty\mechi altindaki 4 ham studio MP4 klibini:
  1. 480x270 @ 15 fps olcegine indirir,
  2. Her klibin arka plan kenar tonunu sabit #262928 studio zeminine hizalar,
  3. Dis %10 kenar seridini sinuzoidal gecisle #262928'e eritir (sifir cerceve izi),
  4. Bas/son 0.35 sn fade-in / fade-out ile sahneler arasi dikissiz gecis saglar,
  5. Animated WebP (birincil) + 2-pass paletli GIF (yedek) uretir.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
from typing import List, Tuple

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

STAGE_BG_RGB = (38, 41, 40)  # #262928
DEFAULT_SOURCE_DIR = r"E:\ab ayçam\antigravty\mechi"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MECHI_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_TARGET_DIR = os.path.join(MECHI_DIR, "assets", "mascot")

CLIP_MAPPING: List[Tuple[str, str]] = [
    ("Fluffy_cat_turning_to_camera_*.mp4", "01_tekir_acilis"),
    ("Black_cat_walking_toward_camera_*.mp4", "02_bombay_cam"),
    ("Two_cats_meeting_in_studio_*.mp4", "03_ikili_selam"),
    ("Two_cats_walking_in_studio_*.mp4", "04_kuyruk_takip"),
]


def _measure_border_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    bw = max(6, min(h, w) // 24)
    border = np.concatenate(
        [
            frame_rgb[:bw].reshape(-1, 3),
            frame_rgb[-bw:].reshape(-1, 3),
            frame_rgb[:, :bw].reshape(-1, 3),
            frame_rgb[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border, axis=0).astype(np.float32)


def _build_gamma_lut(bg_clip_rgb: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    luts = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        src = float(np.clip(bg_clip_rgb[c], 12.0, 240.0)) / 255.0
        dst = float(np.clip(target_rgb[c], 12.0, 240.0)) / 255.0
        gamma = math.log(dst) / math.log(src)
        gamma = float(np.clip(gamma, 0.85, 1.45))
        mapped = np.clip(np.round(np.power(x, gamma) * 255.0), 0, 255).astype(np.uint8)
        luts[c] = mapped
    return luts


def _build_feather_mask(width: int, height: int) -> np.ndarray:
    """Radyal-eliptik studyo isigi yumusatmasi: dikdortgen kose izi birakmadan #262928'e eritir."""
    m_left = max(12, int(round(width * 0.14)))
    m_right = m_left
    m_top = max(8, int(round(height * 0.08)))
    m_bot = max(16, int(round(height * 0.18)))

    wx = np.ones(width, dtype=np.float32)
    tx = np.linspace(0.0, 1.0, m_left, dtype=np.float32)
    ease_x = 0.5 * (1.0 - np.cos(math.pi * tx))
    wx[:m_left] = ease_x
    wx[-m_right:] = ease_x[::-1]

    wy = np.ones(height, dtype=np.float32)
    ty_t = np.linspace(0.0, 1.0, m_top, dtype=np.float32)
    wy[:m_top] = 0.5 * (1.0 - np.cos(math.pi * ty_t))
    ty_b = np.linspace(0.0, 1.0, m_bot, dtype=np.float32)
    wy[-m_bot:] = (0.5 * (1.0 - np.cos(math.pi * ty_b)))[::-1]

    box_mask = wy[:, None] * wx[None, :]

    # Koseleri yumusatmak icin genis eliptik vignette ile carp
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx - (width - 1) * 0.5) / ((width - 1) * 0.5)
    ny = (yy - (height - 1) * 0.5) / ((height - 1) * 0.5)
    r = np.sqrt(nx * nx + ny * ny)
    # r <= 0.78 tam parlak, r >= 1.25 sifir
    t_rad = np.clip((r - 0.78) / (1.25 - 0.78), 0.0, 1.0)
    radial = 1.0 - (0.5 * (1.0 - np.cos(math.pi * t_rad)))

    mask = np.clip(box_mask * radial, 0.0, 1.0)
    return mask[..., None]


def _process_video_frames(
    src_path: str,
    target_size: Tuple[int, int] = (480, 270),
    target_fps: float = 15.0,
    fade_sec: float = 0.35,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {src_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0.0

    raw_frames_rgb: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        raw_frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not raw_frames_rgb:
        raise RuntimeError(f"Video kare icermiyor: {src_path}")

    # Ilk ve son bos karelerden kenar arka plan rengini olc
    bg_start = _measure_border_rgb(raw_frames_rgb[0])
    bg_end = _measure_border_rgb(raw_frames_rgb[-1])
    bg_clip = 0.5 * (bg_start + bg_end)
    luts = _build_gamma_lut(bg_clip, STAGE_BG_RGB)

    out_count = max(2, int(round(duration * target_fps)))
    tw, th = target_size
    feather = _build_feather_mask(tw, th)
    bg_arr = np.array(STAGE_BG_RGB, dtype=np.float32).reshape(1, 1, 3)

    fade_frames = max(1, int(round(fade_sec * target_fps)))
    processed: List[np.ndarray] = []

    for out_idx in range(out_count):
        src_idx = min(
            len(raw_frames_rgb) - 1,
            int(round(out_idx * (len(raw_frames_rgb) - 1) / max(1, out_count - 1))),
        )
        frame = raw_frames_rgb[src_idx]
        resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

        # Kanal bazli arka plan ton hizalamasi
        aligned = np.empty_like(resized)
        for c in range(3):
            aligned[..., c] = luts[c][resized[..., c]]

        # Kenar yumusatma (%10 cerceve eritme -> #262928)
        blended = aligned.astype(np.float32) * feather + bg_arr * (1.0 - feather)

        # Bas/son fade-in / fade-out
        alpha_t = 1.0
        if out_idx < fade_frames:
            alpha_t = (out_idx / float(fade_frames))
        elif out_idx >= out_count - fade_frames:
            rem = (out_count - 1) - out_idx
            alpha_t = (rem / float(fade_frames))

        if alpha_t < 1.0:
            blended = blended * alpha_t + bg_arr * (1.0 - alpha_t)

        processed.append(np.clip(np.round(blended), 0, 255).astype(np.uint8))

    return processed


def _save_animated_webp(frames: List[np.ndarray], out_webp: str, fps: float = 15.0) -> None:
    duration_ms = int(round(1000.0 / fps))
    pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
    pil_frames[0].save(
        out_webp,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        quality=82,
        method=4,
    )


def _save_gif_ffmpeg(frames: List[np.ndarray], out_gif: str, fps: float = 15.0) -> None:
    h, w = frames[0].shape[:2]
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-filter_complex",
        "[0:v]split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop",
        "0",
        out_gif,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg GIF hatasi ({rc}): {stderr[-400:]}")
    finally:
        if proc.poll() is None:
            proc.kill()


def convert_all(source_dir: str, target_dir: str, fps: float = 15.0) -> List[dict]:
    os.makedirs(target_dir, exist_ok=True)
    results = []
    for pattern, out_stem in CLIP_MAPPING:
        matches = sorted(glob.glob(os.path.join(source_dir, pattern)))
        if not matches:
            print(f"[ATLA] Bulunamadi: {pattern}")
            continue
        src_path = matches[0]
        print(f"[ISLENIYOR] {os.path.basename(src_path)} -> {out_stem}.webp / .gif")
        frames = _process_video_frames(src_path, target_size=(480, 270), target_fps=fps)
        out_webp = os.path.join(target_dir, f"{out_stem}.webp")
        out_gif = os.path.join(target_dir, f"{out_stem}.gif")
        _save_animated_webp(frames, out_webp, fps=fps)
        _save_gif_ffmpeg(frames, out_gif, fps=fps)
        webp_kb = os.path.getsize(out_webp) / 1024.0
        gif_kb = os.path.getsize(out_gif) / 1024.0
        print(
            f"  [TAMAM] {len(frames)} kare @ {fps:g}fps | "
            f"WebP: {webp_kb:.1f} KB | GIF: {gif_kb:.1f} KB"
        )
        results.append(
            {
                "stem": out_stem,
                "frames": len(frames),
                "webp": out_webp,
                "webp_kb": round(webp_kb, 1),
                "gif": out_gif,
                "gif_kb": round(gif_kb, 1),
            }
        )
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="MeChi Studio maskot MP4 -> WebP/GIF donusturucu")
    ap.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="Ham MP4 klasoru")
    ap.add_argument("--target", default=DEFAULT_TARGET_DIR, help="Hedef mascot varlik klasoru")
    ap.add_argument("--fps", type=float, default=15.0, help="Hedef FPS (varsayilan: 15)")
    args = ap.parse_args()
    res = convert_all(args.source, args.target, fps=args.fps)
    if not res:
        print("[HATA] Hicbir kaynak video bulunamadi.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
