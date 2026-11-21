#!/usr/bin/env python3
"""
Tramsiz (Duz / Katı) Spot Renk Ayrimi — 508 PPI Atolye Standardi (v2).

699-702 uretim PSD standartlarina gore guncellendi:
  - Varsayilan 508 PPI (eski 600 PPI celiskisi giderildi).
  - Otomatik Lab k-means palet tespiti (--renk N) veya ozel palet (--palet "Ad:#RRGGBB,...").
  - Choke varsayilani 2 px (genis lap icin 3 px; ince detayda 1 px korunur).
  - Trap varsayilani 2 px (~0.10 mm @ 508 PPI).
  - Koyu kumasta --ust-beyaz (Highlight White, 0 px choke, Solidity 16%) ve --kumas "#RRGGBB".
  - Zengin kanallar_hibrit.json (mod="tramsiz", rol, solidity=100/16) + geriye donuk kanallar_tramsiz.json.

Kullanim:
  python tools/tramsiz.py girdi.png cikti_klasoru --renk 4 --choke 2 --trap 2 --ust-beyaz
  python tools/tramsiz.py girdi.png cikti_klasoru --palet "Kirmizi:#e72f0f,Mavi:#092b79,Siyah:#18192f"
"""
import argparse
import colorsys
import json
import math
import os
import time

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def parse_hex(h):
    h = h.strip().lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb2lab(a):
    f = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
    return cv2.cvtColor(f, cv2.COLOR_RGB2Lab)


def auto_name(rgb):
    lab = rgb2lab(np.array([[rgb]], np.uint8))[0, 0]
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    C = math.hypot(a, b)
    h01, s01, _ = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    h = h01 * 360.0
    if C < 7.0 or s01 < 0.08:
        if L > 88: return "Beyaz"
        if L > 55: return "Gri"
        if L > 25: return "Antrasit"
        return "Siyah"
    if h < 15 or h >= 340: return "Kirmizi"
    if h < 42: return "Kahve" if L < 52 else "Turuncu"
    if h < 70: return "Sari"
    if h < 165: return "Yesil"
    if h < 198: return "Turkuaz"
    if h < 252: return "Lacivert" if L < 38 else "Mavi"
    if h < 295: return "Mor"
    return "Pembe"


def detect_palette_kmeans(rgb_u8, k_colors, bg_rgb=(255, 255, 255)):
    """Kenar ara tonlarini dislayarak Lab k-means ile gercek spot murekkepleri tespit eder."""
    h, w = rgb_u8.shape[:2]
    sc = min(1.0, 1200.0 / max(h, w))
    sm = cv2.resize(rgb_u8, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    lab = rgb2lab(sm)
    # Yerel std dusuk olan (duz, kenar olmayan) piksellerden ornekle
    mu = cv2.blur(lab, (5, 5))
    mu2 = cv2.blur(lab * lab, (5, 5))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0).sum(axis=2))
    bg_lab = rgb2lab(np.array([[bg_rgb]], np.uint8))[0, 0]
    de_bg = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
    mask = (sd < 3.5) & (de_bg > 10.0)
    if mask.sum() < k_colors * 30:
        mask = (de_bg > 8.0)
    pts = lab[mask].reshape(-1, 3)
    if len(pts) < k_colors:
        raise SystemExit("HATA: Palet tespiti icin yeterli renkli piksel bulunamadi.")
    rng = np.random.RandomState(0)
    sel = pts[rng.permutation(len(pts))[:min(60000, len(pts))]].astype(np.float32)
    cv2.setRNGSeed(0)
    _, _, cen = cv2.kmeans(
        sel, k_colors, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3),
        5, cv2.KMEANS_PP_CENTERS,
    )
    rgbs = np.clip(cv2.cvtColor(cen.reshape(1, -1, 3), cv2.COLOR_Lab2RGB).reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    inks = {}
    used = {}
    for c in rgbs:
        t = (int(c[0]), int(c[1]), int(c[2]))
        bn = auto_name(t)
        cnt = used.get(bn, 0) + 1
        used[bn] = cnt
        inks[bn if cnt == 1 else f"{bn}_{cnt}"] = t
    return inks


def dist_in(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def dilate_px(mask, r):
    if r <= 0:
        return mask.copy()
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= r


def main():
    ap = argparse.ArgumentParser(description="Tramsiz (Duz) Spot Renk Ayrimi — 508 PPI Atolye Standardi")
    ap.add_argument("girdi", help="Kaynak gorsel")
    ap.add_argument("cikti", help="Cikti klasoru")
    ap.add_argument("--renk", type=int, default=4, help="Otomatik palet renk sayisi (varsayilan 4)")
    ap.add_argument("--palet", default=None, help="Ozel palet: 'Ad1:#RRGGBB,Ad2:#RRGGBB'")
    ap.add_argument("--ppi", type=int, default=508, help="Cozunurluk (varsayilan 508)")
    ap.add_argument("--genislik-cm", type=float, default=0.0, help="Baski genisligi cm (0 = kaynak boyutu koru)")
    ap.add_argument("--trap", type=int, default=2, help="Trap tasirma px (varsayilan 2 = ~0.10 mm @ 508 PPI)")
    ap.add_argument("--choke", type=int, default=2, help="Underbase choke px (varsayilan 2; genis lap 3)")
    ap.add_argument("--medyan", type=int, default=3, help="Medyan temizlik cekirdegi (tek sayi, 0=kapali)")
    ap.add_argument("--kumas", default="#1e1e1e", help="Koyu set kumas rengi hex (varsayilan #1e1e1e)")
    ap.add_argument("--ust-beyaz", action="store_true", help="Koyu sette ic beyaz alanlar icin ayri Ust Beyaz (Solidity 16%)")
    ap.add_argument("--siyah-fon-oy", action="store_true", help="En koyu duz blok altindan beyaz fonu oy (kumas koyu degilse/antrasit fonsuzsa)")
    args = ap.parse_args()

    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
    os.makedirs(args.cikti, exist_ok=True)
    fabric_rgb = list(parse_hex(args.kumas))

    im0 = Image.open(args.girdi)
    if im0.mode in ("RGBA", "LA") or ("transparency" in im0.info):
        rgba = np.asarray(im0.convert("RGBA"))
        af = rgba[..., 3:4].astype(np.float32) / 255.0
        src = np.clip(rgba[..., :3] * af + 255.0 * (1.0 - af), 0, 255).astype(np.uint8)
    else:
        src = np.asarray(im0.convert("RGB"))
    sh, sw = src.shape[:2]

    if args.genislik_cm > 0:
        W = int(round(args.genislik_cm / 2.54 * args.ppi))
        s = W / sw
        H = int(round(sh * s))
    else:
        W, H, s = sw, sh, 1.0

    if args.palet:
        INKS = {}
        for part in args.palet.split(","):
            nm, hx = part.split(":")
            INKS[nm.strip()] = parse_hex(hx)
    else:
        INKS = detect_palette_kmeans(src, args.renk)

    names = list(INKS)
    rgbs = np.array([INKS[n] for n in names], np.uint8)
    labs = rgb2lab(rgbs.reshape(1, -1, 3)).reshape(-1, 3)
    order = np.argsort(-labs[:, 0])  # acik -> koyu baski sirasi
    names = [names[i] for i in order]
    rgbs = rgbs[order]
    labs = labs[order]
    WHITE = rgb2lab(np.array([[[255, 255, 255]]], np.uint8))[0, 0]
    pal = np.vstack([WHITE, labs]).astype(np.float32)  # 0 = beyaz/kagit
    m = len(pal)
    log("baski sirasi:", list(zip(names, [tuple(int(x) for x in c) for c in rgbs])))
    log(f"hedef {W}x{H} @ {args.ppi} PPI ({W/args.ppi*2.54:.1f}x{H/args.ppi*2.54:.1f} cm) | choke={args.choke}px trap={args.trap}px")

    # Tile bazli Lab siniflandirma (saf renk + 2'li kenar karisimi yuvarlama)
    L = np.zeros((H, W), np.uint8)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    TH = 256
    for y0 in range(0, H, TH):
        h = min(TH, H - y0)
        N = h * W
        if s == 1.0:
            tile = src[y0:y0 + h]
        else:
            M = np.float32([[1 / s, 0, 0.5 / s - 0.5], [0, 1 / s, (y0 + 0.5) / s - 0.5]])
            tile = cv2.warpAffine(src, M, (W, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        lab = rgb2lab(tile).reshape(-1, 3)
        best = np.full(N, 1e9, np.float32)
        cls = np.zeros(N, np.uint8)
        for i in range(m):
            d = lab - pal[i]
            e = (d * d).sum(1)
            mk = e < best
            best[mk] = e[mk]
            cls[mk] = i
        for (i, j) in pairs:
            ab = pal[j] - pal[i]
            den = float((ab * ab).sum())
            if den < 1e-6:
                continue
            d = lab - pal[i]
            t = np.clip((d @ ab) / den, 0, 1)
            r = d - t[:, None] * ab
            e = (r * r).sum(1)
            mk = e < best
            best[mk] = e[mk]
            cls[mk] = np.where(t[mk] >= 0.5, j, i)
        L[y0:y0 + h] = cls.reshape(h, W)

    if args.medyan >= 3 and args.medyan % 2 == 1:
        L = cv2.medianBlur(L, args.medyan)

    # Kagit (disari) ve tasarim beyazi ayrimi
    white = (L == 0)
    core = dist_in(white) > 2
    n_cc, cc = cv2.connectedComponents(core.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    border = np.unique(np.concatenate([cc[0, :], cc[-1, :], cc[:, 0], cc[:, -1]]))
    lut = np.zeros(n_cc + 1, bool)
    lut[border] = True
    lut[0] = False
    outside = dilate_px(lut[cc], 3) & white
    dwhite = white & ~outside
    masks = {k: (L == (k + 1)) for k in range(len(names))}

    # Trap: acik renkler kendinden koyu renklerin altina trap px tasar
    final = {}
    for k in range(len(names)):
        darker = np.zeros((H, W), bool)
        for q in range(k + 1, len(names)):
            darker |= masks[q]
        final[k] = masks[k] | (dilate_px(masks[k], args.trap) & darker) if (args.trap > 0 and darker.any()) else masks[k]

    # Underbase (fon): 2-3 px choke, ince detayda 1 px korunur
    U = dwhite.copy()
    for k in masks:
        U |= masks[k]
    if args.choke > 0:
        E = dist_in(U) > args.choke
        recon = cv2.distanceTransform((~E).astype(np.uint8), cv2.DIST_L2, 5) <= args.choke
        thin = U & ~recon
        UB = E | (dist_in(thin) > 1 if args.choke >= 2 else thin)
    else:
        UB = U.copy()

    if args.siyah_fon_oy and len(names) >= 1:
        black = masks[len(names) - 1]
        UB &= ~(dist_in(black) > max(args.trap, 2))

    def save_1bit(mask, path):
        im = Image.fromarray((~mask).astype(bool))
        assert im.mode == "1"
        im.save(path, dpi=(args.ppi, args.ppi), optimize=True)

    acik_dir = os.path.join(args.cikti, "Acik_Tisort")
    koyu_dir = os.path.join(args.cikti, "Koyu_Tisort")
    os.makedirs(acik_dir, exist_ok=True)
    os.makedirs(koyu_dir, exist_ok=True)

    sets_v2 = {"Acik_Tisort": [], "Koyu_Tisort": []}
    sets_old = {"Acik_Tisort": [], "Koyu_Tisort": []}

    # Koyu set: 01 fon (Solidity 100%) + istege bagli 02 beyaz_ust (Solidity 16%)
    save_1bit(UB, os.path.join(koyu_dir, "01_fon.png"))
    sets_v2["Koyu_Tisort"].append({"ad": "fon", "dosya": "01_fon.png", "rgb": [255, 255, 255], "rol": "fon", "solidity": 100, "aile": -10})
    sets_old["Koyu_Tisort"].append(["01 fon", "01_fon.png", [255, 255, 255]])

    k_idx = 2
    if args.ust_beyaz and dwhite.mean() > 0.0005:
        save_1bit(dwhite, os.path.join(koyu_dir, "02_beyaz_ust.png"))
        sets_v2["Koyu_Tisort"].append({"ad": "beyaz_ust", "dosya": "02_beyaz_ust.png", "rgb": [255, 255, 255], "rol": "ust_beyaz", "solidity": 16, "aile": -11})
        sets_old["Koyu_Tisort"].append(["02 beyaz_ust", "02_beyaz_ust.png", [255, 255, 255]])
        k_idx = 3

    for k, nm in enumerate(names):
        safe_nm = nm.replace(" ", "_")
        fa = f"{k + 1:02d}_{safe_nm}.png"
        fk = f"{k_idx + k:02d}_{safe_nm}.png"
        save_1bit(final[k], os.path.join(acik_dir, fa))
        save_1bit(final[k], os.path.join(koyu_dir, fk))
        rgb_l = rgbs[k].tolist()
        sets_v2["Acik_Tisort"].append({"ad": nm, "dosya": fa, "rgb": rgb_l, "rol": "lap", "solidity": 16, "aile": k})
        sets_v2["Koyu_Tisort"].append({"ad": nm, "dosya": fk, "rgb": rgb_l, "rol": "lap", "solidity": 16, "aile": k})
        sets_old["Acik_Tisort"].append([f"{k + 1:02d} {nm}", fa, rgb_l])
        sets_old["Koyu_Tisort"].append([f"{k_idx + k:02d} {nm}", fk, rgb_l])

    meta_v2 = {
        "mod": "tramsiz", "W": W, "H": H, "ppi": args.ppi,
        "trap_px": args.trap, "choke_px": args.choke,
        "kumas_rgb": fabric_rgb, "sets": sets_v2,
    }
    with open(os.path.join(args.cikti, "kanallar_hibrit.json"), "w", encoding="utf-8") as fp:
        json.dump(meta_v2, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(args.cikti, "kanallar_tramsiz.json"), "w", encoding="utf-8") as fp:
        json.dump({"mod": "tramsiz", "W": W, "H": H, "ppi": args.ppi, "trap_px": args.trap, "choke_px": args.choke, "sets": sets_old}, fp, ensure_ascii=False, indent=2)

    # Onizlemeler (dinamik olcek)
    def composite(bg, layers):
        out = np.full((H, W, 3), bg, dtype=np.uint8)
        for mk, col in layers:
            out[mk] = col
        return out

    sc = min(1.0, 1400.0 / max(W, H))
    tw, th = max(1, int(W * sc)), max(1, int(H * sc))
    light_layers = [(final[k], rgbs[k]) for k in range(len(names))]
    dark_layers = [(UB, (255, 255, 255))] + ([(dwhite, (255, 255, 255))] if args.ust_beyaz else []) + light_layers
    for tag, bg, lyrs in (("acik", (250, 250, 250), light_layers), ("koyu", tuple(fabric_rgb), dark_layers)):
        comp = composite(bg, lyrs)
        Image.fromarray(cv2.resize(comp, (tw, th), interpolation=cv2.INTER_AREA)).save(os.path.join(args.cikti, f"onizleme_{tag}.png"))
    log("BITTI")


if __name__ == "__main__":
    main()
