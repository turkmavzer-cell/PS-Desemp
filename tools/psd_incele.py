#!/usr/bin/env python3
"""
Genel Cok Kanalli (Spot / CMYK / RGB / Multichannel) PSD Derin Denetci.

Buyuk (200-600 MB) uretim PSD/PSB dosyalarini ham RLE cozucuyle tam cozunurlukte okur:
  - PPI, fiziksel boyut (cm), renk modu, XMP/ICC, katman agaci.
  - Tum spot/alpha kanallarin DisplayInfo renk uzayi, tahmini RGB ve Solidity (%).
  - 1-bit / surekli ton dagilimi, kapsama %, 2D FFT tram olcumu (LPI, aci, periyot).
  - (--derin) Fon choke mesafesi (distanceTransform), kanallar arasi ortusme ve
    tramli kanallar arasi dot-on-dot faz olcumu.
  - (--onizleme klasor) Kanal kontak sayfasi ve %100 1:1 piksel kesiti uretimi.

Kullanim:
  python tools/psd_incele.py dosya.psd [--json rapor.json] [--onizleme klasor] [--derin]
"""
import argparse
import colorsys
import json
import math
import os
import re
import struct
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.compression.rle import decode as decode_rle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COLOR_MODES = {
    0: "Bitmap", 1: "Grayscale", 2: "Indexed", 3: "RGB",
    4: "CMYK", 7: "Multichannel", 8: "Duotone", 9: "Lab",
}
DISPLAY_CS = {
    0: "RGB", 1: "HSB", 2: "CMYK", 3: "Pantone", 7: "Lab", 8: "Grayscale",
}
DISPLAY_KINDS = {0: "Selected", 1: "Protected", 2: "Spot"}


def ascii_safe(s):
    return str(s).translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")).encode("ascii", "replace").decode("ascii")


def decode_display_rgb(cs, comps):
    c0, c1, c2, c3 = comps
    if cs == 0:
        return [int(c0 / 257), int(c1 / 257), int(c2 / 257)]
    if cs == 1:
        r, g, b = colorsys.hsv_to_rgb(c0 / 65535.0, c1 / 65535.0, c2 / 65535.0)
        return [int(r * 255), int(g * 255), int(b * 255)]
    if cs == 2:
        c, m, y, k = 1.0 - c0 / 65535.0, 1.0 - c1 / 65535.0, 1.0 - c2 / 65535.0, 1.0 - c3 / 65535.0
        return [max(0, min(255, int(255 * (1 - x) * (1 - k)))) for x in (c, m, y)]
    if cs == 7:
        L = c0 / 100.0
        a = struct.unpack(">h", struct.pack(">H", c1))[0] / 100.0
        b = struct.unpack(">h", struct.pack(">H", c2))[0] / 100.0
        fy = (L + 16.0) / 116.0
        fx, fz = a / 500.0 + fy, fy - b / 200.0
        finv = lambda t: t**3 if t > 0.206893 else (t - 16.0 / 116.0) / 7.787
        X, Y, Z = 0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)
        rl, gl, bl = 3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z
        gam = lambda u: 12.92 * max(0.0, min(1.0, u)) if u <= 0.0031308 else 1.055 * (max(0.0, min(1.0, u)) ** (1 / 2.4)) - 0.055
        return [int(round(gam(u) * 255)) for u in (rl, gl, bl)]
    if cs == 8:
        g = max(0, min(255, int(round(255 * (1.0 - c0 / 10000.0)))))
        return [g, g, g]
    return [128, 128, 128]


def parse_display_info(data, version=1):
    items, pos = [], 4 if version == 2 else 0
    step = 13 if version == 2 else 14
    fmt = ">HHHHHHB" if version == 2 else ">HHHHHHBB"
    while pos + step <= len(data):
        vals = struct.unpack(fmt, data[pos:pos + step])
        cs, c0, c1, c2, c3, op, kind = vals[:7]
        pos += step
        items.append({
            "color_space": DISPLAY_CS.get(cs, f"CS({cs})"),
            "approx_rgb": decode_display_rgb(cs, (c0, c1, c2, c3)),
            "solidity": op,
            "kind": DISPLAY_KINDS.get(kind, f"Kind({kind})"),
        })
    return items


def parse_unicode_names(data):
    names, pos = [], 0
    while pos + 4 <= len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if pos + n * 2 > len(data):
            break
        names.append(data[pos:pos + n * 2].decode("utf-16-be", errors="replace").rstrip("\x00"))
        pos += n * 2
    return names


def parse_pascal_names(data):
    names, pos = [], 0
    while pos < len(data):
        n = data[pos]
        pos += 1
        if pos + n > len(data):
            break
        names.append(data[pos:pos + n].decode("latin1", errors="replace"))
        pos += n
    return names


def parse_psd_header_and_channels(filepath):
    with open(filepath, "rb") as f:
        sig, ver, _, chs, h, w, depth, mode = struct.unpack(">4sH6sHIIHH", f.read(26))
        is_psb = (ver == 2)
        cm_len = struct.unpack(">I", f.read(4))[0]
        f.seek(cm_len, 1)
        ir_len = struct.unpack(">I", f.read(4))[0]
        ir_end = f.tell() + ir_len
        res = {}
        while f.tell() < ir_end:
            if f.read(4) != b"8BIM":
                break
            rid = struct.unpack(">H", f.read(2))[0]
            nl = struct.unpack(">B", f.read(1))[0]
            f.seek(nl + (1 if (1 + nl) % 2 else 0), 1)
            rsize = struct.unpack(">I", f.read(4))[0]
            res[rid] = f.read(rsize)
            if rsize % 2:
                f.seek(1, 1)
        f.seek(ir_end)
        lm_len = struct.unpack(">Q" if is_psb else ">I", f.read(8 if is_psb else 4))[0]
        f.seek(lm_len, 1)
        comp = struct.unpack(">H", f.read(2))[0]
        arrays = []
        if depth != 8:
            raise ValueError(f"Desteklenmeyen bit derinligi: {depth}")
        if comp == 0:
            for _ in range(chs):
                arrays.append(np.frombuffer(f.read(h * w), dtype=np.uint8).reshape((h, w)))
        elif comp == 1:
            rc_all = struct.unpack(f">{chs * h}{'I' if is_psb else 'H'}", f.read(chs * h * (4 if is_psb else 2)))
            for c in range(chs):
                buf = bytearray(h * w)
                off, base = 0, c * h
                for r in range(h):
                    buf[off:off + w] = decode_rle(f.read(rc_all[base + r]), w)
                    off += w
                arrays.append(np.frombuffer(buf, dtype=np.uint8).reshape((h, w)))
        else:
            raise ValueError(f"Desteklenmeyen sikistirma: {comp}")
    return {
        "is_psb": is_psb, "channels": chs, "width": w, "height": h,
        "depth": depth, "mode": COLOR_MODES.get(mode, f"Mode({mode})"),
        "resources": res, "arrays": arrays,
    }


def detect_am_fft(arr, ppi):
    h, w = arr.shape
    ink = 255 - arr
    ps = min(512, h, w)
    if ps < 64:
        return {"detected": False}
    best_sc, best_p = -1.0, None
    sy, sx = max(ps // 2, h // 12), max(ps // 2, w // 12)
    for y in range(0, h - ps + 1, sy):
        for x in range(0, w - ps + 1, sx):
            p = ink[y:y + ps, x:x + ps]
            m = p.mean()
            if 15 < m < 235:
                sc = float(np.abs(np.diff(p.astype(np.int16), axis=1)).mean() + np.abs(np.diff(p.astype(np.int16), axis=0)).mean())
                if sc > best_sc:
                    best_sc, best_p = sc, p
    if best_p is None or best_sc < 2.0:
        return {"detected": False}
    pf = best_p.astype(np.float32) - best_p.mean()
    win = np.outer(np.hanning(ps), np.hanning(ps))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(pf * win)))
    cy, cx = ps // 2, ps // 2
    yy, xx = np.ogrid[:ps, :ps]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (rr >= max(6.0, 15.0 * ps / ppi)) & (rr <= min(ps * 0.45, 130.0 * ps / ppi))
    mm = np.where(mask, mag, 0.0)
    peak = float(mm.max())
    snr = peak / max(float(np.median(mag[mask])) if np.any(mask) else 1.0, 1e-6)
    py, px = np.unravel_index(np.argmax(mm), mm.shape)
    dy, dx = py - cy, px - cx
    fc = math.hypot(dx, dy)
    per = ps / fc if fc > 0 else 0
    lpi = ppi / per if per > 0 else 0
    ang = math.degrees(math.atan2(-dy, dx)) % 90.0
    rv = mag[np.abs(rr - fc) <= 2.0]
    ring = float(rv.max() / max(np.median(rv), 1e-6))
    return {
        "detected": bool(snr > 18.0 and ring > 8.0),
        "lpi": round(lpi, 1), "angle_deg": round(ang, 1),
        "period_px": round(per, 2), "snr": round(snr, 1),
    }


def measure_dot_on_dot(arr_a, arr_b):
    a, b = (arr_a < 128), (arr_b < 128)
    h, w = a.shape
    blk = 32
    bh, bw = h // blk, w // blk
    am = a[:bh * blk, :bw * blk].reshape(bh, blk, bw, blk).mean(axis=(1, 3))
    bm = b[:bh * blk, :bw * blk].reshape(bh, blk, bw, blk).mean(axis=(1, 3))
    both = (am > 0.15) & (am < 0.65) & (bm > 0.15) & (bm < 0.65)
    if not both.any():
        return None
    ys, xs = np.where(both)
    step = max(1, len(ys) // 250)
    obs, maxp = 0, 0
    for k in range(0, len(ys), step):
        y0, x0 = ys[k] * blk, xs[k] * blk
        pa, pb = a[y0:y0 + blk, x0:x0 + blk], b[y0:y0 + blk, x0:x0 + blk]
        obs += int((pa & pb).sum())
        maxp += min(int(pa.sum()), int(pb.sum()))
    return round(obs / max(maxp, 1) * 100.0, 1)


def main():
    ap = argparse.ArgumentParser(description="Genel Cok Kanalli PSD Derin Denetci (508 PPI, Solidity, FFT, Choke, Dot-on-Dot)")
    ap.add_argument("psd", help="Incelenecek PSD/PSB dosyasi")
    ap.add_argument("--json", default=None, help="JSON rapor cikti yolu")
    ap.add_argument("--onizleme", default=None, help="Kanal grid ve %100 kesit PNG cikti klasoru")
    ap.add_argument("--derin", action="store_true", help="Fon choke (distanceTransform) ve dot-on-dot faz olcumu yap")
    args = ap.parse_args()

    data = parse_psd_header_and_channels(args.psd)
    res = data["resources"]
    ppi_h = ppi_v = 72.0
    if 1005 in res and len(res[1005]) >= 16:
        hr, _, _, vr, _, _ = struct.unpack(">IHHIHH", res[1005][:16])
        ppi_h, ppi_v = hr / 65536.0, vr / 65536.0

    names = parse_unicode_names(res[1045]) if 1045 in res else (parse_pascal_names(res[1006]) if 1006 in res else [])
    disp = parse_display_info(res[1077], 2) if 1077 in res else (parse_display_info(res[1007], 1) if 1007 in res else [])

    base_map = {"RGB": ["Red", "Green", "Blue"], "CMYK": ["Cyan", "Magenta", "Yellow", "Black"], "Grayscale": ["Gray"], "Multichannel": []}
    base_names = base_map.get(data["mode"], [])
    n_base = len(base_names)
    has_trans = (data["channels"] - n_base == len(names) + 1)

    psd_obj = PSDImage.open(args.psd)
    layer_count = len(list(psd_obj))

    print(f"=== {os.path.basename(args.psd)} ({os.path.getsize(args.psd)/1024/1024:.1f} MB) ===")
    print(f"  Mod: {data['mode']} {data['depth']}-bit | {data['width']}x{data['height']} @ {ppi_h:.2f} PPI "
          f"({data['width']/ppi_h*2.54:.1f}x{data['height']/ppi_v*2.54:.1f} cm) | Katman: {layer_count} | Kanal: {data['channels']}")

    ch_reps = []
    am_spots = []
    for idx, arr in enumerate(data["arrays"]):
        if idx < n_base:
            role, name, dinfo = "base", base_names[idx], None
        elif has_trans and idx == n_base:
            role, name, dinfo = "transparency", "Transparency", None
        else:
            ei = idx - n_base - (1 if has_trans else 0)
            name = names[ei] if ei < len(names) else f"Extra_{ei}"
            dinfo = disp[ei] if ei < len(disp) else None
            role = dinfo["kind"].lower() if dinfo else "extra"

        hist = np.bincount(arr.ravel(), minlength=256)
        tot = arr.size
        n_uniq = int(np.count_nonzero(hist))
        p_bin = float((hist[0] + hist[255]) / tot * 100.0)
        ink_any = float((tot - hist[255]) / tot * 100.0)
        min_v = int(np.argmax(hist > 0))
        max_v = int(255 - np.argmax(hist[::-1] > 0))

        fft = None
        if role == "spot" or data["mode"] == "Multichannel":
            if min_v < 255 and max_v > 0:
                fft = detect_am_fft(arr, ppi_h)
                if fft.get("detected") and fft["lpi"] >= 30.0:
                    am_spots.append(idx)

        rep = {
            "index": idx, "name": name, "role": role, "display": dinfo,
            "min": min_v, "max": max_v, "n_unique": n_uniq,
            "p_binary": round(p_bin, 2), "ink_any_pct": round(ink_any, 2), "fft": fft,
        }
        ch_reps.append(rep)
        dstr = f" {dinfo['color_space']} rgb={dinfo['approx_rgb']} sol={dinfo['solidity']}%" if dinfo else ""
        fstr = f" | AM {fft['lpi']}LPI@{fft['angle_deg']}°" if (fft and fft.get("detected")) else ""
        print(f"  [{idx:2d}] {role:10s} '{name}'{dstr} | uniq={n_uniq:3d} bin={p_bin:6.2f}% ink={ink_any:6.2f}%{fstr}")

    deep = {}
    if args.derin:
        # Fon choke kontrolu (adi 'fon' iceren ilk kanal vs diger aktif spotlar)
        fon_cands = [r["index"] for r in ch_reps if "fon" in r["name"].lower() and r["ink_any_pct"] < 99.0 and r["ink_any_pct"] > 0.1]
        active_spots = [r["index"] for r in ch_reps if r["role"] == "spot" and 0.05 < r["ink_any_pct"] < 99.0]
        if fon_cands and len(active_spots) >= 2:
            fi = fon_cands[0]
            m_fon = data["arrays"][fi] < 128
            others = np.zeros_like(m_fon)
            for si in active_spots:
                if si != fi and "fon" not in ch_reps[si]["name"].lower():
                    others |= (data["arrays"][si] < 128)
            dist = cv2.distanceTransform((~m_fon).astype(np.uint8), cv2.DIST_L2, 3)
            edge = others & (cv2.erode(others.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
            out_e = edge & (~m_fon) & (dist <= 20.0)
            if out_e.any():
                deep["fon_choke_px"] = {
                    "fon_channel": ch_reps[fi]["name"],
                    "median_px": round(float(np.median(dist[out_e])), 2),
                    "p90_px": round(float(np.percentile(dist[out_e], 90)), 2),
                }
                print(f"  [Derin] Fon choke ({ch_reps[fi]['name']}): medyan={deep['fon_choke_px']['median_px']} px, p90={deep['fon_choke_px']['p90_px']} px")

        # Dot-on-dot faz kontrolu
        dod = []
        for i in range(len(am_spots)):
            for j in range(i + 1, len(am_spots)):
                ia, ib = am_spots[i], am_spots[j]
                pct = measure_dot_on_dot(data["arrays"][ia], data["arrays"][ib])
                if pct is not None:
                    dod.append({"pair": f"{ch_reps[ia]['name']} vs {ch_reps[ib]['name']}", "dot_on_dot_pct": pct})
                    print(f"  [Derin] Dot-on-Dot ({ch_reps[ia]['name']} vs {ch_reps[ib]['name']}): %{pct}")
        deep["dot_on_dot"] = dod

    if args.onizleme:
        os.makedirs(args.onizleme, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.psd))[0]
        n = len(data["arrays"])
        cols = min(5, max(1, n))
        rows = math.ceil(n / cols)
        cw = 320
        ch_h = max(120, min(400, int(cw * data["height"] / data["width"])))
        lh = 40
        grid = Image.new("RGB", (cols * cw, rows * (ch_h + lh)), (235, 235, 235))
        dr = ImageDraw.Draw(grid)
        for idx, arr in enumerate(data["arrays"]):
            ri, ci = divmod(idx, cols)
            x0, y0 = ci * cw, ri * (ch_h + lh)
            grid.paste(Image.fromarray(arr).resize((cw - 8, ch_h - 8), Image.BILINEAR).convert("RGB"), (x0 + 4, y0 + lh + 4))
            rep = ch_reps[idx]
            sw = tuple(rep["display"]["approx_rgb"]) if rep["display"] else (180, 180, 180)
            dr.rectangle([x0 + 4, y0 + 4, x0 + 24, y0 + 24], fill=sw, outline=(0, 0, 0))
            dr.text((x0 + 30, y0 + 4), ascii_safe(f"[{idx}] {rep['name'][:24]}"), fill=(0, 0, 0))
            sol = f" s={rep['display']['solidity']}%" if rep["display"] else ""
            dr.text((x0 + 30, y0 + 20), ascii_safe(f"ink={rep['ink_any_pct']:.1f}% bin={rep['p_binary']:.0f}%{sol}"), fill=(40, 40, 40))
        grid.save(os.path.join(args.onizleme, f"{stem}_channels_grid.png"))

    out = {
        "filename": os.path.basename(args.psd), "mode": data["mode"], "depth": data["depth"],
        "width": data["width"], "height": data["height"], "ppi": [round(ppi_h, 2), round(ppi_v, 2)],
        "layers": layer_count, "channels": ch_reps, "deep": deep,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        print(f"  JSON yazildi: {args.json}")


if __name__ == "__main__":
    main()
