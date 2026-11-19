#!/usr/bin/env python3
"""
Saf 1-bit Kayit Krosu (+), 4 Alanli Atolye Film Bilgi Etiketi ve On+Arka Yerlesim Araci.

699, 700, 701, 702 uretim PSD dosyalarindaki film zarfini uretir — ancak etiket yazilarini
anti-aliasing OLMADAN (saf 0/255 1-bit) basarak gri kenar artigi birakmaz.

Standart 4 alanli etiket yapisi (sol altta) + sag altta '+' kayit krosu:
  [Beden/Kalip + Baski Yuzu]  |  [Kanal Adi]
  ----------------------------+-----------------------------
                              |  [Xr  IsNo  KisaAd]
                              |  [Tarih]

Kullanim:
  # Tek set klasorune kros + etiket ekleme:
  python tools/film_etiket_ve_kros.py film_klasoru --cikti cikti_klasoru \
      --beden "s-xl" --yuz "arka baski" --is-no "703" --is-ad "desen"

  # Arka (ustte) + On (altta) tek tuvalde birlestirme:
  python tools/film_etiket_ve_kros.py arka_klasoru --on-klasor on_klasoru \
      --cikti birlesik_klasor --beden "s-xl" --is-no "703" --is-ad "desen"
"""
import argparse
import datetime
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_crisp_font(size_px):
    """Sistemden net sans-serif TrueType font bul (yoksa PIL varsayilan)."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size_px)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_registration_cross(draw, cx, cy, arm_px, thick_px):
    """Saf 1-bit '+' kayit krosu cizer (0 = siyah murekkep)."""
    half_t = max(1, thick_px // 2)
    draw.rectangle([cx - arm_px, cy - half_t, cx + arm_px, cy + half_t], fill=0)
    draw.rectangle([cx - half_t, cy - arm_px, cx + half_t, cy + arm_px], fill=0)


def draw_label_block(canvas_L, left_x, top_y, beden, yuz, kanal_adi, renk_sayisi, is_no, is_ad, tarih, ppi=508):
    """
    4 alanli atolye etiket blogu + sagda '+' kayit krosu cizer (grayscale uzerinde, sonra <128 1-bit).
    """
    draw = ImageDraw.Draw(canvas_L)
    # 508 PPI'da 1 mm = 20 px
    px_per_mm = ppi / 25.4
    font_sz = max(18, int(round(3.2 * px_per_mm)))  # ~3.2 mm yukseklik (~64 px @ 508 PPI)
    font = load_crisp_font(font_sz)
    line_t = max(2, int(round(0.25 * px_per_mm)))  # 0.25 mm cizgi (~5 px)

    col1_w = int(round(28 * px_per_mm))  # ~28 mm sol sutun
    row1_h = int(round(10 * px_per_mm))  # ~10 mm ust satir
    row2_h = int(round(12 * px_per_mm))  # ~12 mm alt satir
    pad = int(round(1.5 * px_per_mm))

    # Dikey bolme cizgisi ve yatay bolme cizgisi (699/700/701 stili arti-tablo)
    div_x = left_x + col1_w
    div_y = top_y + row1_h
    draw.rectangle([div_x - line_t // 2, top_y, div_x + line_t // 2, top_y + row1_h + row2_h], fill=0)
    draw.rectangle([left_x, div_y - line_t // 2, left_x + col1_w + int(round(34 * px_per_mm)), div_y + line_t // 2], fill=0)

    # Sol ust: Beden / Baski yuzu
    draw.text((left_x + pad, top_y + pad), f"{beden}", fill=0, font=font)
    draw.text((left_x + pad, top_y + pad + int(font_sz * 1.1)), f"{yuz}", fill=0, font=font)

    # Sag ust: Kanal adi
    draw.text((div_x + pad * 2, top_y + pad + int(font_sz * 0.4)), f"{kanal_adi}", fill=0, font=font)

    # Sag alt: Xr IsNo IsAd + Tarih
    code_str = f"{renk_sayisi}r {is_no} {is_ad}".strip()
    draw.text((div_x + pad * 2, div_y + pad), code_str, fill=0, font=font)
    draw.text((div_x + pad * 2, div_y + pad + int(font_sz * 1.15)), tarih, fill=0, font=font)


def stamp_single_film(ink_bool, kanal_adi, renk_sayisi, beden, yuz, is_no, is_ad, tarih, ppi=508):
    """Tek filme alt bant + 4 alanli etiket + '+' kayit krosu (saf 1-bit) ekler."""
    h, w = ink_bool.shape
    px_per_mm = ppi / 25.4
    strip_h = int(round(30 * px_per_mm))  # 3 cm alt bilgi + kros bandi
    new_h = h + strip_h

    # Yeni 1-bit tuval (False = beyaz/bos, True = siyah/murekkep)
    out = np.zeros((new_h, w), dtype=bool)
    out[:h, :] = ink_bool

    # Etiket + krosu ayri 8-bit seritte cizip <128 ile %100 saf 1-bit'e cevir
    strip = Image.new("L", (w, strip_h), 255)
    left_x = max(int(round(10 * px_per_mm)), int(w * 0.14))
    top_y = int(round(4 * px_per_mm))
    draw_label_block(strip, left_x, top_y, beden, yuz, kanal_adi, renk_sayisi, is_no, is_ad, tarih, ppi=ppi)

    # Sag altta '+' kayit krosu (15 mm kol, 0.3 mm cizgi)
    draw = ImageDraw.Draw(strip)
    cross_x = min(w - int(round(15 * px_per_mm)), int(w * 0.78))
    cross_y = int(round(15 * px_per_mm))
    arm = int(round(7.5 * px_per_mm))
    thick = max(2, int(round(0.30 * px_per_mm)))
    draw_registration_cross(draw, cross_x, cross_y, arm, thick)

    # Sol ust/sag ust ek denge kroslari (opsiyonel degil, tek alt kros 699-702 ile birebir)
    strip_ink = np.array(strip) < 128
    out[h:, :] = strip_ink
    return out


def clean_channel_label(filename):
    stem = os.path.splitext(filename)[0]
    return re.sub(r"^\d+[a-z]?_", "", stem).replace("_", " ")


def load_1bit_ink(path):
    im = Image.open(path)
    arr = np.array(im.convert("L"))
    return arr < 128


def save_1bit(ink_bool, path, ppi=508):
    im = Image.fromarray((~ink_bool).astype(bool))
    assert im.mode == "1"
    im.save(path, dpi=(ppi, ppi), optimize=True)


def main():
    ap = argparse.ArgumentParser(description="Saf 1-bit '+' kayit krosu, 4 alanli film etiketi ve On+Arka yerlesim.")
    ap.add_argument("film_klasoru", help="1-bit PNG filmlerin bulundugu klasor (veya Arka baski klasoru)")
    ap.add_argument("--cikti", required=True, help="Etiketli 1-bit filmlerin yazilacagi klasor")
    ap.add_argument("--on-klasor", default=None, help="Istege bagli On (gogus) baski film klasoru (altta birlestirilir)")
    ap.add_argument("--beden", default="s-xl", help="Beden/kalip etiketi (orn. 's-xl', 'xs-L', '8-16 garson')")
    ap.add_argument("--yuz", default="arka baski", help="Baski yuzu (varsayilan 'arka baski')")
    ap.add_argument("--on-yuz", default="on baski", help="On klasor verildiginde alt yuz etiketi")
    ap.add_argument("--is-no", default="000", help="Is/desen numarasi (orn. 703)")
    ap.add_argument("--is-ad", default="desen", help="Kisa anonim desen adi")
    ap.add_argument("--tarih", default=None, help="Tarih (varsayilan bugun GG.AA.YYYY)")
    ap.add_argument("--ppi", type=int, default=508, help="Cozunurluk (varsayilan 508)")
    args = ap.parse_args()

    tarih = args.tarih or datetime.date.today().strftime("%d.%m.%Y")
    os.makedirs(args.cikti, exist_ok=True)

    back_files = sorted([f for f in os.listdir(args.film_klasoru) if f.lower().endswith(".png")])
    if not back_files:
        raise SystemExit(f"HATA: {args.film_klasoru} icinde PNG bulunamadi.")

    renk_sayisi = len(back_files)
    front_map = {}
    if args.on_klasor:
        for f in sorted(os.listdir(args.on_klasor)):
            if f.lower().endswith(".png"):
                front_map[f] = os.path.join(args.on_klasor, f)

    gap_px = int(round(15 * (args.ppi / 25.4)))  # 1.5 cm arka-on arasi bosluk

    for fname in back_files:
        back_ink = load_1bit_ink(os.path.join(args.film_klasoru, fname))
        label = clean_channel_label(fname)
        stamped_back = stamp_single_film(
            back_ink, label, renk_sayisi, args.beden, args.yuz, args.is_no, args.is_ad, tarih, ppi=args.ppi
        )

        if args.on_klasor and fname in front_map:
            front_ink = load_1bit_ink(front_map[fname])
            stamped_front = stamp_single_film(
                front_ink, label, len(front_map), args.beden, args.on_yuz, args.is_no, args.is_ad, tarih, ppi=args.ppi
            )
            W = max(stamped_back.shape[1], stamped_front.shape[1])
            H = stamped_back.shape[0] + gap_px + stamped_front.shape[0]
            merged = np.zeros((H, W), dtype=bool)
            bx = (W - stamped_back.shape[1]) // 2
            fx = (W - stamped_front.shape[1]) // 2
            merged[:stamped_back.shape[0], bx:bx + stamped_back.shape[1]] = stamped_back
            y_f = stamped_back.shape[0] + gap_px
            merged[y_f:y_f + stamped_front.shape[0], fx:fx + stamped_front.shape[1]] = stamped_front
            final_ink = merged
        else:
            final_ink = stamped_back

        out_path = os.path.join(args.cikti, fname)
        save_1bit(final_ink, out_path, ppi=args.ppi)
        print(f"  [1-bit etiket+kros] {fname} -> {final_ink.shape[1]}x{final_ink.shape[0]} @ {args.ppi} PPI")

    print(f"[TAMAM] {len(back_files)} film etiketlendi: {args.cikti}")


if __name__ == "__main__":
    main()
