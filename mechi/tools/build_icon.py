"""MeChi Studio icin yuksek cozunurluklu kedi-kulakli 'M' amblem ikonu uretir."""
from __future__ import annotations

import os
from PIL import Image, ImageDraw

MECHI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(MECHI_DIR, "assets")


def build_icon(out_dir: str = ASSETS_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # 2x supersampling (512 -> 256) ile yumusak anti-aliasing
    S = 512
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    pad = 24
    radius = 112
    # Koyu komur arka plan + ince cerceve
    draw.rounded_rectangle(
        [pad, pad, S - pad, S - pad],
        radius=radius,
        fill=(27, 31, 34, 255),
        outline=(54, 62, 69, 255),
        width=8,
    )

    amber = (224, 163, 77, 255)      # #E0A34D
    amber_soft = (236, 186, 107, 255)

    # Kedi kulakli 'M' harfi ve altina tekir/bombay bıyık noktasi
    # Sol ve sag kedi kulagi sivri uclari dis direklerin tepesinde
    pts = [
        (114, 376),  # sol alt dis
        (114, 148),  # sol direk yukari
        (148, 102),  # sol kulak tepesi
        (194, 166),  # sol kulak ic inis
        (256, 260),  # orta V dibi
        (318, 166),  # sag kulak ic cikis
        (364, 102),  # sag kulak tepesi
        (398, 148),  # sag direk yukari
        (398, 376),  # sag alt dis
        (344, 376),  # sag alt ic
        (344, 226),  # sag ic direk
        (274, 326),  # orta V sag
        (238, 326),  # orta V sol
        (168, 226),  # sol ic direk
        (168, 376),  # sol alt ic
    ]
    draw.polygon(pts, fill=amber)

    # Kulak iclerinde ince koyu vurgu ucgenleri (kedi siluetini belirginlestirir)
    inner_l = [(134, 152), (149, 126), (168, 156)]
    inner_r = [(378, 152), (363, 126), (344, 156)]
    draw.polygon(inner_l, fill=(27, 31, 34, 255))
    draw.polygon(inner_r, fill=(27, 31, 34, 255))

    # Alt orta kucuk burun/pati noktasi
    draw.ellipse([240, 356, 272, 388], fill=amber_soft)

    final_256 = im.resize((256, 256), Image.LANCZOS)
    png_path = os.path.join(out_dir, "mechi.png")
    ico_path = os.path.join(out_dir, "mechi.ico")

    final_256.save(png_path, format="PNG")
    final_256.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    return png_path, ico_path


if __name__ == "__main__":
    p, i = build_icon()
    print("Icon generated:", p, i)
