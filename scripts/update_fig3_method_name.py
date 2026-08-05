#!/usr/bin/env python3
"""Synchronize the paper method name in the Visio source and its JPEG export."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "manuscript_current"
VSDX_PATH = FIGURE_DIR / "fig3.vsdx"
JPEG_PATH = FIGURE_DIR / "fig3.jpg"
FONT_PATH = Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf")


def update_visio_source() -> None:
    with zipfile.ZipFile(VSDX_PATH, "r") as source:
        page_xml = source.read("visio/pages/page1.xml")
        updated_xml = page_xml.replace(
            b"VICM/SRBM\r\nModel Predictive Control",
            b"Centroidal MPC\r\nSRBM / IR-CMPC",
        ).replace(
            b"IR-CMPC/SRBM\r\nModel Predictive Control",
            b"Centroidal MPC\r\nSRBM / IR-CMPC",
        )
        if updated_xml == page_xml:
            return

        with tempfile.NamedTemporaryFile(
            dir=VSDX_PATH.parent, suffix=".vsdx", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)

        try:
            with zipfile.ZipFile(temporary_path, "w") as target:
                for item in source.infolist():
                    data = updated_xml if item.filename == "visio/pages/page1.xml" else source.read(item.filename)
                    target.writestr(item, data)
            os.replace(temporary_path, VSDX_PATH)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def update_jpeg_export() -> None:
    image = Image.open(JPEG_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Redraw the complete label area so both lines share the module centerline.
    draw.rectangle((5280, 330, 7860, 1180), fill="white")

    def draw_centered(label: str, font_size: int, top: int) -> None:
        font = ImageFont.truetype(str(FONT_PATH), font_size)
        bbox = draw.textbbox((0, 0), label, font=font)
        x = 6600 - (bbox[2] - bbox[0]) / 2 - bbox[0]
        y = top - bbox[1]
        draw.text((x, y), label, fill="black", font=font)

    draw_centered("Centroidal MPC", 280, 530)
    draw_centered("SRBM / IR-CMPC", 250, 830)
    draw.rounded_rectangle(
        (5225, 255, 7935, 1270), radius=145, outline="black", width=36
    )
    image.save(JPEG_PATH, quality=95, subsampling=0)


def main() -> None:
    update_visio_source()
    update_jpeg_export()
    print(VSDX_PATH.relative_to(ROOT))
    print(JPEG_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
