#!/usr/bin/env python3
"""Rastert forgepix.svg zu PNGs + baut macOS .icns."""
import os
import subprocess
import sys

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
SVG = os.path.join(HERE, "forgepix.svg")


def render(size, out):
    r = QSvgRenderer(QByteArray(open(SVG, "rb").read()))
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    r.render(p)
    p.end()
    img.save(out)


def main():
    QApplication(sys.argv)
    # Fenster-/Dock-Icon
    render(512, os.path.join(HERE, "forgepix_512.png"))
    render(256, os.path.join(HERE, "forgepix_256.png"))
    # .iconset für iconutil
    iconset = os.path.join(HERE, "ForgePix.iconset")
    os.makedirs(iconset, exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        render(base, os.path.join(iconset, f"icon_{base}x{base}.png"))
        render(base * 2, os.path.join(iconset, f"icon_{base}x{base}@2x.png"))
    icns = os.path.join(HERE, "ForgePix.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
    print("OK:", icns)


if __name__ == "__main__":
    main()
