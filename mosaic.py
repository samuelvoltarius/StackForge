#!/usr/bin/env python3
"""
mosaic.py — Mosaik/Panorama-Zusammensetzen für Modul „Hybrid" (Mond-/Sonnen-Mosaik).

Setzt überlappende Kacheln (Panels) zu einem großen Bild zusammen — z.B. mehrere
Aufnahmen vom Mond/Sonne, die zusammen die ganze Scheibe ergeben. Reine OpenCV-Lösung.
"""
import cv2
import numpy as np


def _to8(img):
    if img is None:
        return None
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def stitch(paths, mode="panorama", log=print):
    """Überlappende Kacheln zu einem Mosaik zusammensetzen.
    Gibt (Ergebnis-BGR-uint8, status_text) zurück."""
    imgs = [_to8(cv2.imread(p, cv2.IMREAD_UNCHANGED)) for p in paths]
    imgs = [im if (im is None or im.ndim == 3) else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            for im in imgs]
    imgs = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        raise RuntimeError("Mindestens 2 überlappende Kacheln nötig")
    log(f"  {len(imgs)} Kacheln zusammensetzen ({mode}) …")
    m = cv2.Stitcher_SCANS if mode == "scans" else cv2.Stitcher_PANORAMA
    st = cv2.Stitcher_create(m)
    status, pano = st.stitch(imgs)
    if status != cv2.Stitcher_OK:
        msgs = {1: "zu wenig Überlappung / zu wenige gemeinsame Merkmale",
                2: "Homographie-Schätzung fehlgeschlagen",
                3: "Kamera-Parameter-Anpassung fehlgeschlagen"}
        raise RuntimeError(f"Mosaik fehlgeschlagen ({msgs.get(status, status)}). "
                           f"Tipp: mehr Überlappung (~30 %) zwischen den Kacheln.")
    return pano, "ok"
