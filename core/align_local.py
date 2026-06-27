#!/usr/bin/env python3
"""
align_local.py — gemeinsame Ausrichtungs-Bausteine (über Module hinweg).

Zwei Stufen, die in den Profi-Tools den Unterschied machen und ForgePix bisher fehlten:

1) `ecc_refine` — **subpixel-genaue, helligkeitsinvariante** Verfeinerung einer groben (Feature-)
   Ausrichtung per `cv2.findTransformECC`. Robust auch auf defokussierten/teils unscharfen Frames,
   wo ORB/SIFT wenig finden.

2) `flow_warp` — **lokale, nicht-rigide** Ausrichtung per dichtem Optical-Flow (`DISOpticalFlow`),
   geglättet und in der Verschiebung **gedeckelt**, dann `cv2.remap`. Korrigiert lokale Verzerrung
   (Seeing, Focus-Breathing, leichte Motiv-Verformung), die eine globale Transformation nicht kann.

Reine OpenCV/NumPy-Abhängigkeiten (MIT-kompatibel).
"""
import numpy as np
import cv2


def _gray8(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.dtype == np.uint16:
        g = (g / 256).astype(np.uint8)
    elif g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def ecc_refine(ref, mov, init=None, motion="euclidean", iters=50, eps=1e-4, gauss=5):
    """Verfeinert die Ausrichtung von `mov` auf `ref` subpixel-genau (ECC, helligkeitsinvariant).
    init: 2x3-Startmatrix (z. B. aus ORB/RANSAC) oder None (Identität).
    Gibt (warp_2x3, korrelationskoeffizient) zurück, oder (init_or_eye, 0.0) bei Fehlschlag."""
    gr, gm = _gray8(ref), _gray8(mov)
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE}.get(motion, cv2.MOTION_EUCLIDEAN)
    warp = np.eye(2, 3, dtype=np.float32) if init is None else init.astype(np.float32)[:2]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(gr, gm, warp, mode, crit, None, gauss)
    except cv2.error:
        return (init.astype(np.float32)[:2] if init is not None
                else np.eye(2, 3, dtype=np.float32)), 0.0
    return warp, float(cc)


def apply_warp(img, warp_2x3, inverse=False):
    """Warpt `img` mit einer 2x3-Matrix. inverse=True für ECC-Matrizen (die template→input
    abbilden — zum Ausrichten von input auf template als Inverse anwenden)."""
    h, w = img.shape[:2]
    flags = cv2.INTER_LANCZOS4 | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpAffine(img, warp_2x3, (w, h), flags=flags,
                          borderMode=cv2.BORDER_REPLICATE)


def flow_warp(ref, mov, cap_px=6.0, smooth_sigma=9.0, log=None):
    """Lokale, nicht-rigide Ausrichtung von `mov` auf `ref` per dichtem Optical-Flow.
    Der Flow wird geglättet (smooth_sigma) und in der Länge auf `cap_px` gedeckelt (verhindert
    wilde Warps in strukturlosen Flächen), dann wird `mov` per remap gewarpt. Gibt das gewarpte
    Bild (gleicher dtype) zurück. Auf grobe globale Ausrichtung ANWENDEN, nie als erster Schritt."""
    gr, gm = _gray8(ref), _gray8(mov)
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(gr, gm, None)                       # (h,w,2): mov→ref
    except Exception as e:
        if log:
            log(f"    (lokale Ausrichtung übersprungen: {e})")
        return mov
    if smooth_sigma and smooth_sigma > 0:
        flow = cv2.GaussianBlur(flow, (0, 0), smooth_sigma)
    if cap_px and cap_px > 0:                                # Verschiebung deckeln
        mag = np.linalg.norm(flow, axis=2, keepdims=True)
        scale = np.minimum(1.0, cap_px / np.maximum(mag, 1e-6))
        flow = flow * scale
    h, w = gr.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = (gx + flow[..., 0]).astype(np.float32)
    mapy = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(mov, mapx, mapy, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE)


def align_pair(ref, mov, init=None, motion="euclidean", local=False,
               cap_px=6.0, min_cc=0.0, log=None):
    """Komplett-Ausrichtung eines Frames `mov` auf `ref`:
    1) ECC-Subpixel-Verfeinerung (auf `init` aufsetzend), 2) optional lokaler Optical-Flow.
    Gibt (ausgerichtetes_bild, korrelationskoeffizient) zurück. cc < min_cc → Aufrufer kann verwerfen."""
    warp, cc = ecc_refine(ref, mov, init=init, motion=motion)
    out = apply_warp(mov, warp, inverse=True)               # ECC-Matrix als Inverse anwenden
    if local:
        out = flow_warp(ref, out, cap_px=cap_px, log=log)
    return out, cc
