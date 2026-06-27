#!/usr/bin/env python3
"""
develop.py — RAW-Entwicklungs-/Editor-Primitive (RawTherapee/darktable-Niveau, treu/nicht-generativ).

Enthält die hochwertigen Bausteine, die ForgePix' Editor bisher fehlten:
  • highlight_reconstruct — ausgebrannte Lichter rekonstruieren (Kanal-Verhältnis-Füllung +
    Entsättigen-zu-Weiß gegen magenta Lichter)
  • tone_curve_lut / apply_lut — Tonwertkurve (PCHIP-Punkt-Kurve, ohne Überschwingen)
  • fast_denoise — kantenerhaltendes Entrauschen (Non-Local-Means, Luma/Chroma)
  • gradient_mask / radial_mask / refine_mask — lokale Anpassungs-Masken (Smoothstep + Guided Filter)

Schärfung (Wavelet/Unsharp) liegt in wavelet.py. Reine OpenCV/NumPy(/scipy)-Abhängigkeiten.
"""
import numpy as np
import cv2


def _as_float(img):
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, np.uint16, 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, np.uint8, 255.0
    return img.astype(np.float32), np.float32, 1.0


def highlight_reconstruct(img, thresh=0.92, sigma=8.0):
    """Ausgebrannte Lichter rekonstruieren. Für teil-geclippte Pixel werden geclippte Kanäle aus
    dem maskierten Weichzeichner der ungeclippten Nachbarn desselben Kanals gefüllt (LCh-artig);
    voll ausgebrannte Pixel werden zu **neutralem Weiß** gezogen (gegen den magenta Lichter-Stich,
    den rohes Clipping erzeugt). Treu, kein Erfinden von Struktur."""
    f, dtype, maxv = _as_float(img)
    if f.ndim != 3:
        return img
    out = f.copy()
    for c in range(3):
        ch = f[..., c]
        clipped = ch >= thresh
        if not clipped.any():
            continue
        valid = (~clipped).astype(np.float32)
        num = cv2.GaussianBlur(ch * valid, (0, 0), sigma)
        den = cv2.GaussianBlur(valid, (0, 0), sigma) + 1e-6
        filled = num / den
        out[..., c] = np.where(clipped, np.maximum(ch, filled), ch)
    anyclip = (f >= thresh).any(axis=2)
    if anyclip.any():
        mx = out.max(axis=2, keepdims=True)
        neutral = np.maximum(out, mx * 0.92)                  # Kanäle Richtung Pixel-Maximum → weiß
        m = anyclip[..., None]
        out = np.where(m, neutral, out)
    out = np.clip(out, 0, 1)
    return (out * maxv).astype(dtype) if dtype != np.float32 else out


def tone_curve_lut(points, bits=16):
    """Monotone PCHIP-Punkt-Kurve → LUT (kein Überschwingen, anders als natürlicher Spline).
    points: Liste (x,y) in [0,1], (0,0)/(1,1) werden ergänzt."""
    try:
        from scipy.interpolate import PchipInterpolator
        pts = dict(points)
        pts.setdefault(0.0, 0.0); pts.setdefault(1.0, 1.0)
        xs = np.array(sorted(pts)); ys = np.array([pts[x] for x in xs])
        spl = PchipInterpolator(xs, ys)
        n = 1 << bits
        grid = np.linspace(0, 1, n)
        return np.clip(spl(grid), 0, 1).astype(np.float32)
    except Exception:
        n = 1 << bits
        return np.linspace(0, 1, n, dtype=np.float32)


def apply_lut(img, lut01):
    """LUT (float [0,1], Länge 2^bits) auf das Bild anwenden — auf der Luminanz, Farbe erhalten."""
    f, dtype, maxv = _as_float(img)
    n = len(lut01)
    if f.ndim == 3:
        y = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
        idx = np.clip(y * (n - 1), 0, n - 1).astype(np.int32)
        gain = np.where(y > 1e-4, lut01[idx] / np.maximum(y, 1e-4), 1.0)[..., None]
        out = np.clip(f * gain, 0, 1)
    else:
        idx = np.clip(f * (n - 1), 0, n - 1).astype(np.int32)
        out = lut01[idx]
    return (out * maxv).astype(dtype) if dtype != np.float32 else out


def fast_denoise(img, luma=7.0, chroma=10.0):
    """Kantenerhaltendes Entrauschen (Non-Local-Means, getrennt Luma/Chroma). Schnell genug für
    die Vorschau. Für stärkstes Entrauschen multiskalig → wavelet.wavelet_denoise."""
    f, dtype, maxv = _as_float(img)
    u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if u8.ndim == 3:
        d = cv2.fastNlMeansDenoisingColored(u8, None, float(luma), float(chroma), 7, 21)
    else:
        d = cv2.fastNlMeansDenoising(u8, None, float(luma), 7, 21)
    out = d.astype(np.float32) / 255.0
    return (out * maxv).astype(dtype) if dtype != np.float32 else out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def gradient_mask(shape, cx, cy, angle_deg, feather=0.3):
    """Verlaufs-Maske (0..1): linearer Übergang senkrecht zur Achse durch (cx,cy). feather in
    Anteil der Bilddiagonale."""
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle_deg)
    d = (xs - cx) * np.cos(a) + (ys - cy) * np.sin(a)
    fw = feather * np.hypot(h, w)
    return _smoothstep(-fw / 2, fw / 2, d).astype(np.float32)


def radial_mask(shape, cx, cy, rx, ry, feather=0.3):
    """Radial-/Ellipsen-Maske (1 innen, 0 außen) mit weichem Rand."""
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs - cx) / max(rx, 1e-6)) ** 2 + ((ys - cy) / max(ry, 1e-6)) ** 2)
    return (1.0 - _smoothstep(1.0 - feather, 1.0, r)).astype(np.float32)


def refine_mask(mask, guide_img, radius=16, eps=1e-3):
    """Maske kanten-treu an die Bildkanten anschmiegen (Guided Filter), falls verfügbar."""
    try:
        g = cv2.cvtColor(guide_img, cv2.COLOR_BGR2GRAY) if guide_img.ndim == 3 else guide_img
        g = g.astype(np.float32) / (65535.0 if g.dtype == np.uint16 else 255.0) \
            if g.dtype != np.float32 else g
        return cv2.ximgproc.guidedFilter(g, mask.astype(np.float32), radius, eps)
    except Exception:
        return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
