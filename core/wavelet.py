#!/usr/bin/env python3
"""
wavelet.py — à-trous (stationäre) Wavelet-Zerlegung + RegiStax-artige Multi-Skalen-Schärfung.

Statt eines einzelnen Unsharp-Masks (eine Skala, Halo-anfällig) zerlegt die à-trous-Transformation
das Bild in mehrere **Frequenzbänder** (fein → grob) und erlaubt, jedes **einzeln** zu verstärken
und zu entrauschen — wie die RegiStax-Wavelet-Regler. Halo-arm und rausch-kontrollierbar.

à trous: rekursiv mit einem B3-Spline-Kern [1,4,6,4,1]/16 tiefpassfiltern, den Kern je Ebene
**dilatieren** (Löcher einfügen) → keine Verkleinerung, Bild bleibt voll aufgelöst.
Detail-Ebene i = approx[i-1] − approx[i]. Rekonstruktion = Σ gain_i · detail_i + Rest-Approx.

Geteilt von: Lucky-Imaging (Final-Schärfung), Astro (Detail), RAW-Editor (Capture-Schärfung).
Reine OpenCV/NumPy-Abhängigkeiten (MIT-kompatibel).
"""
import numpy as np
import cv2

_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16.0


def _dilated_kernel(level):
    """B3-Spline-Kern mit 2^level-1 Nullen zwischen den Taps (à-trous-„Löcher")."""
    step = 1 << level
    k = np.zeros(4 * step + 1, np.float32)
    k[::step] = _B3
    return k


def atrous(gray, levels=5):
    """à-trous-Zerlegung. Gibt (detail_ebenen[fein..grob], rest_approx) zurück (alle float32)."""
    approx = gray.astype(np.float32)
    details = []
    for i in range(levels):
        k = _dilated_kernel(i)
        sm = cv2.sepFilter2D(approx, cv2.CV_32F, k, k, borderType=cv2.BORDER_REFLECT)
        details.append(approx - sm)
        approx = sm
    return details, approx


def wavelet_sharpen(img, gains=(2.0, 1.6, 1.3, 1.1, 1.0), denoise=0.0, levels=None):
    """Multi-Skalen-Schärfung (RegiStax-Stil). gains[i] = Verstärkung der i-ten Detail-Ebene
    (Index 0 = feinste). denoise>0 = Soft-Threshold auf den feinen Ebenen (gegen Rausch-Verstärkung).
    Wirkt auf die Luminanz (Farbe bleibt erhalten). Gibt das Bild im Eingabe-dtype zurück."""
    dtype = img.dtype
    maxv = 65535.0 if dtype == np.uint16 else 255.0
    levels = levels if levels else len(gains)
    color = img.ndim == 3
    f = img.astype(np.float32)
    # Luminanz schärfen, das DELTA auf alle Kanäle addieren → farbtreu, kein cvtColor-Konventions-Trap
    y = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]) if color else f
    details, approx = atrous(y, levels)
    out = approx.copy()
    for i, d in enumerate(details):
        g = gains[i] if i < len(gains) else 1.0
        if denoise > 0:                                     # feine Ebenen stärker entrauschen
            t = denoise * float(np.std(d)) * (0.6 ** i)
            d = np.sign(d) * np.maximum(np.abs(d) - t, 0.0)
        out = out + g * d
    if color:
        delta = (out - y)[..., None]
        return np.clip(f + delta, 0, maxv).astype(dtype)
    return np.clip(out, 0, maxv).astype(dtype)


def wavelet_denoise(img, strength=1.0, levels=4):
    """Reines Multi-Skalen-Entrauschen (Soft-Threshold je Ebene, BayesShrink-artig) — feine Ebenen
    stärker. Geteilt mit dem RAW-Editor. Gibt Eingabe-dtype zurück."""
    return wavelet_sharpen(img, gains=tuple([1.0] * levels), denoise=strength, levels=levels)
