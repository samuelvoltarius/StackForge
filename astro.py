#!/usr/bin/env python3
"""
astro.py — Astro-Stacking-Modul für StackForge (Siril-inspiriert, eigenständig).

ANDERER Algorithmus als Fokus-Stacking: Ziel = Rauschen senken (SNR), nicht Schärfentiefe.
Pipeline:
  1. (optional) Kalibrierung: Master-Dark abziehen, durch Master-Flat teilen.
  2. Registrierung: Frames per Phasenkorrelation aufs Referenzbild ausrichten (Translation,
     sub-pixel; robust für nachgeführte Aufnahmen). -> ausgerichtete Temp-Frames auf Platte.
  3. Stacking mit Rejection: average / median / sigma (Kappa-Sigma) / winsor / max.
     Zweistufig über die Platte gerechnet -> speicherschonend auch bei 100+ Frames.
  4. (optional) Auto-Stretch (asinh) fürs Anzeigen des linearen Ergebnisses.

Speicher: hält nie alle Frames gleichzeitig im RAM (anders als der Fokus-Stacker).
Reine OpenCV/NumPy-Abhängigkeiten.
"""
import os
import numpy as np
import cv2

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".dng", ".orf", ".pef", ".srw"}


def _read_float(path):
    """Bild als float32 [0..1] (BGR) lesen — TIFF/PNG/JPG/FITS; RAW via rawpy."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits", ".fts"):
        from astropy.io import fits
        d = np.asarray(fits.getdata(path)).astype(np.float32)
        if d.ndim == 3 and d.shape[0] in (3, 4):     # (C,H,W) -> (H,W,C)
            d = np.moveaxis(d[:3], 0, -1)
        mx = float(np.nanmax(d)) if d.size else 1.0
        f = d / mx if mx > 1.5 else np.clip(d, 0, 1)  # ADU -> 0..1
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)    # FITS = RGB -> BGR
        return np.nan_to_num(f)
    if ext in RAW_EXTS:
        import rawpy
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True,
                                  output_color=rawpy.ColorSpace.sRGB)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); maxv = 65535.0
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        maxv = 65535.0 if img.dtype == np.uint16 else 255.0
    f = img.astype(np.float32) / maxv
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    return f


def _master(paths):
    """Master-Frame (Median) aus mehreren Kalibrier-Frames, oder einzelnes Frame."""
    if isinstance(paths, str):
        return _read_float(paths)
    fs = [_read_float(p) for p in paths]
    return np.median(np.stack(fs), axis=0)


def calibrate(f, dark=None, flat=None):
    out = f
    if dark is not None:
        out = out - dark
    if flat is not None:
        fn = flat / (float(flat.mean()) + 1e-6)
        out = out / np.clip(fn, 0.2, None)
    return np.clip(out, 0, None)


def _gray(f):
    return f.mean(axis=2).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)


def register_and_cache(paths, out_dir, dark=None, flat=None, do_register=True, log=print):
    """Frames kalibrieren + per Phasenkorrelation aufs Referenzbild ausrichten,
    als 16-bit-TIFF in out_dir ablegen. Gibt die Liste der ausgerichteten Pfade zurück."""
    os.makedirs(out_dir, exist_ok=True)
    ref = calibrate(_read_float(paths[len(paths) // 2]), dark, flat)
    refg = _gray(ref)
    win = cv2.createHanningWindow((refg.shape[1], refg.shape[0]), cv2.CV_32F)
    aligned = []
    for i, p in enumerate(paths):
        f = calibrate(_read_float(p), dark, flat)
        if f.shape[:2] != ref.shape[:2]:
            f = cv2.resize(f, (ref.shape[1], ref.shape[0]))
        if do_register:
            (dx, dy), _resp = cv2.phaseCorrelate(refg * win, _gray(f) * win)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        op = os.path.join(out_dir, f"reg_{i:04d}.tif")
        cv2.imwrite(op, np.clip(f * 65535, 0, 65535).astype(np.uint16),
                    [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
        aligned.append(op)
        log(f"    registriert {i + 1}/{len(paths)}")
    return aligned


def stack(paths, method="sigma", kappa=2.5, normalize=True, log=print):
    """Speicherschonendes Stacken über die Platte (zweistufig bei sigma/winsor).
    Gibt float32-Ergebnis [0..1] (BGR) zurück."""
    n = len(paths)
    first = _read_float(paths[0])
    shape = first.shape

    # additive Normalisierung: Hintergrund (Median) je Frame angleichen
    offs = np.zeros(n, np.float32)
    if normalize:
        meds = [float(np.median(_read_float(p))) for p in paths]
        gm = float(np.median(meds))
        offs = np.array([gm - m for m in meds], np.float32)

    if method in ("average", "median", "max"):
        if method == "median":
            # Median braucht alle Werte -> in Kacheln über die Höhe, speicherschonend
            res = np.empty(shape, np.float32)
            rows = max(1, 2_000_000 // (shape[1] * shape[2]))  # ~Zeilen pro Kachel
            for y in range(0, shape[0], rows):
                band = np.stack([_read_float(p)[y:y + rows] + offs[i]
                                 for i, p in enumerate(paths)])
                res[y:y + rows] = np.median(band, axis=0)
                log(f"    median Zeilen {y}/{shape[0]}")
            return np.clip(res, 0, 1)
        acc = np.zeros(shape, np.float32) if method == "average" else None
        mx = np.zeros(shape, np.float32) if method == "max" else None
        for i, p in enumerate(paths):
            f = _read_float(p) + offs[i]
            if method == "average":
                acc += f
            else:
                mx = np.maximum(mx, f)
            log(f"    {method} {i + 1}/{n}")
        return np.clip(acc / n if method == "average" else mx, 0, 1)

    # sigma / winsor: Pass 1 Mittel+Std, Pass 2 Rejection
    s = np.zeros(shape, np.float32); s2 = np.zeros(shape, np.float32)
    for i, p in enumerate(paths):
        f = _read_float(p) + offs[i]
        s += f; s2 += f * f
        log(f"    Statistik {i + 1}/{n}")
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0))
    lo = mean - kappa * std; hi = mean + kappa * std
    acc = np.zeros(shape, np.float32); cnt = np.zeros(shape, np.float32)
    for i, p in enumerate(paths):
        f = _read_float(p) + offs[i]
        if method == "winsor":
            f = np.clip(f, lo, hi)
            acc += f; cnt += 1
        else:  # sigma: Ausreißer verwerfen
            m = (f >= lo) & (f <= hi)
            acc += np.where(m, f, 0); cnt += m
        log(f"    {method}-Rejection {i + 1}/{n}")
    return np.clip(acc / np.clip(cnt, 1, None), 0, 1)


def background_extract(f, strength=0.12):
    """Klassische Hintergrund-/Gradienten-Entfernung (Lichtverschmutzung, Vignette).
    Modelliert den glatten Hintergrund (sternunterdrückt + stark geglättet) und zieht ihn ab.
    Kein KI-Tool wie GraXpert, aber wirksam gegen weiche Gradienten."""
    u16 = (np.clip(f, 0, 1) * 65535).astype(np.uint16)
    star_suppressed = cv2.medianBlur(u16, 5).astype(np.float32) / 65535.0
    sigma = max(8.0, min(f.shape[0], f.shape[1]) * strength / 3.0)
    bg = cv2.GaussianBlur(star_suppressed, (0, 0), sigma)
    out = f - bg + float(np.median(bg))
    return np.clip(out, 0, 1)


def autostretch(f, black_clip=0.001):
    """asinh-Stretch fürs Anzeigen des (linearen, dunklen) Astro-Ergebnisses."""
    g = _gray(f)
    bg = np.quantile(g, black_clip)
    x = np.clip(f - bg, 0, None)
    x = x / (np.quantile(x, 0.9995) + 1e-6)
    out = np.arcsinh(x * 10.0) / np.arcsinh(10.0)
    return np.clip(out, 0, 1)
