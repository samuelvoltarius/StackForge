#!/usr/bin/env python3
"""
astro.py — Astro-Stacking-Modul für ForgePix (Siril-inspiriert, eigenständig).

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

from constants import RAW_EXTS

# OSC-Bayer-Muster (FITS BAYERPAT) -> OpenCV-Debayer-Code. Achtung: OpenCVs Bayer-Benennung ist
# ggü. der üblichen FITS-Konvention um eine Zeile/Spalte verschoben (bekannte Falle).
_BAYER2CV = {
    "RGGB": cv2.COLOR_BayerBG2BGR,
    "BGGR": cv2.COLOR_BayerRG2BGR,
    "GRBG": cv2.COLOR_BayerGB2BGR,
    "GBRG": cv2.COLOR_BayerGR2BGR,
}


def detect_bayer(d):
    """CFA-Muster selbst erkennen, wenn kein BAYERPAT im Header steht. Probiert alle 4 Muster,
    debayert einen zentralen Ausschnitt und wählt das mit den GERINGSTEN Farb-Artefakten
    (falsche Muster erzeugen starkes Farb-Zipper/Schachbrett). Default RGGB bei Fehler."""
    try:
        a = np.nan_to_num(np.asarray(d)).astype(np.float32)
        if a.ndim != 2:
            return "RGGB"
        mx = float(a.max()) or 1.0
        h, w = a.shape
        cy, cx = (h // 4) * 2, (w // 4) * 2        # zentriert, gerade Offsets (CFA-Phase wahren)
        s = min(400, (min(h, w) // 2) * 2)
        crop = a[cy:cy + s, cx:cx + s]
        raw16 = np.clip(crop / mx * 65535.0, 0, 65535).astype(np.uint16)
        best, best_score = "RGGB", None
        for pat, code in _BAYER2CV.items():
            bgr = cv2.cvtColor(raw16, code).astype(np.float32) / 65535.0
            chroma = bgr - bgr.mean(axis=2, keepdims=True)   # Farbabweichung vom Grau
            score = float(np.mean(np.abs(cv2.Laplacian(chroma, cv2.CV_32F))))
            if best_score is None or score < best_score:
                best, best_score = pat, score
        return best
    except Exception:
        return "RGGB"


def _read_float(path):
    """Bild als float32 [0..1] (BGR) lesen — TIFF/PNG/JPG/FITS; RAW via rawpy."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits", ".fts"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdu = hdul[0]
            d = np.asarray(hdu.data).astype(np.float32)
            bayer = str(hdu.header.get("BAYERPAT", "")).strip().upper()
        if d.ndim == 3 and d.shape[0] in (3, 4):     # (C,H,W) -> (H,W,C)
            d = np.moveaxis(d[:3], 0, -1)
        # OSC-Kameras (z. B. Seestar/ASI) liefern Bayer-Rohdaten als 2D-FITS -> debayern = Farbe.
        # BAYERPAT aus dem Header, sonst SELBST erkennen (gegen Dateien ohne Header-Eintrag).
        if d.ndim == 2:
            if bayer not in _BAYER2CV:
                bayer = detect_bayer(d)
            if bayer in _BAYER2CV:
                mx = float(np.nanmax(d)) or 1.0
                raw16 = np.clip(np.nan_to_num(d) / mx * 65535.0, 0, 65535).astype(np.uint16)
                bgr = cv2.cvtColor(raw16, _BAYER2CV[bayer])      # -> BGR (Farbe!)
                return bgr.astype(np.float32) / 65535.0
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
        if img is None:
            raise RuntimeError(f"Bild nicht lesbar: {path}")
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


def cosmetic_correct(f, strength=3.0):
    """Hot-/Cold-Pixel entfernen (kosmetische Korrektur): einzelne Ausreißer ggü. dem lokalen
    Median ersetzen. Beseitigt helle/dunkle Einzelpixel (Sensor-Defekte/Cosmics) ohne Sterne
    anzutasten. Klassisch, kein ML."""
    u16 = (np.clip(f, 0, 1) * 65535).astype(np.uint16)
    med = cv2.medianBlur(u16, 3).astype(np.float32) / 65535.0
    diff = f - med
    sigma = float(np.std(diff)) + 1e-6
    mask = np.abs(diff) > strength * sigma
    out = f.copy()
    out[mask] = med[mask]
    return out


def _star_centroids(g, max_stars=200):
    """Sternzentren (sub-pixel) als Punktwolke: Hintergrund abziehen, **rauschadaptive Schwelle
    (Median + 5·MAD)**, kleine helle Blobs als Sterne, nach Fläche sortiert.

    Wichtig: Otsu lieferte auf dünnen Astro-Frames nur eine Handvoll Sterne (zu strenge Schwelle),
    wodurch das Ausrichten zu wenig Stützpunkte hatte und Sterne im Stack verschmierten. Die
    MAD-Schwelle findet zuverlässig 100–200 Sterne — genug für robustes Offset-Voting + RANSAC."""
    a = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    bg = cv2.medianBlur(a, 31)
    sub = cv2.subtract(a, bg).astype(np.float32)
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) * 1.4826 + 1e-6
    th = (sub > max(med + 5.0 * mad, 3.0)).astype(np.uint8) * 255
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(th, connectivity=8)
    stars = [(cent[i][0], cent[i][1], int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n) if 2 <= stats[i, cv2.CC_STAT_AREA] <= 600]
    stars.sort(key=lambda s: -s[2])
    return np.array([[s[0], s[1]] for s in stars[:max_stars]], np.float32)


def _coarse_offset_vote(ref_pts, img_pts, nbright=80, tol=2.5, min_votes=8):
    """Dominanten Versatz (ref − img) aus den hellsten Sternpaaren per Voting bestimmen.

    Robust gegen feste Hotpixel/Amp-Glow (die würden für Versatz (0,0) stimmen) und gegen
    fehlende/zusätzliche Sterne: der echte Sternversatz bekommt die meisten übereinstimmenden
    Stimmen. Ersetzt die Phasenkorrelation, die bei Astro-Frames auf dem festen Fixed-Pattern
    statt auf den (gewanderten) Sternen einrastet. Gibt (ox, oy) oder None."""
    if len(ref_pts) < min_votes or len(img_pts) < min_votes:
        return None
    R, I = ref_pts[:nbright], img_pts[:nbright]
    offs = (R[:, None, :] - I[None, :, :]).reshape(-1, 2)        # alle Paar-Versätze
    best, bestc = None, 0
    for o in offs:
        c = int((np.abs(offs - o).max(1) < tol).sum())          # Übereinstimmungen
        if c > bestc:
            bestc, best = c, o
    if bestc < min_votes:
        return None
    # Mittel der zustimmenden Versätze (sub-pixel-genauer als ein einzelner Paar-Versatz)
    near = offs[np.abs(offs - best).max(1) < tol]
    return near.mean(0)


def _estimate_star_transform(refg, img_g):
    """Translation + Feldrotation aus tatsächlichen STERNPOSITIONEN schätzen (genauer = rundere
    Sterne). Grober Versatz per Offset-Voting (robust gegen Hotpixel), dann Nearest-Neighbor-Match
    + RANSAC-Affine. Gibt 2x3-Matrix oder None (dann Fallback ORB)."""
    ref_pts, img_pts = _star_centroids(refg), _star_centroids(img_g)
    if len(ref_pts) < 8 or len(img_pts) < 8:
        return None
    off = _coarse_offset_vote(ref_pts, img_pts)
    if off is None:
        return None
    shifted = img_pts + off                                     # img ≈ in ref-Raster gebracht
    src, dst = [], []
    for rp in ref_pts:                                          # ref-Stern -> nächster img-Stern
        d = np.linalg.norm(shifted - rp, axis=1)
        j = int(np.argmin(d))
        if d[j] < 4.0:                                          # Toleranz in px (nach Grobversatz)
            src.append(img_pts[j]); dst.append(rp)
    if len(src) < 6:
        return None
    M, inl = cv2.estimateAffinePartial2D(np.array(src, np.float32), np.array(dst, np.float32),
                                         method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or (inl is not None and int(inl.sum()) < 6):
        return None
    return M


def _estimate_rotation(refg, img_g, detector="ORB", min_inliers=25):
    """Partielle Affine (Translation + Rotation, kein Scherung) per ORB-Merkmalen schätzen —
    Fallback, wenn das stern-basierte Voting den Versatz nicht findet (z. B. großer Dither-Sprung
    in einen wenig überlappenden Bereich). Gibt 2x3-Matrix nur bei genügend Inliern zurück, sonst
    None — damit unsicher ausgerichtete Frames lieber verworfen als verschmiert gestackt werden."""
    a = (np.clip(refg, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(img_g, 0, 1) * 255).astype(np.uint8)
    det = cv2.ORB_create(5000)
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:300]
    if len(m) < min_inliers:
        return None
    src = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    return M


def _compose_affine(A, B):
    """Zwei 2x3-Affinen verketten: Ergebnis bildet ab wie erst B, dann A (A∘B)."""
    A3 = np.vstack([A, [0, 0, 1]]); B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def _tps_refine(fw, refg_out, max_ctrl=150, min_resid=0.5, log=print):
    """Lokale (nicht-rigide) Feinregistrierung per Thin-Plate-Spline gegen RESTVERZEICHNUNG —
    Feldkrümmung bei Weitwinkel/Refraktor, atmosphärische Refraktion, leichtes Tilt. Nach der
    globalen Affin-Ausrichtung bleibende Restversätze der Sterne werden als glattes Warp-Feld
    herausgerechnet (Sterne werden über das ganze Feld rund). Nur aktiv, wenn genug Sternpaare
    mit echtem Restversatz da sind — sonst bleibt der Frame unverändert (kein Verschlimmbessern)."""
    try:
        from scipy.interpolate import RBFInterpolator
    except Exception:
        return fw
    fg = _gray(fw)
    rp = _star_centroids(refg_out, max_stars=max_ctrl)
    ip = _star_centroids(fg, max_stars=max_ctrl * 3)
    if len(rp) < 12 or len(ip) < 12:
        return fw
    src, dst = [], []
    for r in rp:                                            # ref-Pos -> nächste Frame-Pos
        d = np.linalg.norm(ip - r, axis=1)
        j = int(np.argmin(d))
        if d[j] < 6.0:
            dst.append(r); src.append(ip[j])
    if len(src) < 12:
        return fw
    src = np.array(src, np.float32); dst = np.array(dst, np.float32)
    resid = np.linalg.norm(src - dst, axis=1)
    if float(np.median(resid)) < min_resid:
        return fw                                           # global schon sauber → nichts zu tun
    h, w = fg.shape[:2]
    try:
        rbf = RBFInterpolator(dst, src, kernel="thin_plate_spline", smoothing=1.0)
        gs = 48                                             # grobes Gitter, TPS ist glatt → hochskalieren
        gx, gy = np.meshgrid(np.linspace(0, w - 1, gs), np.linspace(0, h - 1, gs))
        q = np.stack([gx.ravel(), gy.ravel()], 1)
        mapped = rbf(q).reshape(gs, gs, 2).astype(np.float32)
        mapx = cv2.resize(mapped[..., 0], (w, h))
        mapy = cv2.resize(mapped[..., 1], (w, h))
        out = cv2.remap(fw, mapx, mapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        log(f"      TPS-Feinregistrierung: {len(src)} Sterne, Rest {float(np.median(resid)):.2f}px")
        return out
    except Exception:
        return fw


def _warp_and_save(f, M, out_size, op, drizzle, tps_refg=None):
    if M is not None:
        if drizzle > 1:
            M = M.copy(); M[:, 2] *= drizzle
        f = cv2.warpAffine(f, M, out_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    elif drizzle > 1:
        f = cv2.resize(f, out_size, interpolation=cv2.INTER_LANCZOS4)
    if tps_refg is not None:
        f = _tps_refine(f, tps_refg, log=lambda *a: None)
    cv2.imwrite(op, np.clip(f * 65535, 0, 65535).astype(np.uint16),
                [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
    return op


def register_and_cache(paths, out_dir, dark=None, flat=None, do_register=True,
                       align_mode="shift", cosmetic=False, drizzle=1, detector="ORB",
                       tps=False, log=print):
    """Frames kalibrieren + ausrichten, als 16-bit-TIFF in out_dir ablegen.

    align_mode: 'shift'/'rotate' (stern-basiert; phaseCorrelate wird bewusst NICHT genutzt — rastet
                bei Astro auf dem festen Fixed-Pattern statt auf den gewanderten Sternen ein).
    cosmetic:   Hot-/Cold-Pixel vor dem Ausrichten entfernen.
    drizzle:    Ausgabe-Hochskalierung (1=aus, 2=doppelte Kantenlänge, „Drizzle-lite").

    Parallel über alle Kerne (OpenCV gibt den GIL frei). Frames, die sich nicht an die Referenz
    ausrichten lassen (großer Dither-Sprung), werden in einem 2. Pass über eine Cluster-Brücke
    zurückgeholt statt verworfen. Gibt die Liste der ausgerichteten Pfade zurück."""
    from parallel import pmap
    os.makedirs(out_dir, exist_ok=True)
    drizzle = max(1, int(drizzle))
    ref = calibrate(_read_float(paths[len(paths) // 2]), dark, flat)
    if cosmetic:
        ref = cosmetic_correct(ref)
    refg = _gray(ref)
    out_size = (ref.shape[1] * drizzle, ref.shape[0] * drizzle)
    tps_refg = (cv2.resize(refg, out_size) if drizzle > 1 else refg) if tps else None
    if tps:
        log("    TPS-Feinregistrierung aktiv (lokale Restverzeichnung wird korrigiert)")

    def _prep(i):
        f = calibrate(_read_float(paths[i]), dark, flat)
        if f.shape[:2] != ref.shape[:2]:
            f = cv2.resize(f, (ref.shape[1], ref.shape[0]))
        if cosmetic:
            f = cosmetic_correct(f)
        return f

    def _one(i):
        f = _prep(i)
        op = os.path.join(out_dir, f"reg_{i:04d}.tif")
        if not do_register:
            return (i, _warp_and_save(f, None, out_size, op, drizzle))
        fg = _gray(f)
        M = _estimate_star_transform(refg, fg)
        if M is None:
            M = _estimate_rotation(refg, fg, detector)
        if M is None:
            return (i, None)                                 # 2. Pass versucht Cluster-Brücke
        return (i, _warp_and_save(f, M, out_size, op, drizzle, tps_refg))

    results = pmap(_one, list(range(len(paths))), memory_heavy=True)
    aligned = [op for _i, op in sorted(results) if op]
    skipped = [i for i, op in sorted(results) if op is None]
    log(f"    registriert {len(aligned)}/{len(paths)} (Pass 1)")

    # ---- Pass 2: weit weggeditherte Frames über eine Cluster-Brücke zurückholen ----
    # Sub-Referenz im Cluster wählen → per ORB an die Hauptreferenz brücken → jeden Frame an die
    # Sub-Referenz ausrichten und die Transforms verketten. JEDER zurückgeholte Frame wird verifiziert
    # (seine Sterne müssen nach der Transformation gut auf die Referenz fallen), sonst bleibt er außen
    # vor — so kann eine schwache Brücke kein Verschmieren zurückbringen.
    if do_register and len(skipped) >= 3:
        ref_pts = _star_centroids(refg)
        grays = {i: _gray(_prep(i)) for i in skipped}
        subref = max(skipped, key=lambda i: len(_star_centroids(grays[i])))
        bridge = _estimate_rotation(refg, grays[subref], detector, min_inliers=10)  # subref -> ref
        rescued = 0
        if bridge is not None and len(ref_pts) >= 8:
            for i in skipped:
                S = (np.float32([[1, 0, 0], [0, 1, 0]]) if i == subref
                     else _estimate_star_transform(grays[subref], grays[i]))   # frame -> subref
                if S is None:
                    continue
                M = _compose_affine(bridge, S)               # frame -> subref -> ref
                # Verifizieren: Sterne des Frames mit M ins Ref-Raster bringen, gute Treffer zählen
                ip = _star_centroids(grays[i])
                if len(ip) < 8:
                    continue
                ext = np.hstack([ip, np.ones((len(ip), 1), np.float32)])
                tp = (M @ ext.T).T
                good = sum(1 for r in ref_pts if np.min(np.linalg.norm(tp - r, axis=1)) < 1.5)
                if good < 25:                                # zu wenige saubere Treffer → lieber lassen
                    continue
                op = os.path.join(out_dir, f"reg_{i:04d}.tif")
                aligned.append(_warp_and_save(_prep(i), M, out_size, op, drizzle, tps_refg))
                rescued += 1
        if rescued:
            log(f"    +{rescued} weit geditherte Frames über Cluster-Brücke zurückgeholt ({len(aligned)}/{len(paths)})")
    return aligned


def drizzle_stack(paths, scale=2, pixfrac=0.7, dark=None, flat=None, cosmetic=False,
                  detector="ORB", log=print):
    """ECHTES Drizzle (Variable-Pixel Linear Reconstruction, Fruchter & Hook — Punktkernel,
    inverse Formulierung). Anders als „Drizzle-lite“ (jeden Frame einzeln hochskalieren und mitteln,
    was die Interpolation verschmiert) wird hier das Resampling AUFGESCHOBEN: jeder Roh-Frame wird
    über seine Sub-Pixel-Registrierung mit einem geschrumpften „Drop“ (pixfrac) direkt auf das feine
    Ausgabegitter getropft, Fluss UND Gewicht akkumuliert. Aus geditherten Subs entsteht so echte
    Auflösungsrückgewinnung (kleinere, schärfere Sterne) statt nur Hochskalierung.

    scale: Gitter-Faktor (2 = doppelte Kantenlänge). pixfrac: Drop-Größe 0.1..1 (kleiner = schärfer,
    braucht aber mehr Frames für volle Abdeckung; 0.7 ist ein guter Kompromiss)."""
    scale = int(max(2, scale))
    pf = float(np.clip(pixfrac, 0.1, 1.0))
    ref = calibrate(_read_float(paths[len(paths) // 2]), dark, flat)
    if cosmetic:
        ref = cosmetic_correct(ref)
    refg = _gray(ref)
    H, W = ref.shape[:2]
    ch = ref.shape[2] if ref.ndim == 3 else 1
    OH, OW = H * scale, W * scale
    flux = np.zeros((OH, OW, ch), np.float32)
    wsum = np.zeros((OH, OW), np.float32)
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    used = 0
    for k, p in enumerate(paths):
        f = calibrate(_read_float(p), dark, flat)
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H))
        if cosmetic:
            f = cosmetic_correct(f)
        if f.ndim == 2:
            f = f[..., None]
        fg = _gray(f)
        M = _estimate_star_transform(refg, fg)
        if M is None:
            M = _estimate_rotation(refg, fg, detector)
        if M is None:
            continue
        Ms = (scale * M).astype(np.float32)              # frame -> Ausgabegitter
        Minv = cv2.invertAffineTransform(Ms)             # Ausgabegitter -> frame
        xs = Minv[0, 0] * xx + Minv[0, 1] * yy + Minv[0, 2]
        ys = Minv[1, 0] * xx + Minv[1, 1] * yy + Minv[1, 2]
        rx = np.round(xs); ry = np.round(ys)
        inb = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
        wgt = ((np.abs(xs - rx) <= pf / 2) & (np.abs(ys - ry) <= pf / 2) & inb).astype(np.float32)
        val = cv2.remap(f, xs, ys, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT)
        if val.ndim == 2:
            val = val[..., None]
        flux += val * wgt[..., None]
        wsum += wgt
        used += 1
        log(f"    Drizzle {used}/{len(paths)} (pixfrac {pf:.2f}, {scale}×)")
    if used == 0:
        raise RuntimeError("Drizzle: kein Frame ausrichtbar")
    out = flux / np.clip(wsum[..., None], 1e-6, None)
    holes = wsum < 1e-6
    if holes.any():                                       # nie getroffene Pixel sanft füllen
        u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        filled = cv2.inpaint(u8, holes.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        out[holes] = filled[holes].astype(np.float32) / 255.0
        log(f"    Drizzle: {int(holes.sum())} unbedeckte Pixel gefüllt (mehr Frames/größeres pixfrac hilft)")
    return np.clip(out, 0, 1)


def stack(paths, method="sigma", kappa=2.5, normalize=True, local_norm=False,
          log=print, preview_cb=None):
    """Speicherschonendes Stacken über die Platte (zweistufig bei sigma/winsor).
    Gibt float32-Ergebnis [0..1] (BGR) zurück.

    preview_cb(img01_bgr, i, n): optionaler Callback für die Live-Vorschau — wird während des
    Stackens periodisch mit dem laufenden (Teil-)Ergebnis aufgerufen."""
    if not paths:
        raise RuntimeError("keine Frames zum Stacken")
    n = len(paths)
    _pv_every = max(1, n // 12)              # ~12 Vorschau-Updates über den Lauf
    first = _read_float(paths[0])
    shape = first.shape

    # additive Normalisierung: Hintergrund (Median) je Frame angleichen
    offs = np.zeros(n, np.float32)
    if normalize:
        meds = [float(np.median(_read_float(p))) for p in paths]
        gm = float(np.median(meds))
        offs = np.array([gm - m for m in meds], np.float32)
    # lokale Normalisierung: örtliche Hintergrund-Fläche statt nur Skalar (gegen Gradienten)
    ref_surf = _bg_surface(first) if (normalize and local_norm) else None
    if ref_surf is not None:
        log("    lokale Normalisierung aktiv (örtlicher Hintergrundabgleich)")

    def rd(i, p):
        f = _read_float(p)
        if ref_surf is not None:
            return local_normalize(f, ref_surf)
        return f + offs[i]

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
            f = rd(i, p)
            if method == "average":
                acc += f
            else:
                mx = np.maximum(mx, f)
            log(f"    {method} {i + 1}/{n}")
            if preview_cb and (i % _pv_every == 0 or i == n - 1):
                preview_cb(np.clip((acc / (i + 1)) if method == "average" else mx, 0, 1), i + 1, n)
        return np.clip(acc / n if method == "average" else mx, 0, 1)

    if method == "linearfit":
        # Linear-Fit-Clipping (PixInsight-Stil): pro Pixel die sortierten Werte über die Frames
        # mit einer Geraden modellieren, Streuung der Residuen messen, Ausreißer (Satelliten,
        # Flugzeuge, kosmische Treffer, Hotpixel) jenseits kappa·sigma verwerfen. Robuster als
        # Sigma-Clipping bei WENIGEN Subs und systematisch ungleicher Transparenz/Helligkeit.
        res = np.empty(shape, np.float32)
        rows = max(1, 2_000_000 // (shape[1] * shape[2]))
        x = np.arange(n, dtype=np.float32)
        xm = x.mean(); xv = float(((x - xm) ** 2).sum()) + 1e-9
        for y in range(0, shape[0], rows):
            band = np.stack([rd(i, p)[y:y + rows] for i, p in enumerate(paths)])  # (n, r, w, c)
            v = np.sort(band, axis=0)
            mask = np.ones_like(v, dtype=bool)
            for _ in range(2):                       # 2 Iterationen reichen praktisch
                w_ = mask.astype(np.float32)
                sw = np.clip(w_.sum(axis=0), 1.0, None)
                ym = (v * w_).sum(axis=0) / sw
                slope = ((x[:, None, None, None] - xm) * (v - ym) * w_).sum(axis=0) / xv
                fit = slope * (x[:, None, None, None] - xm) + ym
                resid = v - fit
                sig = np.sqrt((resid * resid * w_).sum(axis=0) / sw) + 1e-9
                mask = np.abs(resid) <= kappa * sig
            w_ = mask.astype(np.float32)
            res[y:y + rows] = (v * w_).sum(axis=0) / np.clip(w_.sum(axis=0), 1.0, None)
            log(f"    linearfit-Rejection Zeilen {y}/{shape[0]}")
        return np.clip(res, 0, 1)

    # sigma / winsor: Pass 1 Mittel+Std, Pass 2 Rejection
    s = np.zeros(shape, np.float32); s2 = np.zeros(shape, np.float32)
    for i, p in enumerate(paths):
        f = rd(i, p)
        s += f; s2 += f * f
        log(f"    Statistik {i + 1}/{n}")
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0))
    lo = mean - kappa * std; hi = mean + kappa * std
    acc = np.zeros(shape, np.float32); cnt = np.zeros(shape, np.float32)
    for i, p in enumerate(paths):
        f = rd(i, p)
        if method == "winsor":
            f = np.clip(f, lo, hi)
            acc += f; cnt += 1
        else:  # sigma: Ausreißer verwerfen
            m = (f >= lo) & (f <= hi)
            acc += np.where(m, f, 0); cnt += m
        log(f"    {method}-Rejection {i + 1}/{n}")
        if preview_cb and (i % _pv_every == 0 or i == n - 1):
            preview_cb(np.clip(acc / np.clip(cnt, 1, None), 0, 1), i + 1, n)
    return np.clip(acc / np.clip(cnt, 1, None), 0, 1)


def bin_image(f, factor=2):
    """Software-Binning: factor×factor-Blöcke mitteln → halbe (bei 2×) Auflösung, aber höheres
    Signal-Rausch-Verhältnis und kleinere/rundere Sterne. Sinnvoll bei überabgetasteten Daten
    (FWHM ≫ 2 px). factor=1 → unverändert."""
    factor = max(1, int(factor))
    if factor == 1 or f is None:
        return f
    h, w = f.shape[:2]
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    f = f[:h2, :w2]
    if f.ndim == 3:
        f = f.reshape(h2 // factor, factor, w2 // factor, factor, f.shape[2]).mean(axis=(1, 3))
    else:
        f = f.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    return f.astype(np.float32)


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


def _extract_ha_oiii(bgr, unmix=0.20):
    """Hα und OIII aus Dual-Band-OSC SAUBER trennen (normalisiert, [0..1]).
    Übersprechen beim OSC-Sensor: Hα (656 nm) → v. a. Rot (leckt etwas in Grün), OIII (500 nm) →
    Grün+Blau (leckt etwas in Rot). Darum Hα=Rot, OIII=Blau (Grün am stärksten Hα-kontaminiert),
    Hintergrund pro Kanal abziehen, leichte lineare Entmischung, einzeln normalisieren."""
    f = bgr.astype(np.float32)
    b, _, r = f[..., 0], f[..., 1], f[..., 2]

    def _sub_bg(x):
        return np.clip(x - float(np.quantile(x, 0.30)), 0, None)

    ha, oiii = _sub_bg(r), _sub_bg(b)
    ha2 = np.clip(ha - unmix * oiii, 0, None)
    oiii2 = np.clip(oiii - unmix * ha, 0, None)

    def _norm(x):
        return np.clip(x / max(float(np.quantile(x, 0.999)), 1e-6), 0, 1)

    return _norm(ha2), _norm(oiii2)


def _star_desat(out, ha_n, oiii_n, strength=0.92):
    """Sterne (Kontinuum-Quellen) **neutral/weiß** ziehen — in Schmalband ist Sternfarbe ein
    Artefakt (durchs Dual-Band-Filter kommen nur Hα-Rot + OIII-Cyan → türkise Sternkugeln).
    Ausgedehnte Nebel behalten ihre Farbe.

    Zwei Stufen: kompakte Sternkerne über lokalen Kontrast erkennen (niedriges Helligkeits-Gate,
    damit auch mittelhelle Sterne erfasst werden) und die Maske um die **Sternhöfe** aufweiten —
    sonst bleibt der Glow heller Sterne farbig, während nur der Kern entsättigt wird."""
    lum = np.maximum(ha_n, oiii_n).astype(np.float32)
    smooth = cv2.medianBlur((lum * 255).astype(np.uint8), 9).astype(np.float32) / 255.0
    detail = np.clip(lum - smooth, 0, 1)
    core = np.clip(detail * 6.0, 0, 1) * np.clip((lum - 0.06) / 0.06, 0, 1)   # kompakte Sternkerne
    coreb = (core > 0.25).astype(np.uint8)
    halo = cv2.dilate(coreb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))  # Hof drumherum
    mask = np.maximum(core, halo.astype(np.float32))
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
    gray = out.mean(axis=2, keepdims=True)
    return np.clip(out * (1 - strength * mask) + gray * (strength * mask), 0, 1)


def dualband_hoo(bgr, unmix=0.20):
    """HOO-Palette: Rot=Hα, Grün+Blau=OIII → rote Hα-Nebel + tealfarbene OIII-Bereiche (zwei echte
    Signale, datentreu). Sterne werden neutralisiert."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    ha_n, oiii_n = _extract_ha_oiii(bgr, unmix)
    out = np.zeros((*ha_n.shape, 3), np.float32)
    out[..., 2] = ha_n                          # R = Hα
    out[..., 1] = oiii_n                        # G = OIII
    out[..., 0] = oiii_n                        # B = OIII → teal
    return _star_desat(out, ha_n, oiii_n)


def dualband_sho(bgr, unmix=0.20):
    """SYNTHETISCHE SHO-/Hubble-Palette aus Dual-Band (Ha+OIII) — **gold + blau** (klassisch).
    ⚠️ KEIN echtes SII in Dual-Band; SII wird aus Hα synthetisiert. Mapping wie in den Anleitungen:
    Rot = SII(≈Hα), Grün = 0.8·Hα + 0.2·OIII, Blau = OIII → Hα-Bereiche werden gold, OIII blau.
    Forciert den Gold-Look (auch bei reinen Hα-Zielen). Nicht wissenschaftlich, nur fürs Aussehen."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    ha, oiii = _extract_ha_oiii(bgr, unmix)
    out = np.zeros((*ha.shape, 3), np.float32)
    out[..., 2] = ha                            # R = SII(synthetisch ≈ Hα)
    out[..., 1] = np.clip(0.8 * ha + 0.2 * oiii, 0, 1)        # G → R+G = gold
    out[..., 0] = oiii                          # B = OIII → blau
    return _star_desat(out, ha, oiii)


def dualband_foraxx(bgr, unmix=0.20):
    """SYNTHETISCHE SHO-Palette im **Foraxx-Stil** (dynamisch, thecoldestnights.com): der Grün-Kanal
    wird je nach Hα·OIII-Stärke gemischt — G = f·Hα + (1−f)·OIII mit f = (Hα·OIII)^(1−Hα·OIII).
    Dadurch: reines Hα → rot, Hα+OIII gemischt → gold, reines OIII → blau. Nuancierter als das
    flache SHO, aber rein Hα-Ziele bleiben rot (kein erzwungenes Gold). SII synthetisch = Hα."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    ha, oiii = _extract_ha_oiii(bgr, unmix)
    prod = np.clip(ha * oiii, 1e-6, 1.0)
    fg = prod ** (1.0 - prod)
    out = np.zeros((*ha.shape, 3), np.float32)
    out[..., 2] = ha                            # R = SII(synthetisch ≈ Hα)
    out[..., 1] = np.clip(fg * ha + (1.0 - fg) * oiii, 0, 1)  # G = dynamischer Hα/OIII-Blend
    out[..., 0] = oiii                          # B = OIII
    return _star_desat(out, ha, oiii)


def dualband_bicolor(bgr, unmix=0.20):
    """Bicolor-Technik (nach Cannistra): aus zwei Kanälen (Hα, OIII) wird der fehlende **synthetisch
    errechnet**, damit Farben/Sterne natürlicher werden (weniger Magenta als reines HOO).
    Hier: Rot = Hα, Blau = OIII, **Grün = synthetisch** aus beiden (Mittel, OIII-betont):
    G = max(OIII, 0.5·Hα). Ergebnis: Hα-Bereiche bernstein/rot, OIII cyan-blau, Übergänge weich;
    Sterne werden neutraler. SII bleibt außen vor (nur Hα+OIII)."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    ha, oiii = _extract_ha_oiii(bgr, unmix)
    g = np.maximum(oiii, 0.5 * ha)              # synthetisches Grün aus den beiden Kanälen
    out = np.zeros((*ha.shape, 3), np.float32)
    out[..., 2] = ha                            # R = Hα
    out[..., 1] = np.clip(g, 0, 1)              # G = synthetisch (errechnet)
    out[..., 0] = oiii                          # B = OIII
    return _star_desat(out, ha, oiii)


def color_balance(f, strength=1.0):
    """Farbkalibrierung fürs Anzeigen (gegen Rotstich von OSC + LP-Filter):
      1. Himmelshintergrund PRO KANAL neutralisieren (Sky -> neutrales Grau),
      2. Kanäle so abgleichen, dass helle, unklippte Referenzen (Sterne) ~neutral werden.
    So treten die echten Nebelfarben hervor (rotes Ha, blaue Reflexion, teal O-III) statt alles rot.
    strength 0..1 blendet zwischen Original (0) und voller Kalibrierung (1) — einstellbar / KI-gesteuert.
    Wirkt nur aufs Vorschau-/JPG-Bild; die linearen Exports bleiben unangetastet."""
    if f is None or f.ndim != 3 or f.shape[2] != 3 or strength <= 0:
        return f
    src = f.astype(np.float32)
    bg = np.array([np.quantile(src[..., c], 0.30) for c in range(3)], np.float32)
    out = np.clip(src - bg.reshape(1, 1, 3), 0, None)          # Hintergrund neutral
    hi = np.array([np.quantile(out[..., c], 0.995) for c in range(3)], np.float32)
    scale = np.clip(hi.mean() / np.clip(hi, 1e-6, None), 0.4, 2.5).astype(np.float32)
    out = np.clip(out * scale.reshape(1, 1, 3), 0, None)        # Sterne ~neutral -> echte Farben
    s = float(min(1.0, max(0.0, strength)))
    return out if s >= 1.0 else np.clip(src * (1 - s) + out * s, 0, None)


def photometric_balance(f, strength=1.0, max_stars=300, log=print):
    """Photometrischer Farbabgleich (PCC-lite, OHNE Online-Sternkatalog): Anders als die einfache
    Quantil-Kalibrierung misst dies die ECHTEN Farben vieler einzelner, UNGESÄTTIGTER Sterne und
    gleicht die Kanäle so ab, dass die mittlere Sternfarbe neutral wird (Sterne sind im Mittel ~weiß).
    Robuster als der 99.5%-Quantil-Weißpunkt, weil gesättigte Sternkerne und gefärbter Nebel
    ausgeschlossen werden → echte, glaubwürdige Nebelfarben statt Farbstich.

    Hinweis: Das ist KEIN katalogbasiertes SPCC (das bräuchte Plate-Solving + Gaia-Abfrage online);
    es nutzt nur die statistische Weiß-Annahme der Sternpopulation im Bild — treu und reproduzierbar."""
    if f is None or f.ndim != 3 or f.shape[2] != 3 or strength <= 0:
        return f
    src = f.astype(np.float32)
    bg = np.array([np.quantile(src[..., c], 0.30) for c in range(3)], np.float32)
    out = np.clip(src - bg.reshape(1, 1, 3), 0, None)          # 1) Hintergrund neutralisieren
    g = _gray(out)
    pts = _star_centroids(g / (g.max() + 1e-6), max_stars=max_stars)
    cols = []
    H, W = g.shape[:2]
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 2 <= xi < W - 2 and 2 <= yi < H - 2:
            patch = out[yi - 2:yi + 3, xi - 2:xi + 3].reshape(-1, 3)
            peak = float(patch.max())
            if 0.02 < peak < 0.95:                            # ungesättigt UND über dem Rauschen
                cols.append(patch.mean(0))
    if len(cols) < 15:                                        # zu wenige Sterne → Quantil-Fallback
        log(f"    PCC-lite: nur {len(cols)} brauchbare Sterne → Standard-Farbabgleich")
        return color_balance(f, strength)
    med = np.median(np.array(cols, np.float32), axis=0) + 1e-6   # mittlere Sternfarbe (BGR)
    scale = np.clip(float(med.mean()) / med, 0.4, 2.5).astype(np.float32)
    out = np.clip(out * scale.reshape(1, 1, 3), 0, None)      # 2) mittleren Stern → neutral
    log(f"    PCC-lite: {len(cols)} Sterne, Kanal-Skalierung BGR={np.round(scale, 3)}")
    s = float(min(1.0, max(0.0, strength)))
    return out if s >= 1.0 else np.clip(src * (1 - s) + out * s, 0, None)


def remove_green_cast(f, amount=1.0):
    """SCNR-artige Grün-Entfernung (Average Neutral): Grün wird auf den Schnitt von Rot/Blau
    begrenzt. In der Deep-Sky-Fotografie ist Grün praktisch nie echtes Signal (Nebel sind rot/blau),
    ein Grünstich kommt von OSC-Bayer/Lichtverschmutzung. Subtraktiv/treu — fügt nichts hinzu.
    Entfernt zugleich grüne Hot-Pixel-/Stern-Sprenkel. amount 0..1."""
    if f is None or f.ndim != 3 or f.shape[2] != 3 or amount <= 0:
        return f
    out = f.astype(np.float32).copy()
    b, g, r = out[..., 0], out[..., 1], out[..., 2]      # BGR
    neutral = np.minimum(g, (b + r) * 0.5)
    out[..., 1] = g * (1 - amount) + neutral * amount
    return out


def _bg_surface(f, ds=8):
    """Glatte Hintergrund-/Gradienten-Fläche eines Frames (grob downsamplen + stark glätten →
    Sterne mitteln sich weg). Für die lokale Normalisierung."""
    g = f if f.ndim == 2 else f.mean(2)
    h, w = g.shape
    sw, sh = max(8, w // ds), max(8, h // ds)
    small = cv2.resize(g.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(2.0, sw / 10.0))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def local_normalize(frame, ref_surface):
    """Frame **örtlich** an die Referenz-Hintergrundfläche angleichen (statt nur per Skalar-Offset).
    Da die Frames registriert sind, hebt sich der gemeinsame Nebel in (ref_surf − frame_surf) auf —
    übrig bleibt der **örtliche Hintergrund-/Gradienten-Unterschied**, der korrigiert wird. Das macht
    die Ausreißer-Rejection erst korrekt (gegen Gradienten & Mehrfach-Sessions)."""
    corr = ref_surface - _bg_surface(frame)
    return frame + (corr[..., None] if frame.ndim == 3 else corr)


def _mtf(x, m):
    """Midtones Transfer Function (PixInsight-Primitive): MTF(x,m) = (m−1)x / ((2m−1)x − m).
    Reversibel, definiert auf [0,1]. m = Mitteltonbalance (klein = stark aufhellen)."""
    x = np.asarray(x, np.float32)
    den = (2.0 * m - 1.0) * x - m
    out = np.where(np.abs(den) < 1e-9, x, ((m - 1.0) * x) / den)
    return np.clip(out, 0, 1)


def mtf_stretch(f, target_bg=0.25, shadow=-2.8, saturation=1.05, denoise_chroma=True):
    """MTF-Auto-Stretch (PixInsight-AutoSTF-Stil): Schwarzpunkt aus Median+shadow·MAD, dann
    Mitteltonbalance so, dass der Himmelshintergrund auf `target_bg` (≈0.25) gehoben wird.
    Korrekt und reversibel — kontrollierter als asinh, mit definiertem Schwarzpunkt."""
    g = _gray(f)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med))) * 1.4826 + 1e-6
    c0 = float(np.clip(med + shadow * mad, 0, 0.99))            # Schwarzpunkt (shadow<0 → unter Median)
    x = np.clip((f - c0) / max(1e-6, 1 - c0), 0, 1)
    mn = float(np.clip((med - c0) / max(1e-6, 1 - c0), 1e-4, 0.9999))
    T = target_bg
    m = mn * (T - 1.0) / (2 * T * mn - T - mn)                  # MTF(mn,m)=T nach m aufgelöst
    m = float(np.clip(m, 1e-3, 1 - 1e-3))
    out = _mtf(x, m)
    if saturation and saturation != 1.0 and out.ndim == 3:
        lum = _gray(out)[..., None]
        out = np.clip(lum + (out - lum) * saturation, 0, 1)
    if denoise_chroma and out.ndim == 3:
        lum = _gray(out)[..., None]
        out = np.clip(lum + cv2.GaussianBlur(out - lum, (0, 0), 3.0), 0, 1)
    return out


def ghs_stretch(f, D=2.5, b=-0.5, SP=0.18, black_clip=None, saturation=1.08,
                denoise_chroma=True, samples=4096):
    """Generalised-Hyperbolic-Stretch (GHS-Familie) — frei steuerbarer High-Dynamic-Stretch,
    der schwaches Nebel-Signal kräftig anhebt, ohne den hellen Kern/Sterne auszubrennen.
    Ergänzt MTF (fester Schwarzpunkt) und asinh um eine voll parametrische Kurve.

      • D  = Intensität (Stärke der Streckung; höher = aggressiver)
      • b  = Charakter der Kurve:  b<0 weicher Knick (asinh-artig), gegen 0 sanfter,
             stärker negativ = härterer, konzentrierter Knick (hyperbolisch)
      • SP = Symmetrie-/Pivotpunkt (0..1): die Helligkeit, um die herum am stärksten gestreckt
             wird — typ. knapp über dem Himmelshintergrund.

    Konstruiert über die kumulierte lokale Streckung (Integral einer überall positiven
    Streckfunktion) → garantiert monoton, bildet [0..1] streng auf [0..1] ab, erhält Schwarz/Weiß.
    Identische Kurve je Kanal (linked, wie in Siril)."""
    g = _gray(f)
    if black_clip is not None:
        bg = float(np.quantile(g, black_clip))
    else:
        med = float(np.median(g))
        mad = float(np.median(np.abs(g - med))) * 1.4826
        bg = med + 0.25 * mad
    # Schwarzpunkt setzen UND das Signal in den aktiven Bereich der Kurve normieren (wie asinh):
    # ohne diese Normierung liegt schwaches (lineares) Nebel-Signal nahe 0 und die Kurve hebt es nicht.
    sub = np.clip(f - bg, 0, None)
    norm = float(np.quantile(_gray(sub), 0.9997)) + 1e-6
    x0 = np.clip(sub / norm, 0, 1)

    xs = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    k = float(D) * float(D)
    ls = (1.0 + k * (xs - float(SP)) ** 2) ** float(b)         # lokale Streckung, Maximum bei SP
    cdf = np.cumsum(ls)
    cdf -= cdf[0]
    cdf /= (cdf[-1] + 1e-12)                                    # → streng [0..1], monoton
    out = np.interp(x0.ravel(), xs, cdf).reshape(x0.shape).astype(np.float32)

    if out.ndim == 3:
        if denoise_chroma:
            lum = _gray(out)[..., None]
            out = np.clip(lum + cv2.GaussianBlur(out - lum, (0, 0), 3.0), 0, 1)
        if saturation and saturation != 1.0:
            lum = _gray(out)[..., None]
            out = np.clip(lum + (out - lum) * saturation, 0, 1)
    return np.clip(out, 0, 1)


def autostretch(f, black_clip=None, strength=6.0, protect_core=True, saturation=1.05,
                denoise_chroma=True):
    """asinh-Auto-Stretch fürs Anzeigen des (linearen, dunklen) Astro-Ergebnisses.

    Zurückhaltend gehalten — Ziel ist eine *echte* Bearbeitung, kein Neon-Comic:
    schwaches Signal wird sichtbar, aber der Hintergrund bleibt dunkel und das Rauschen unten.

    strength  : wie stark schwaches Signal angehoben wird (höher = heller/aggressiver).
    protect_core: helle Bereiche (Nebel-Kern, helle Sterne) werden sanfter gestreckt, damit der
                  Kern nicht zu einem flachen weißen Klecks ausbleicht — Detail/Farbe bleibt.
    saturation: leichter Farb-Boost (Astro-Farben sind nach dem Strecken oft blass).
    denoise_chroma: Farb-Rauschen glätten (Luminanz bleibt scharf) — killt den bunten Grieß im
                    Hintergrund, ohne Schärfe zu kosten.
    black_clip: optionaler fester Schwarzpunkt als Quantil. Standard (None) = **robuster
                Himmelshintergrund** (Median + 0.5·MAD), damit der Hintergrund wirklich nach
                Schwarz geht und das Rauschen nicht hochgezogen wird."""
    g = _gray(f)
    if black_clip is not None:
        bg = np.quantile(g, black_clip)
    else:
        med = float(np.median(g))
        mad = float(np.median(np.abs(g - med))) * 1.4826      # robustes Sigma
        bg = med + 0.25 * mad                                  # Schwarzpunkt knapp über dem Himmel
        #  (weicher als 0.5·MAD: zeigt schwache Nebel-Außenbereiche, Hintergrund bleibt dunkel)
    x = np.clip(f - bg, 0, None)
    norm = np.quantile(_gray(x), 0.9997) + 1e-6
    x = x / norm
    out = np.clip(np.arcsinh(x * strength) / np.arcsinh(strength), 0, 1)
    if protect_core:
        # In den hellsten ~20 % nur sanft strecken (Kern-Schutz) und mit der starken Kurve mischen.
        gentle = np.clip(np.arcsinh(x * (strength * 0.25)) / np.arcsinh(strength * 0.25), 0, 1)
        lum = _gray(out)
        hi = np.clip((lum - 0.80) / 0.20, 0, 1)
        hi = cv2.GaussianBlur(hi, (0, 0), 2)[..., None] if hi.ndim == 2 else hi[..., None]
        out = out * (1 - hi) + gentle * hi
    if denoise_chroma and out.ndim == 3:
        # Farb-Rauschen ist niederfrequent tolerierbar: Chroma weichzeichnen, Luminanz scharf lassen.
        lum = _gray(out)[..., None]
        chroma = cv2.GaussianBlur(out - lum, (0, 0), 3.0)
        out = np.clip(lum + chroma, 0, 1)
    if saturation != 1.0 and out.ndim == 3:
        lum = _gray(out)[..., None]
        out = np.clip(lum + (out - lum) * saturation, 0, 1)
    return np.clip(out, 0, 1)
