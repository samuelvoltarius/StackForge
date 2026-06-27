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


def stitch_from_points(img_a, img_b, pts_a, pts_b, log=print):
    """MANUELLES Zusammensetzen über Kontrollpunkte (Hugin/PTGui-Prinzip): wenn die automatische
    Erkennung versagt (wenig Überlappung/Struktur, sich wiederholende Muster), gibt der/die Nutzer:in
    selbst zusammengehörige Punktpaare an (≥4) — daraus wird die Homographie B→A geschätzt, B in die
    Ebene von A gewarpt und nahtlos (distanzgewichtet gefedert) eingeblendet. Gibt das Panorama (BGR).

    pts_a/pts_b: Listen von (x, y) in Bild A bzw. B (gleiche Reihenfolge = zusammengehörig)."""
    a = _to8(img_a); b = _to8(img_b)
    pa = np.asarray(pts_a, np.float32); pb = np.asarray(pts_b, np.float32)
    if len(pa) < 4 or len(pa) != len(pb):
        raise ValueError("mindestens 4 zusammengehörige Punktpaare nötig")
    H, _ = cv2.findHomography(pb, pa, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homographie aus den Punkten nicht bestimmbar")
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    cb = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)
    cbw = cv2.perspectiveTransform(cb, H).reshape(-1, 2)
    allp = np.vstack([cbw, [[0, 0], [wa, 0], [wa, ha], [0, ha]]])
    xmin, ymin = np.floor(allp.min(0)).astype(int)
    xmax, ymax = np.ceil(allp.max(0)).astype(int)
    T = np.float32([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    W, Hh = int(xmax - xmin), int(ymax - ymin)
    if W <= 0 or Hh <= 0 or W * Hh > 80_000_000:
        raise ValueError("ungültige Panorama-Größe (Punkte prüfen)")
    aw = cv2.warpPerspective(a, T, (W, Hh))
    bw = cv2.warpPerspective(b, T @ H, (W, Hh))
    ma = cv2.warpPerspective(np.full((ha, wa), 255, np.uint8), T, (W, Hh))
    mb = cv2.warpPerspective(np.full((hb, wb), 255, np.uint8), T @ H, (W, Hh))
    da = cv2.distanceTransform((ma > 0).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((mb > 0).astype(np.uint8), cv2.DIST_L2, 3)
    wsum = da + db + 1e-6
    blend = (aw.astype(np.float32) * da[..., None] + bw.astype(np.float32) * db[..., None]) / wsum[..., None]
    valid = ((ma > 0) | (mb > 0))[..., None]
    out = np.where(valid, np.clip(blend, 0, 255), 0).astype(np.uint8)
    log(f"    Kontrollpunkt-Stitch: {len(pa)} Punktpaare → {W}×{Hh}")
    return out


def stitch_detail(imgs, projection="spherical", log=print):
    """Explizite cv2.detail-Pipeline (statt Black-Box-Stitcher) mit Kontrolle über **Projektion**,
    **Belichtungsausgleich** (BlocksGain) und **MultiBand-Nahtmischung** (enblend-Äquivalent) +
    GraphCut-Nähte. Gibt das Panorama (uint8 BGR) zurück oder wirft bei Fehlschlag."""
    work_scale = min(1.0, 1000.0 / max(im.shape[1] for im in imgs))
    finder = cv2.SIFT_create() if hasattr(cv2, "SIFT_create") else cv2.ORB_create(1000)
    feats = []
    for im in imgs:
        s = cv2.resize(im, (0, 0), fx=work_scale, fy=work_scale) if work_scale < 1 else im
        feats.append(cv2.detail.computeImageFeatures2(finder, s))
    matcher = cv2.detail_BestOf2NearestMatcher(False, 0.55)
    pw = matcher.apply2(feats)
    matcher.collectGarbage()
    est = cv2.detail_HomographyBasedEstimator()
    ok, cams = est.apply(feats, pw, None)
    if not ok:
        raise RuntimeError("Kamera-Schätzung fehlgeschlagen")
    for c in cams:
        c.R = c.R.astype(np.float32)
    adj = cv2.detail_BundleAdjusterRay()
    adj.setConfThresh(0.7)
    ok, cams = adj.apply(feats, pw, cams)
    if not ok:
        raise RuntimeError("Bündelausgleich fehlgeschlagen")
    rmats = [np.copy(c.R) for c in cams]
    cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
    for c, R in zip(cams, rmats):
        c.R = R
    scale = float(np.median([c.focal for c in cams]))
    warper = cv2.PyRotationWarper(projection, scale * work_scale)
    corners, masks_w, imgs_w, sizes = [], [], [], []
    for im, c in zip(imgs, cams):
        s = cv2.resize(im, (0, 0), fx=work_scale, fy=work_scale) if work_scale < 1 else im
        K = c.K().astype(np.float32)
        corner, wimg = warper.warp(s, K, c.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * np.ones(s.shape[:2], np.uint8)
        _, wmask = warper.warp(mask, K, c.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        corners.append(corner); imgs_w.append(wimg.astype(np.float32))
        masks_w.append(wmask); sizes.append((wimg.shape[1], wimg.shape[0]))
    comp = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    comp.feed(corners, [w.astype(np.uint8) for w in imgs_w], masks_w)
    seamer = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    seam_masks = seamer.find([i.astype(np.float32) for i in imgs_w], corners, masks_w)
    blender = cv2.detail_MultiBandBlender()
    dst_sz = cv2.detail.resultRoi(corners, sizes)
    blender.prepare(dst_sz)
    for idx in range(len(imgs_w)):
        comp.apply(idx, corners[idx], imgs_w[idx].astype(np.uint8), masks_w[idx])
        sm = cv2.resize(seam_masks[idx].get() if hasattr(seam_masks[idx], "get") else seam_masks[idx],
                        (masks_w[idx].shape[1], masks_w[idx].shape[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.bitwise_and(sm, masks_w[idx])
        blender.feed(imgs_w[idx].astype(np.int16), m, corners[idx])
    pano, _ = blender.blend(None, None)
    return _to8(np.clip(pano, 0, 255))


def stitch(paths, mode="panorama", projection="spherical", detail=True, log=print):
    """Überlappende Kacheln zu einem Mosaik zusammensetzen.
    detail=True: explizite cv2.detail-Pipeline (Projektion/Belichtungsausgleich/MultiBand-Nähte),
    bei Fehlschlag Rückfall auf den klassischen cv2.Stitcher. Gibt (BGR-uint8, status) zurück."""
    imgs = [_to8(cv2.imread(p, cv2.IMREAD_UNCHANGED)) for p in paths]
    imgs = [im if (im is None or im.ndim == 3) else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            for im in imgs]
    imgs = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        raise RuntimeError("Mindestens 2 überlappende Kacheln nötig")
    log(f"  {len(imgs)} Kacheln zusammensetzen ({mode}, {projection}) …")
    if detail and mode != "scans":
        try:
            return stitch_detail(imgs, projection=projection, log=log), "ok (detail)"
        except Exception as e:
            log(f"  detail-Pipeline fehlgeschlagen ({e}) → klassischer Stitcher")
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
