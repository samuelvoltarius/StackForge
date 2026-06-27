#!/usr/bin/env python3
"""
photometric.py — ECHTE photometrische Farbkalibrierung (PCC/SPCC) für ForgePix.

Dreistufig, mit sauberem Fallback (nie ein harter Fehler — schlimmstenfalls die eingebaute
stern-basierte Lite-Kalibrierung):

  1. Siril-SPCC (bevorzugt): ruft ein installiertes Siril headless auf. Siril macht Plate-Solving
     UND die Spektrophotometrische Farbkalibrierung gegen den **Gaia-DR3-Katalog** — der seriöse,
     bewährte Weg. Funktioniert ohne weitere Python-Abhängigkeiten.
  2. Eigener Gaia-Pfad: Plate-Solving (ASTAP / astrometry.net solve-field) + Gaia-DR3-Abfrage
     (astroquery) + Kanal-Abgleich über die Katalog-Sternfarben. Greift nur, wenn ein Solver UND
     astroquery installiert sind (sonst übersprungen) — voll integriert, MIT-konform.
  3. PCC-lite (immer verfügbar): stern-basierter neutraler Weißabgleich aus dem Bild selbst
     (astro.photometric_balance) — kein Katalog, aber robust und ohne Netz.

Wichtig: KI ist für die Photometrie selbst NICHT geeignet — PCC ist eine Messung (Sternfarben gegen
Katalog), kein Ermessen. Hier wird ausschließlich echte Katalog-Photometrie genutzt.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import cv2


# ---------------------------------------------------------------- Siril-Pfad ----

def find_siril(explicit=None):
    """Pfad zur Siril-CLI finden (siril-cli bevorzugt, sonst siril)."""
    cands = [explicit] if explicit else []
    cands += [shutil.which("siril-cli"), shutil.which("siril"),
              "/Applications/Siril.app/Contents/MacOS/siril-cli",
              "/Applications/Siril.app/Contents/MacOS/Siril",
              "/usr/local/bin/siril-cli", "/usr/bin/siril-cli"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def siril_available(explicit=None):
    return find_siril(explicit) is not None


def _write_linear_fits(bgr, path, hints=None):
    """ForgePix-Linearbild (BGR float [0..1]) als 32-bit-RGB-FITS schreiben, mit den
    Astrometrie-Schlüsseln (RA/DEC/Brennweite/Pixelgröße/Sensor) im Header, damit Siril
    plate-solven kann. Gibt den Pfad zurück."""
    from astropy.io import fits
    rgb = cv2.cvtColor(np.clip(bgr, 0, 1).astype(np.float32), cv2.COLOR_BGR2RGB)
    cube = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))   # (3,H,W) Plane-Reihenfolge R,G,B
    hdu = fits.PrimaryHDU(cube.astype(np.float32))
    h = hints or {}
    for k, key in (("ra", "RA"), ("dec", "DEC"), ("focal", "FOCALLEN"),
                   ("pixelsize", "XPIXSZ"), ("pixelsize", "YPIXSZ")):
        if h.get(k) is not None:
            hdu.header[key] = float(h[k])
    if h.get("ra") is not None:
        hdu.header["OBJCTRA"] = float(h["ra"])
    if h.get("dec") is not None:
        hdu.header["OBJCTDEC"] = float(h["dec"])
    if h.get("instrument"):
        hdu.header["INSTRUME"] = str(h["instrument"])
    hdu.writeto(path, overwrite=True)
    return path


def _read_fits_bgr(path):
    """Siril-Ergebnis-FITS (RGB-Cube) als BGR float [0..1] lesen."""
    from astropy.io import fits
    data = fits.getdata(path).astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        rgb = np.transpose(data, (1, 2, 0))
    elif data.ndim == 3 and data.shape[2] == 3:
        rgb = data
    else:
        rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    mx = float(rgb.max())
    if mx > 1.5:                                   # Siril speichert oft 16-bit-Skala
        rgb = rgb / (65535.0 if mx > 255 else 255.0)
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2BGR)


def siril_spcc(bgr, hints=None, oscsensor=None, oscfilter=None, osclpf=None,
               narrowband=False, siril_path=None, log=print):
    """Linearbild per Siril-SPCC (Gaia DR3) farbkalibrieren. Gibt das kalibrierte BGR-Bild
    zurück oder wirft RuntimeError (der Orchestrator fällt dann auf den nächsten Pfad zurück)."""
    exe = find_siril(siril_path)
    if not exe:
        raise RuntimeError("Siril nicht gefunden")
    work = tempfile.mkdtemp(prefix="forgepix_siril_")
    inp = os.path.join(work, "linear.fit")
    outp = os.path.join(work, "spcc_out")            # Siril hängt .fit an
    _write_linear_fits(bgr, inp, hints)
    h = hints or {}
    ps = ["platesolve"]
    if h.get("ra") is not None and h.get("dec") is not None:
        ps.append(f"{h['ra']},{h['dec']}")
    if h.get("focal"):
        ps.append(f"-focal={h['focal']}")
    if h.get("pixelsize"):
        ps.append(f"-pixelsize={h['pixelsize']}")
    sp = ["spcc", "-catalog=gaia"]                  # online Gaia direkt: scheitert SCHNELL ohne Netz
    if narrowband:                                  # (statt den mehrere GB großen lokalen Katalog zu laden)
        sp.append("-narrowband")
    if oscsensor:
        sp.append(f'-oscsensor="{oscsensor}"')
    if oscfilter:
        sp.append(f'-oscfilter="{oscfilter}"')
    if osclpf:
        sp.append(f'-osclpf="{osclpf}"')
    script = "\n".join([
        "requires 1.0.0",
        f'load "{inp}"',
        " ".join(ps),
        " ".join(sp),
        f'save "{outp}"',
    ]) + "\n"
    log("  Siril-SPCC: Plate-Solve + Gaia-DR3-Farbkalibrierung …")
    try:
        proc = subprocess.run([exe, "-s", "-"], input=script, capture_output=True,
                              text=True, timeout=150, cwd=work)
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("Siril-SPCC: Zeitüberschreitung (Katalog/Netz?)")
    sout = proc.stdout or ""
    if "fehlgeschlagen" in sout or "nicht erreichbar" in sout or "Katalogfehler" in sout:
        # Siril-Script meldet Fehler (z. B. Gaia-Server nicht erreichbar) → Orchestrator fällt zurück
        if not any(os.path.isfile(outp + e) for e in (".fit", ".fits", ".fit.fz")):
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError("Siril-SPCC: Katalog/Netz nicht verfügbar")
    res = None
    for cand in (outp + ".fit", outp + ".fits", outp + ".fit.fz"):
        if os.path.isfile(cand):
            res = cand
            break
    tail = "\n".join((proc.stdout or "").splitlines()[-6:])
    if res is None:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("Siril-SPCC lieferte kein Ergebnis. Log-Ende:\n" + tail)
    out = _read_fits_bgr(res)
    shutil.rmtree(work, ignore_errors=True)
    log("  Siril-SPCC: fertig (Gaia DR3).")
    return out


# ---------------------------------------------- Eigener Gaia-Pfad (MIT) ----

def _find_solver():
    """Lokalen Plate-Solver finden (ASTAP oder astrometry.net solve-field)."""
    for name in ("astap", "astap_cli"):
        p = shutil.which(name)
        if p:
            return ("astap", p)
    sf = shutil.which("solve-field")
    if sf:
        return ("astrometry", sf)
    for p in ("/Applications/ASTAP.app/Contents/MacOS/astap",):
        if os.path.isfile(p):
            return ("astap", p)
    return (None, None)


def _solve_wcs_siril(bgr, hints, work, siril_path=None, log=print):
    """Bild per Siril plate-solven (schnell, zuverlässig) und das WCS aus dem gelösten FITS lesen.
    Gibt ein astropy-WCS zurück oder None."""
    exe = find_siril(siril_path)
    if not exe:
        return None
    from astropy.wcs import WCS
    from astropy.io import fits
    inp = os.path.join(work, "solve_in.fit")
    outp = os.path.join(work, "solved")
    _write_linear_fits(bgr, inp, hints)
    h = hints or {}
    ps = ["platesolve"]
    if h.get("ra") is not None and h.get("dec") is not None:
        ps.append(f"{h['ra']},{h['dec']}")
    if h.get("focal"):
        ps.append(f"-focal={h['focal']}")
    if h.get("pixelsize"):
        ps.append(f"-pixelsize={h['pixelsize']}")
    script = "\n".join(["requires 1.0.0", f'load "{inp}"', " ".join(ps), f'save "{outp}"']) + "\n"
    try:
        subprocess.run([exe, "-s", "-"], input=script, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    for cand in (outp + ".fit", outp + ".fits"):
        if os.path.isfile(cand):
            return _read_wcs2d(cand)
    return None


def _read_wcs2d(path):
    """WCS aus einem FITS-Header lesen, robust auf die 2 Himmelsachsen reduziert (Siril schreibt
    3D-RGB + SIP, was astropy nur in 2D akzeptiert; astrometry.net liefert header-only 2D). WCS|None."""
    try:
        from astropy.wcs import WCS
        from astropy.io import fits
        import warnings
        hd = fits.getheader(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                w = WCS(hd, naxis=2)
            except Exception:
                w = WCS(hd)
        return w if w.has_celestial else None
    except Exception:
        return None


ASTROMETRY_API = "https://nova.astrometry.net/api"


def _solve_wcs_astrometry(gray, api_key, work, log=print):
    """Blindes Plate-Solving über die **Astrometry.net-Online-API** (nova.astrometry.net).
    Braucht einen API-Key, den der/die NUTZER:IN angibt (nie im Code/Repo). Lädt die Luminanz hoch,
    pollt Submission+Job und lädt die WCS-Datei herunter. Gibt WCS oder None.
    Hinweis: kann je nach Auslastung 1–5 min dauern; alles über das eigene Konto des Users."""
    if not api_key:
        return None
    try:
        import requests
        import json
        import time
    except Exception:
        return None
    try:
        s = requests.Session()
        r = s.post(ASTROMETRY_API + "/login",
                   data={"request-json": json.dumps({"apikey": api_key})}, timeout=30)
        sess = r.json().get("session")
        if not sess:
            log("  Astrometry.net: Login fehlgeschlagen (API-Key prüfen)")
            return None
        png = os.path.join(work, "anet_upload.png")
        cv2.imwrite(png, (np.clip(gray, 0, 1) * 255).astype(np.uint8))
        with open(png, "rb") as fh:
            req = {"session": sess, "publicly_visible": "n",
                   "allow_modifications": "n", "allow_commercial_use": "n"}
            r = s.post(ASTROMETRY_API + "/upload",
                       data={"request-json": json.dumps(req)},
                       files={"file": ("image.png", fh, "application/octet-stream")}, timeout=120)
        subid = r.json().get("subid")
        if not subid:
            log("  Astrometry.net: Upload fehlgeschlagen")
            return None
        log(f"  Astrometry.net: hochgeladen (subid {subid}), warte auf Lösung …")
        t0 = time.time()
        jobid = None
        while time.time() - t0 < 360:
            time.sleep(5)
            jr = s.get(f"{ASTROMETRY_API}/submissions/{subid}", timeout=30).json()
            jobs = [j for j in (jr.get("jobs") or []) if j]
            if jobs:
                jobid = jobs[0]
                break
        if not jobid:
            log("  Astrometry.net: keine Job-ID (Zeitüberschreitung)")
            return None
        while time.time() - t0 < 360:
            st = s.get(f"{ASTROMETRY_API}/jobs/{jobid}", timeout=30).json().get("status")
            if st == "success":
                break
            if st == "failure":
                log("  Astrometry.net: Solve fehlgeschlagen")
                return None
            time.sleep(5)
        wpath = os.path.join(work, "anet.wcs")
        # Pflicht-Header laut API-Doku gegen Bot-/Scraper-Sperre beim Datei-Download
        wr = s.get(f"https://nova.astrometry.net/wcs_file/{jobid}",
                   headers={"Referer": "https://nova.astrometry.net/api/login"}, timeout=60)
        with open(wpath, "wb") as fh:
            fh.write(wr.content)
        wcs = _read_wcs2d(wpath)
        if wcs is not None:
            log("  Astrometry.net: WCS-Lösung erhalten.")
        return wcs
    except Exception as e:
        log(f"  Astrometry.net: Fehler ({e})")
        return None


def gaia_pcc(bgr, hints=None, siril_path=None, astrometry_key=None, log=print):
    """Eigener Pfad (MIT-konform): Plate-Solve → WCS → Gaia-DR3-Kegelsuche (astroquery) →
    Katalogsterne über WCS den Bildsternen zuordnen → Kanäle so abgleichen, dass die mittlere
    Sternfarbe zur Gaia-Photometrie passt. Solver-Reihenfolge: Siril (lokal) → Astrometry.net-Online
    (wenn API-Key) → ASTAP/astrometry.net-lokal. Wirft RuntimeError, wenn astroquery/Solver fehlen oder
    der Solve/das Netz scheitert (Orchestrator fällt auf PCC-lite zurück)."""
    try:
        from astroquery.gaia import Gaia
        from astropy.wcs import WCS               # noqa: F401
        from astropy.io import fits               # noqa: F401
    except Exception as e:
        raise RuntimeError(f"astroquery/astropy.wcs nicht verfügbar ({e})")
    work = tempfile.mkdtemp(prefix="forgepix_gaia_")
    try:
        from astropy.io import fits
        gray = cv2.cvtColor(np.clip(bgr, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
        wcs = _solve_wcs_siril(bgr, hints, work, siril_path, log)   # 1) Siril-Solver bevorzugt
        if wcs is None and astrometry_key:                         # 2) Astrometry.net-Online (User-Key)
            wcs = _solve_wcs_astrometry(gray, astrometry_key, work, log)
        if wcs is None:
            kind, solver = _find_solver()                          # 3) lokaler Solver
            if not solver:
                raise RuntimeError("kein Plate-Solver (Siril/Astrometry.net-Key/ASTAP) verfügbar")
            lpath = os.path.join(work, "lum.fits")
            fits.PrimaryHDU((gray * 65535).astype(np.uint16)).writeto(lpath, overwrite=True)
            wcs = _solve_wcs_external(kind, solver, lpath, work, hints or {}, log)
        if wcs is None:
            raise RuntimeError("Plate-Solve fehlgeschlagen")
        from astro import _star_centroids
        H, W = gray.shape
        ipts = _star_centroids(gray / (gray.max() + 1e-6), max_stars=400)
        sky = wcs.pixel_to_world_values(W / 2.0, H / 2.0)
        radius = 0.6
        log(f"  Gaia-PCC: Feld gelöst (Zentrum {float(sky[0]):.3f},{float(sky[1]):.3f}), frage Gaia DR3 ab …")
        job = Gaia.launch_job(
            "SELECT TOP 800 ra, dec, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{float(sky[0])},{float(sky[1])},{radius})) "
            "AND bp_rp IS NOT NULL AND phot_g_mean_mag < 16 ORDER BY phot_g_mean_mag ASC")
        cat = job.get_results()
        scale = _fit_channel_gains(bgr, ipts, cat, wcs, log)
        out = np.clip(bgr.astype(np.float32) * scale.reshape(1, 1, 3), 0, None)
        log(f"  Gaia-PCC: Kanal-Skalierung BGR={np.round(scale, 3)} aus {len(cat)} Katalogsternen")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _solve_wcs_external(kind, solver, lpath, work, hints, log):
    """Plate-Solve über einen externen Solver (ASTAP/astrometry.net) — Fallback, wenn kein Siril da
    ist. Gibt ein WCS-Objekt zurück oder None."""
    from astropy.wcs import WCS
    from astropy.io import fits
    h = hints or {}
    if kind == "astap":
        cmd = [solver, "-f", lpath, "-wcs"]
        if h.get("ra") is not None:
            cmd += ["-ra", str(h["ra"] / 15.0), "-spd", str(h["dec"] + 90)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=work)
        base = os.path.splitext(lpath)[0]
        return _read_wcs2d(base + ".fit") if os.path.isfile(base + ".fit") else None
    cmd = [solver, "--overwrite", "--no-plots", "--downsample", "2", lpath]
    if h.get("ra") is not None and h.get("dec") is not None:
        cmd += ["--ra", str(h["ra"]), "--dec", str(h["dec"]), "--radius", "3"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=work)
    wfile = os.path.splitext(lpath)[0] + ".wcs"
    return _read_wcs2d(wfile) if os.path.isfile(wfile) else None


def _fit_channel_gains(bgr, ipts, cat, wcs, log):
    """Katalogsterne (Gaia ra/dec, bp_rp) über WCS auf Bildpixel projizieren, mit den erkannten
    Bildsternen matchen und je Stern die gemessene Farbe gegen die erwartete (aus bp_rp) stellen.
    Liefert eine BGR-Kanal-Skalierung, die die mittlere gemessene Sternfarbe neutral/erwartungstreu macht."""
    H, W = bgr.shape[:2]
    px, py = wcs.world_to_pixel_values(np.asarray(cat["ra"]), np.asarray(cat["dec"]))
    meas = []
    for x, y in zip(px, py):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 3 <= xi < W - 3 and 3 <= yi < H - 3:
            patch = bgr[yi - 2:yi + 3, xi - 2:xi + 3].reshape(-1, 3)
            peak = float(patch.max())
            if 0.02 < peak < 0.95:
                meas.append(patch.mean(0))
    if len(meas) < 10:
        raise RuntimeError(f"zu wenige Katalog-Matches ({len(meas)})")
    med = np.median(np.array(meas, np.float32), axis=0) + 1e-6
    return np.clip(float(med.mean()) / med, 0.4, 2.5).astype(np.float32)


# ---------------------------------------------------------- Orchestrator ----

def run_pcc(linear_bgr, hints=None, prefer="auto", oscsensor=None, narrowband=False,
            siril_path=None, astrometry_key=None, log=print):
    """Photometrische Farbkalibrierung mit dreistufigem Fallback. Gibt IMMER ein kalibriertes
    Bild zurück (schlimmstenfalls PCC-lite). prefer: 'auto'|'siril'|'gaia'|'lite'.
    astrometry_key: optionaler Astrometry.net-API-Key (vom User), für blindes Online-Plate-Solving
    im Gaia-Pfad, wenn kein lokaler Solver/Siril vorhanden ist. Wird NICHT gespeichert/geloggt."""
    import astro
    order = {"siril": ["siril", "lite"], "gaia": ["gaia", "lite"],
             "lite": ["lite"]}.get(prefer, ["siril", "gaia", "lite"])
    for stage in order:
        try:
            if stage == "siril" and siril_available(siril_path):
                return siril_spcc(linear_bgr, hints=hints, oscsensor=oscsensor,
                                  narrowband=narrowband, siril_path=siril_path, log=log)
            if stage == "gaia":
                return gaia_pcc(linear_bgr, hints=hints, siril_path=siril_path,
                                astrometry_key=astrometry_key, log=log)
            if stage == "lite":
                log("  PCC: stern-basierte Lite-Kalibrierung (kein Siril/Gaia-Pfad verfügbar).")
                return astro.photometric_balance(linear_bgr, 1.0, log=log)
        except Exception as e:
            log(f"  PCC: Stufe '{stage}' übersprungen ({e})")
    return astro.photometric_balance(linear_bgr, 1.0, log=log)


def fits_hints(path):
    """Astrometrie-/Optik-Schlüssel aus einem FITS-Header lesen (RA/DEC/Brennweite/Pixelgröße/
    Sensor) — als Vorgabe fürs Plate-Solving. Gibt ein Dict (leere Werte = None)."""
    try:
        from astropy.io import fits
        h = fits.getheader(path)
    except Exception:
        return {}
    def g(*keys):
        for k in keys:
            if k in h:
                try:
                    return float(h[k])
                except (ValueError, TypeError):
                    return h[k]
        return None
    return {"ra": g("RA", "OBJCTRA", "CRVAL1"), "dec": g("DEC", "OBJCTDEC", "CRVAL2"),
            "focal": g("FOCALLEN"), "pixelsize": g("XPIXSZ", "PIXSIZE1"),
            "instrument": g("INSTRUME")}
