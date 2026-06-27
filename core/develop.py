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


def local_contrast(img, amount=0.5, scales=4, protect=0.2):
    """Lokaler Kontrast-Equalizer (darktable/RawTherapee-Stil): hebt den Kontrast je Detailgröße
    getrennt an — über mehrere Skalen (Laplace-Pyramide auf der Luminanz). Anders als globaler
    Kontrast oder ein einzelnes Unsharp arbeitet das gleichmäßig über grobe UND feine Strukturen
    und bleibt dabei halo-arm (große Radien werden sanfter angehoben).

      amount  = Stärke (0..~1.5)
      scales  = Anzahl Detailskalen (mehr = auch gröbere Strukturen)
      protect = Schutz von Lichtern/Tiefen 0..0.5 (gegen Clipping/Verstärkung von Rauschen im Dunkeln)

    Treu/nicht-generativ — nur Tonwert/lokaler Kontrast, keine erfundenen Inhalte."""
    f, dtype, maxv = _as_float(img)
    if f.ndim == 3:
        lab = cv2.cvtColor(np.clip(f, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
        L = lab[..., 0] / 100.0
    else:
        L = np.clip(f, 0, 1).astype(np.float32)
    base = L.copy()
    out = L.copy()
    sigma = 2.0
    for s in range(int(max(1, scales))):                 # Detailbänder via Difference-of-Gaussians
        blur = cv2.GaussianBlur(base, (0, 0), sigma)
        detail = base - blur                              # Detail dieser Skala
        gain = amount * (0.6 ** s)                        # gröbere Skalen sanfter (halo-arm)
        out = out + gain * detail
        base = blur
        sigma *= 2.0
    if protect and protect > 0:                           # Lichter/Tiefen schützen (Glocken-Gewicht um 0.5)
        wmid = 1.0 - np.abs(L - 0.5) * 2.0
        wmid = np.clip(wmid, protect, 1.0)
        out = L + (out - L) * wmid
    out = np.clip(out, 0, 1)
    if f.ndim == 3:
        lab[..., 0] = out * 100.0
        res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        res = out
    res = np.clip(res, 0, 1)
    return (res * maxv).astype(dtype) if dtype != np.float32 else res


def capture_sharpen(img, sigma=0.7, iterations=12, protect=0.9):
    """Capture-Sharpening per Richardson-Lucy-Dekonvolution (wie RawTherapee „Capture Sharpening"):
    macht die Aufnahme-Unschärfe (Demosaicing + AA-Filter + leichtes Seeing) rückgängig — holt ECHTE
    Auflösung zurück, nicht nur Kantenkontrast wie Unsharp-Mask. Gauß-PSF (sigma in Pixeln), auf der
    Luminanz, mit weichem Lichter-Schutz (protect) gegen Überschwinger an hellen Kanten. Treu."""
    f, dtype, maxv = _as_float(img)
    if iterations < 1 or sigma <= 0:
        return img
    k = cv2.getGaussianKernel(int(max(3, round(sigma * 6)) | 1), sigma)
    psf = (k @ k.T).astype(np.float32); psf /= psf.sum()
    psf_m = psf[::-1, ::-1].copy()
    lum = (cv2.cvtColor(np.clip(f, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
           if f.ndim == 3 else np.clip(f, 0, 1).astype(np.float32))
    obs = np.clip(lum, 1e-4, None)
    est = obs.copy()
    for _ in range(int(iterations)):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        est = est * cv2.filter2D(obs / np.maximum(conv, 1e-6), -1, psf_m, borderType=cv2.BORDER_REFLECT)
        est = np.clip(est, 0, None)
    ratio = np.clip(est / np.maximum(lum, 1e-4), 0.3, 3.0)
    out = f.astype(np.float32) * ratio[..., None] if f.ndim == 3 else f.astype(np.float32) * ratio
    if protect is not None and protect < 1.0:               # Lichter weich schützen
        hi = cv2.GaussianBlur(np.clip((lum - protect) / max(1e-3, 1 - protect), 0, 1), (0, 0), 2.0)
        m = hi[..., None] if out.ndim == 3 else hi
        out = out * (1 - m) + f.astype(np.float32) * m
    out = np.clip(out, 0, 1)
    return (out * maxv).astype(dtype) if dtype != np.float32 else out


def dehaze(img, strength=1.0, omega=0.9, patch=15):
    """Dunst-/Schleier-Entfernung über den **Dark-Channel-Prior** (He et al. 2009 — wie RawTherapee/
    darktable „Haze removal"): in dunstfreien Außenbereichen ist in jedem Patch mindestens ein
    Farbkanal sehr dunkel; Dunst hebt dieses Minimum an. Daraus werden Atmosphärenlicht und
    Transmission geschätzt, die Transmission wird (Guided Filter, falls verfügbar) kantentreu
    verfeinert, dann das dunstfreie Bild rekonstruiert. strength 0..1 blendet zum Original. Treu."""
    f, dtype, maxv = _as_float(img)
    if f.ndim != 3 or strength <= 0:
        return img
    f = np.clip(f, 0, 1).astype(np.float32)
    dark = cv2.erode(f.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))
    flat = dark.ravel()
    n = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n)[-n:]                    # hellste 0.1% des Dark Channels
    g = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).ravel()
    A = f.reshape(-1, 3)[idx][np.argmax(g[idx])]            # Atmosphärenlicht
    A = np.clip(A, 1e-3, 1.0)
    t = 1.0 - omega * cv2.erode((f / A).min(axis=2),
                                cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))
    try:                                                    # Transmission kantentreu verfeinern
        t = cv2.ximgproc.guidedFilter(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY),
                                      t.astype(np.float32), 40, 1e-3)
    except Exception:
        t = cv2.GaussianBlur(t, (0, 0), 8)
    t = np.clip(t, 0.1, 1.0)[..., None]
    out = (f - A) / t + A
    out = np.clip(out, 0, 1)
    s = float(min(1.0, max(0.0, strength)))
    out = f * (1 - s) + out * s
    return (np.clip(out, 0, 1) * maxv).astype(dtype) if dtype != np.float32 else np.clip(out, 0, 1)


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


def _lensfun_auto(f, exif_path, log=print):
    """Automatische Objektivkorrektur über die lensfun-Datenbank (Vignette + Verzeichnung + TCA),
    anhand der Kamera-/Objektiv-Angaben aus den EXIF-Daten. Nur aktiv, wenn lensfunpy installiert
    UND Kamera/Objektiv in der Datenbank gefunden werden. Gibt (korrigiert, True) oder (f, False)."""
    try:
        import lensfunpy
        import subprocess
        import json
        meta = json.loads(subprocess.run(
            ["exiftool", "-j", "-n", "-Make", "-Model", "-LensModel", "-FocalLength",
             "-FNumber", "-FocusDistance", exif_path], capture_output=True, text=True).stdout)[0]
        db = lensfunpy.Database()
        cams = db.find_cameras(meta.get("Make", ""), meta.get("Model", ""))
        lenses = db.find_lenses(cams[0], None, meta.get("LensModel", "")) if cams else []
        if not cams or not lenses:
            log("    Objektivkorrektur: Kamera/Objektiv nicht in lensfun-DB → manuelle Parameter")
            return f, False
        h, w = f.shape[:2]
        mod = lensfunpy.Modifier(lenses[0], cams[0].crop_factor, w, h)
        mod.initialize(float(meta.get("FocalLength", 0) or 50),
                       float(meta.get("FNumber", 0) or 8),
                       float(meta.get("FocusDistance", 0) or 1000))
        out = f.copy()
        try:
            mod.apply_color_modification(out)            # Vignette
        except Exception:
            pass
        coords = mod.apply_geometry_distortion()         # Verzeichnung (+TCA, falls Profil)
        if coords is not None:
            out = cv2.remap(out, coords, None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        log(f"    Objektivkorrektur (lensfun): {lenses[0].model}")
        return np.clip(out, 0, 1), True
    except Exception as e:
        log(f"    lensfun nicht verfügbar/anwendbar ({e}) → manuelle Parameter")
        return f, False


def lens_correct(img, vignette=0.0, distortion=0.0, ca=0.0, auto=False, exif_path=None, log=print):
    """Objektivkorrekturen (RawTherapee/darktable-Niveau). Zwei Wege:

      • auto=True + exif_path: automatisch aus der **lensfun-Datenbank** (wenn lensfunpy installiert
        und das Objektiv bekannt ist) — Vignette, Verzeichnung und Farbquerfehler nach Profil.
      • manuelle Parameter (immer verfügbar, ohne Datenbank):
          - vignette   : Randabdunklung ausgleichen (>0 hellt die Ecken auf, ~r²-Modell)
          - distortion : Verzeichnung (k1; <0 korrigiert Tonnen-, >0 Kissenverzeichnung)
          - ca         : Farbquerfehler (laterale CA; R/B radial gegen G skalieren)

    Treu/nicht-generativ — nur geometrische/Helligkeits-Korrektur, kein Erfinden von Inhalten."""
    f, dt, mx = _as_float(img)
    used_auto = False
    if auto and exif_path:
        f, used_auto = _lensfun_auto(f, exif_path, log=log)
    if not used_auto and (abs(distortion) > 1e-4 or abs(ca) > 1e-4 or abs(vignette) > 1e-4):
        h, w = f.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        nr2 = cx * cx + cy * cy
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        ux, uy = xs - cx, ys - cy
        r2 = (ux * ux + uy * uy) / nr2                    # normierter Radius² (0 Mitte … 1 Ecke)
        if abs(distortion) > 1e-4:
            fac = 1.0 + float(distortion) * r2
            f = cv2.remap(f, cx + ux * fac, cy + uy * fac, cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)
        if abs(ca) > 1e-4 and f.ndim == 3:
            for ch, sgn in ((2, 1.0), (0, -1.0)):         # R nach außen, B nach innen (oder umgekehrt)
                fac = 1.0 + sgn * float(ca) * r2
                f[..., ch] = cv2.remap(f[..., ch], cx + ux * fac, cy + uy * fac,
                                       cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        if abs(vignette) > 1e-4:
            gain = 1.0 + float(vignette) * r2             # Ecken aufhellen
            f = f * (gain[..., None] if f.ndim == 3 else gain)
        log(f"    Objektivkorrektur (manuell): vignette={vignette}, distortion={distortion}, ca={ca}")
    out = np.clip(f, 0, 1)
    if dt == np.uint16:
        return (out * 65535.0).astype(np.uint16)
    if dt == np.uint8:
        return (out * 255.0).astype(np.uint8)
    return out
