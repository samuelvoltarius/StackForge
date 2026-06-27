#!/usr/bin/env python3
"""
graxpert_engine.py — optionales KI-Backend für Hintergrund-/Gradienten-Entfernung und Entrauschen
über die installierte GraXpert-App (https://github.com/Steffenhir/GraXpert, GPL-3.0).

ForgePix bleibt MIT: GraXpert wird NICHT mitgeliefert, sondern — falls vom Nutzer installiert — als
externes Tool aufgerufen (gleiches Muster wie die Siril-Integration). Liefert auf echten OSC-Daten ein
deutlich saubereres, gradientenfreies Ergebnis als die eingebaute RBF-Methode (VLLM-verifiziert: 3 → 1).
"""
import os
import glob
import shutil
import subprocess
import numpy as np

_CANDIDATES = [
    "/Applications/GraXpert.app/Contents/MacOS/GraXpert",
    os.path.expanduser("~/Applications/GraXpert.app/Contents/MacOS/GraXpert"),
    "/opt/GraXpert/GraXpert",
]


def find_cli(path=None):
    """Pfad zur GraXpert-CLI finden (übergebener Pfad, App-Bundle, oder im PATH), sonst None."""
    for c in ([path] if path else []) + _CANDIDATES + [shutil.which("graxpert"), shutil.which("GraXpert")]:
        if c and os.path.exists(c):
            return c
    return None


def available(path=None):
    return find_cli(path) is not None


def run(linear_bgr, work_dir, command="background-extraction", smoothing=0.2, gpu=False,
        path=None, timeout=900, log=print):
    """`linear_bgr` (float32, HWC BGR, ~0..1) durch GraXpert schicken und das Ergebnis-Array
    (gleiche Form/Reihenfolge) zurückgeben. command: 'background-extraction' | 'denoising'.
    Arbeitet über FITS (GraXperts natives Format) — verlustfrei linear."""
    from astropy.io import fits
    cli = find_cli(path)
    if cli is None:
        raise RuntimeError("GraXpert nicht gefunden (App installiert?)")
    os.makedirs(work_dir, exist_ok=True)
    inp = os.path.join(work_dir, "gx_input.fits")
    arr = np.clip(np.asarray(linear_bgr, np.float32), 0, None)
    if arr.ndim == 3:                                       # BGR→RGB, HWC→CHW (FITS-Konvention)
        data = np.transpose(arr[..., ::-1], (2, 0, 1))
    else:
        data = arr
    if os.path.exists(inp):
        os.remove(inp)
    fits.writeto(inp, data, overwrite=True)
    out_base = os.path.join(work_dir, "gx_out")
    for f in glob.glob(out_base + "*"):
        os.remove(f)
    cmd = [cli, "-cli", "-cmd", command, "-gpu", "true" if gpu else "false", "-output", out_base, inp]
    if command == "background-extraction":
        cmd += ["-smoothing", str(smoothing)]
    log(f"    GraXpert {command} (GPU={gpu}) …")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    outs = sorted(glob.glob(out_base + "*"))
    if not outs:
        raise RuntimeError(f"GraXpert lieferte keine Ausgabe (rc={proc.returncode})")
    d = fits.getdata(outs[0]).astype(np.float32)
    if d.ndim == 3 and d.shape[0] == 3:                    # CHW→HWC
        d = np.transpose(d, (1, 2, 0))
    if d.ndim == 3 and d.shape[2] == 3:                    # RGB→BGR
        d = d[..., ::-1]
    mx = float(d.max())
    if mx > 1.0:
        d = d / mx
    return np.clip(d, 0, None)
