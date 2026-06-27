# ForgePix vs. Pro Tools — Honest Scorecard (v1.22.1)

Status after the parity wave (v1.20) and the gap-closing waves (v1.21–v1.22). Honest:
✅ = at/near parity for the core job · 🟡 = works but a real refinement is missing · ❌ = not built.

ForgePix stays MIT and self-contained (OpenCV/NumPy/astropy). External tools (Siril, GraXpert,
StarNet++, lensfun, Astrometry.net) are **optional** — everything degrades gracefully without them.

---

## 🔬 Focus stacking — vs Helicon Focus / Zerene Stacker

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Pyramid merge (PMax-style) | ✓ | `pyramid` | ✅ |
| Depth-map merge (DMap-style) | ✓ | `depthmap` (power-weighted) | ✅ |
| Weighted average (Method A) | ✓ | `average` | ✅ |
| Radius / Smoothing controls | ✓ | `--focus-radius/-smoothing` | ✅ |
| Halo handling | retouch brush | `halofix` (auto envelope clamp) + retouch brush | ✅ |
| Alignment + crop overlap | ✓ | ECC + optical-flow + crop_to_overlap | ✅ |
| Shot/series & focus-map analysis, DOF assistant | partial | ✓ (explainable) | ✅ |

**Remaining:** full interactive *paint-from-frame-N* retouching (Helicon/Zerene let you brush a
specific source frame over an artefact). ForgePix has auto halo-fix + a halo/ghost retouch brush, but
not per-pixel source-frame painting. → 🟡 (rarely needed thanks to halofix).

## 🌌 Astro stacking — vs Siril / PixInsight / APP / DSS

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Calibration (dark/flat/bias, auto-detect) | ✓ | ✓ | ✅ |
| Registration (translation/rotation) | ✓ | star-based | ✅ |
| Local/distortion registration | PixInsight/APP | **TPS** (`--astro-tps`) | ✅ |
| Rejection sigma/winsor | ✓ | ✓ | ✅ |
| Linear-fit clipping | PixInsight | `linearfit` | ✅ |
| Drizzle (true, pixfrac) | ✓ | `--astro-drizzle-true` | ✅ |
| Stretch (MTF / asinh / **GHS**) | Siril/PI | all three | ✅ |
| Gradient removal | GraXpert/DBE | built-in + GraXpert (auto) | ✅ |
| Star removal (starless) | StarNet/PI | StarNet (auto) | ✅ |
| **Photometric color (PCC/SPCC)** | Siril/PI | **real** Siril Gaia DR3 → astroquery → lite | ✅ |
| Dual-band palettes (HOO/SHO/Foraxx/Bicolor) | PI scripts | ✓ | ✅ |
| Binning, multi-session, live preview | mixed | ✓ | ✅ |

**Remaining:** **deconvolution / PSF sharpening** (PixInsight Deconvolution, BlurXTerminator) — ForgePix
has wavelet sharpening but no PSF-based deconvolution. No PixelMath scripting engine. Comet-mode stacking
(align on the comet) not in the astro path. → 🟡 (deconvolution is the one genuine technique gap).

## 📷 HDR — vs Photomatix / Lightroom HDR

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Exposure fusion (halo-free) | Lightroom | Mertens (default) | ✅ |
| Radiance map + tonemapping | Photomatix | Debevec + Reinhard/Mantiuk/Drago | ✅ |
| Deghosting | ✓ | motion-masked reference | ✅ |
| Tone look presets | ✓ | natural/vivid/dramatic | ✅ |

**Remaining:** Photomatix-level per-zone manual tone sliders. → essentially ✅.

## 🌠 Long exposure — vs Sequator / StarStaX

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Average / median / lighten / comet | ✓ | smooth/declutter/trails/comet/bright | ✅ |
| Sigma-clip outlier rejection | ✓ | `--longexp-sigma` | ✅ |
| Trail gap-fill | StarStaX | `--longexp-gapfill` | ✅ |
| Freeze foreground | Sequator | `--longexp-freeze` (horizontal split) | ✅ |
| Virtual exposure-time slider | — | ✓ (ForgePix extra) | ✅ |

**Remaining:** Sequator **auto-detects the sky region**; ForgePix's freeze is a manual height fraction,
not an automatic sky segmentation. → 🟡 (a sky-mask auto-detect would close it).

## 🌐 Panorama — vs Hugin / PTGui

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Feature match + bundle adjust + wave correct | ✓ | cv2.detail (SIFT/Ray) | ✅ |
| Projections (spherical/cylindrical/…) | ✓ | ✓ | ✅ |
| Exposure compensation + seam + multiband blend | ✓ | GAIN_BLOCKS + GraphCut + MultiBand | ✅ |
| **Interactive control-point editor** | ✓ | — | ❌ |
| Manual masking / vignette polynomial UI | ✓ | — | ❌ |

**Remaining:** the **control-point / masking UI** is a separate large GUI project. Auto-stitching works;
manual rescue of hard panoramas doesn't. → 🟡 overall (auto ✅, manual ❌).

## 🖼️ RAW develop — vs RawTherapee / darktable

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Demosaic (DHT/DCB/VNG/AHD) | ✓ | ✓ (+AMaZE if GPL LibRaw) | ✅ |
| Highlight reconstruction | ✓ | ✓ | ✅ |
| Tone curve (no overshoot) | ✓ | PCHIP | ✅ |
| Per-color HSL | ✓ | ✓ | ✅ |
| Denoise (luma/chroma) | ✓ | NLM/BM3D | ✅ |
| Local masks (gradient/radial/guided) | ✓ | ✓ | ✅ |
| Lens corrections | lensfun | lensfun auto + manual vignette/distortion/CA | ✅ |

**Remaining:** full module-graph editor (darktable), parametric masks everywhere, ICC color management,
local contrast equalizer. ForgePix is a faithful developer + editor, not a full RT/darktable clone.
→ 🟡 (essentials covered).

## 🪐 Lucky imaging / planetary — vs AutoStakkert! / RegiStax

| Capability | Pro tool | ForgePix | |
|---|---|---|---|
| Multi-point (MAP) alignment | ✓ | experimental `lucky_stack_map` | 🟡 |
| Per-AP quality sort + stack | ✓ | partial | 🟡 |
| Wavelet sharpening | RegiStax | à-trous | ✅ |

**Remaining — the weakest module.** On featureless/low-res discs the single best frame can still beat
the MAP stack. Needs a good Moon/planet **video** to validate and tune (the dev set was a Seestar RAW
`.avi` that cv2 mis-decodes — use `.mp4`). → 🟡/❌ honest.

---

## Summary

| Module | Verdict |
|---|---|
| Focus stacking | ✅ parity (minus paint-from-frame retouch) |
| Astro stacking | ✅ parity (minus deconvolution) |
| HDR | ✅ parity |
| Long exposure | ✅ parity (minus auto sky-mask) |
| Panorama | 🟡 auto ✅, manual control-point UI ❌ |
| RAW develop | 🟡 essentials ✅, not a full RT/darktable |
| Lucky imaging | 🟡 experimental — the one to fix next |

**Top remaining items, honestly ranked:**
1. **Lucky-imaging MAP** — needs a real Moon/planet video to fix properly.
2. **Panorama control-point UI** — a sizeable standalone GUI feature.
3. **Astro deconvolution (PSF)** — the one missing pro *technique* in an otherwise complete astro chain.
4. Sky-segmentation auto-mask for freeze-foreground; paint-from-frame focus retouch (both nice-to-have).
