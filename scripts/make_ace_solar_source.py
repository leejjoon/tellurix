#!/usr/bin/env python
"""The ACE-FTS solar spectrum as a source for the window fit (GitHub #2).

ACE-FTS looks at the centre of the solar disc from orbit (1.25 mrad of the Sun's 9 mrad,
mu >= 0.99), so it observes the quantity Kitt Peak's disc-centre files do, with no
atmosphere in the way. The 2026 release (Bhusal et al.; 1829.4-4439.875 cm-1 on a
0.005 cm-1 grid) is used: its interferogram is cut at 25.2-25.6 cm everywhere, where the
2010 release (Hase et al.) is cut shorter above 2200 cm-1.

**It is apodized.** Hase et al. 2010 applied Norton-Beer strong apodization to the
averaged spectra; the READMEs quote only the unapodized 0.02 cm-1. Degrading file 5
(telluric divided out) to match, Norton-Beer strong beats a plain cut at every width.
Fitted as published, its lines are 2-3x wider than file 5's sinc. So the apodization is
divided out (``deapod``): Norton-Beer strong never falls below 0.09 before the cut, and
the transform beyond 25.4 cm is zeroed, leaving a 25.4 cm boxcar spectrum. What that
loses -- solar content beyond 25 cm -- changes file 5's fitted solar model by 0.002% in
its lines. ``raw`` is the published spectrum, for the control.

Both are Fourier interpolated x4 onto 0.00125 cm-1, since the window fit refuses a source
coarser than its grid, and normalized by their median.

    uv run python scripts/make_ace_solar_source.py   # -> data/stellar/solar_ace_fts_2026_{deapod,raw}.npz
"""
import argparse, hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACE = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/references/aux/"
           "bhusal2026_ace_fts_2026/ACE-FTS_Solar_Spectrum_2026.csv")
# Measured with tellurix_fts.ils.describe_truncation on 100 cm-1 stretches, 1900-4400 cm-1.
MOPD_CM = 25.4
UPSAMPLE = 4


def norton_beer_strong(u):
    s = np.clip(1.0 - u**2, 0.0, None)
    return np.where(u <= 1.0, 0.09 + 0.5875 * s**2 + 0.3225 * s**4, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ace", type=Path, default=ACE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/stellar")
    args = parser.parse_args()

    data = np.loadtxt(args.ace, delimiter=",", skiprows=1)
    nu, flux = data[:, 0], data[:, 1]
    step = 0.005
    if not np.allclose(np.diff(nu), step, atol=1e-9):
        raise SystemExit(f"{args.ace} is not on a uniform {step} cm-1 grid")
    n = flux.size
    # Mirrored at both ends, so the transform's wrap-around meets a continuous signal.
    extended = np.concatenate([flux[::-1], flux, flux[::-1]])
    transform = np.fft.rfft(extended)
    opd_cm = np.fft.rfftfreq(extended.size, d=step)
    sha = hashlib.sha256(args.ace.read_bytes()).hexdigest()
    for name in ("deapod", "raw"):
        if name == "deapod":
            spectrum = np.where(opd_cm <= MOPD_CM,
                                transform / np.maximum(norton_beer_strong(opd_cm / MOPD_CM), 1e-30), 0.0)
        else:
            spectrum = transform
        padded = np.zeros(extended.size * UPSAMPLE // 2 + 1, complex)
        padded[:spectrum.size] = spectrum
        fine = np.fft.irfft(padded, n=extended.size * UPSAMPLE)[n * UPSAMPLE:2 * n * UPSAMPLE] * UPSAMPLE
        grid = nu[0] + np.arange(fine.size) * step / UPSAMPLE
        output = args.output_dir / f"solar_ace_fts_2026_{name}.npz"
        np.savez_compressed(output, wavenumber_cm1=grid, flux=fine / np.median(fine),
                            note=f"ACE-FTS 2026 (Bhusal et al.), {name}, Norton-Beer strong at {MOPD_CM} cm; "
                                 f"input sha256 {sha}; make_ace_solar_source.py")
        print(f"{output}: {grid[0]:.3f}-{grid[-1]:.3f} cm-1, {fine.size} samples")


if __name__ == "__main__":
    main()
