#!/usr/bin/env python
"""Build a site profile for one IGRINS night from the frames' own headers.

`make_site_profile.py` wants a surface pressure, a temperature and a
precipitable water column. The first two are in every header; the third was
being chosen by eye, and on three nights that was out by up to a factor of two
-- which matters, because `precompute_opacity` linearizes self-broadening about
the profile's own water content and a large scaling walks outside that
expansion. This fills all three in from the night itself and then calls
`make_site_profile.py`, so there is still only one piece of code that writes a
profile.

    uv run python scripts/igrins_site_profile.py \\
        --spec data/igrins/20181220_*/SDCH_*.spec.fits \\
        --output data/profiles/dct_2018.csv

The weather cards are unreliably populated across the archive, so the surface
values are medians over whatever frames carry them, and the script says how many
that was. If nothing carries a dewpoint or a humidity it refuses rather than
inventing a column.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+",
                        help="the night's PLP .spec.fits files, one band is enough")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", default="2020", help="trace-gas abundances")
    parser.add_argument("--precipitable-water-mm", type=float, default=None,
                        help="override the estimate from the dewpoint")
    parser.add_argument("--scale-height-km", type=float, default=1.4,
                        help="effective water scale height used for the estimate; the layer "
                             "profile's own 2.0 km is a different quantity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from jax_telluric import precipitable_water_mm, read_igrins_observation

    temperature, pressure, water, sites = [], [], [], set()
    for path in args.spec:
        observation = read_igrins_observation(path)
        surface = observation.surface
        sites.add((surface["site"], surface["altitude_km"]))
        if surface["temperature_k"] is not None:
            temperature.append(surface["temperature_k"])
        if surface["pressure_hpa"] is not None:
            pressure.append(surface["pressure_hpa"])
        estimate = precipitable_water_mm(surface, args.scale_height_km)
        if estimate is not None:
            water.append(estimate)

    if len(sites) != 1:
        raise SystemExit(f"these frames are from more than one site: {sorted(sites)}")
    site, altitude = sites.pop()
    if not temperature or not pressure:
        raise SystemExit(
            f"{len(args.spec)} frames carry no usable surface temperature or pressure; "
            "recent Gemini frames drop those cards, so pass a profile built by hand")
    seed = args.precipitable_water_mm
    if seed is None:
        if not water:
            raise SystemExit(
                "no frame carries a dewpoint or a humidity, so the water column cannot be "
                "estimated; pass --precipitable-water-mm")
        seed = float(np.median(water))

    print(f"{site} at {altitude:.3f} km, from {len(args.spec)} frames")
    print(f"  surface temperature {np.median(temperature):.2f} K  "
          f"(from {len(temperature)} frames, spread {np.ptp(temperature):.2f})")
    print(f"  surface pressure    {np.median(pressure):.1f} hPa station "
          f"(from {len(pressure)}, spread {np.ptp(pressure):.1f})")
    print(f"  precipitable water  {seed:.2f} mm "
          + (f"(estimated from {len(water)} dewpoints, spread {np.ptp(water):.2f})"
             if args.precipitable_water_mm is None else "(given)"))

    command = [
        sys.executable, str(root / "scripts/make_site_profile.py"),
        "--site", site, "--site-altitude-km", f"{altitude:.3f}",
        "--surface-pressure-hpa", f"{np.median(pressure):.1f}",
        "--surface-temperature-k", f"{np.median(temperature):.2f}",
        "--precipitable-water-mm", f"{seed:.2f}",
        "--epoch", args.epoch, "--output", str(args.output),
    ]
    print("\n" + " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
