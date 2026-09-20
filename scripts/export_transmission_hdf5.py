#!/usr/bin/env python
"""Export the unconvolved telluric transmission on the model's own fine grid.

The spectra export (`export_spectra_hdf5.py`) is for a consumer who wants our
corrected spectrum. This is for the opposite consumer: someone running their own
synthesis, who wants the atmosphere as a *multiplicand* to put inside their own
convolution,

    model(nu) = Conv_ILS[ their_continuum(nu) x their_source(nu) x T(nu) ]

fitted against the raw `observed` column. That never divides by a convolved
quantity, so the non-commutation that forces `corrected` to be written as
`(observed / model_flux) * stellar_only` does not arise, and our stellar model
never touches their data.

The transmission in the `.npz` cache and in the spectra export is interpolated
to the atlas pixels, which sample about 2.3 points per resolution element --
enough to plot, not enough to convolve, because the lines are undersampled
before the interpolation ever runs. This writes T on the model grid itself:
4 samples per resolution element at R = 100,000, the grid the opacity was
evaluated on, with no interpolation anywhere in the path.

    uv run python scripts/export_transmission_hdf5.py \\
        --record data/corrected/atlas/arcturus_atlas.h5 \\
        --output data/corrected/arcturus_transmission.h5

Grids are shared: a page window is one (v1, v2) and both epochs sit on it, so
the file stores 310 grids for 598 rows and `grid_index` says which row uses
which. The atmosphere the columns scale is embedded in /profile, so T can be
regenerated on any other grid from this file plus a line list.

Rebuilding is per *window*, not per row -- the grid, the line selection, the
opacity backend and the XLA compilation all belong to the window, and only the
column scales change between epochs. That is the same loop inversion the IGRINS
driver uses and it halves the work here.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path,
                        default=root / "data/corrected/atlas/arcturus_atlas.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many windows; for a smoke test")
    parser.add_argument("--check", action="store_true",
                        help="interpolate back to the pixels and compare against the cached "
                             ".npz transmission, which is the only independent copy")
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import h5py
    import numpy as np

    from jax_telluric import (
        AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, TelluricModel,
        file_sha256, igrins_wavenumber_grid, load_atmosphere_csv, parameters_from_row,
        read_record, select_significant_lines, trim_wavenumber_grid,
    )
    from jax_telluric.record import text

    record = read_record(args.record)
    config, physics, inputs = record.config, record.physics, record.inputs
    pages = record.pages
    species_all = [n[len("log_column_"):] for n in pages.dtype.names
                   if n.startswith("log_column_")]

    def verified(key: str, path: Path) -> Path:
        stored = inputs.get(f"{key}_sha256")
        if stored and path.exists() and file_sha256(path) != stored:
            raise SystemExit(f"{key} has changed since the run: {path}")
        return path

    profile = load_atmosphere_csv(verified("profile", Path(inputs["profile"])))
    line_root = Path(inputs["aer_line_root"])
    zenith = float(config["zenith_angle_deg"])

    # Group the rows by window. Rounding to 1e-6 cm-1 is far below the 5 cm-1
    # grid margin and only exists so a float repr cannot split a shared window.
    windows: dict[tuple[float, float], list[int]] = {}
    for index in range(len(pages)):
        key = (round(float(pages["v1"][index]), 6), round(float(pages["v2"][index]), 6))
        windows.setdefault(key, []).append(index)
    ordered = sorted(windows)
    if args.limit:
        ordered = ordered[:args.limit]
    rows = sorted(i for key in ordered for i in windows[key])
    print(f"{len(rows)} page-epochs on {len(ordered)} distinct windows")

    width = int(max(pages["grid_points"][i] for i in rows))
    grids = np.full((len(ordered), width), np.nan, dtype="f8")
    transmission = np.full((len(pages), width), np.nan, dtype="f4")
    grid_index = np.full(len(pages), -1, dtype="i4")
    grid_points = np.zeros(len(ordered), dtype="i4")
    worst_check = 0.0

    started = time.time()
    for position, key in enumerate(ordered):
        v1, v2 = key
        grid = trim_wavenumber_grid(
            igrins_wavenumber_grid(1.0e7 / v2, 1.0e7 / v1,
                                   resolving_power=float(config["resolving_power"]),
                                   samples_per_resolution=float(config["samples_per_resolution"]),
                                   margin_cm1=float(config["margin_cm1"])),
            v1, v2, float(config["grid_margin_cm1"]))

        databases = {}
        for name, molecule_id in sorted(MOLECULE_IDS.items()):
            stem = f"{molecule_id:02d}_{name}"
            try:
                databases[name] = AERLineDatabase(
                    verified(f"aer_{name}", line_root / stem / stem), name, (v1, v2),
                    margin_cm1=float(config["margin_cm1"]))
            except ValueError as exc:
                if not str(exc).startswith(f"no {name} lines found"):
                    raise
        budget = float(config.get("line_budget", 0.0))
        if budget > 0.0:
            databases = {n: select_significant_lines(d, profile, n, budget,
                                                     maximum_column_scale=float(np.exp(2.0)))
                         for n, d in databases.items()}

        opacity = ExoJAXOpacityBackend.prepare(
            databases, grid, methods=physics["opacity_method"],
            temperature_range_k=(float(np.min(profile.temperature_k)),
                                 float(np.max(profile.temperature_k))),
            maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
            vectorize_layers=bool(physics["vectorize_layers"]),
            mixed_precision=bool(physics["mixed_precision"]),
            pressure_shift=bool(physics["pressure_shift"]))
        continuum = MTCKDWaterContinuum.from_netcdf(
            verified("mt_ckd", Path(inputs["mt_ckd"])), grid)
        model = TelluricModel(profile, grid, opacity, continuum=continuum,
                              accuracy_mode=physics["accuracy_mode"],
                              max_lsf_sigma_kms=float(physics["max_lsf_sigma_kms"]),
                              pixel_integration=physics["pixel_integration"])

        grids[position, :grid.size] = grid
        grid_points[position] = grid.size

        # One compile per window, one evaluation per epoch on it. `transmission`
        # reads only the column scales and the zenith angle, so the fitted
        # velocity, LSF and continuum play no part and no instrument model is
        # needed here at all.
        for index in windows[key]:
            row = pages[index]
            present = [s for s in species_all if s in model.species]
            parameters = parameters_from_row(row, present)
            values = np.asarray(model.transmission(parameters, zenith))
            transmission[index, :grid.size] = values
            grid_index[index] = position

            if args.check:
                cached = args.record.parent / f"{text(row['page'])}_{text(row['epoch'])}.npz"
                if cached.exists():
                    with np.load(cached) as stored:
                        pixels = np.asarray(stored["wavenumber_cm1"])
                        reference = np.asarray(stored["transmission"])
                        mask = np.asarray(stored["mask"]).astype(bool)
                    again = np.interp(pixels, grid, values)
                    drift = float(np.max(np.abs(again[mask] - reference[mask])))
                    worst_check = max(worst_check, drift)

        elapsed = time.time() - started
        rate = elapsed / (position + 1)
        print(f"[{position + 1:3d}/{len(ordered)}] {v1:8.1f}-{v2:8.1f} cm-1  "
              f"{grid.size:5d} samples  {len(windows[key])} epochs  "
              f"{elapsed / 60:5.1f} min elapsed, {rate * (len(ordered) - position - 1) / 60:5.1f} to go",
              flush=True)

    keep = grid_index >= 0
    print(f"\nwriting {int(keep.sum())} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["about"] = (
            "Unconvolved telluric transmission from jax-telluric, on the forward model's own "
            "grid: 4 samples per resolution element at R = 100,000, no interpolation. This is "
            "a multiplicand, not a divisor -- put it inside your own instrument convolution, "
            "Conv[continuum x source x T], and fit that against the raw observed spectrum. "
            "Row i uses grid grid_index[i], valid over its first grid_points entry. The "
            "zenith angle is 0, so this is the vertical transmission; the fitted column "
            "scales in /parameters already absorbed any slant path.")
        handle.attrs["format"] = "jax-telluric transmission 1"
        handle.attrs["record"] = str(args.record)
        handle.attrs["zenith_angle_deg"] = zenith
        handle.attrs["resolving_power"] = float(config["resolving_power"])
        handle.attrs["samples_per_resolution"] = float(config["samples_per_resolution"])
        for key, value in record.run.items():
            handle.attrs[key] = "" if value is None else value

        index_map = np.cumsum(keep) - 1
        keys = [record.key(i) for i in range(len(pages)) if keep[i]]
        handle.create_dataset("key", data=[" ".join(k).encode() for k in keys])
        for position, field in enumerate(record.key_fields):
            handle.create_dataset(field, data=[k[position].encode() for k in keys])

        handle.create_dataset("wavenumber_cm1", data=grids,
                              compression=args.compression, shuffle=True)
        handle["wavenumber_cm1"].attrs["units"] = "cm-1"
        handle["wavenumber_cm1"].attrs["description"] = (
            "one grid per distinct page window, ascending, uniform in log wavenumber "
            "(a constant velocity step); valid over the first grid_points entries")
        handle.create_dataset("grid_points", data=grid_points)
        handle.create_dataset("grid_index", data=grid_index[keep].astype("i4"))
        handle["grid_index"].attrs["description"] = "which wavenumber_cm1 row this spectrum uses"
        handle.create_dataset("transmission", data=transmission[keep],
                              compression=args.compression, shuffle=True)
        handle["transmission"].attrs["units"] = "fraction"
        handle["transmission"].attrs["description"] = (
            "exp(-sum_layers tau) at the fitted column scales, vertical path, before any "
            "instrument profile. Multiply this into your synthesis; do not divide by it.")

        # The atmosphere the scales multiply, so T can be regenerated elsewhere.
        group = handle.create_group("profile")
        group.attrs["source"] = str(inputs["profile"])
        group.attrs["description"] = (
            "the layered atmosphere, top to bottom. A fitted log_column_X scales vmr/X "
            "uniformly over every layer; tau_X = cross_section(T, P, p_X) * air_column_cm2 * vmr.")
        for name, values in (("pressure_layer_bar", profile.pressure_layer_bar),
                             ("pressure_edges_bar", profile.pressure_edges_bar),
                             ("temperature_k", profile.temperature_k),
                             ("altitude_km", profile.altitude_km),
                             ("air_column_cm2", profile.air_column_cm2)):
            group.create_dataset(name, data=np.asarray(values))
        vmr = group.create_group("vmr")
        for name, values in profile.vmr.items():
            vmr.create_dataset(name, data=np.asarray(values))

        handle.create_dataset("parameters", data=pages[keep], compression=args.compression)
        handle["parameters"].attrs["description"] = (
            "the fitted parameters and quality numbers, one row per spectrum, in the same "
            "order. log_column_X is the natural log of the scale applied to that species' "
            "VMR profile. Formal sigmas are in /sigma and are NOT uncertainties.")
        handle.create_dataset("parameter_names", data=[n.encode() for n in record.parameter_names])
        handle.create_dataset("sigma", data=record.sigma[keep], compression=args.compression)
        handle.create_dataset("correlation", data=record.correlation[keep],
                              compression=args.compression)
        for group_name, block in (("config", config), ("physics", physics), ("inputs", inputs)):
            written = handle.create_group(group_name)
            for key, value in block.items():
                written.attrs[key] = value

    size = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size:.1f} MB)")
    if args.check:
        print(f"max |interpolated back to pixels - cached transmission| = {worst_check:.3e}")


if __name__ == "__main__":
    main()
