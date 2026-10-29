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

A co-add record (`physics["coadd"]`, the IAG atlases) is rebuilt as its
`CoaddedTelluricModel`: one atmosphere per day, each node at its own zenith angle
and moved by its own velocity, averaged by weight. That mean is what the atlas
saw, so it is what this writes; there is no single profile or zenith angle, and
the file carries the co-add's components and nodes in /coadd instead of /profile.

    uv run python scripts/export_transmission_hdf5.py \\
        --record data/corrected/solar/iag_nir/iag_nir.h5 \\
        --output data/corrected/iag_nir_transmission.h5 --check
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Which molecules to build line lists for comes from the **record**, not from a
# table here. A private species list in this file would be a third copy of the
# mistake that hid OCS and O3 for as long as it did: a solar record carries
# whatever its per-window scan found, and a hardcoded six would silently drop
# the rest. Iterating the record's own species is also faster than iterating all
# 46 AER molecules, which is what a naive fix would do.


def write_profile(group, profile, source: str) -> None:
    import numpy as np

    group.attrs["source"] = source
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


def write_coadd(handle, coadd, physics) -> None:
    import numpy as np

    group = handle.create_group("coadd")
    group.attrs["description"] = (
        "the atmospheres the transmission averages: one component per observing day, each "
        "with its own profile and nodes (weight, zenith angle, velocity). The weights are "
        "normalized over all nodes. One set of column scales serves every component.")
    group.attrs["source"] = str(physics["coadd"])
    group.attrs["sha256"] = str(physics.get("coadd_sha256", ""))
    total = sum(node.weight for _, nodes, _ in coadd for node in nodes)
    for number, (component, nodes, label) in enumerate(coadd):
        item = group.create_group(f"component_{number:02d}")
        item.attrs["label"] = label
        for name, values in (("weight", [node.weight / total for node in nodes]),
                             ("zenith_angle_deg", [node.zenith_angle_deg for node in nodes]),
                             ("velocity_kms", [node.velocity_kms for node in nodes])):
            item.create_dataset(name, data=np.asarray(values, dtype="f8"))
        item["velocity_kms"].attrs["description"] = (
            "positive moves that node's transmission to the red, toward lower wavenumber")
        write_profile(item.create_group("profile"), component, label)


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
    import jax
    import numpy as np

    from tellurix import (
        AER_MOLECULE_IDS, AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, TelluricModel,
        file_sha256, constant_velocity_grid, load_atmosphere_csv, parameters_from_row,
        read_record, select_significant_lines, trim_wavenumber_grid,
    )
    from tellurix.record import text
    from tellurix.quality import write_upper_bound_flags

    record = read_record(args.record)
    config, physics, inputs = record.config, record.physics, record.inputs
    pages = record.pages
    species_all = [n[len("log_column_"):] for n in pages.dtype.names
                   if n.startswith("log_column_")]
    # A solar batch (fit_fts_batch.py, keyed by file and window) builds each
    # window's model from that window's scanned species alone, `free_species`, and
    # stores 0 for the rest of the run's union. Arcturus and IGRINS keep every
    # species in the model and pin the ones not free at their profile column, so
    # there the union is right. Building the union for a solar window adds the
    # absorbers its scan left out at full column: up to 1.6e-3 in the IAG windows.
    # An Arcturus run of fit_fts_batch.py (keyed by page and epoch) scans its species
    # per window too, and says so in its physics.
    windowed = (tuple(record.key_fields) == ("file", "window")
                or physics.get("species") == "computed per window by the scan")

    def verified(key: str, path: Path) -> Path:
        stored = inputs.get(f"{key}_sha256")
        if stored and path.exists() and file_sha256(path) != stored:
            raise SystemExit(f"{key} has changed since the run: {path}")
        return path

    profile = load_atmosphere_csv(verified("profile", Path(inputs["profile"])))
    # An Arcturus run may give each epoch its own atmosphere ("summer=PATH,winter=PATH").
    epoch_profiles = {}
    for item in filter(None, str(physics.get("epoch_profiles", "")).split(",")):
        epoch, _, _ = item.partition("=")
        epoch_profiles[epoch] = load_atmosphere_csv(
            verified(f"profile_{epoch}", Path(inputs[f"profile_{epoch}"])))
    epoch_field = "epoch" if "epoch" in record.key_fields else None

    def profile_key(index: int) -> str:
        if epoch_profiles and epoch_field:
            epoch = str(text(pages[epoch_field][index]))
            if epoch in epoch_profiles:
                return epoch
        return ""
    # A co-add's own profiles, one per day, held to the hashes the run recorded.
    coadd = None
    if physics.get("coadd"):
        from tellurix import CoaddedTelluricModel
        from tellurix_fts.window import load_coadd

        coadd_path = Path(inputs.get("coadd") or root / physics["coadd"])
        coadd, provenance = load_coadd(coadd_path, root)
        if provenance["coadd_sha256"] != physics.get("coadd_sha256"):
            raise SystemExit(f"{coadd_path} has changed since the run")
        for item in json.loads(coadd_path.read_text())["components"]:
            key = f"coadd_profile_{item.get('label', item['profile'])}"
            if f"{key}_sha256" not in inputs:
                raise SystemExit(f"the record holds no hash for the co-add profile {item['profile']}")
            verified(key, root / item["profile"])
    # The opacity's validity bounds cover every profile it serves, as the fit's did.
    bounding = [profile, *epoch_profiles.values(), *(component for component, _, _ in coadd or ())]
    line_root = Path(inputs.get("aer_line_root")
                     or root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule")
    # A record without the key predates HITRAN's H2O shifts and used AER's.
    shift_table = None
    if physics.get("h2o_pressure_shift", "aer") == "hitran":
        shift_table = verified("hitran_h2o_pressure_shift", Path(
            inputs.get("hitran_h2o_pressure_shift")
            or root / "data/databases/hitran/h2o_pressure_shift.npz"))
    # Likewise, one without this used AER's widths.
    width_scale = float(physics.get("pressure_width_scale", 1.0))
    # And one without this used AER's strengths.
    from tellurix import recorded_line_strength_table
    strength_table = recorded_line_strength_table(physics, root / "data/line_strength")
    widened = {s.strip().upper() for s in physics.get("pressure_width_species", "").split(",")}

    # The atlas fits everything at the zenith and carries one angle in the
    # config; an IGRINS night carries the frame's own angle per row, 18.7 to
    # 67.3 degrees on one night. Take the row's when it has one, so the saved T
    # is the slant transmission that frame actually saw.
    # The atlas and IGRINS hold a resolving power fixed for the whole run; the
    # NSO FTS spectra hold the interferogram truncation fixed instead, so the
    # resolution element is constant in WAVENUMBER and R rises across the band
    # -- 114,075 at 2000 cm-1 to 513,338 at 9000. Those records carry the FWHM
    # rather than an R, and each window's grid is sized from its own centre.
    fwhm_cm1 = physics.get("fwhm_cm1")
    if "resolving_power" in config:
        resolving_power_for = lambda v1, v2: float(config["resolving_power"])
    elif fwhm_cm1:
        resolving_power_for = lambda v1, v2: 0.5 * (v1 + v2) / float(fwhm_cm1)
    else:
        raise SystemExit("this record carries neither a resolving power nor an ILS width")

    per_row_zenith = "zenith_angle_deg" in pages.dtype.names
    if coadd is not None:
        # Each node carries its own angle; the row has none to report.
        zenith_of = lambda row: float("nan")
    elif not per_row_zenith and "zenith_angle_deg" not in config:
        raise SystemExit("this record carries no zenith angle, per row or in its config")
    else:
        zenith_of = ((lambda row: float(row["zenith_angle_deg"])) if per_row_zenith
                     else (lambda row: float(config["zenith_angle_deg"])))

    # Group the rows by window. Rounding to 1e-6 cm-1 is far below the 5 cm-1
    # grid margin and only exists so a float repr cannot split a shared window.
    windows: dict[tuple[float, float, bool], list[int]] = {}
    for index in range(len(pages)):
        # Coupling is per row: two epochs of one Arcturus page are two fits and may
        # choose differently, and then they need two opacities on one grid.
        key = (round(float(pages["v1"][index]), 6), round(float(pages["v2"][index]), 6),
               bool(pages["line_coupling"][index]) if "line_coupling" in pages.dtype.names
               else bool(physics.get("line_coupling", False)))
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

    # A solar batch names its arrays in its summary, not by the record's key.
    summary_path = args.record.with_name(args.record.stem + "_summary.json")
    arrays_by_window = {}
    if args.check and summary_path.exists():
        for entry in json.loads(summary_path.read_text())["results"]:
            if entry.get("npz"):
                arrays_by_window[(round(float(entry["v1"]), 6), round(float(entry["v2"]), 6),
                                  entry.get("epoch") or "")] = (
                    args.record.parent / Path(entry["npz"]).name)

    started = time.time()
    for position, key in enumerate(ordered):
        v1, v2, coupled = key
        grid = trim_wavenumber_grid(
            constant_velocity_grid(1.0e7 / v2, 1.0e7 / v1,
                                   resolving_power=resolving_power_for(v1, v2),
                                   samples_per_resolution=float(config["samples_per_resolution"]),
                                   margin_cm1=float(config["margin_cm1"])),
            v1, v2, float(config["grid_margin_cm1"]))

        # Per row when the record says so (a coupled run refits a window without
        # coupling where its coupled opacity goes too negative, or where the
        # uncoupled fit is better); a record without the column used the run's setting.
        if windowed:
            chosen = {str(text(pages["free_species"][i])) for i in windows[key]}
            if len(chosen) != 1:
                raise SystemExit(f"rows of window {v1}-{v2} disagree on their species")
            window_species = sorted(n for n in chosen.pop().split("+") if n)
        else:
            window_species = sorted(species_all)
        databases = {}
        for name in window_species:
            stem = f"{AER_MOLECULE_IDS[name]:02d}_{name}"
            try:
                databases[name] = AERLineDatabase(
                    verified(f"aer_{name}", line_root / stem / stem), name, (v1, v2),
                    margin_cm1=float(config["margin_cm1"]),
                    line_coupling=(verified("aer_line_coupling", line_root.parent / "lncpl_lines")
                                   if coupled else None),
                    pressure_shift_table=shift_table if name == "H2O" else None,
                    pressure_width_scale=width_scale if name in widened else 1.0,
                    line_strength_table=strength_table)
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
            temperature_range_k=(min(float(np.min(p.temperature_k)) for p in bounding),
                                 max(float(np.max(p.temperature_k)) for p in bounding)),
            maximum_pressure_bar=max(float(np.max(p.pressure_layer_bar)) for p in bounding),
            vectorize_layers=bool(physics["vectorize_layers"]),
            mixed_precision=bool(physics["mixed_precision"]),
            pressure_shift=bool(physics["pressure_shift"]),
            line_coupling=coupled)
        # A "fast" record fitted lines only -- the windows past MT_CKD's 20000
        # cm-1 -- and must be rebuilt the same way.
        continuum = None if physics["accuracy_mode"] == "fast" else MTCKDWaterContinuum.from_netcdf(
            verified("mt_ckd", Path(inputs.get("mt_ckd")
                     or root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc")), grid)
        if physics.get("o2_cia"):
            from tellurix import ContinuumSum, O2CollisionInducedContinuum

            o2 = O2CollisionInducedContinuum(grid)
            if o2.terms:
                continuum = o2 if continuum is None else ContinuumSum((continuum, o2))
        grids[position, :grid.size] = grid
        grid_points[position] = grid.size
        if not databases:
            # Nothing above the scan's cut: the fit had no atmosphere here.
            for index in windows[key]:
                transmission[index, :grid.size] = 1.0
                grid_index[index] = position
            continue

        def single(atmosphere):
            return TelluricModel(atmosphere, grid, opacity, continuum=continuum,
                                 accuracy_mode=physics["accuracy_mode"],
                                 max_lsf_sigma_kms=float(physics["max_lsf_sigma_kms"]),
                                 pixel_integration=physics["pixel_integration"])

        # One model per atmosphere the window's rows were fitted through: the
        # run's profile, or each epoch's own.
        models = {name: (single(epoch_profiles.get(name, profile)) if coadd is None
                         else CoaddedTelluricModel(
                             [(single(component), nodes) for component, nodes, _ in coadd]))
                  for name in sorted({profile_key(i) for i in windows[key]})}
        model = next(iter(models.values()))

        # One compile per window, one evaluation per epoch on it. Calling
        # `transmission` eagerly instead costs 35.9 s on the first call of a
        # window against 6.1 s to compile and 0.013 s to run -- the uncompiled
        # kernel dispatches operation by operation, which is the trap in
        # CLAUDE.md under "What a fit costs". `transmission` reads only the
        # column scales and the zenith angle, so the fitted velocity, LSF and
        # continuum play no part and no instrument model is needed here at all,
        # and the two epochs of a window differ only in those scales.
        present = [s for s in species_all if s in model.species]
        template = parameters_from_row(pages[windows[key][0]], present)

        # The zenith angle is an operand, not a captured constant: IGRINS rows
        # in one window come from frames at different airmasses and would
        # otherwise recompile per frame.
        def compiled(chosen):
            @jax.jit
            def transmission_at(scales, angle, _model=chosen, _template=template):
                return _model.transmission(_template._replace(log_column_scales=scales), angle)
            return transmission_at

        evaluators = {name: compiled(chosen) for name, chosen in models.items()}
        for index in windows[key]:
            row = pages[index]
            transmission_at = evaluators[profile_key(index)]
            parameters = parameters_from_row(row, present)
            values = np.asarray(transmission_at(
                {name: float(value)
                 for name, value in parameters.log_column_scales.items()},
                zenith_of(row)))
            transmission[index, :grid.size] = values
            grid_index[index] = position

            if args.check:
                cached = arrays_by_window.get(
                    (v1, v2, str(text(pages[epoch_field][index])) if epoch_field else "")) or (
                    args.record.parent / ("_".join(record.key(index)) + ".npz"))
                if cached.exists():
                    with np.load(cached) as stored:
                        pixels = np.asarray(stored["wavenumber_cm1"])
                        # The solar fit names it for the pixels it is on.
                        reference = np.asarray(stored["transmission"] if "transmission" in stored.files
                                               else stored["transmission_pixels"])
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
            "Unconvolved telluric transmission from tellurix, on the forward model's own "
            f"grid: {float(config['samples_per_resolution']):g} samples per resolution element"
            + (f" at R = {float(config['resolving_power']):,.0f}" if "resolving_power" in config else "")
            + ", no interpolation. This is "
            "a multiplicand, not a divisor -- put it inside your own instrument convolution, "
            "Conv[continuum x source x T], and fit that against the raw observed spectrum. "
            "Row i uses grid grid_index[i], valid over its first grid_points entry. The "
            "zenith angle is in /zenith_angle_deg, one per row: 0 for the Arcturus atlas, so "
            "that is the vertical transmission and the fitted column scales in /parameters "
            "already absorbed any slant path, and the frame's own angle for an IGRINS night, "
            "so that is the slant transmission along the line of sight. A co-add (the IAG "
            "atlases) has no angle of its own: its rows are the weighted mean over the "
            "atmospheres and nodes described in /coadd, which replaces /profile. A run that "
            "gave each epoch its own atmosphere has /profile_<epoch> in place of /profile.")
        handle.attrs["format"] = "tellurix transmission 1"
        handle.attrs["record"] = str(args.record)
        if "resolving_power" in config:
            handle.attrs["resolving_power"] = float(config["resolving_power"])
        else:
            # Constant in wavenumber, not in R: quote what is actually fixed.
            handle.attrs["fwhm_cm1"] = float(fwhm_cm1)
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
        handle.create_dataset("zenith_angle_deg", data=np.array(
            [zenith_of(pages[i]) for i in range(len(pages)) if keep[i]], dtype="f8"))
        handle["zenith_angle_deg"].attrs["description"] = (
            "the angle each row's transmission was evaluated at; 0 means a vertical column, "
            "NaN a co-add, whose nodes each carry their own (see /coadd)")
        handle.create_dataset("transmission", data=transmission[keep],
                              compression=args.compression, shuffle=True)
        handle["transmission"].attrs["units"] = "fraction"
        handle["transmission"].attrs["description"] = (
            "exp(-sum_layers tau) at the fitted column scales, vertical path, before any "
            "instrument profile. Multiply this into your synthesis; do not divide by it."
            if coadd is None else
            "the co-add's weighted mean of exp(-sum_layers tau / cos z) over its nodes, each "
            "node at its own zenith angle z and moved by its own velocity, at the fitted column "
            "scales and before any instrument profile. The fitted velocity_kms in /parameters "
            "is not applied, as in every row of every record. Multiply this into your "
            "synthesis; do not divide by it.")

        if coadd is not None:
            write_coadd(handle, coadd, physics)
        # The atmosphere the scales multiply, so T can be regenerated elsewhere;
        # with one per epoch, /profile_<epoch> beside it.
        group = handle.create_group("profile") if coadd is None and not epoch_profiles else None
        if group is not None:
            write_profile(group, profile, str(inputs["profile"]))
        for epoch, atmosphere in sorted(epoch_profiles.items()):
            write_profile(handle.create_group(f"profile_{epoch}"), atmosphere,
                          str(inputs[f"profile_{epoch}"]))

        write_upper_bound_flags(handle, pages[keep], species_all)


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
