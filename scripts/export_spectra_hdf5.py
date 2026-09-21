#!/usr/bin/env python
"""Pack a run's corrected spectra into one self-contained HDF5 file.

The run record (`src/jax_telluric/record.py`) holds the fitted *parameters* and
is the archival product: 530 KB for the whole Arcturus atlas, from which
`rebuild_arcturus_page.py` regenerates every array to 2.8e-7. That is the right
thing to keep in git and the wrong thing to hand someone who wants spectra,
because regenerating needs this repository, the AER line files, the original
data and a GPU.

This writes the other artefact: every corrected spectrum in one file, readable
with nothing but `h5py`, carrying its own units, descriptions and the full
provenance copied from the record.

    uv run python scripts/export_spectra_hdf5.py \\
        --record data/corrected/atlas/arcturus_atlas.h5 \\
        --output data/corrected/arcturus_spectra.h5

Rows are padded to the longest spectrum with NaN, which costs nothing after
compression and makes every dataset a plain 2-D array. Read one back with:

    import h5py
    with h5py.File("arcturus_spectra.h5") as f:
        i = list(f["key"].asstr()).index("ab5000_ summer")
        good = f["reliable"][i]
        nu, flux = f["wavenumber_cm1"][i][good], f["corrected"][i][good]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# name -> (dtype, units, what it is). `corrected` is the product; the rest are
# there so a reader can check the work or rebuild the correction differently.
COLUMNS = {
    "wavenumber_cm1": ("f8", "cm-1", "vacuum wavenumber, ascending"),
    "observed": ("f4", "continuum units", "the input spectrum, NaN where not fitted"),
    "corrected": ("f4", "continuum units",
                  "THE PRODUCT: (observed / model_flux) * stellar_only -- the telluric "
                  "absorption removed and the stellar model retained"),
    "model_flux": ("f4", "continuum units", "the full forward model, convolved"),
    "stellar_only": ("f4", "continuum units", "the same model with the atmosphere removed"),
    "effective_transmission": ("f4", "fraction",
                               "model_flux / stellar_only -- the operator the correction "
                               "applied. NOT the unconvolved transmission, which differs "
                               "from it by up to 0.48 in a deep core"),
    "transmission_unconvolved": ("f4", "fraction",
                                 "atmospheric transmission before the instrument profile; "
                                 "diagnostic only, do not divide by this"),
    "continuum": ("f4", "continuum units",
                  "the FITTED Chebyshev continuum: a free polynomial carrying the instrument, "
                  "the input's own normalisation and whatever else did not fit. It is not the "
                  "star's continuum -- see stellar_continuum"),
    "stellar_continuum": ("f4", "model flux units",
                          "the stellar model's own physical continuum, interpolated to the "
                          "pixels. Unlike the fitted continuum this is a prediction, not a "
                          "free function, and it carries the bound-free edges: the Brackett "
                          "edge at 1458.8 nm is a 0.19% step for Arcturus and 5.6% for A0V. "
                          "Its units are the model's, so only its shape is meaningful"),
    "uncertainty": ("f4", "continuum units", "per-pixel sigma where the reduction supplies one"),
    "reliable": ("?", "", "True where the pixel was fitted and the transmission exceeds the "
                          "run's floor (see the transmission_floor attribute). That floor is "
                          "permissive by design -- recut on effective_transmission for line work"),
}

ABOUT = (
    "Telluric-corrected spectra from jax-telluric. Use 'corrected'; select pixels with "
    "'reliable'. Do NOT divide by 'transmission_unconvolved' -- convolution does not "
    "commute with multiplication, so that leaves a derivative-shaped spike beside every "
    "strong line, 86% of which is the method rather than the fit. The fitted parameters, "
    "the input hashes and the run configuration are in /parameters and the /config, "
    "/physics and /inputs groups, copied from the run record."
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, required=True,
                        help="the run record; the .npz arrays are read from beside it "
                             "unless --arrays says otherwise")
    parser.add_argument("--arrays", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression", default="gzip")
    args = parser.parse_args()

    import h5py

    from jax_telluric import read_record
    from jax_telluric.record import text

    record = read_record(args.record)
    arrays = args.arrays or args.record.parent
    keys = [record.key(index) for index in range(len(record.pages))]

    # Arcturus writes <page>_<epoch>.npz, IGRINS <frame>_<order>.npz -- both are
    # the key joined by an underscore, which is the only naming this needs to know.
    paths, missing = [], []
    for key in keys:
        candidate = arrays / (("_".join(key)) + ".npz")
        (paths if candidate.exists() else missing).append(candidate)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(keys)} arrays are not beside the record, starting with "
            f"{missing[0]}. They are a regenerable cache: rebuild them, or point --arrays "
            "at where they live.")

    widths = []
    for path in paths:
        with np.load(path) as values:
            widths.append(int(values["wavenumber_cm1"].size))
    rows, width = len(paths), max(widths)
    print(f"{rows} spectra, up to {width} pixels each")

    packed = {}
    for name, (dtype, _, _) in COLUMNS.items():
        packed[name] = np.zeros((rows, width), dtype=dtype) if dtype == "?" else \
            np.full((rows, width), np.nan, dtype=dtype)

    # A run older than the stellar_continuum column can still have it: the
    # continuum is a pure function of the stellar model and the pixel
    # wavenumbers, both of which the record identifies. Computing it here beats
    # refitting an atlas to add a column that was always derivable.
    stellar = None
    if "stellar" in record.inputs:
        from jax_telluric import StellarSpectrum, resample_stellar_continuum

        stellar_path = Path(record.inputs["stellar"])
        if stellar_path.exists():
            stellar = StellarSpectrum.from_npz(stellar_path)
            if stellar.continuum is None:
                print(f"note: {stellar_path.name} carries no usable flux_continuum; "
                      "stellar_continuum will be absent")
                stellar = None

    present = set()
    for index, path in enumerate(paths):
        with np.load(path) as values:
            n = int(values["wavenumber_cm1"].size)
            star = np.asarray(values["stellar_only"], dtype=float)
            effective = np.where(np.abs(star) > 1.0e-12,
                                 np.asarray(values["model_flux"], dtype=float)
                                 / np.where(np.abs(star) > 1.0e-12, star, 1.0), np.nan)
            for name in COLUMNS:
                if name == "effective_transmission":
                    packed[name][index, :n] = effective
                elif name == "transmission_unconvolved":
                    packed[name][index, :n] = values["transmission"]
                elif name in values:
                    packed[name][index, :n] = values[name]
                elif name == "stellar_continuum" and stellar is not None:
                    packed[name][index, :n] = resample_stellar_continuum(
                        stellar, np.asarray(values["wavenumber_cm1"]))
                else:
                    continue
                present.add(name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["about"] = ABOUT
        handle.attrs["format"] = "jax-telluric spectra 1"
        handle.attrs["record"] = str(args.record)
        handle.attrs["key_fields"] = [f.encode() for f in record.key_fields]
        for key, value in record.run.items():
            handle.attrs[key] = "" if value is None else value

        joined = [" ".join(key) for key in keys]
        handle.create_dataset("key", data=[k.encode() for k in joined])
        handle["key"].attrs["description"] = (
            "row identifier, the key fields joined by a space: "
            + ", ".join(record.key_fields))
        for position, field in enumerate(record.key_fields):
            handle.create_dataset(field, data=[k[position].encode() for k in keys])
        handle.create_dataset("pixels", data=np.asarray(widths, dtype="i4"))
        handle["pixels"].attrs["description"] = "valid pixels in each row; the rest is padding"

        for name, (dtype, units, description) in COLUMNS.items():
            if name not in present:
                continue
            dataset = handle.create_dataset(name, data=packed[name],
                                            compression=args.compression, shuffle=True)
            dataset.attrs["units"] = units
            dataset.attrs["description"] = description
            if name == "reliable":
                dataset.attrs["transmission_floor"] = float(record.config["min_transmission"])

        # The parameters and provenance, so the file stands on its own.
        handle.create_dataset("parameters", data=record.pages, compression=args.compression)
        handle["parameters"].attrs["description"] = (
            "the fitted parameters and quality numbers, one row per spectrum, in the same "
            "order. Formal sigmas are in /sigma and are NOT uncertainties: they assume "
            "independent pixel errors and measure three to nine times too small.")
        handle.create_dataset("parameter_names", data=[n.encode() for n in record.parameter_names])
        handle.create_dataset("sigma", data=record.sigma, compression=args.compression)
        handle.create_dataset("correlation", data=record.correlation, compression=args.compression)
        for group_name, block in (("config", record.config), ("physics", record.physics),
                                  ("inputs", record.inputs)):
            group = handle.create_group(group_name)
            for key, value in block.items():
                group.attrs[key] = value

    size = args.output.stat().st_size / 1e6
    raw = sum(path.stat().st_size for path in paths) / 1e6
    print(f"wrote {args.output} ({size:.1f} MB, against {raw:.1f} MB of .npz)")
    print(f"  arrays: {', '.join(sorted(present))}")


if __name__ == "__main__":
    main()
