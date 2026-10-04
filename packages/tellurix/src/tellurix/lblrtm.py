"""Reproducible LBLRTM reference runs for validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .reference import LBLRTMSpectrum, read_tape12_single_precision
from .types import AtmosphereProfile


# HITRAN molecule order, which is what record 3.6 is positional in: there is no
# way to give LBLRTM a subset, so reaching OCS at number 19 means declaring all
# nineteen and leaving the ones we have no profile for at zero. The list stops
# at OCS because that is the highest molecule this package models; extending it
# further is adding names, not logic.
_LBLRTM_SPECIES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "NO", "SO2",
                   "NO2", "NH3", "HNO3", "OH", "HF", "HCL", "HBR", "HI", "CLO", "OCS")
# Record 3.6 is read as (8E15.8) because record 3.5 sets JLONG, so the
# abundances wrap every eight molecules.
_ABUNDANCE_PER_LINE = 8
_GAS_CONSTANT_J_MOL_K = 8.314462618


def _layer_values_at_edges(values: np.ndarray) -> np.ndarray:
    """Choose edge values whose adjacent means reproduce layer values."""
    layer = np.asarray(values, dtype=float)
    edge = np.empty(len(layer) + 1)
    edge[0] = layer[0]
    for index, value in enumerate(layer):
        edge[index + 1] = 2.0 * value - edge[index]
    if np.any(edge < 0.0):
        # Some sharply non-monotonic profiles cannot have nonnegative edge
        # values with every adjacent mean exact. Interpolation is safer than
        # emitting negative molecular abundances into LBLRTM.
        centers = np.arange(len(layer), dtype=float) + 0.5
        edge = np.interp(np.arange(len(layer) + 1), centers, layer)
    return edge


def _hydrostatic_altitude_edges(profile: AtmosphereProfile) -> np.ndarray:
    """Derive top-to-bottom altitude edges from the same pressure layers."""
    pressure = np.asarray(profile.pressure_edges_bar)
    temperature = np.asarray(profile.temperature_k)
    molar_mass = np.asarray(profile.mean_molecular_weight_g_mol) * 1.0e-3
    gravity = np.asarray(profile.gravity_m_s2)
    thickness_km = (
        _GAS_CONSTANT_J_MOL_K * temperature / (molar_mass * gravity)
        * np.log(pressure[1:] / pressure[:-1]) / 1000.0
    )
    altitude = np.empty(len(pressure))
    altitude[-1] = float(np.asarray(profile.altitude_km)[-1])
    for index in range(len(thickness_km) - 1, -1, -1):
        altitude[index] = altitude[index + 1] + thickness_km[index]
    return altitude


@dataclass(frozen=True)
class LBLRTMRunConfig:
    """Settings for a ground-to-space, user-profile LBLRTM calculation."""

    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    zenith_angle_deg: float = 0.0
    continuum_flag: int = 1
    description: str = "tellurix reference atmosphere"
    # Pass the profile's own layers -- pressure, temperature and molecular
    # column of each, as LBLRTM's IATM=0 layer input -- instead of levels for
    # LBLRTM's path calculation. Levels make LBLRTM rebuild each layer by
    # interpolating between them, which in the 5000-5020 cm-1 template gave it
    # 13.8% less water than tellurix had (docs/lblrtm_corrected_mode.md); layers
    # make the two codes integrate the same atmosphere, so a difference between
    # them is the physics.
    user_layers: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.wavenumber_min_cm1 < self.wavenumber_max_cm1:
            raise ValueError("wavenumber limits must be positive and increasing")
        if not 0.0 <= self.zenith_angle_deg < 90.0:
            raise ValueError("zenith angle must be in [0, 90) degrees")
        if self.continuum_flag not in range(6):
            raise ValueError("continuum_flag must be one of 0, 1, 2, 3, 4, or 5")


def write_tape5(
    path: str | Path,
    profile: AtmosphereProfile,
    config: LBLRTMRunConfig,
) -> None:
    """Write a fixed-format TAPE5 for an atmospheric transmission run.

    Pressure and altitude edges define the same hydrostatic path used by the
    JAX model. Edge temperature and VMR values are reconstructed so adjacent
    means reproduce the supplied layer values. Abundances use the ``A`` unit
    code (ppmv by volume).
    """

    available = set(profile.vmr)
    unsupported = available - set(_LBLRTM_SPECIES)
    if unsupported:
        raise ValueError(f"unsupported LBLRTM TAPE3 species: {', '.join(sorted(unsupported))}")
    nlayers = len(profile.temperature_k)
    if nlayers < 1:
        raise ValueError("LBLRTM user profiles need at least one layer")
    if config.user_layers:
        _write_layer_tape5(path, profile, config)
        return

    # LBLRTM expects user profile levels from the observer upward.
    altitude = _hydrostatic_altitude_edges(profile)[::-1]
    pressure_hpa = np.asarray(profile.pressure_edges_bar)[::-1] * 1000.0
    temperature = _layer_values_at_edges(profile.temperature_k)[::-1]
    # LBLRTM's ``A`` abundance unit is ppmv relative to dry air, whereas the
    # public profile stores fractions of total moist air.  Convert at each
    # reconstructed level so H2O and the dry gases retain their stated wet-air
    # volume fractions inside LBLRTM.
    edge_vmr = {
        name: _layer_values_at_edges(profile.vmr.get(name, np.zeros(nlayers)))[::-1]
        for name in _LBLRTM_SPECIES
    }
    dry_air_fraction = 1.0 - edge_vmr["H2O"]
    if np.any(dry_air_fraction <= 0.0):
        raise ValueError("LBLRTM profiles require H2O VMR below one")
    abundance_ppmv = np.column_stack(
        [edge_vmr[name] / dry_air_fraction * 1.0e6 for name in _LBLRTM_SPECIES]
    )
    observer_altitude = float(altitude[0])
    space_altitude = float(altitude[-1])

    lines = [f"${config.description[:79]}"]
    flags = (1, 1, config.continuum_flag, 0, 1, 0, 0, 0, 0, 1)
    lines.append("".join(f"{value:5d}" for value in flags) + f"{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}")
    lines.append(
        f"{config.wavenumber_min_cm1:10.3f}{config.wavenumber_max_cm1:10.3f}"
        f"{4.0:10.3f}{0.0:10.3f}{0.04:10.3f}{36.0:10.3f}{-1.0:10.3f}{-1.0:10.3f}"
        f"{0:5d}{0.0:15.3f}{0:5d}"
    )
    lines.append(
        f"{float(temperature[0]):10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}"
        f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}    s"
    )
    # MODEL=0, ITYPE=3, NMOL=7. HSPACE is the top supplied level.
    #
    # IBMAX is the count of layer boundaries, given explicitly on record 3.3B
    # rather than left at 0 for LBLRTM to generate with AUTLAY. Two reasons,
    # and the second is why this is not just a workaround:
    #
    # AUTLAY fails on these profiles. With IBMAX = 0 and AVTRAT/TDIFF left at
    # zero it subdivides without making progress -- on the 12-layer Kitt Peak
    # profiles it emits 600 boundaries all at the same altitude and stops with
    # "THE NUMBER OF GENERATED LAYER BOUNDARIES EXCEEDS THE DIMENSION IBDIM".
    # It happens to survive the 6-layer example_midlatitude.csv, which is the
    # only profile this writer had ever been run on.
    #
    # And the layers are the point. The JAX model integrates optical depth over
    # exactly these edges; letting LBLRTM choose its own means the two codes
    # integrate different atmospheres, so a disagreement between them is no
    # longer attributable to the line physics. Supplying the boundaries makes
    # the comparison a comparison.
    boundaries = np.asarray(altitude, dtype=float)
    lines.append(
        f"{0:5d}{3:5d}{len(boundaries):5d}{0:5d}{0:5d}{len(_LBLRTM_SPECIES):5d}{0:5d}"
        f"{0:2d} {0:2d}{0.0:10.3f}{space_altitude:10.3f}"
        f"{0.5 * (config.wavenumber_min_cm1 + config.wavenumber_max_cm1):10.3f}"
        f"{'':10s}{0.0:10.3f}"
    )
    lines.append(
        f"{observer_altitude:10.3f}{0.0:10.3f}{config.zenith_angle_deg:10.3f}"
        f"{0.0:10.3f}{0.0:10.3f}{0:5d}{'':5s}{observer_altitude:10.3f}"
    )
    # Record 3.3B, 8F10.3: the boundaries themselves, ascending from the
    # observer. This replaces record 3.3A, which LBLRTM reads only when
    # IBMAX is zero.
    for start in range(0, len(boundaries), 8):
        lines.append("".join(f"{z:10.3f}" for z in boundaries[start : start + 8]))
    nlevels = nlayers + 1
    lines.append(f"{nlevels:5d}{' tellurix profile':24s}")
    jchar = "A" * len(_LBLRTM_SPECIES)
    for z_km, pressure, temp, abundances in zip(altitude, pressure_hpa, temperature, abundance_ppmv):
        lines.append(f"{z_km:10.3E}{pressure:10.3E}{temp:10.3E}     AA L {jchar}")
        for start in range(0, len(abundances), _ABUNDANCE_PER_LINE):
            lines.append("".join(f"{value:15.8E}"
                                 for value in abundances[start : start + _ABUNDANCE_PER_LINE]))
    lines.extend((f"{-1.0:4.1f}", f"{-1.0:4.1f}", "%"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_layer_tape5(path, profile: AtmosphereProfile, config: LBLRTMRunConfig) -> None:
    """TAPE5 with IATM=0: records 2.1-2.1.3, one layer each, from the observer up.

    Each layer gets ``profile.pressure_layer_bar`` -- the pressure tellurix
    evaluates its lines at -- its temperature, and the columns tellurix
    integrates, ``air_column_cm2 * vmr``. LBLRTM takes one pressure per layer for
    lines and continua alike, so a profile whose continua should agree too must
    carry ``mean_pressure_bar``.
    """

    altitude = _hydrostatic_altitude_edges(profile)[::-1]
    pressure_edges_hpa = np.asarray(profile.pressure_edges_bar)[::-1] * 1000.0
    # Edge temperatures enter only LBLRTM's Planck function, not transmission.
    temperature_edges = _layer_values_at_edges(profile.temperature_k)[::-1]
    pressure_hpa = np.asarray(profile.pressure_layer_bar)[::-1] * 1000.0
    temperature = np.asarray(profile.temperature_k)[::-1]
    air = np.asarray(profile.air_column_cm2)[::-1]
    present = [_LBLRTM_SPECIES.index(name) for name in profile.vmr]
    nmol = max(7, max(present) + 1)
    columns = np.zeros((len(air), nmol))
    for name, values in profile.vmr.items():
        columns[:, _LBLRTM_SPECIES.index(name)] = air * np.asarray(values)[::-1]
    # WBROAD is every molecule of air not given a profile of its own.
    broadening = air - columns.sum(axis=1)
    secant = 1.0 / np.cos(np.deg2rad(config.zenith_angle_deg))

    lines = [f"${config.description[:79]}"]
    flags = (1, 1, config.continuum_flag, 0, 1, 0, 0, 0, 0, 0)
    lines.append("".join(f"{value:5d}" for value in flags) + f"{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}")
    lines.append(
        f"{config.wavenumber_min_cm1:10.3f}{config.wavenumber_max_cm1:10.3f}"
        f"{4.0:10.3f}{0.0:10.3f}{0.04:10.3f}{36.0:10.3f}{-1.0:10.3f}{-1.0:10.3f}"
        f"{0:5d}{0.0:15.3f}{0:5d}"
    )
    lines.append(
        f"{float(temperature_edges[0]):10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}"
        f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}    s"
    )
    # Record 2.1: IFORM=1 (E15.7 amounts), NLAYRS, NMOL, SECNTO > 0 looking up.
    lines.append(f" {1:1d}{len(air):3d}{nmol:5d}{secant:10.6f}{'':20s}"
                 f"{altitude[0]:8.3f}{'':4s}{altitude[-1]:8.3f}{'':5s}{config.zenith_angle_deg:8.3f}")
    for layer in range(len(air)):
        # Record 2.1.1 in its IFORM=1 layout.
        lines.append(
            f"{pressure_hpa[layer]:15.7E}{temperature[layer]:10.4f}{0.0:10.4f}{'':3s}{0:2d}"
            f" {altitude[layer]:7.2f}{pressure_edges_hpa[layer]:8.3f}{temperature_edges[layer]:7.2f}"
            f"{altitude[layer + 1]:7.2f}{pressure_edges_hpa[layer + 1]:8.3f}"
            f"{temperature_edges[layer + 1]:7.2f}"
        )
        # Records 2.1.2 and 2.1.3: molecules 1-7 and WBROAD, then 8 onward.
        lines.append("".join(f"{value:15.7E}" for value in columns[layer, :7])
                     + f"{broadening[layer]:15.7E}")
        for start in range(7, nmol, 8):
            lines.append("".join(f"{value:15.7E}" for value in columns[layer, start:start + 8]))
    lines.extend((f"{-1.0:4.1f}", f"{-1.0:4.1f}", "%"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def run_lblrtm(
    workdir: str | Path,
    profile: AtmosphereProfile,
    config: LBLRTMRunConfig,
    executable: str | Path,
    tape3: str | Path,
    mt_ckd_data: str | Path,
) -> LBLRTMSpectrum:
    """Create an isolated LBLRTM run directory, execute it, and read TAPE12."""

    directory = Path(workdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_tape5(directory / "TAPE5", profile, config)
    for source, name in ((tape3, "TAPE3"), (mt_ckd_data, "absco-ref_wv-mt-ckd.nc")):
        destination = directory / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(Path(source).resolve())
    binary = directory / "lblrtm"
    shutil.copy2(executable, binary)
    tape12 = directory / "TAPE12"
    if tape12.exists():
        tape12.unlink()
    result = subprocess.run([str(binary)], cwd=directory, capture_output=True, text=True)
    if result.returncode != 0 or not tape12.exists():
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"LBLRTM failed with status {result.returncode}:\n{details[-4000:]}")
    return read_tape12_single_precision(tape12)


def run_lnfl(
    workdir: str | Path,
    species: tuple[str, ...],
    wavenumber_min_cm1: float,
    wavenumber_max_cm1: float,
    line_file: str | Path,
    executable: str | Path,
    line_coupling: bool = True,
) -> Path:
    """Build an LBLRTM TAPE3 with LNFL and return its path.

    ``species`` are names from the HITRAN-ordered list LBLRTM uses (record 3 of
    LNFL's TAPE5 is a positional on/off mask). ``line_coupling=False`` passes
    LNFL's NOCPL option, which drops AER's first-order coupling coefficients --
    the one way to see what LBLRTM's line mixing contributes, since LBLRTM
    applies whatever the TAPE3 carries.
    """

    unknown = set(species) - set(_LBLRTM_SPECIES)
    if unknown:
        raise ValueError(f"unknown LNFL species: {', '.join(sorted(unknown))}")
    directory = Path(workdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mask = "".join("1" if name in species else "0" for name in _LBLRTM_SPECIES).ljust(47, "0")
    (directory / "TAPE5").write_text(
        f"tellurix LNFL {' '.join(species)}{'' if line_coupling else ' (NOCPL)'}\n"
        f"{wavenumber_min_cm1:10.3f}{wavenumber_max_cm1:10.3f}\n"
        f"{mask}    {'' if line_coupling else 'NOCPL'}\n", encoding="ascii")
    for name in ("TAPE1", "TAPE3", "TAPE10"):
        target = directory / name
        if target.exists() or target.is_symlink():
            target.unlink()
    (directory / "TAPE1").symlink_to(Path(line_file).resolve())
    binary = directory / "lnfl"
    shutil.copy2(executable, binary)
    result = subprocess.run([str(binary)], cwd=directory, capture_output=True, text=True)
    tape3 = directory / "TAPE3"
    if result.returncode != 0 or not tape3.exists():
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"LNFL failed with status {result.returncode}:\n{details[-4000:]}")
    return tape3


def lblrtm_line_shape_optical_depth(
    database,
    profile: AtmosphereProfile,
    wavenumber_cm1: np.ndarray,
    pressure_shift: bool = True,
    cutoff_cm1: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical optical depth of ``database``'s lines as LBLRTM and as tellurix shape them.

    Returns ``(truncated, full)``. ``full`` is the Voigt line everywhere, as
    tellurix evaluates it. ``truncated`` is LBLRTM 12.17's with ILBLF4=1
    (``oprop.f90``, CONVF4): every line cut at ``cutoff_cm1`` less its Lorentz
    value there, L(B) for most molecules and (2 - x^2/B^2) L(B) for CO2, whose
    chi factor 12.17 overrides to 1. ``truncated - full`` is what the cutoff
    changes for lines both codes carry; ``truncated`` alone is what LBLRTM adds
    for lines only it carries. Line coupling is not included. Evaluated in
    numpy, line by line, for validation rather than fitting.
    """

    from scipy.special import voigt_profile

    from .direct import SparseCoreDirect

    nu = np.asarray(wavenumber_cm1, dtype=float)
    species = database.simple_molecule_name.upper()
    temperature = np.asarray(profile.temperature_k)
    pressure = np.asarray(profile.pressure_layer_bar)
    vmr = np.asarray(profile.vmr[species])
    # The same line parameters the fitting kernel uses, from the same class.
    calculator = SparseCoreDirect(database, np.linspace(nu[0], nu[-1], 16),
                                  pressure_shift=pressure_shift,
                                  minimum_temperature_k=float(temperature.min()),
                                  maximum_temperature_k=float(temperature.max()),
                                  maximum_pressure_bar=float(pressure.max()))
    truncated = np.zeros_like(nu)
    full = np.zeros_like(nu)
    for layer, column in enumerate(profile.air_column_cm2 * vmr):
        sigma, gamma, strength = (np.asarray(a) for a in calculator._line_parameters(
            temperature[layer], pressure[layer], pressure[layer] * vmr[layer]))
        centre = np.asarray(database.nu_lines, dtype=float)
        if pressure_shift:
            centre = centre + np.asarray(calculator._line_shift(temperature[layer], pressure[layer]))
        for start in range(0, centre.size, 500):
            part = slice(start, start + 500)
            x = nu[None, :] - centre[part, None]
            g, s = gamma[part, None], strength[part, None]
            voigt = s * voigt_profile(x, sigma[part, None], g)
            pedestal = s * g / (np.pi * (g * g + cutoff_cm1 ** 2))
            if species == "CO2":
                pedestal = pedestal * (2.0 - x * x / cutoff_cm1 ** 2)
            full += column * voigt.sum(axis=0)
            truncated += column * np.where(np.abs(x) <= cutoff_cm1, voigt - pedestal, 0.0).sum(axis=0)
    return truncated, full
