"""Fit telluric absorption and a continuum to one window of an NSO FTS spectrum.

Uses ``accuracy_mode="mt_ckd"`` with the native differentiable continuum, so
LBLRTM is not involved at fit time. ``scripts/fit_fts_window.py`` is the command
line for one window and ``scripts/fit_fts_batch.py`` walks a whole spectrum;
both run :func:`fit_window`, and the review export rebuilds a fitted model
through :func:`prepare_window`, so there is one description of the fit.

Three things differ from the Arcturus driver and all three are load-bearing.

**The resolving power is not a free choice.** These spectra hold the
interferogram truncation fixed rather than the resolving power, so the
resolution element is constant in wavenumber (0.017532 cm-1, measured; see
docs/solar_ils.md) and R rises with it -- 114,075 at 2000 cm-1 to 513,338 at
9000. The grid is sized from the window's own centre. A single atlas-wide
velocity step, which is what the Arcturus driver uses, would oversample the red
end by a factor of 4.5.

**The MOPD is one constant, not a per-window measurement.** The page-to-page
spread is 1.010, inside the FFT bin, so fitting it per window would be fitting
noise. This is the ``--sinc-resolving-power`` lesson from Arcturus, on firmer
ground.

**The zenith angle comes from the header.** The atlas fits at zero and folds
air mass into the column scale; these spectra record it, so the fit can use it.
``zenith_angle_deg=0`` restores the Arcturus configuration, which is the
diagnostic half of the slant-path test: run both ways, and the two files' column
scales must agree at the header zenith and differ by their air-mass ratio at
zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import numpy as np

from tellurix import (
    NegativeOpacityError,
    AER_MOLECULE_IDS, AERLineDatabase, ArrayOpacityBackend, BoxcarFTSInstrumentProfile,
    CoaddNode, CoaddedTelluricModel, ContinuumSum, ExoJAXOpacityBackend, LBLRTMOpticalDepthCorrection, MTCKDWaterContinuum,
    O2CollisionInducedContinuum, OrderObjective, StellarSpectrum, TelluricModel,
    TelluricParameters, chebyshev_continuum, constant_velocity_grid, fit_order, ils_fingerprint,
    line_strength_provenance, load_atmosphere_csv, prepare_stellar_source, resample_stellar_continuum,
    trim_wavenumber_grid,
)
from tellurix.download import DataPaths
from tellurix.hitran import pressure_shift_provenance

from .atlas import epoch_velocity_kms, read_arcturus_page
from .nso import (
    MEASURED_FWHM_CM1, arcturus_as_fts_spectrum, fts_spectral_order, read_solar_spectrum,
    zenith_angle_deg_for_airmass,
)

# Every molecule AER ships, from the package rather than a local copy: keeping
# a second list here is what let O3 be 'unknown' after OCS had been added, and
# what let OCS be unreachable for as long as it was. Which species are worth
# fitting is a per-window question, answered by `species` and by `at_bound`.
MOLECULE_IDS = AER_MOLECULE_IDS
# Stages are cumulative: each frees more parameters starting from the last fit.
STAGES = {
    "continuum": ("continuum_*", "log_jitter"),
    "velocity": ("continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms"),
    "columns": ("continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species_*"),
    "stellar": ("continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species_*",
                "stellar_velocity_kms", "stellar_broadening_kms"),
}


@dataclass(frozen=True)
class WindowSettings:
    """How one window is fitted. The defaults are the command line's.

    ``profile`` and ``correction`` are resolved against the ``base`` directory
    given to :func:`prepare_window` and recorded in the report as given, so a
    report written from a checkout names them the way the checkout does.
    """

    spectrum: Path
    profile: Path
    v1: float = 6000.0
    v2: float = 6030.0
    # How far outside the window a line may still contribute.
    margin_cm1: float = 25.0
    # How far outside the window the model grid extends.
    grid_margin_cm1: float = 5.0
    samples_per_resolution: float = 4.0
    # The measured sinc FWHM; sets both the grid and the instrument.
    fwhm_cm1: float = MEASURED_FWHM_CM1
    # Comma-separated molecules, or 'all'. The right set is per window: O2 has
    # no lines at 6000 cm-1 and 818 at 7874.
    species: str = "H2O,CO2,CH4"
    # 'flat', 'auto' to pick the band covering the window, or a path.
    stellar: str = "auto"
    # Zero for disc centre, where solar rotation is transverse.
    vsini_kms: float = 0.0
    macroturbulence_kms: float = 1.5
    # Feed the source as flux_total, so it carries Payne Zero's own predicted
    # continuum; implies normalize_source.
    source_continuum: bool = False
    normalize_source: bool = False
    continuum_degree: int = 3
    stages: str = "continuum,velocity,columns"
    # NAME=VALUE entries held fixed at that value.
    pin: tuple = ()
    # None takes the header's mean air mass; 0 folds it into the column scales.
    zenith_angle_deg: float | None = None
    accuracy_mode: str = "mt_ckd"
    # The LBLRTM correction template npz; required by accuracy_mode lblrtm_corrected.
    correction: Path | None = None
    gaussian_ils: bool = False
    # LBLRTM's O2 collision-induced bands (tellurix.o2_cia).
    o2_cia: bool = False
    # AER's first-order line coupling (CO2, CH4, O2), as LBLRTM applies it.
    # fit_window fits a window with coupled lines both ways and keeps the better.
    # A coupled optical depth more negative than negative_tolerance in any layer
    # cannot be precomputed and loses; smaller negatives are clipped to zero.
    line_coupling: bool = False
    # Fit a Gaussian broadening of the solar source alone (stellar_broadening_kms),
    # freed with stellar_velocity_kms. Without it the only free width is
    # lsf_sigma_kms, which the solar lines set -- Payne Zero's are narrower than
    # the Sun's -- and which then widens the telluric lines too.
    stellar_broadening: bool = False
    # Whose H2O air pressure shifts: "hitran" (DataPaths.h2o_pressure_shift), or
    # "aer", LBLRTM's own. AER gives some 0.8-1.1 um lines a shift of the wrong
    # sign, and they sit 100-300 m/s off in niratl; see tellurix.hitran.
    h2o_pressure_shift: str = "hitran"
    # Multiplies gamma_air and gamma_self of these species' lines. Kitt Peak's
    # water and O2 lines are broader than AER's widths -- ~5% in every spectrum,
    # ~11% in niratl -- with a Lorentzian shape the free Gaussian cannot take
    # (docs/solar_fit_plan.md §4r).
    pressure_width_scale: float = 1.0
    pressure_width_species: str = "H2O,O2"
    # Gamma2/Gamma0 of a quadratic speed-dependent Voigt for these species' lines;
    # 0 keeps the Voigt. Baker et al.'s isolated water lines want 0.09-0.13 at
    # AER's widths (docs/line_intensity_corrections.md); LBLRTM has none.
    speed_dependence: float = 0.0
    speed_dependence_species: str = "H2O"
    # A tellurix.LineStrengthTable (scripts/build_line_strength_table.py), relative to
    # base: per-line strength factors measured from observed spectra. None keeps
    # AER's strengths (docs/line_intensity_corrections.md).
    line_strength_table: Path | None = None
    # Fit the telluric parameters (columns, velocity, lsf_sigma_kms) on pixels the
    # solar model calls clean, then everything else with them pinned; see
    # fit_window. A solar line deeper than clean_solar_depth below the model's
    # local maximum is masked; past clean_max_masked of the window masked, the
    # all-pixel fit is kept. clean_envelope "closing" takes that maximum as a
    # morphological closing instead, which follows a broad wing's slope: against
    # the running maximum the Ca II 8542 and H-alpha wings counted as solar line
    # and refused their windows (solar_line_mask).
    telluric_from_clean_pixels: bool = False
    clean_solar_depth: float = 0.01
    clean_max_masked: float = 0.6
    clean_envelope: str = "max"
    negative_tolerance: float = 1.0e-3
    # A window whose continuum is fewer sigma above zero is refused as opaque.
    # Arcturus ab2119_ (~26 sigma, in the CO fundamental) is real data under the solar 30.
    minimum_continuum_snr: float = 30.0
    vectorize_layers: bool = True
    mixed_precision: bool = True
    precompute_opacity: bool = True
    self_broadening: str = "linear"
    # False skips the formal covariance, and with it the Hessian's compilation:
    # 6 of the 15 s a Baker et al. telluric window costs. The report then has no
    # sigma, correlation or at_bound; only a fit read for its residuals wants that.
    covariance: bool = True
    layer_chunk_size: int = 0
    # A co-add description (scripts/make_iag_coadd.py), relative to base: the
    # spectrum is a mean over observations through several atmospheres, each
    # with its own profile, air masses and telluric velocities
    # (tellurix.CoaddedTelluricModel). The order is then fitted at zenith 0,
    # since the nodes carry the geometry, and ``profile`` serves only the
    # source-alone model.
    coadd: Path | None = None
    # An Arcturus atlas page's epoch, 'summer' or 'winter': ``spectrum`` is then an
    # ``abNNNN_`` page, read through arcturus_as_fts_spectrum. None for the Sun.
    epoch: str | None = None
    # Where the source's velocity starts and what its +-5 km/s bound is centred
    # on. None: 0 for the Sun, the epoch's readme.dat value for Arcturus, whose
    # epochs sit at +14 and -26 km/s.
    stellar_velocity_kms: float | None = None

    @property
    def stellar_velocity_centre_kms(self) -> float:
        if self.stellar_velocity_kms is not None:
            return float(self.stellar_velocity_kms)
        return 0.0 if self.epoch is None else epoch_velocity_kms(self.epoch)


def read_window_spectrum(settings: WindowSettings):
    """The spectrum a window is cut from: an NSO file or page, or an Arcturus page's epoch."""

    if settings.epoch is not None:
        return arcturus_as_fts_spectrum(read_arcturus_page(settings.spectrum, settings.epoch))
    return read_solar_spectrum(settings.spectrum)


def solar_source_for(v1_cm1: float, v2_cm1: float, directory: Path) -> Path:
    """The synthesized band whose range covers this window, with margin.

    The source is generated per band because the intrinsic sampling it needs is
    set by the bluest wavenumber it contains; see generate_payne_zero_solar.py.
    ``directory`` holds the ``solar_payne_zero_*`` npz files and their json.
    """

    wanted = (1.0e7 / v2_cm1, 1.0e7 / v1_cm1)
    available = []
    for metadata in sorted(Path(directory).glob("solar_payne_zero_*.json")):
        covered = json.loads(metadata.read_text())["wavelength_nm"]
        available.append((metadata.stem, covered))
        if covered[0] <= wanted[0] and wanted[1] <= covered[1]:
            return metadata.with_suffix(".npz")
    raise ValueError(
        f"no solar source covers {wanted[0]:.1f}-{wanted[1]:.1f} nm; have "
        + ", ".join(f"{name} {lo:.0f}-{hi:.0f}" for name, (lo, hi) in available)
    )


def load_coadd(path: Path, base: Path = Path(".")):
    """The components of a co-add description: ``[(profile, nodes, label)]``, and its provenance."""

    path = Path(base) / path
    text = path.read_bytes()
    spec = json.loads(text)
    components = [(load_atmosphere_csv(Path(base) / item["profile"]),
                   [CoaddNode(node["weight"], node["zenith_angle_deg"], node["velocity_kms"])
                    for node in item["nodes"]], item.get("label", item["profile"]))
                  for item in spec["components"]]
    provenance = {"coadd": path.name, "coadd_sha256": hashlib.sha256(text).hexdigest(),
                  "coadd_components": len(components),
                  "coadd_nodes": sum(len(nodes) for _, nodes, _ in components)}
    return components, provenance


def build_bounds(species, continuum_degree, free, initial, include_stellar, pinned=(),
                 stellar_broadening=False, stellar_velocity_centre_kms=0.0):
    """Bounds that pin everything except the named free parameters."""

    def window(name, lower, upper, pinned_at):
        if name in pinned:
            return (pinned_at, pinned_at)
        if name in free or (name.startswith("continuum_") and "continuum_*" in free) or (
            name in species and "species_*" in free
        ):
            return (lower, upper)
        return (pinned_at, pinned_at)

    bounds = {}
    for name in species:
        bounds[name] = window(name, -2.0, 2.0, float(initial.log_column_scales[name]))
    bounds["velocity_kms"] = window("velocity_kms", -5.0, 5.0, float(initial.velocity_kms))
    # An FTS wavenumber scale is exactly linear, so the only physical freedom is
    # a multiplicative factor, which is already the velocity.
    bounds["wavelength_stretch"] = (0.0, 0.0)
    bounds["lsf_sigma_kms"] = window("lsf_sigma_kms", 0.05, 4.0, float(initial.lsf_sigma_kms))
    for index in range(continuum_degree + 1):
        current = float(np.asarray(initial.continuum_coeffs)[index])
        limit = 2.0 if index == 0 else 0.5
        bounds[f"continuum_{index}"] = window(f"continuum_{index}", -limit, limit, current)
    bounds["log_jitter"] = window("log_jitter", np.log(1e-6), np.log(0.1), float(initial.log_jitter))
    if include_stellar:
        # The Sun's own offset is a few hundred m/s -- gravitational redshift
        # +636, convective blueshift about -400 and depth dependent, plus
        # Earth's orbital component. A star's tens of km/s are its centre's.
        centre = float(stellar_velocity_centre_kms)
        bounds["stellar_velocity_kms"] = window(
            "stellar_velocity_kms", centre - 5.0, centre + 5.0, float(initial.stellar_velocity_kms))
        if stellar_broadening:
            # The same ceiling as lsf_sigma_kms: the model's kernel is sized
            # for max_lsf_sigma_kms, and a sigma past it would be truncated.
            bounds["stellar_broadening_kms"] = window(
                "stellar_broadening_kms", 0.05, 4.0, float(initial.stellar_broadening_kms))
    return bounds


def prepare_window(settings: WindowSettings, paths: DataPaths, *, stellar_directory: Path,
                   base: Path = Path("."), spectrum=None):
    """Everything a window needs before anything is fitted.

    Split out so that a consumer which only wants the *fitted* model back --
    the review export, say -- rebuilds it through this function rather than
    through a second description of the same grid, line selection, instrument
    and source. A decomposition shown next to the observed spectrum has to come
    from the model that produced the fit, not from one that resembles it.
    """

    started = time.time()

    # Re-reading the file per window costs 1.3 s to take a 3,166-pixel slice out
    # of 831,488 lines. A caller that walks many windows of one file passes the
    # spectrum in; it is immutable, so sharing it is safe.
    spectrum = read_window_spectrum(settings) if spectrum is None else spectrum
    window = spectrum.select(settings.v1, settings.v2)
    profile = load_atmosphere_csv(Path(base) / settings.profile)

    centre = 0.5 * (settings.v1 + settings.v2)
    # The instrument's resolution element is constant in wavenumber, so the
    # resolving power this window needs follows from its own centre.
    resolving_power = centre / settings.fwhm_cm1
    grid = constant_velocity_grid(
        1.0e7 / settings.v2, 1.0e7 / settings.v1,
        resolving_power=resolving_power,
        samples_per_resolution=settings.samples_per_resolution,
        margin_cm1=settings.margin_cm1,
    )
    grid = trim_wavenumber_grid(grid, settings.v1, settings.v2, settings.grid_margin_cm1)

    wanted = (set(MOLECULE_IDS) if settings.species == "all"
              else {name.strip().upper() for name in settings.species.split(",")})
    unknown = wanted - set(MOLECULE_IDS)
    if unknown:
        raise ValueError(f"unknown species: {', '.join(sorted(unknown))}")
    widened = {name.strip().upper() for name in settings.pressure_width_species.split(",")}
    strength_table = None if settings.line_strength_table is None else Path(base) / settings.line_strength_table
    databases, skipped = {}, []
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        if species not in wanted:
            continue
        try:
            databases[species] = AERLineDatabase(
                paths.line_file(species, molecule_id), species, (settings.v1, settings.v2),
                margin_cm1=settings.margin_cm1,
                line_coupling=paths.line_coupling if settings.line_coupling else None,
                pressure_shift_table=paths.pressure_shift_table(species, settings.h2o_pressure_shift),
                pressure_width_scale=(settings.pressure_width_scale if species in widened else 1.0),
                line_strength_table=strength_table)
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
            skipped.append(species)
    if not databases:
        raise ValueError(
            f"none of {sorted(wanted)} has lines in {settings.v1}-{settings.v2} cm-1")
    lines_per_species = {name: int(db.nu_lines.size) for name, db in databases.items()}

    speed_dependent = ({name.strip().upper() for name in settings.speed_dependence_species.split(",")}
              & set(databases) if settings.speed_dependence else set())
    # The SDV's core list reaches 36 Gamma2 from each line, so it is bounded by
    # the self pressure too: the largest the column bounds allow (log scale <= 2).
    self_pressure_bar = max((float(np.max(np.asarray(profile.vmr[name]) * profile.pressure_layer_bar))
                             for name in speed_dependent if name in profile.vmr), default=0.0)
    coadd, coadd_provenance = ((None, {}) if settings.coadd is None
                               else load_coadd(settings.coadd, base))
    # One opacity serves every component: its bounds only decide where the
    # sparse kernel is valid, so they must cover every profile it will see.
    profiles = [profile, *(component for component, _, _ in coadd or ())]
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(min(float(np.min(p.temperature_k)) for p in profiles),
                             max(float(np.max(p.temperature_k)) for p in profiles)),
        maximum_pressure_bar=max(float(np.max(p.pressure_layer_bar)) for p in profiles),
        vectorize_layers=settings.vectorize_layers,
        mixed_precision=settings.mixed_precision,
        pressure_shift=True,
        layer_chunk_size=settings.layer_chunk_size or None,
        line_coupling=settings.line_coupling,
        speed_dependence={name: settings.speed_dependence for name in speed_dependent},
        maximum_self_pressure_bar=float(np.exp(2.0)) * self_pressure_bar,
    )
    # The two modes are exclusive by construction: lblrtm_corrected carries the
    # continua inside its template and TelluricModel rejects a separate one.
    if settings.accuracy_mode == "lblrtm_corrected":
        if settings.correction is None:
            raise ValueError("accuracy_mode lblrtm_corrected needs a correction template")
        correction = LBLRTMOpticalDepthCorrection.load(Path(base) / settings.correction)
        continuum = None
    elif settings.accuracy_mode == "fast":
        correction, continuum = None, None
    else:
        correction = None
        continuum = MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, grid)
    if settings.o2_cia:
        # LBLRTM's O2 collision-induced bands (tellurix.o2_cia). Without them the
        # A-band's O2 column absorbs a structured continuum it cannot represent.
        o2 = O2CollisionInducedContinuum(grid)
        if o2.terms:
            continuum = o2 if continuum is None else ContinuumSum((continuum, o2))

    mopd_cm = None if settings.gaussian_ils else 1.20671 / (2.0 * settings.fwhm_cm1)
    instrument = None if settings.gaussian_ils else BoxcarFTSInstrumentProfile(
        mopd_cm=mopd_cm, wavenumber_center_cm1=centre, max_residual_sigma_kms=4.0)
    def single(atmosphere):
        return TelluricModel(
            atmosphere, grid, opacity, continuum=continuum, correction=correction,
            accuracy_mode=settings.accuracy_mode,
            max_lsf_sigma_kms=4.0,
            # The FTS point-samples; it does not integrate over a pixel the way a
            # grating spectrograph's detector does.
            pixel_integration="point",
            instrument=instrument,
        )

    model = (single(profile) if coadd is None
             else CoaddedTelluricModel([(single(component), nodes) for component, nodes, _ in coadd]))

    if settings.stellar == "flat":
        source_path, stellar = None, StellarSpectrum.flat(grid)
        source = prepare_stellar_source(stellar, model)
    else:
        source_path = (solar_source_for(settings.v1, settings.v2, stellar_directory)
                       if settings.stellar == "auto" else Path(settings.stellar))
        stellar = StellarSpectrum.from_npz(source_path)
        normalize = settings.normalize_source
        if settings.source_continuum:
            if stellar.continuum is None:
                raise ValueError(f"{source_path} carries no flux_continuum to put back")
            # flux * continuum is flux_total, reconstructed rather than re-read
            # so it inherits whichever orientation from_npz verified.
            stellar = StellarSpectrum(
                stellar.wavenumber_cm1, stellar.flux * stellar.continuum,
                {**stellar.meta, "source_flux": "flux_total"}, continuum=stellar.continuum)
            normalize = True
        source = prepare_stellar_source(
            stellar, model, vsini_kms=settings.vsini_kms,
            macroturbulence_kms=settings.macroturbulence_kms, normalize=normalize)

    zenith = settings.zenith_angle_deg
    if coadd is not None:
        if zenith not in (None, 0.0):
            raise ValueError("a co-add's nodes carry its geometry; leave zenith_angle_deg unset or 0")
        zenith = 0.0
    elif zenith is None:
        if spectrum.airmass_mean is None:
            raise ValueError(f"{Path(settings.spectrum).name} has no air mass; "
                             "give zenith_angle_deg")
        zenith = zenith_angle_deg_for_airmass(spectrum.airmass_mean)
    order = fts_spectral_order(window, zenith_angle_deg=zenith, source_flux_model_grid=source,
                               minimum_continuum_snr=settings.minimum_continuum_snr)

    fit_model = (model.precompute_opacity(self_broadening=settings.self_broadening,
                                          negative_tolerance=settings.negative_tolerance)
                 if settings.precompute_opacity else model)

    parameters = TelluricParameters(
        log_column_scales={name: 0.0 for name in model.species},
        velocity_kms=0.0,
        wavelength_stretch=0.0,
        lsf_sigma_kms=0.5,
        continuum_coeffs=np.concatenate(
            [[float(np.log(np.median(np.asarray(order.flux)[np.asarray(order.mask)])))],
             np.zeros(settings.continuum_degree)]),
        log_jitter=float(np.log(np.median(order.uncertainty))),
        stellar_velocity_kms=settings.stellar_velocity_centre_kms,
        # Away from zero, where a Gaussian's width has no gradient.
        stellar_broadening_kms=0.5 if settings.stellar_broadening and source is not None else 0.0,
    )

    return SimpleNamespace(
        spectrum=spectrum, window=window, profile=profile, grid=grid, centre=centre,
        resolving_power=resolving_power, databases=databases, skipped=skipped,
        lines_per_species=lines_per_species, opacity=opacity, correction=correction,
        continuum=continuum, mopd_cm=mopd_cm, instrument=instrument, model=model,
        stellar=stellar, source=source, source_path=source_path, zenith=zenith,
        order=order, fit_model=fit_model, parameters=parameters, started=started,
        strength_table=strength_table, coadd_provenance=coadd_provenance,
    )


def solar_line_mask(arrays, depth: float = 0.01, width_cm1: float = 3.0,
                    envelope: str = "max") -> np.ndarray:
    """Pixels where a fitted window's solar model dips more than ``depth``.

    The depth is taken below the model's own running maximum over ``width_cm1``,
    not below the fitted continuum: deep in a water band that continuum does
    not sit on the solar model's top, and measured against it 90% of a page
    looked like solar line (ph11675). On a broad wing the running maximum sits
    up the slope, half a width away, so a smooth wing reads as line: 66% of
    Wallace 2011's 11680 window (Ca II 8542) and 83% of 15220 (H-alpha).
    ``envelope="closing"`` follows the slope instead -- the running minimum of
    that maximum, which fills only dips narrower than ``width_cm1`` -- and
    masks 19% and 21%. Returned in the arrays' pixel order.
    """

    from scipy.ndimage import grey_closing, maximum_filter

    if envelope not in ("max", "closing"):
        raise ValueError(f"unknown solar envelope {envelope!r}")

    nu = np.asarray(arrays["wavenumber_cm1"])
    solar = np.asarray(arrays["stellar_only_pixels"]) / np.asarray(arrays["continuum"])
    order = np.argsort(nu)
    size = max(int(round(width_cm1 / float(np.median(np.diff(nu[order]))))), 1)
    top = np.empty_like(solar)
    if envelope == "closing":
        top[order] = grey_closing(solar[order], size=size, mode="nearest")
    else:
        top[order] = maximum_filter(solar[order], size, mode="nearest")
    return solar < (1.0 - depth) * top


def fit_window(settings: WindowSettings, paths: DataPaths, *, stellar_directory: Path,
               base: Path = Path("."), spectrum=None):
    """Fit one window and return its report and its diagnostic arrays.

    With ``telluric_from_clean_pixels`` the window is fitted three times. The
    first fit, on every pixel, supplies a solar model; the second sees only the
    pixels that model calls free of solar lines (:func:`solar_line_mask`) and
    sets the columns, ``velocity_kms`` and ``lsf_sigma_kms``; the third fits
    every pixel again with those pinned, freeing the continuum and the solar
    parameters. A wrong solar line otherwise sets the water column: on niratl
    ph09250 one blend at the page edge took it from 0.13 to 0.21 and
    over-corrected every water line on the page by up to 7%. Across niratl it
    put the corrected spectra level with or ahead of the authors' on 48 of 71
    scorable pages, against 24 (docs/solar_fit_plan.md §4r). The report's
    ``physics.telluric_from_clean_pixels`` says what was masked and pinned, or
    why the all-pixel fit was kept; the pinned parameters' ``sigma`` is the
    second fit's.

    With ``line_coupling``, a window that carries coupled lines is fitted twice,
    with coupling and without, and the fit with the lower residual is returned.
    Coupling is right at the CO2 Q-branches and wrong in the 4.3 um band wing,
    where tellurix has no CO2 continuum and the result depends on the line
    margin (``docs/arcturus_line_coupling.md``); no physical test separates the
    two in advance, so the observations decide, window by window. The report's
    ``physics`` records what was used (``line_coupling``) and both residuals
    (``line_coupling_trial``). A coupled opacity too negative to precompute
    (:class:`tellurix.NegativeOpacityError`) counts as a loss, with the reason in
    ``line_coupling_refused``.
    """

    if settings.telluric_from_clean_pixels:
        return _fit_from_clean_pixels(settings, paths, stellar_directory=stellar_directory,
                                      base=base, spectrum=spectrum)
    return _fit_with_coupling(settings, paths, stellar_directory=stellar_directory,
                              base=base, spectrum=spectrum)


def _fit_from_clean_pixels(settings: WindowSettings, paths: DataPaths, *, stellar_directory: Path,
                           base: Path, spectrum):
    """:func:`fit_window`'s three fits; see its docstring."""

    def run(chosen, data):
        return _fit_with_coupling(chosen, paths, stellar_directory=stellar_directory,
                                  base=base, spectrum=data)

    spectrum = read_window_spectrum(settings) if spectrum is None else spectrum
    plain = replace(settings, telluric_from_clean_pixels=False)
    report, arrays = run(plain, spectrum)
    solar_lines = solar_line_mask(arrays, settings.clean_solar_depth, envelope=settings.clean_envelope)
    note = {"clean_solar_depth": settings.clean_solar_depth, "clean_envelope": settings.clean_envelope,
            "masked_fraction": float(solar_lines.mean()), "used": False}
    if note["masked_fraction"] > settings.clean_max_masked:
        note["reason"] = f"more than {settings.clean_max_masked:g} of the window is solar line"
        report["physics"]["telluric_from_clean_pixels"] = note
        return report, arrays

    nu = np.asarray(arrays["wavenumber_cm1"])
    order = np.argsort(nu)
    hidden = np.interp(spectrum.wavenumber_vacuum_cm1, nu[order],
                       solar_lines[order].astype(float), left=0.0, right=0.0) > 0.0
    try:
        clean, clean_arrays = run(plain, replace(spectrum, flux=np.where(hidden, np.nan, spectrum.flux)))
    except ValueError as exc:
        # Too little left to set a continuum level, typically.
        note["reason"] = f"the clean-pixel fit failed: {exc}"
        report["physics"]["telluric_from_clean_pixels"] = note
        return report, arrays

    already = {entry.partition("=")[0].strip() for entry in settings.pin}
    telluric = {name: clean["parameters"][name]
                for name in (*clean["physics"]["species"], "velocity_kms", "lsf_sigma_kms")
                if name not in already}
    # The final fit uses the coupling the clean fit chose, so the pinned
    # columns describe the same opacity they are applied to.
    coupled = clean["physics"].get("line_coupling_trial", {}).get("chosen", "coupled") == "coupled"
    final = replace(plain, pin=(*settings.pin, *(f"{k}={v!r}" for k, v in telluric.items())),
                    line_coupling=plain.line_coupling and coupled)
    report, arrays = run(final, spectrum)
    names = list(np.asarray(arrays["parameter_names"]))
    sigma = np.array(arrays["sigma"], copy=True)
    for name in telluric:
        if name in names:
            sigma[names.index(name)] = clean_arrays["sigma"][list(clean_arrays["parameter_names"]).index(name)]
    arrays["sigma"] = sigma
    note.update(used=True, pinned=telluric,
                clean_fit_rms=clean["residuals"]["rms"],
                clean_fit_at_bound=clean["at_bound"])
    report["physics"]["telluric_from_clean_pixels"] = note
    return report, arrays


def _fit_with_coupling(settings: WindowSettings, paths: DataPaths, *, stellar_directory: Path,
                       base: Path, spectrum):
    """One fit, or with ``line_coupling`` the better of two; see :func:`fit_window`."""

    def run(chosen):
        return _fit_window(chosen, paths, stellar_directory=stellar_directory,
                           base=base, spectrum=spectrum)

    if not settings.line_coupling:
        return run(settings)
    try:
        coupled = run(settings)
    except NegativeOpacityError as exc:
        report, arrays = run(replace(settings, line_coupling=False))
        report["physics"]["line_coupling_refused"] = {
            "species": exc.species, "min_layer_optical_depth": exc.minimum,
            "tolerance": exc.tolerance}
        report["physics"]["line_coupling_trial"] = {
            "rms_coupled": None, "rms_uncoupled": report["residuals"]["rms"],
            "chosen": "uncoupled"}
        return report, arrays
    if not coupled[0]["physics"]["coupled_species"]:
        # No coupled line reaches this window: the two fits would be identical.
        return coupled
    uncoupled = run(replace(settings, line_coupling=False))
    rms = {"coupled": coupled[0]["residuals"]["rms"],
           "uncoupled": uncoupled[0]["residuals"]["rms"]}
    chosen = "coupled" if rms["coupled"] < rms["uncoupled"] else "uncoupled"
    report, arrays = coupled if chosen == "coupled" else uncoupled
    report["physics"]["line_coupling_trial"] = {
        "rms_coupled": rms["coupled"], "rms_uncoupled": rms["uncoupled"], "chosen": chosen}
    return report, arrays


def _fit_window(settings: WindowSettings, paths: DataPaths, *, stellar_directory: Path,
                base: Path = Path("."), spectrum=None):
    """Fit one window as configured; :func:`fit_window` adds the coupling fallback.

    The single-window command and the batch driver both run this function
    rather than two descriptions of the same fit, because a batch product
    assembled by a second implementation is not the thing the recorded
    single-window fits validated.
    """

    prepared = prepare_window(settings, paths, stellar_directory=stellar_directory,
                              base=base, spectrum=spectrum)
    spectrum = prepared.spectrum
    window = prepared.window
    profile = prepared.profile
    grid = prepared.grid
    resolving_power = prepared.resolving_power
    lines_per_species = prepared.lines_per_species
    skipped = prepared.skipped
    correction = prepared.correction
    mopd_cm = prepared.mopd_cm
    instrument = prepared.instrument
    model = prepared.model
    stellar = prepared.stellar
    source = prepared.source
    source_path = prepared.source_path
    order = prepared.order
    fit_model = prepared.fit_model
    parameters = prepared.parameters
    started = prepared.started

    broadening = settings.stellar_broadening and order.source_flux_model_grid is not None
    pinned = {}
    for entry in settings.pin:
        name, _, value = entry.partition("=")
        try:
            pinned[name.strip()] = float(value)
        except ValueError:
            raise ValueError(f"pin expects NAME=VALUE, got {entry!r}")
    allowed = set(model.species) | {
        "velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms", "log_jitter",
        *(("stellar_broadening_kms",) if broadening else ()),
        *(f"continuum_{i}" for i in range(settings.continuum_degree + 1))}
    if set(pinned) - allowed:
        raise ValueError(f"cannot pin: {', '.join(sorted(set(pinned) - allowed))}")
    if pinned:
        scales = dict(parameters.log_column_scales)
        scales.update({k: v for k, v in pinned.items() if k in model.species})
        parameters = parameters._replace(log_column_scales=scales, **{
            k: v for k, v in pinned.items()
            if k in ("velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms", "log_jitter",
                     "stellar_broadening_kms")})
        print(f"pinned: {', '.join(f'{k}={v:g}' for k, v in sorted(pinned.items()))}")

    stage_reports = []
    shared = OrderObjective(fit_model, order, settings.continuum_degree + 1,
                            stellar_broadening=broadening)
    for stage in settings.stages.split(","):
        stage = stage.strip()
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}; choose from {', '.join(STAGES)}")
        bounds = build_bounds(model.species, settings.continuum_degree, STAGES[stage], parameters,
                              include_stellar=order.source_flux_model_grid is not None,
                              pinned=pinned, stellar_broadening=broadening,
                              stellar_velocity_centre_kms=settings.stellar_velocity_centre_kms)
        stage_started = time.time()
        result = fit_order(fit_model, order, parameters, bounds, objective=shared,
                           covariance=settings.covariance)
        parameters = result.parameters
        stage_reports.append({
            "stage": stage,
            "free": [name for name, (lo, hi) in bounds.items() if lo < hi],
            "objective": result.objective,
            "success": bool(result.success),
            "message": result.message,
            "iterations": result.iterations,
            "seconds": round(time.time() - stage_started, 2),
        })
        print(f"[{stage}] objective={result.objective:.6g} success={result.success} "
              f"iterations={result.iterations} ({stage_reports[-1]['seconds']} s)")

    final_bounds = build_bounds(
        model.species, settings.continuum_degree, STAGES[settings.stages.split(",")[-1].strip()],
        parameters, include_stellar=order.source_flux_model_grid is not None, pinned=pinned,
        stellar_broadening=broadening,
        stellar_velocity_centre_kms=settings.stellar_velocity_centre_kms)
    packed = {
        **{name: float(v) for name, v in result.parameters.log_column_scales.items()},
        "velocity_kms": float(result.parameters.velocity_kms),
        "stellar_velocity_kms": float(result.parameters.stellar_velocity_kms),
        "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
        "log_jitter": float(result.parameters.log_jitter),
        **({"stellar_broadening_kms": float(result.parameters.stellar_broadening_kms)}
           if broadening else {}),
        **{f"continuum_{i}": float(v)
           for i, v in enumerate(np.asarray(result.parameters.continuum_coeffs))},
    }
    at_bound = sorted(
        name for name, value in packed.items()
        if name in final_bounds and final_bounds[name][0] < final_bounds[name][1]
        and min(abs(value - final_bounds[name][0]), abs(value - final_bounds[name][1])) < 1e-6)
    if at_bound:
        print(f"WARNING: parameters resting on a bound: {', '.join(at_bound)}")

    # The same forward model with the atmosphere removed: the Sun through the
    # same instrument, sampled the same way. Correcting by the model ratio needs
    # it, and that is the only way to avoid dividing by a convolved
    # transmission, which does not recover the source.
    star_only_model = TelluricModel(
        profile, grid,
        ArrayOpacityBackend({name: np.zeros((len(profile.temperature_k), grid.size))
                             for name in model.species}),
        accuracy_mode="fast", max_lsf_sigma_kms=4.0,
        pixel_integration=model.pixel_integration, instrument=instrument)
    stellar_only_pixels = np.asarray(star_only_model.predict(order, result.parameters))

    # Refreezing at the fitted parameters is exact there -- the expansion's
    # offset from its own reference is zero -- and it reuses the compilation.
    exact_model = model.precompute_opacity(result.parameters,
                                           negative_tolerance=settings.negative_tolerance)
    exact_flux = np.asarray(exact_model.predict(order, result.parameters))
    exact_transmission = np.asarray(
        exact_model.transmission(result.parameters, order.zenith_angle_deg))
    continuum_pixels = np.asarray(chebyshev_continuum(
        result.parameters.continuum_coeffs,
        np.linspace(-1.0, 1.0, len(order.wavelength_vacuum_nm))))
    residual = np.asarray(order.flux) - exact_flux
    mask = np.asarray(order.mask)
    scaled = residual[mask] / np.asarray(order.uncertainty)[mask]
    transmission_pixels = np.interp(
        np.asarray(order.wavelength_vacuum_nm),
        (1.0e7 / np.asarray(model.wavenumber_cm1))[::-1], exact_transmission[::-1])

    # Each absorber's own transmission, taken here rather than rebuilt later.
    # Anything downstream that wants the split -- the review page does -- would
    # otherwise reconstruct this exact model from scratch: same grid, same line
    # lists, same precomputed opacity, to call one more method on a model that
    # existed here and was discarded. Optical depth is additive, so these
    # multiply back to `exact_transmission` (checked in packages/tellurix/tests/test_model.py),
    # and taking them from the fitted model is what makes them the fit's own
    # numbers rather than a later approximation of them.
    species_on_grid = {name: np.asarray(values) for name, values
                       in exact_model.species_transmission(
                           result.parameters, order.zenith_angle_deg).items()}
    # Optical depth is additive, so on the model grid the pieces multiply back
    # to the whole exactly. Check it here, where that is true. Once each piece
    # is interpolated to the pixels separately the identity no longer holds --
    # interpolation does not commute with multiplication -- so a check made
    # after interpolation would be testing the interpolation, not the split.
    _product = np.ones_like(exact_transmission)
    for _values in species_on_grid.values():
        _product = _product * _values
    species_split_drift = float(np.max(np.abs(_product - exact_transmission)))
    species_transmission = {
        name: np.interp(np.asarray(order.wavelength_vacuum_nm),
                        (1.0e7 / np.asarray(model.wavenumber_cm1))[::-1], values[::-1])
        for name, values in species_on_grid.items()}

    spectrum_path = Path(settings.spectrum)
    report = {
        "spectrum": {
            "path": str(settings.spectrum), "name": spectrum_path.name, "sha256": spectrum.sha256,
            "source_name": spectrum.source_name, "epoch": settings.epoch,
            "observed_utc": (None if spectrum.observed_utc_mid is None
                             else spectrum.observed_utc_mid.isoformat()),
            "airmass_start": spectrum.airmass_start, "airmass_stop": spectrum.airmass_stop,
            "airmass_mean": spectrum.airmass_mean,
            "window_cm1": [settings.v1, settings.v2],
            "pixels": int(mask.size), "masked": int((~mask).sum()),
            "spacing_cm1": window.spacing_cm1,
            "grid_residual_cm1": window.grid_residual_cm1,
            "uncertainty": float(order.uncertainty[0]),
            "stated_resolution_cm1": spectrum.stated_resolution_cm1,
        },
        "grid": {
            "resolving_power": resolving_power,
            "samples_per_resolution": settings.samples_per_resolution,
            "margin_cm1": settings.margin_cm1, "grid_margin_cm1": settings.grid_margin_cm1,
            "points": int(grid.size), "velocity_step_kms": model.velocity_step_kms,
        },
        "physics": {
            "accuracy_mode": model.accuracy_mode,
            "continuum": ("LBLRTM correction template" if correction is not None
                          else "native MT_CKD 4.3"),
            "correction": None if settings.correction is None else str(settings.correction),
            "pressure_shift": True,
            **pressure_shift_provenance(
                settings.h2o_pressure_shift,
                getattr(prepared.databases.get("H2O"), "pressure_shift_table", None)),
            "line_coupling": settings.line_coupling,
            # Ratio Gamma2/Gamma0 per species; absent in records made before it.
            "speed_dependence": {name: settings.speed_dependence
                                 for name, calculator in sorted(prepared.opacity.calculators.items())
                                 if getattr(calculator, "speed_dependence", None) is not None},
            **line_strength_provenance(prepared.strength_table),
            "strengths_scaled": {name: db.strengths_scaled for name, db in prepared.databases.items()},
            "pressure_width_scale": settings.pressure_width_scale,
            "pressure_width_species": sorted(name for name, db in prepared.databases.items()
                                             if db.pressure_width_scale != 1.0),
            "stellar_broadening": broadening,
            # Which species' lines the coupling actually reached, if any.
            "coupled_species": (sorted(name for name, db in prepared.databases.items()
                                       if getattr(db, "has_line_coupling", False))
                                if settings.line_coupling else []),
            "mixed_precision": settings.mixed_precision,
            "precomputed_opacity": (settings.self_broadening if settings.precompute_opacity
                                    else None),
            "pixel_integration": model.pixel_integration,
            "instrument": "gaussian" if instrument is None else "boxcar FTS sinc",
            "fwhm_cm1": settings.fwhm_cm1,
            "mopd_cm": mopd_cm,
            "instrument_resolving_power": None if instrument is None else instrument.resolving_power,
            "profile": str(settings.profile),
            "species": list(model.species),
            "lines_per_species": lines_per_species,
            "species_without_lines": skipped,
            "zenith_angle_deg": order.zenith_angle_deg,
            "airmass_used": float(1.0 / np.cos(np.radians(order.zenith_angle_deg))),
            **prepared.coadd_provenance,
        },
        "stellar": {
            "source": "flat" if source_path is None else str(source_path),
            "vsini_kms": settings.vsini_kms, "macroturbulence_kms": settings.macroturbulence_kms,
            "normalized": settings.normalize_source or settings.source_continuum,
            "carries_stellar_continuum": settings.source_continuum,
            **({} if settings.epoch is not None else {"emergent_quantity_note": (
                "Payne Zero returns Eddington flux and has no mu option; this is a disc-centre "
                "observation, so the source mode is mismatched. See docs/solar_fit_plan.md.")}),
        },
        "pinned": pinned,
        "stages": stage_reports,
        "at_bound": at_bound,
        "parameters": {
            **{name: float(v) for name, v in result.parameters.log_column_scales.items()},
            "velocity_kms": float(result.parameters.velocity_kms),
            "stellar_velocity_kms": float(result.parameters.stellar_velocity_kms),
            "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
            **({"stellar_broadening_kms": float(result.parameters.stellar_broadening_kms)}
               if broadening else {}),
            "continuum_coeffs": [float(v) for v in np.asarray(result.parameters.continuum_coeffs)],
            "log_jitter": float(result.parameters.log_jitter),
        },
        "residuals": {
            "reduced_chi2": float(np.sum(scaled**2) / scaled.size),
            "rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "percentile_99_absolute": float(np.percentile(np.abs(residual[mask]), 99.0)),
            "jitter_over_uncertainty": float(
                np.exp(result.parameters.log_jitter) / np.median(order.uncertainty)),
        },
        "median_transmission": float(np.median(transmission_pixels[mask])),
        # How far the per-species split sits from the total, on the model grid.
        "species_split_drift": species_split_drift,
        "condition_number": float(result.condition_number or 0.0),
        "all_stages_converged": all(s["success"] for s in stage_reports),
        "runtime_seconds": round(time.time() - started, 1),
        "platform": str(jax.devices()[0]),
        "jax": jax.__version__,
    }

    _stellar_continuum = (None if source_path is None else
                          resample_stellar_continuum(stellar, window.wavenumber_vacuum_cm1[::-1]))
    names = list(shared.codec.names)
    deviation = (np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))
                 if result.covariance is not None else np.zeros(len(names)))
    correlation = (result.correlation if result.correlation is not None
                   else np.zeros((len(names), len(names))))
    ils_velocity, ils_profile = ils_fingerprint(
        instrument, float(result.parameters.lsf_sigma_kms), model.velocity_step_kms)
    arrays = dict(
        # The record's per-parameter block. Built here, in the fit's own
        # parameter order, because that order depends on which species this
        # window has -- and a batch that reconstructed it from the outside
        # would be free to get it wrong.
        parameter_names=np.asarray(names), sigma=deviation, correlation=correlation,
        ils_velocity_kms=ils_velocity, ils_profile=ils_profile,
        wavelength_vacuum_nm=np.asarray(order.wavelength_vacuum_nm),
        wavenumber_cm1=window.wavenumber_vacuum_cm1[::-1],
        flux=np.asarray(order.flux), uncertainty=np.asarray(order.uncertainty), mask=mask,
        # SpectralOrder.flux carries a placeholder of 1.0 at masked pixels,
        # because it must be finite and positive everywhere. That placeholder
        # sits in the core of every saturated line and will look like a feature
        # if it is plotted, so keep the untouched column alongside it.
        observed_raw=window.flux[::-1],
        model_flux=exact_flux, residual=residual, continuum=continuum_pixels,
        transmission_pixels=transmission_pixels,
        grid_wavenumber_cm1=np.asarray(model.wavenumber_cm1),
        grid_transmission=exact_transmission,
        stellar_source=np.asarray(source),
        stellar_only_pixels=stellar_only_pixels,
        **({} if _stellar_continuum is None else {"stellar_continuum": _stellar_continuum}),
        **{"species_transmission_" + name: values
           for name, values in species_transmission.items()},
    )
    return report, arrays


def arrays_in_page_units(arrays) -> dict:
    """:func:`fit_window`'s pixel arrays, ascending in wavenumber and in the page's own units.

    The fit sees the window over its 99th percentile (``flux``) and saves the page
    as it came (``observed_raw``), ascending in wavelength. A consumer that wants the
    page's units -- the Arcturus spectra file and review page, whose columns were
    always the atlas's -- gets the model, star, continuum, residual and uncertainty
    multiplied back by that level, so ``corrected`` is ``observed / effective
    transmission`` in either. Transmissions are dimensionless and only reordered.
    """

    order = np.argsort(np.asarray(arrays["wavenumber_cm1"]))
    take = lambda name: np.asarray(arrays[name])[order]
    mask = take("mask").astype(bool)
    raw = take("observed_raw")
    level = float(np.median(raw[mask] / take("flux")[mask]))
    out = {
        "level": level,
        "wavenumber_cm1": take("wavenumber_cm1"),
        "mask": mask,
        "observed": np.where(mask, raw, np.nan),
        "model_flux": take("model_flux") * level,
        "stellar_only": take("stellar_only_pixels") * level,
        "continuum": take("continuum") * level,
        "residual": take("residual") * level,
        "uncertainty": take("uncertainty") * level,
        "transmission": take("transmission_pixels"),
    }
    out["corrected"] = raw / np.maximum(out["model_flux"], 1e-6) * out["stellar_only"]
    for name in arrays:
        if name.startswith("species_transmission_") or name in ("stellar_continuum", "reliable"):
            out[name] = take(name)
    return out

