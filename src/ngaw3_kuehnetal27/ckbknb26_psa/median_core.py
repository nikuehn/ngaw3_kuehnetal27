"""
Median PSA computation for the NGA-West3 CKBKNB26 GMM, ported from
`PSA_NGAW3_GMM.R` (function `NGAW3medianPSA`) to Python/JAX.

Scope of this port
-------------------
Only the **median** ground motion is implemented (matching the "mainly
interested in the median predictions" use case) -- the standard-deviation
machinery (Tau/Phi/GenSigmoid) and the lowess spectral smoothing in the R
function are not ported.

**Nonlinear soil amplification** is implemented for the Ilhan et al.
(2026) model (`nonlinear_site.py`, ported from `ILHAN26.R`) -- pass
`nl_coef`/`coef_pga` to `calculate_ln_median_psa_campbelletal27` to enable
it (see that function's docstring). With `nl_coef=None` (the default) the
site term is linear-only (`fsite = flin`), i.e. what the R code computes
when `NlnMod == FALSE`.

Two simplifications relative to the R code, both noted at the point they
apply below:

1. **PGA row.** The R code always treats `Coeffs[1, ]` (period 0.01 s) as
   a separate "PGA" coefficient set for the nonlinear model, independent
   of whatever period subset is being predicted. Here that's the
   `coef_pga` argument -- load it explicitly with
   `load_coefficients(csv_path, periods=[0.01])` rather than relying on
   0.01 s being present in `coef`.

2. **Rx/Rjb back-fill.** The R code back-fills missing `Rx`/`Rjb` inside
   a single vectorized `ifelse(is.na(Rx) | is.na(Rjb), <block>, NA)`,
   which (because `ifelse`'s true/false branches are each evaluated as
   whole vectors) would zero out valid `Rx`/`Rjb` entries elsewhere in
   the batch whenever *any* entry in the batch is missing. That's almost
   certainly not intended. Here each scenario is back-filled independently
   based on its own `Rx`/`Rjb` being NaN, which is the sane reading of the
   R geometry and is what you'd want when running many scenarios (e.g.
   from `scenario_prediction.py`) at once.

Coefficients are read once via `load_coefficients` into a `Coefficients`
NamedTuple of JAX arrays, one entry per spectral period -- treat it as a
static pytree to pass into `calculate_ln_median_psa_campbelletal27`/
`calculate_median_psa_campbelletal27`
(both are vmap/jit-friendly: no data-dependent Python control flow, only
`jnp.where`).

Precision note: to match the R (double-precision) implementation closely,
enable x64 once in your own entry point, before importing anything that
calls into this module:

    from jax import config
    config.update("jax_enable_x64", True)

Without that, JAX's default float32 will introduce small (~1e-6 relative)
numerical differences from R -- fine for most uses, but worth knowing about.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .nonlinear_site import Ilhan26Coefficients, compute_ilhan26_ir, compute_ln_nonlinearity_ilhan26

__all__ = [
    "Coefficients",
    "load_coefficients",
    "calculate_ln_median_psa_campbelletal27",
    "calculate_median_psa_campbelletal27",
]

# Geometric-spreading period breakpoints (R: per1.T / per2.T) -- fixed
# constants in the R code, not fitted per period.
_PER1 = 0.75
_PER2 = 2.00

_COEF_FIELDS = [
    "Per",
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
    "c10", "c11", "c12", "c13", "c14", "c15", "c16",
    "Zseis", "Rcross",
    "v0", "v1", "v2", "v3",
    "m1", "m2", "m3",
    "d1", "d2",
    "z1", "z2", "z3", "z4", "z5", "z6",
    "h1", "h2", "h3", "h4", "h5", "h6",
]


class Coefficients(NamedTuple):
    """One JAX array per field, each shape ``(n_periods,)``."""
    Per: jnp.ndarray
    c0: jnp.ndarray
    c1: jnp.ndarray
    c2: jnp.ndarray
    c3: jnp.ndarray
    c4: jnp.ndarray
    c5: jnp.ndarray
    c6: jnp.ndarray
    c7: jnp.ndarray
    c8: jnp.ndarray
    c9: jnp.ndarray
    c10: jnp.ndarray
    c11: jnp.ndarray
    c12: jnp.ndarray
    c13: jnp.ndarray
    c14: jnp.ndarray
    c15: jnp.ndarray
    c16: jnp.ndarray
    Zseis: jnp.ndarray
    Rcross: jnp.ndarray
    v0: jnp.ndarray
    v1: jnp.ndarray
    v2: jnp.ndarray
    v3: jnp.ndarray
    m1: jnp.ndarray
    m2: jnp.ndarray
    m3: jnp.ndarray
    d1: jnp.ndarray
    d2: jnp.ndarray
    z1: jnp.ndarray
    z2: jnp.ndarray
    z3: jnp.ndarray
    z4: jnp.ndarray
    z5: jnp.ndarray
    z6: jnp.ndarray
    h1: jnp.ndarray
    h2: jnp.ndarray
    h3: jnp.ndarray
    h4: jnp.ndarray
    h5: jnp.ndarray
    h6: jnp.ndarray


def _select_periods(df: pd.DataFrame, periods: Sequence[float], rtol: float = 1e-6, atol: float = 1e-9) -> pd.DataFrame:
    """
    Select the rows of `df` (must have a `Per` column) matching `periods`,
    tolerant of small floating-point noise -- in particular, a `periods`
    list sourced from a `Coefficients.Per` JAX array will have been
    silently rounded to float32 (~1e-7 relative error) unless the caller
    has enabled `jax_enable_x64`, which an exact-equality match (e.g.
    pandas `.isin`) would then reject even though it's really the same
    period (e.g. 0.01 read back as 0.009999999776482582). Keeps the
    file's original row order, not `periods`' order.
    """
    per_values = df["Per"].to_numpy(dtype=np.float64)
    periods_arr = np.asarray(list(periods), dtype=np.float64)
    is_selected = np.any(np.isclose(per_values[:, None], periods_arr[None, :], rtol=rtol, atol=atol), axis=1)
    selected = df[is_selected].reset_index(drop=True)
    matched_per = selected["Per"].to_numpy(dtype=np.float64)
    missing = [p for p in periods_arr if not np.any(np.isclose(matched_per, p, rtol=rtol, atol=atol))]
    if missing:
        raise ValueError(f"Periods not found: {sorted(missing)}")
    return selected


def load_coefficients(csv_path: str, periods: Optional[Sequence[float]] = None) -> Coefficients:
    """
    Load `Coefficients` from the model's coefficient CSV (e.g. `CoefMean.csv`).

    Parameters
    ----------
    csv_path : str
    periods : sequence of float, optional
        Restrict to these periods (matched to `Per` values in the file
        with a small floating-point tolerance -- see `_select_periods`).
        Defaults to every period in the file, in file order. Pass e.g.
        `periods=[0.01]` to get a single-period `Coefficients` for use as
        `coef_pga`.
    """
    df = pd.read_csv(csv_path)
    if periods is not None:
        df = _select_periods(df, periods)
    return Coefficients(**{
        field: jnp.asarray(df[field].values, dtype=jnp.float64) for field in _COEF_FIELDS
    })


def _fill_geometry(M, Rrup, Frv, Dip, W, Ztor, Zhyp, Zbot, HW, Rjb, Rx):
    """
    Back-fill missing (NaN) rupture-geometry inputs, mirroring the "Test
    for missing ... Values" sections of the R code. All inputs/outputs are
    scenario-only (shape ``(n,)``) -- none of this depends on period.
    """
    deg2rad = jnp.pi / 180.0

    # Ztor
    ztor_rv = jnp.clip(2.704 - 1.226 * jnp.clip(M - 5.849, 0.0, None), 0.0, None) ** 2
    ztor_other = jnp.clip(2.673 - 1.136 * jnp.clip(M - 4.970, 0.0, None), 0.0, None) ** 2
    Ztor = jnp.where(jnp.isnan(Ztor), jnp.where(Frv == 1, ztor_rv, ztor_other), Ztor)

    # W (down-dip rupture width)
    W = jnp.where(jnp.isnan(W), jnp.sqrt(10.0 ** ((M - 4.07) / 0.98)), W)

    # Depth to bottom of rupture
    Zbor = Ztor + W * jnp.sin(Dip * deg2rad)

    # Zbot (depth to bottom of seismogenic zone)
    Zbot = jnp.where(jnp.isnan(Zbot), 20.0, Zbot)
    Zbot = jnp.maximum(Zbot, Zbor)

    # Zhyp (hypocentral depth)
    f_dZM = jnp.where(M < 6.75, -4.317 + 0.984 * M, 2.325)
    f_dZDip = jnp.where(Dip > 40.0, 0.0, 0.0445 * (Dip - 40.0))
    zhyp_default = Ztor + jnp.exp(jnp.minimum(f_dZM + f_dZDip, jnp.log(0.9 * (Zbor - Ztor))))
    Zhyp = jnp.where(jnp.isnan(Zhyp), zhyp_default, Zhyp)

    # Rx / Rjb -- see module docstring note 2 on the departure from the R
    # code's whole-vector ifelse.
    L1 = Ztor / jnp.cos(Dip * deg2rad)
    L2 = Zbor / jnp.cos(Dip * deg2rad)
    H1 = Ztor / jnp.tan(Dip * deg2rad)
    H2 = W * jnp.cos(Dip * deg2rad)

    rjb_missing = jnp.isnan(Rjb)
    rx_missing = jnp.isnan(Rx)

    Rjb_geom = jnp.sqrt(jnp.where(HW == 0, Rrup ** 2 - Ztor ** 2, Rjb ** 2))
    Rx_geom = jnp.where((HW == 0) & (Dip == 90), Rjb_geom, -Rjb_geom)
    Rx_geom = jnp.sqrt(jnp.where(
        (HW == 1) & (Rrup <= L1), jnp.clip(Rrup ** 2 - Ztor ** 2, 0.0, None), Rx_geom ** 2))
    Rx_geom = jnp.where(
        (HW == 1) & (Rrup <= L2) & (Rrup > L1), Rrup / jnp.sin(Dip * deg2rad) - H1, Rx_geom)
    Rx_geom = jnp.where(
        (HW == 1) & (Rrup > L2),
        H2 + jnp.sqrt(jnp.where(Rrup > Zbor, Rrup ** 2 - Zbor ** 2, Rx_geom ** 2)),
        Rx_geom)
    Rjb_geom = jnp.where((HW == 1) & (Rx_geom <= H2), 0.0, Rx_geom - H2)

    Rjb = jnp.where(rjb_missing, Rjb_geom, Rjb)
    Rx = jnp.where(rx_missing, Rx_geom, Rx)

    return Ztor, W, Zbot, Zhyp, Rjb, Rx


def _base_terms(c, M_, Rrup_, Rjb_, Rx_, Frv_, Fnm_, W_, Dip_, Ztor_, Zhyp_, HW_):
    """
    Sum of every median term that does *not* depend on Vs30/Z25 (constant,
    magnitude, distance, attenuation, style-of-faulting, hypocentral-depth
    and hanging-wall terms) -- shared between the site-level prediction
    (using `coef`) and any rock-reference prediction needed for the
    nonlinear model (using `coef` or `coef_pga`, both with Vs30 fixed to
    the reference 800 m/s implicitly via the sediment term, see
    `_site_and_rock_sediment_terms`).

    All `*_` arguments are already-broadcast (n, 1) column vectors; `c` is
    a `Coefficients` (or single-period subset of one), giving (n,
    n_periods_of_c) output.
    """
    deg2rad = jnp.pi / 180.0

    fcon = c.c0

    mterm2 = (M_ - c.m1) * (M_ >= c.m1)
    mterm3 = (M_ - c.m2) * (M_ >= c.m2)
    fmag = c.c1 * M_ + c.c2 * mterm2 + c.c3 * mterm3

    HTerm = jnp.exp(c.c6 + c.c7 * M_)
    Rseis = Rrup_ + 0.001
    Rseis = jnp.where((c.Per <= _PER1) & (Rrup_ < c.Zseis), c.Zseis, Rseis)
    Rseis = jnp.where(
        (c.Per > _PER1) & (c.Per < _PER2) & (Rrup_ < c.Zseis),
        c.Zseis * (jnp.log(_PER2) - jnp.log(c.Per)) / (jnp.log(_PER2) - jnp.log(_PER1)),
        Rseis)
    beyond = Rseis > c.Rcross
    R_eff = jnp.where(beyond, c.Rcross, Rseis)
    fdis = (c.c4 + c.c5 * M_) * jnp.log(jnp.sqrt(R_eff ** 2 + HTerm ** 2)) \
        - jnp.where(beyond, 0.5 * jnp.log(Rseis / c.Rcross), 0.0)

    fatn = c.c8 / 100.0 * (Rrup_ - c.Rcross) * (Rrup_ > c.Rcross)

    fflt = c.c9 * Frv_ + c.c10 * Fnm_

    fhyp1 = c.c11 * (Zhyp_ * (Zhyp_ <= c.d1) + c.d1 * (Zhyp_ > c.d1))
    fhyp2 = c.c12 * ((Zhyp_ - c.d1) * (Zhyp_ > c.d1) * (Zhyp_ <= c.d2)
                     + (c.d2 - c.d1) * (Zhyp_ > c.d2))
    fhyp = fhyp1 + fhyp2

    R1 = W_ * jnp.cos(Dip_ * deg2rad)
    R2 = 62.0 * M_ - 350.0
    f_hngRrup = jnp.where(Rrup_ == 0, 1.0, (Rrup_ - Rjb_) / Rrup_)
    f_hngZ = jnp.where(Ztor_ > 16.66, 0.0, 1.0 - 0.06 * Ztor_)
    f_hngDip = (90.0 - Dip_) / 45.0
    f_1Rx = c.h1 + c.h2 * (Rx_ / R1) + c.h3 * (Rx_ / R1) ** 2
    f_2Rx = c.h4 + c.h5 * (Rx_ - R1) / (R2 - R1) + c.h6 * ((Rx_ - R1) / (R2 - R1)) ** 2
    f_hngRx = jnp.where(Rx_ < 0, 0.0, jnp.where(Rx_ < R1, f_1Rx, jnp.clip(f_2Rx, 0.0, None)))
    fhng = c.c13 * (HW_ * f_hngRx * f_hngRrup * f_hngZ * f_hngDip)

    return fcon + fmag + fdis + fatn + fflt + fhyp + fhng


def _site_and_rock_sediment_terms(c, Vs30_, Z25_):
    """
    Centered basin-depth (sediment) term, both at the site's own Vs30
    (`fsed`, R: `fsed.T`) and relative to the rock reference `v0`
    (`fsed_rock`, R: `fsed.Ir`/`fsed.PGAr`) -- the latter only used when
    building a rock-reference (PSAr/PGAr) prediction for the nonlinear
    model. Both reuse the same site Z25 (`Z25_filled`); only the
    reference `Z25` they're centered on differs.
    """
    lnZ25ref = jnp.where(
        Vs30_ >= c.z6,
        c.z1 + c.z2 * jnp.log(Vs30_) + c.z3 * jnp.log(Vs30_) ** 2 + c.z4 * jnp.log(Vs30_) ** 3,
        jnp.log(c.z5))
    Z25ref = jnp.exp(lnZ25ref)
    Z25_filled = jnp.where(jnp.isnan(Z25_) | (Z25_ == -999.0), Z25ref, Z25_)
    dlnZ25 = jnp.log(Z25_filled / Z25ref)
    fsed = c.c15 * dlnZ25 + c.c16 * dlnZ25 ** 2

    lnZ25refRock = jnp.where(
        c.v0 >= c.z6,
        c.z1 + c.z2 * jnp.log(c.v0) + c.z3 * jnp.log(c.v0) ** 2 + c.z4 * jnp.log(c.v0) ** 3,
        jnp.log(c.z5))
    Z25refRock = jnp.exp(lnZ25refRock)
    dlnZ25_rock = jnp.log(Z25_filled / Z25refRock)
    fsed_rock = c.c15 * dlnZ25_rock + c.c16 * dlnZ25_rock ** 2

    return fsed, fsed_rock


def calculate_ln_median_psa_campbelletal27(
    coef: Coefficients,
    M,
    Rrup,
    Rjb=jnp.nan,
    Rx=jnp.nan,
    Frv=0.0,
    Fnm=0.0,
    W=jnp.nan,
    Dip=90.0,
    Ztor=jnp.nan,
    Zhyp=jnp.nan,
    Zbot=jnp.nan,
    Vs30=760.0,
    Vs30_meas=0.0,
    Z25=jnp.nan,
    HW=0.0,
    nl_coef: Optional[Ilhan26Coefficients] = None,
    coef_pga: Optional[Coefficients] = None,
) -> jnp.ndarray:
    """
    ln(median PSA) for every (scenario, period) combination.

    Every scenario argument is broadcast to a common shape ``(n,)``, so
    they can be passed as Python/NumPy scalars, or arrays of matching
    length for a batch of ``n`` independent scenarios. Use `jnp.nan`
    (the default) for any of `Rjb`, `Rx`, `W`, `Ztor`, `Zhyp`, `Zbot`,
    `Z25` to have it back-filled from the other parameters, exactly as
    the R code does when the corresponding column is `NA`.

    Parameters
    ----------
    coef : Coefficients
        From `load_coefficients`; defines both the coefficients and the
        set of periods predicted.
    M : moment magnitude
    Rrup : closest distance to rupture (km)
    Rjb : closest distance to surface projection of rupture (km), or NaN
    Rx : horizontal distance from top of rupture, perpendicular to strike
        (km), or NaN
    Frv, Fnm : 1/0 reverse, 1/0 normal faulting flags (0/0 = strike-slip)
    W : down-dip rupture width (km), or NaN
    Dip : fault dip (degrees)
    Ztor : depth to top of rupture (km), or NaN
    Zhyp : hypocentral depth (km), or NaN
    Zbot : depth to bottom of seismogenic zone (km), or NaN
    Vs30 : time-averaged shear-wave velocity, top 30 m (m/s)
    Vs30_meas : 0 measured, 1 proxy/estimated
    Z25 : depth to Vs = 2.5 km/s horizon (km), or NaN
    HW : 1 if site is on the hanging wall, else 0
    nl_coef : Ilhan26Coefficients, optional
        From `nonlinear_site.load_ilhan26_coefficients`, loaded with the
        **same** `periods` as `coef` (so the two line up period-for-
        period). If given, the Ilhan et al. (2026) nonlinear
        soil-amplification term is added to the site term; `coef_pga` is
        then required too. If omitted (default), the site term is
        linear-only.
    coef_pga : Coefficients, optional
        Single-period (0.01 s / PGA) `Coefficients`, e.g.
        `load_coefficients(csv_path, periods=[0.01])`. Required, and only
        used, when `nl_coef` is given -- see module docstring note 1 on
        why PGA needs its own coefficient set rather than reusing `coef`.

    Returns
    -------
    jnp.ndarray, shape (n, n_periods)
    """
    if (nl_coef is None) != (coef_pga is None):
        raise ValueError(
            "nl_coef and coef_pga must be given together (both None for a "
            "linear-only site term, or both set to enable the Ilhan26 "
            "nonlinear model)."
        )

    (M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot,
     Vs30, Vs30_meas, Z25, HW) = jnp.broadcast_arrays(*[
        jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)) for x in
        (M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot, Vs30, Vs30_meas, Z25, HW)
    ])

    Ztor, W, Zbot, Zhyp, Rjb, Rx = _fill_geometry(M, Rrup, Frv, Dip, W, Ztor, Zhyp, Zbot, HW, Rjb, Rx)

    # scenario (n,) -> (n, 1) column vectors; coefficients (n_periods,)
    # broadcast as (1, n_periods) row vectors -> every term below is (n, n_periods)
    col = lambda x: x[:, None]  # noqa: E731
    M_, Rrup_, Rjb_, Rx_, Frv_, Fnm_, W_, Dip_, Ztor_, Zhyp_, Vs30_, Z25_, HW_ = (
        col(v) for v in (M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Vs30, Z25, HW)
    )

    f_base = _base_terms(coef, M_, Rrup_, Rjb_, Rx_, Frv_, Fnm_, W_, Dip_, Ztor_, Zhyp_, HW_)
    fsed, fsed_rock = _site_and_rock_sediment_terms(coef, Vs30_, Z25_)

    # Linear shallow-site-response term
    Vs30_clipped = jnp.clip(Vs30_, coef.v1, coef.v2)
    flin = coef.c14 * jnp.log(Vs30_clipped / coef.v0)

    if nl_coef is None:
        return f_base + flin + fsed

    if nl_coef.Per.shape != coef.Per.shape or not np.allclose(
        np.asarray(nl_coef.Per), np.asarray(coef.Per)
    ):
        raise ValueError(
            "nl_coef and coef cover different periods -- load both with the "
            "same `periods` argument so they line up period-for-period."
        )

    # Rock-reference (Vs30 = 800 m/s) median PSA, needed to drive the
    # nonlinear model (R: PSAr.T).
    PSAr = jnp.exp(f_base + fsed_rock)

    # Rock-reference median PGA from its own (period = 0.01 s) coefficient
    # set (R: PGAr, via Coeffs.PGA = Coeffs[1, ]) -- see module docstring note 1.
    f_base_pga = _base_terms(coef_pga, M_, Rrup_, Rjb_, Rx_, Frv_, Fnm_, W_, Dip_, Ztor_, Zhyp_, HW_)
    _, fsed_rock_pga = _site_and_rock_sediment_terms(coef_pga, Vs30_, Z25_)
    PGAr = jnp.exp(f_base_pga + fsed_rock_pga)  # shape (n, 1)

    ir = compute_ilhan26_ir(PSAr, PGAr)
    fnln = compute_ln_nonlinearity_ilhan26(Vs30_, ir, nl_coef)

    return f_base + flin + fnln + fsed


def calculate_median_psa_campbelletal27(coef: Coefficients, *args, **kwargs) -> jnp.ndarray:
    """`exp(calculate_ln_median_psa_campbelletal27(...))` -- median PSA in g, shape (n, n_periods)."""
    return jnp.exp(calculate_ln_median_psa_campbelletal27(coef, *args, **kwargs))
