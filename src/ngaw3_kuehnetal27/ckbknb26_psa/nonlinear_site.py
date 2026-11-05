"""
Ilhan et al. (2026) soil-nonlinearity model for PSA, ported from
`ILHAN26.R`.

This is the same functional form as the "new" Hashash model already
implemented for EAS/PSA in `nl_models.py`
(`compute_ln_nonlinearity_hashash_new`) -- same `f2`/`delta_f2` terms, same
`TEM_NL` transition, same cosine frequency taper. Confirmed independently:
`ILHAN26_Coeffs.csv`/`ILHAN26_StdDevs.csv` (here) are the same tables as
`N2_RS_ay26.csv`/`N2_RS_ay26-STDEV.csv` (there), just with columns
`V_f`/`V_g` instead of `Vf`/`Vg`.

The one substantive difference -- why this isn't just a call into the
existing function -- is how the reference intensity ("Ir") going into the
nonlinearity term is determined. The existing EAS implementation caps the
reference motion at a fixed 0.3 g-sec (`IR_LIM`). Ilhan26 instead caps the
rock-reference PSA at 1 g (`IR_CAP`):

    Ir = min(PSAr, 1 g)

(An earlier version of the model replaced PSAr by PGAr once PSAr exceeded
1 g; the model developers have since changed this to a plain 1 g cap,
which also means no separate rock-reference PGA prediction is needed.)
`median_core.calculate_ln_median_psa_campbelletal27` applies the cap
before calling `compute_ln_nonlinearity_ilhan26`.

Besides the median term (`compute_ln_nonlinearity_ilhan26`) this module has
the two pieces `ILHAN26.R` hands to the total-sigma calculation:

* `compute_alpha_ilhan26` -- `Alpha`, the derivative of the nonlinear term
  with respect to ln(rock motion), which rescales the linear standard
  deviations by ``(1 + Alpha)``.
* `compute_sig_fnln_ilhan26` -- `sig_fnln`, the standard deviation of the
  nonlinear amplification itself (includes epistemic uncertainty), read from
  `ILHAN26_StdDevs.csv` via `load_ilhan26_std_coefficients`.

Both take the same capped reference intensity `Ir = min(PSAr, 1 g)` as the
median term. How they are combined with the linear model's standard
deviations is in `sigma_core.calculate_sigma_campbelletal27`.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

__all__ = [
    "Ilhan26Coefficients",
    "load_ilhan26_coefficients",
    "Ilhan26StdCoefficients",
    "load_ilhan26_std_coefficients",
    "compute_ln_nonlinearity_ilhan26",
    "compute_alpha_ilhan26",
    "compute_sig_fnln_ilhan26",
]

V_REF = 800.0             # Reference (rock) Vs30, m/s -- matches v0 in the median-model coefficients
V_REF2 = 360.0            # Second reference velocity, m/s
FREQ_TAPER_CENTER = 4.998  # Hz, center of cosine taper for the frequency weight
IR_CAP = 1.0              # g -- Ilhan26's cap on the rock-reference PSA

_ILHAN26_FIELDS = ["Per", "f3", "f4", "f5", "Vf", "Vg"]
_ILHAN26_STD_FIELDS = ["Per", "sig_c", "sig_f", "V1", "V2"]


class Ilhan26Coefficients(NamedTuple):
    """One JAX array per field, each shape ``(n_periods,)``."""
    Per: jnp.ndarray
    f3: jnp.ndarray
    f4: jnp.ndarray
    f5: jnp.ndarray
    Vf: jnp.ndarray
    Vg: jnp.ndarray


class Ilhan26StdCoefficients(NamedTuple):
    """Standard-deviation parameters, one JAX array per field, shape ``(n_periods,)``."""
    Per: jnp.ndarray
    sig_c: jnp.ndarray
    sig_f: jnp.ndarray
    V1: jnp.ndarray
    V2: jnp.ndarray


def _select_periods(df: pd.DataFrame, periods: Sequence[float], rtol: float = 1e-6, atol: float = 1e-9) -> pd.DataFrame:
    """
    Select the rows of `df` (must have a `Per` column) matching `periods`,
    tolerant of small floating-point noise -- see
    `median_core._select_periods` (same logic, duplicated here to keep
    this module independent of `median_core`).
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


def load_ilhan26_coefficients(
    csv_path: str, periods: Optional[Sequence[float]] = None
) -> Ilhan26Coefficients:
    """
    Load `Ilhan26Coefficients` from `ILHAN26_Coeffs.csv`.

    Parameters
    ----------
    csv_path : str
    periods : sequence of float, optional
        Restrict to (and order by) these periods -- pass the **same**
        `periods` used for `median_core.load_coefficients` so the two
        coefficient sets line up period-for-period. Matched to `Per`
        values in the file with a small floating-point tolerance (see
        `_select_periods`) rather than requiring an exact match: the
        frequency-domain EAS/PSA coefficients elsewhere in this codebase
        need interpolation, but `ILHAN26_Coeffs.csv` is already finely
        enough sampled -- 124 periods from 0.001 to 10 s -- that every
        period in the CKBKNB26 median model's own coefficient file has a
        matching entry here to begin with.

    Raises
    ------
    ValueError
        If any requested period has no match (within tolerance) in the file.
    """
    df = pd.read_csv(csv_path).rename(columns={"V_f": "Vf", "V_g": "Vg"})
    if periods is not None:
        df = _select_periods(df, periods)
    return Ilhan26Coefficients(**{
        field: jnp.asarray(df[field].values, dtype=jnp.float64) for field in _ILHAN26_FIELDS
    })


def load_ilhan26_std_coefficients(
    csv_path: str, periods: Optional[Sequence[float]] = None
) -> Ilhan26StdCoefficients:
    """
    Load `Ilhan26StdCoefficients` from `ILHAN26_StdDevs.csv` (columns
    `Per, sig_c, sig_f, V1, V2`).

    `periods` works exactly as in `load_ilhan26_coefficients` -- pass the
    same list used for the median model's `Coefficients` so everything
    lines up period-for-period.

    Raises
    ------
    ValueError
        If any requested period has no match (within tolerance) in the file.
    """
    df = pd.read_csv(csv_path)
    if periods is not None:
        df = _select_periods(df, periods)
    return Ilhan26StdCoefficients(**{
        field: jnp.asarray(df[field].values, dtype=jnp.float64) for field in _ILHAN26_STD_FIELDS
    })


def _f2_and_delta_f2(vs30, c):
    """`f2` and `delta_f2` of `ILHAN26.R` (shared by the median term and Alpha)."""
    vs30_lim = jnp.clip(vs30, 200.0, V_REF)
    tem_nl = jnp.clip((V_REF - vs30_lim) / (V_REF - c.Vg), 0.0, 1.0)

    f2 = c.f4 * (jnp.exp(c.f5 * (vs30_lim - V_REF2)) - jnp.exp(c.f5 * (V_REF - V_REF2)))
    delta_f2 = c.f4 * (
        jnp.exp(c.f5 * (vs30_lim - V_REF2)) - jnp.exp(c.f5 * (c.Vf - V_REF2))
    ) * tem_nl
    return f2, delta_f2


def _frequency_weights(per):
    """
    Clipped cosine frequency taper `wf` of `ILHAN26.R`: 1 below `f_a`, 0
    above `f_b`, cosine transition in between. Used by both the median term
    and `Alpha`.
    """
    freq = 1.0 / per
    f_a = 0.9 * FREQ_TAPER_CENTER
    f_b = 1.1 * FREQ_TAPER_CENTER
    w_cos = 0.5 * (1.0 + jnp.cos(
        jnp.pi * (jnp.log(freq) - jnp.log(f_a)) / (jnp.log(f_b) - jnp.log(f_a))
    ))
    return jnp.where(freq <= f_a, 1.0, jnp.where(freq >= f_b, 0.0, w_cos))


def compute_ln_nonlinearity_ilhan26(
    vs30: jnp.ndarray, ir: jnp.ndarray, nl_coef: Ilhan26Coefficients
) -> jnp.ndarray:
    """
    ln(nonlinear amplification), Ilhan et al. (2026).

    Same functional form as `nl_models.compute_ln_nonlinearity_hashash_new`
    (`TEM_NL` transition + cosine frequency taper), evaluated at a
    caller-supplied reference intensity `ir` rather than an internally
    capped one -- see module docstring for why.

    Parameters
    ----------
    vs30 : jnp.ndarray, shape (n, 1)
        Site Vs30 (m/s), as a column vector.
    ir : jnp.ndarray, shape (n, n_periods)
        Reference intensity (g): the rock-reference PSA, already capped at
        `IR_CAP` by the caller (``jnp.minimum(PSAr, IR_CAP)``).
    nl_coef : Ilhan26Coefficients, fields shape (n_periods,)
        Must cover the same periods, in the same order, as the median
        model's own `Coefficients` (both loaded with the same `periods`
        argument).

    Returns
    -------
    jnp.ndarray, shape (n, n_periods)
    """
    c = nl_coef
    f2, delta_f2 = _f2_and_delta_f2(vs30, c)
    wf = _frequency_weights(c.Per)

    return (f2 + delta_f2 * wf) * jnp.log((ir + c.f3) / c.f3)


def compute_alpha_ilhan26(
    vs30: jnp.ndarray, ir: jnp.ndarray, nl_coef: Ilhan26Coefficients
) -> jnp.ndarray:
    """
    `Alpha` of `ILHAN26.R`: derivative of the nonlinear term `fnln` with
    respect to ln(PSAr), evaluated at the (capped) reference intensity
    `ir`. The linear standard deviations are scaled by ``(1 + Alpha)`` in
    the nonlinear total sigma.

    Parameters are as for `compute_ln_nonlinearity_ilhan26`.

    This is the exact derivative of `compute_ln_nonlinearity_ilhan26`:
    ``Alpha = (f2 + df2*wf) * Ir/(Ir + f3)``, with the same clipped taper
    weight `wf` as the median term. (`ILHAN26.R` has the raw cosine weight
    `Wf` here instead, which looks like a typo -- it only coincides with
    `wf` inside the taper band -- so this deliberately differs from the R
    file; the model developers have been notified.)
    """
    c = nl_coef
    f2, delta_f2 = _f2_and_delta_f2(vs30, c)
    wf = _frequency_weights(c.Per)

    return (f2 + delta_f2 * wf) * ir / (ir + c.f3)


def compute_sig_fnln_ilhan26(
    vs30: jnp.ndarray, ir: jnp.ndarray, std_coef: Ilhan26StdCoefficients
) -> jnp.ndarray:
    """
    `sig_fnln` of `ILHAN26.R`: standard deviation of the nonlinear
    amplification (includes epistemic uncertainty). Called `Phi_nln` in the
    total-sigma calculation.

    Parameters
    ----------
    vs30 : jnp.ndarray, shape (n, 1)
        Site Vs30 (m/s), as a column vector.
    ir : jnp.ndarray, shape (n, n_periods)
        Capped reference intensity (g), ``jnp.minimum(PSAr, IR_CAP)``.
    std_coef : Ilhan26StdCoefficients, fields shape (n_periods,)
        From `load_ilhan26_std_coefficients`, same periods/order as the
        median model's `Coefficients`.

    Returns
    -------
    jnp.ndarray, shape (n, n_periods)
    """
    c = std_coef
    sig_f2_mid = (-c.sig_c / jnp.log(c.V2 / c.V1)) * jnp.log(vs30 / c.V1) + c.sig_c
    sig_f2 = jnp.where(vs30 < c.V1, c.sig_c, jnp.where(vs30 > c.V2, 0.0, sig_f2_mid))

    return sig_f2 * jnp.log((ir + c.sig_f) / c.sig_f)
