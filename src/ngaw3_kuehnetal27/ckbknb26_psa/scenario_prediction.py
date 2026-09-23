"""
Scenario-grid median PSA prediction for the CKBKNB26 (campbelletal27) GMM
-- the same DEFAULTS + itertools.product pattern as the EAS model's
`scenario_predict`, built on top of
`median_core.calculate_ln_median_psa_campbelletal27`.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .median_core import Coefficients, calculate_ln_median_psa_campbelletal27, load_coefficients
from .nonlinear_site import Ilhan26Coefficients, load_ilhan26_coefficients

__all__ = ["DEFAULTS", "scenario_predict_campbelletal27", "scenario_predict_campbelletal27_reference"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_COEF_MEAN_PATH = os.path.join(_DATA_DIR, "CoefMean.csv")
_ILHAN26_COEFFS_PATH = os.path.join(_DATA_DIR, "ILHAN26_Coeffs.csv")

# Defaults for every scenario variable. NaN marks the geometry fields that
# `calculate_ln_median_psa` back-fills itself (see median_core.py) --
# matching the R model's `NA` convention.
DEFAULTS: Dict[str, Any] = {
    "M": 6.5,
    "Rrup": 30.0,
    "Rjb": float("nan"),
    "Rx": float("nan"),
    "Frv": 0.0,
    "Fnm": 0.0,
    "W": float("nan"),
    "Dip": 90.0,
    "Ztor": float("nan"),
    "Zhyp": float("nan"),
    "Zbot": float("nan"),
    "Vs30": 760.0,
    "Vs30_meas": 0.0,
    "Z25": float("nan"),
    "HW": 0.0,
}


def scenario_predict_campbelletal27(
    coef: Coefficients,
    output: str = "ln_median",
    nl_coef: Optional[Ilhan26Coefficients] = None,
    coef_pga: Optional[Coefficients] = None,
    **kwargs,
):
    """
    Generate median-PSA predictions for all combinations of the supplied
    scenario variables.

    Parameters
    ----------
    coef : Coefficients
        From `median_core.load_coefficients`; also fixes the set of
        periods predicted (`coef.Per`).
    output : {"ln_median", "median"}, default "ln_median"
        Whether to return ln(median PSA) (matches
        `calculate_ln_median_psa_campbelletal27`, and is what you want
        for e.g. residual/likelihood work) or median PSA in g.
    nl_coef, coef_pga : optional
        Enable the Ilhan et al. (2026) nonlinear site term -- passed
        straight through to `calculate_ln_median_psa_campbelletal27`; see
        its docstring. Both or neither.
    **kwargs
        Arrays (or scalars) for any subset of the scenario variables in
        `DEFAULTS` (M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot,
        Vs30, Vs30_meas, Z25, HW). Unspecified variables take their
        DEFAULTS value. Every combination of the supplied arrays is
        predicted (Cartesian product).

    Returns
    -------
    df_scenarios : DataFrame, shape (n_combos, n_vars + n_periods)
        Scenario variables plus one prediction column per period -- named
        "lnT0.3" (output="ln_median") or "T0.3" (output="median") for a
        0.3 s period.
    values : jnp.ndarray, shape (n_combos, n_periods)
        ln(median PSA) or median PSA (g), per `output`, in the same order
        as `coef.Per`.

    Example
    -------
        df, ln_median = scenario_predict_campbelletal27(
            coef, M=np.linspace(4, 8, 40), Rrup=np.array([10., 30., 100.]),
        )
    """
    if output not in ("ln_median", "median"):
        raise ValueError(f"output must be 'ln_median' or 'median', got {output!r}")

    varied_keys = list(kwargs.keys())
    varied_vals = [np.atleast_1d(kwargs[k]) for k in varied_keys]
    grid = list(itertools.product(*varied_vals))
    df_scenarios = pd.DataFrame(grid, columns=varied_keys)

    for col, default in DEFAULTS.items():
        if col not in df_scenarios.columns:
            df_scenarios[col] = default

    ln_median = calculate_ln_median_psa_campbelletal27(
        coef,
        M=jnp.asarray(df_scenarios["M"].values),
        Rrup=jnp.asarray(df_scenarios["Rrup"].values),
        Rjb=jnp.asarray(df_scenarios["Rjb"].values),
        Rx=jnp.asarray(df_scenarios["Rx"].values),
        Frv=jnp.asarray(df_scenarios["Frv"].values),
        Fnm=jnp.asarray(df_scenarios["Fnm"].values),
        W=jnp.asarray(df_scenarios["W"].values),
        Dip=jnp.asarray(df_scenarios["Dip"].values),
        Ztor=jnp.asarray(df_scenarios["Ztor"].values),
        Zhyp=jnp.asarray(df_scenarios["Zhyp"].values),
        Zbot=jnp.asarray(df_scenarios["Zbot"].values),
        Vs30=jnp.asarray(df_scenarios["Vs30"].values),
        Vs30_meas=jnp.asarray(df_scenarios["Vs30_meas"].values),
        Z25=jnp.asarray(df_scenarios["Z25"].values),
        HW=jnp.asarray(df_scenarios["HW"].values),
        nl_coef=nl_coef,
        coef_pga=coef_pga,
    )
    values = ln_median if output == "ln_median" else jnp.exp(ln_median)
    col_prefix = "lnT" if output == "ln_median" else "T"

    period_cols = {
        f"{col_prefix}{p:g}": np.asarray(values)[:, i] for i, p in enumerate(np.asarray(coef.Per))
    }
    df_pred = pd.concat([df_scenarios, pd.DataFrame(period_cols)], axis=1)

    return df_pred, values


def scenario_predict_campbelletal27_reference(**kwargs):
    """
    Self-contained version of `scenario_predict_campbelletal27`: reads the
    package's own bundled coefficients (`data/CoefMean.csv`,
    `data/ILHAN26_Coeffs.csv`) itself -- no `Coefficients`/
    `Ilhan26Coefficients` objects to load or pass in beforehand -- and
    always predicts the full spectrum (every period present in
    `CoefMean.csv`, in file order), with the Ilhan et al. (2026)
    nonlinear site term included. PGA, needed internally to drive the
    nonlinear model, is taken from `CoefMean.csv`'s first row (period
    0.01 s), matching the R model's `Coeffs.PGA = Coeffs[1, ]`.

    Use this when you just want "the model's predictions" without
    thinking about coefficient files; use `scenario_predict_campbelletal27`
    directly (with coefficients loaded once and reused) when predicting
    many batches, or a restricted set of periods, since this function
    re-reads and re-interpolates nothing -- it reloads the coefficient
    CSVs from disk on every call.

    Parameters
    ----------
    **kwargs
        Scenario variables, exactly as for `scenario_predict_campbelletal27`
        (M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot, Vs30,
        Vs30_meas, Z25, HW). Unspecified variables take their `DEFAULTS`
        value. Every combination of the supplied arrays is predicted
        (Cartesian product).

    Returns
    -------
    df_scenarios : DataFrame, shape (n_combos, n_vars + n_periods)
        Scenario variables plus one "lnT<period>" column per period in
        `CoefMean.csv`.
    ln_median : jnp.ndarray, shape (n_combos, n_periods)
        ln(median PSA), including nonlinear site amplification, for every
        period in `CoefMean.csv`, in file order.

    Example
    -------
        df, ln_median = scenario_predict_campbelletal27_reference(
            M=np.linspace(4, 8, 40), Rrup=np.array([10., 30., 100.]), Vs30=300.0,
        )
    """
    coef = load_coefficients(_COEF_MEAN_PATH)
    periods = np.asarray(coef.Per).tolist()

    # First row of the coefficient file -- not necessarily literally 0.01
    # if CoefMean.csv were ever swapped out, so read it off rather than
    # hard-coding it, but it's period 0.01 s in the file shipped here.
    coef_pga = load_coefficients(_COEF_MEAN_PATH, periods=[periods[0]])
    nl_coef = load_ilhan26_coefficients(_ILHAN26_COEFFS_PATH, periods=periods)

    return scenario_predict_campbelletal27(
        coef, output="ln_median", nl_coef=nl_coef, coef_pga=coef_pga, **kwargs
    )
