"""
Scenario-grid predictions from fitted PSA models -- the PSA analogue of
`scenario_prediction.py`'s `scenario_predict` for the EAS model.

Currently has the NN version (`scenario_predict_psa_nn`); the parametric
functional-form PSA model's version can be added here too, sharing the
same DEFAULTS/grid-building logic via `_build_grid`.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict

import numpy as np
import pandas as pd

from ngaw3_kuehnetal27.psa_generation.nn_psa_model import GMMNet, predict as predict_nn
from sklearn.preprocessing import StandardScaler

# Defaults for every scenario variable not explicitly supplied.
# "subregion_id" matches the column name used by `sample_scenarios`
# (EAS side) and `build_features` (NN side) -- NOT "reg_id".
# "magbin_id" is intentionally absent: it was a leftover from an
# earlier, now-unused temporary model (see EAS sample_scenarios).
DEFAULTS: Dict[str, Any] = {
    "M": 6.5,
    "R": 30.0,
    "Z": 3.0,
    "VS": 800.0,
    "Frev": 0,
    "Fnm": 0,
    "Dip": 90.0,
    "FW": 10.0,
    "Rx": 0.0,
    "Ry0": 0.0,
    "subregion_id": 0,
    "basin_id": 1,
    "vsmeas_id": 0,
}


def _build_grid(**kwargs) -> pd.DataFrame:
    """
    Cartesian product of the supplied scenario-variable arrays, with
    every unsupplied `DEFAULTS` column filled in.
    """
    varied_keys = list(kwargs.keys())
    varied_vals = [np.atleast_1d(kwargs[k]) for k in varied_keys]
    grid = list(itertools.product(*varied_vals))
    df_scenarios = pd.DataFrame(grid, columns=varied_keys)

    for col, default in DEFAULTS.items():
        if col not in df_scenarios.columns:
            df_scenarios[col] = default

    return df_scenarios


def scenario_predict_psa_nn(
    model: GMMNet,
    scaler: StandardScaler,
    periods: np.ndarray,
    **kwargs,
):
    """
    Generate NN PSA predictions for all combinations of the supplied
    scenario variables.

    Parameters
    ----------
    model, scaler : from `nn_psa_model.train` (or `load_model`)
    periods : array, length 21
        Period values, for column naming only.
    **kwargs
        Arrays for any subset of scenario variables: M, R, Z, VS, Frev,
        Fnm, Dip, FW, Rx, Ry0, subregion_id, basin_id, vsmeas_id.
        Unspecified variables take their `DEFAULTS` value.
        `subregion_id` is only meaningful if `model.include_region`;
        otherwise it's carried in `df_scenarios` but ignored by the
        model (see `nn_psa_model.predict`).

    Returns
    -------
    df_pred : DataFrame, shape (n_combos, n_vars + 21)
        Scenario columns plus one ln(PSA) column per period
        (named "T{period:.3f}").
    ln_psa_pred : Array, shape (n_combos, 21)

    Example
    -------
        df_sc, ln_psa = scenario_predict_psa_nn(
            model, scaler, periods,
            M=np.linspace(4, 8, 40), R=np.array([10., 30., 100.]),
        )
    """
    df_scenarios = _build_grid(**kwargs)
    ln_psa_pred = predict_nn(model, scaler, df_scenarios)

    period_cols = {f"T{p:.3f}": ln_psa_pred[:, i] for i, p in enumerate(periods)}
    df_pred = pd.concat([df_scenarios, pd.DataFrame(period_cols)], axis=1)

    return df_pred, ln_psa_pred
