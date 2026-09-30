"""
Scenario median prediction from MCMC results (arviz InferenceData) --
the MCMC counterpart to `scenario_prediction.scenario_predict`.

`scenario_predict` works on ONE set of resolved site values (SVI point
estimate). Here the same physics (`predict_median_categorical`) is
evaluated for every posterior draw via `jax.vmap`, so the output carries
the full posterior uncertainty of the median (no event/site terms, no
aleatory variability -- median only).

Random effects that are needed for prediction but may not have been saved
(`save_ranef=False`, `save_kappa_adj=False`) are rebuilt from the samples
that ARE always in the posterior:

  c_region   <- c_region_raw, sigma_region   (and L_freq, if not identity)
  kappa term <- kappa_region_table, or m_region, or
                ln_kappa_region_raw + sigma_ln_kappa_region, with c_0_kappa_star
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ngaw3_kuehnetal27.median_core import (
    EventParams,
    ModelConstants,
    SiteParams,
    predict_median_categorical,
)
from ngaw3_kuehnetal27.scenario_prediction import DEFAULTS, coefficients_from_site_values
from ngaw3_kuehnetal27.site_amplification.amp1d import amp1d_adj_from_vs30

# Posterior variables that may be needed for prediction. Anything else in
# the posterior (deltaB, deltaS, deltaB_*_raw, ...) is never touched, so
# large random-effect arrays are not pulled into memory.
_COEF_NAMES = [
    "c_0", "c_m1", "c_m2", "c_m3", "c_hw", "c_nft_1", "c_nft_2", "c_nm", "c_rev",
    "c_gs1", "c_zt", "c_vs_meas", "c_vs_est", "c_vs", "c_attn", "Q_0", "Q_exp",
    "mu_Q_0", "gs_break", "gs_exp", "zt_break",
]
_REGION_NAMES = [
    "c_basin", "c_region", "c_region_raw", "sigma_region",
    "c_0_kappa_star", "kappa_region_table", "m_region",
    "ln_kappa_region_raw", "sigma_ln_kappa_region", "sigma_ln_kappa_station",
]
_WANTED = _COEF_NAMES + [n + "_gl" for n in _COEF_NAMES] + _REGION_NAMES

_SCENARIO_FLOAT_KEYS = ["M", "R", "Rx", "Ry0", "Z", "VS", "Frev", "Fnm", "Dip", "FW"]


# ----------------------------------------------------------------------
# posterior -> dict of stacked draws
# ----------------------------------------------------------------------
def posterior_site_values(
    idata,
    n_draws: Optional[int] = None,
    thin: int = 1,
    seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    """
    Stack chains and draws of the prediction-relevant posterior variables.

    Returns a dict {name: array of shape (n_post, ...)}, where
    n_post = n_chain * n_draw, optionally thinned (`thin`) and/or randomly
    subsampled to `n_draws` (`seed`). Call this once and pass the result
    instead of `idata` if you make many prediction calls.
    """
    post = idata.posterior
    names = [n for n in _WANTED if n in post.data_vars]
    if not names:
        raise KeyError("No prediction-relevant variables found in idata.posterior.")

    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    total = n_chain * n_draw
    idx = np.arange(total)[:: max(int(thin), 1)]
    if n_draws is not None and n_draws < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=n_draws, replace=False))

    out = {}
    for name in names:
        a = np.asarray(post[name].transpose("chain", "draw", ...).values)
        out[name] = jnp.asarray(a.reshape((total,) + a.shape[2:])[idx])
    return out


def _as_site_values(idata_or_dict, n_draws, thin, seed):
    if isinstance(idata_or_dict, dict):
        return idata_or_dict
    return posterior_site_values(idata_or_dict, n_draws=n_draws, thin=thin, seed=seed)


# ----------------------------------------------------------------------
# per-draw region / kappa tables (called inside vmap: one draw at a time)
# ----------------------------------------------------------------------
def _region_tables(sv, dataset_region, n_freq, n_sub_needed, L_freq, c0_parametric):
    """
    Returns (c_basin_table, c_subregion_table, kappa_adj_table) for ONE
    posterior draw. kappa_adj_table is (n_subregion,) or None.
    """
    if dataset_region == "global":
        return jnp.zeros((1, n_freq)), jnp.zeros((1, n_freq)), None

    c_basin = sv["c_basin"]

    if "c_region" in sv:
        c_sub = sv["c_region"]
    elif "c_region_raw" in sv:
        if "sigma_region" not in sv:
            raise KeyError("'c_region_raw' found but 'sigma_region' is missing from the posterior.")
        L_sub = sv["sigma_region"][..., None] * L_freq
        c_sub = sv["c_region_raw"] @ L_sub.T          # mu_freq is zero in the model
    else:
        # include_region=False: no subregion effect
        c_sub = jnp.zeros((n_sub_needed, n_freq))

    has_region_kappa = any(k in sv for k in ("kappa_region_table", "m_region", "ln_kappa_region_raw"))
    has_station_kappa = "sigma_ln_kappa_station" in sv

    if not (has_region_kappa or has_station_kappa):
        kappa = None
    elif has_region_kappa:
        if "kappa_region_table" in sv:
            kappa = sv["kappa_region_table"]
        else:
            kstar = sv["c_0_kappa_star"]
            if "m_region" in sv:
                m = sv["m_region"]
            else:
                m = jnp.exp(sv["ln_kappa_region_raw"] * sv["sigma_ln_kappa_region"])
            kappa = kstar * (m - 1.0) if c0_parametric else kstar * m
    else:
        # station-level kappa only: a new/unknown station has m_station = 1.
        # Parametric c_0 already contains -c_0_kappa_star*F -> deviation is 0.
        # Spline c_0 has no kappa term -> the whole term is c_0_kappa_star.
        kappa = None if c0_parametric else jnp.full((c_sub.shape[0],), sv["c_0_kappa_star"])

    return c_basin, c_sub, kappa


# ----------------------------------------------------------------------
# scenario handling
# ----------------------------------------------------------------------
def _build_scenarios(kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Cartesian grid of the supplied variables, filled up with DEFAULTS."""
    keys = list(kwargs.keys())
    vals = [np.atleast_1d(kwargs[k]) for k in keys]
    df = pd.DataFrame(list(itertools.product(*vals)), columns=keys)
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


def _scenario_arrays(df, F, dataset_region, amp1d_dataset, amp1d_target_name):
    a = {k: jnp.asarray(df[k].values) for k in _SCENARIO_FLOAT_KEYS}
    a["vsmeas_id"] = jnp.asarray(df["vsmeas_id"].values, dtype=int)
    if dataset_region == "global":
        a["basin_id"] = jnp.zeros(len(df), dtype=int)
        a["subregion_id"] = jnp.zeros(len(df), dtype=int)
    else:
        a["basin_id"] = jnp.asarray(df["basin_id"].values, dtype=int)
        a["subregion_id"] = jnp.asarray(df["subregion_id"].values, dtype=int)
    if amp1d_dataset is not None and dataset_region == "wus":
        a["amp1d_adj"] = amp1d_adj_from_vs30(
            amp1d_dataset, a["VS"], jnp.asarray(F), key=amp1d_target_name
        )
    else:
        a["amp1d_adj"] = None
    return a


def _make_predictor(F, nl_model_dict, dataset_region, func_gs_scaling, const,
                    n_sub_needed, L_freq, c0_parametric):
    """jit(vmap over draws) of the median prediction for a block of scenarios."""
    F = jnp.asarray(F)
    n_freq = len(F)
    L_freq = jnp.eye(n_freq) if L_freq is None else jnp.asarray(L_freq)

    def one_draw(sv, a):
        coef = coefficients_from_site_values(sv, dataset_region=dataset_region)
        c_basin, c_sub, kappa = _region_tables(
            sv, dataset_region, n_freq, n_sub_needed, L_freq, c0_parametric
        )
        evt = EventParams(
            M_model=a["M"], Dip_eq=a["Dip"], FW_eq=a["FW"], Zt_eq=a["Z"],
            Zt_eq_scaled=a["Z"] / 10.0, Fnm_eq=a["Fnm"], Frev_eq=a["Frev"],
        )
        site = SiteParams(
            VS_stat=a["VS"], lnVS=jnp.log(a["VS"]) - jnp.log(800.0),
            vs_measured_id=a["vsmeas_id"],
        )
        ln_med, _ = predict_median_categorical(
            a["R"], a["Rx"], a["Ry0"], F, a["R"] / 100.0,
            evt, site, coef, const,
            func_gs_scaling=func_gs_scaling, nl_model_dict=nl_model_dict,
            c_basin_table=c_basin, basin_id=a["basin_id"],
            c_subregion_table=c_sub, subregion_id=a["subregion_id"],
            kappa_adj_table=kappa, amp1d_adj=a["amp1d_adj"],
        )
        return ln_med                                   # (n_scenarios, n_freq)

    return jax.jit(jax.vmap(one_draw, in_axes=(0, None)))


def _n_sub_needed(df, dataset_region):
    return 1 if dataset_region == "global" else int(np.max(df["subregion_id"].values)) + 1


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def single_scenario_predict_mcmc(
    idata,
    F: np.ndarray,
    nl_model_dict: Optional[dict],
    dataset_region: str = "wus",
    func_gs_scaling: str = "stafford",
    const: Optional[ModelConstants] = None,
    amp1d_dataset=None,
    amp1d_target_name: str = "amp_predicted",
    n_draws: Optional[int] = None,
    thin: int = 1,
    seed: int = 0,
    L_freq=None,
    c0_parametric: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    ln(median) of ONE scenario for every posterior draw.

    Parameters
    ----------
    idata : arviz.InferenceData, or dict from `posterior_site_values`
        MCMC results (`az.from_numpyro(mcmc, ...)`, or loaded netcdf).
    F, nl_model_dict, dataset_region, func_gs_scaling, const,
    amp1d_dataset, amp1d_target_name
        As in `scenario_predict`.
    n_draws, thin, seed
        Optional thinning / random subsampling of the (chain x draw) samples.
        Ignored if `idata` is already a dict of stacked draws.
    L_freq : array (n_freq, n_freq), optional
        Only needed if the model was fit with a non-identity `L_freq` AND
        `c_region` was not saved (it enters the c_region reconstruction).
    c0_parametric : bool
        Must match the fit. Only used to rebuild the kappa term when
        `kappa_region_table` is absent or kappa is station-level only.
    **kwargs
        Scalar values for any of M, R, Rx, Ry0, Z, VS, Frev, Fnm, Dip, FW,
        subregion_id, basin_id, vsmeas_id; unspecified ones take DEFAULTS.

    Returns
    -------
    ln_median : ndarray, shape (n_post, n_freq)
    """
    for k, v in kwargs.items():
        if np.size(v) != 1:
            raise ValueError(
                f"'{k}' has {np.size(v)} values; single_scenario_predict_mcmc takes "
                "one scenario. Use scenario_predict_mcmc for several."
            )
    const = const if const is not None else ModelConstants()
    sv = _as_site_values(idata, n_draws, thin, seed)
    df = _build_scenarios(kwargs)
    a = _scenario_arrays(df, F, dataset_region, amp1d_dataset, amp1d_target_name)
    predictor = _make_predictor(
        F, nl_model_dict, dataset_region, func_gs_scaling, const,
        _n_sub_needed(df, dataset_region), L_freq, c0_parametric,
    )
    return np.asarray(predictor(sv, a))[:, 0, :]


def scenario_predict_mcmc(
    idata,
    F: np.ndarray,
    nl_model_dict: Optional[dict],
    dataset_region: str = "wus",
    func_gs_scaling: str = "stafford",
    const: Optional[ModelConstants] = None,
    amp1d_dataset=None,
    amp1d_target_name: str = "amp_predicted",
    n_draws: Optional[int] = None,
    thin: int = 1,
    seed: int = 0,
    L_freq=None,
    c0_parametric: bool = False,
    quantiles: Sequence[float] = (0.05, 0.95),
    scenario_chunk_size: int = 100,
    return_draws: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Posterior summaries of ln(median) for all combinations of the supplied
    scenario variables (same Cartesian grid as `scenario_predict`).

    The scenario grid is processed in blocks of `scenario_chunk_size` rows
    (all draws at once within a block), so memory is roughly
    n_post * scenario_chunk_size * n_freq floats per intermediate array;
    lower it for many draws / large grids. Quantiles are exact.

    Parameters
    ----------
    quantiles : sequence of float
        Reported as keys 'q05', 'q95', ... in the output.
    return_draws : bool
        Also return the full (n_post, n_scenarios, n_freq) array -- can be big.

    Other parameters as in `single_scenario_predict_mcmc`.

    Returns
    -------
    df_scenarios : DataFrame, shape (n_scenarios, n_vars)
    summary : dict of ndarray, each (n_scenarios, n_freq), on the ln scale
        keys: 'mean', 'median', 'std', and one 'qXX' per entry of `quantiles`.
    draws : ndarray (n_post, n_scenarios, n_freq), only if return_draws=True
    """
    const = const if const is not None else ModelConstants()
    sv = _as_site_values(idata, n_draws, thin, seed)
    df = _build_scenarios(kwargs)
    predictor = _make_predictor(
        F, nl_model_dict, dataset_region, func_gs_scaling, const,
        _n_sub_needed(df, dataset_region), L_freq, c0_parametric,
    )

    q_keys = [f"q{int(round(100 * q)):02d}" for q in quantiles]
    parts = {k: [] for k in ["mean", "median", "std"] + q_keys}
    draws_all = []

    for start in range(0, len(df), scenario_chunk_size):
        df_chunk = df.iloc[start:start + scenario_chunk_size]
        a = _scenario_arrays(df_chunk, F, dataset_region, amp1d_dataset, amp1d_target_name)
        y = np.asarray(predictor(sv, a))                 # (n_post, chunk, n_freq)
        parts["mean"].append(y.mean(axis=0))
        parts["median"].append(np.median(y, axis=0))
        parts["std"].append(y.std(axis=0, ddof=1) if y.shape[0] > 1 else np.zeros(y.shape[1:]))
        for q, key in zip(quantiles, q_keys):
            parts[key].append(np.quantile(y, q, axis=0))
        if return_draws:
            draws_all.append(y)

    summary = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    if return_draws:
        return df, summary, np.concatenate(draws_all, axis=1)
    return df, summary
