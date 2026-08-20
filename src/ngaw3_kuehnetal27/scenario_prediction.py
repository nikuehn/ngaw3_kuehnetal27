"""
Scenario-grid median prediction from fitted SVI results -- analogous to
the NN `scenario_predict` (same DEFAULTS + itertools.product pattern),
but reading coefficients from resolved SVI site values (see
`svi_fitting.resolve_svi_site_values`) instead of a trained NN, and with
a WUS-vs-global choice for which dataset's coefficients to use.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

from ngaw3_kuehnetal27.median_core import (
    Coefficients,
    EventParams,
    ModelConstants,
    RecordIndex,
    SiteParams,
    calculate_median_training,
    predict_median_categorical,
)

# Defaults for every scenario variable. Names follow the GMM's own
# convention (M, R, Rx, Ry0, Z for Ztor, VS, ...) rather than the NN
# version's, since these map straight onto EventParams/SiteParams.
DEFAULTS: Dict[str, Any] = {
    "M": 6.5,
    "R": 30.0,
    "Rx": 0.0,
    "Ry0": 0.0,
    "Z": 3.0,       # Ztor
    "VS": 800.0,
    "Frev": 0,
    "Fnm": 0,
    "Dip": 90.0,
    "FW": 10.0,     # fault width; must be > 0 -- R1 = FW * cos(dip) divides the hanging-wall term
    "subregion_id": 0,   # geology subregion -- WUS only, ignored for dataset_region='global'
    "basin_id": 1,       # WUS only, ignored for dataset_region='global'
    "vsmeas_id": 0,      # 0 = measured, 1 = estimated (WUS); only category 0 exists for global
}

# Bounds for random scenario sampling. Only variables that get sampled;
# anything else falls back to DEFAULTS or FOOTWALL_GEOMETRY.
SAMPLE_BOUNDS: Dict[str, tuple] = {
    "M": (4.0, 8.0),
    "R": (0.0, 300.0),
    "Z": (0.0, 25.0),
    "VS": (200.0, 1100.0),
}

# Footwall-only geometry -- hanging-wall sites are excluded (harder to
# sample consistently, and poorly constrained); add a separate
# hanging-wall term on top later if needed.
FOOTWALL_GEOMETRY: Dict[str, float] = {
    "Dip": 90.0,
    "FW": 60.0,
    "Rx": -20.0,
    "Ry0": 0.0,
}


def coefficients_from_site_values(
    site_values: Dict[str, Any],
    dataset_region: str = "wus",
    gl_suffix: str = "_gl",
) -> Coefficients:
    """
    Build a `Coefficients` object from resolved SVI site values (e.g.
    `resolve_svi_site_values` / `write_svi_results` output).

    Parameters
    ----------
    site_values : dict
        Must contain c_0, c_m1, c_m2, c_m3, c_hw, c_nft_1, c_nft_2,
        c_nm, c_rev, c_gs1, c_zt, c_vs_meas, c_vs_est (and their _gl
        counterparts, if fit with global data and dataset_region='global').
    dataset_region : {'wus', 'global'}
        Which dataset's coefficients to use for every coefficient that
        has a dataset-local version. Coefficients that were 'shared'
        (see coefficient_sharing.py) have no _gl entry at all and are
        used as-is regardless of `dataset_region`.
    gl_suffix : str

    Returns
    -------
    Coefficients

    Notes/limitations
    ------------------
    - Attenuation: if 'c_attn' (or, for global, 'c_attn_gl') is present
      in `site_values`, attn_mode='c_attn' is used. Otherwise falls back
      to Q_0/Q_exp ('Q' mode). `dist_cell` attenuation mode isn't
      supported here -- it needs per-path cell distances that don't fit
      a scenario grid; raise if that's what you actually fit.
    - gs_break / gs_exp / zt_break: uses the fitted WUS value if present
      in site_values, else the same literal defaults numpyro_models.py uses
      (50.0, 2.0, 1.5). For dataset_region='global' these are always the
      literal defaults, matching numpyro_models.py (global never estimates them).
    """
    def get(name):
        if dataset_region == "global":
            gl_name = name + gl_suffix
            if gl_name in site_values:
                return jnp.asarray(site_values[gl_name])
        if name not in site_values:
            raise KeyError(
                f"'{name}' not found in site_values (dataset_region={dataset_region!r})."
            )
        return jnp.asarray(site_values[name])

    if dataset_region == "global":
        # Global always uses attn_mode='c_attn' in numpyro_models.py, regardless of
        # what WUS uses -- no Q_0/dist_cell check needed here.
        attn_mode = "c_attn"
        c_attn = get("c_attn")
        Q_0 = Q_exp = None
        c_vs = jnp.stack([get("c_vs_gl")])  # global: single category
        gs_break, gs_exp, zt_break = 50.0, 2.0, 1.5
    else:
        if "c_attn" in site_values:
            attn_mode = "c_attn"
            c_attn = get("c_attn")
            Q_0 = Q_exp = None
        elif "mu_Q_0" in site_values:
            # dist_cell mode: Q_0 is per-cell, shape (n_cell,) -- needs
            # per-path cell distances that don't correspond to any
            # scenario variable here.
            raise NotImplementedError(
                "This model was fit with 'dist_cell' attenuation (per-cell "
                "Q_0) -- scenario prediction isn't supported for that mode, "
                "since it needs per-path cell distances, not just R."
            )
        elif "Q_0" in site_values:
            attn_mode = "Q"
            c_attn = None
            Q_0 = jnp.asarray(site_values["Q_0"])
            Q_exp = jnp.asarray(site_values["Q_exp"])
        else:
            raise NotImplementedError(
                "Could not find 'c_attn' or 'Q_0' in site_values."
            )
        c_vs = jnp.stack([get("c_vs_meas"), get("c_vs_est")])
        gs_break = jnp.asarray(site_values.get("gs_break", 50.0))
        gs_exp = jnp.asarray(site_values.get("gs_exp", 2.0))
        zt_break = jnp.asarray(site_values.get("zt_break", 1.5))

    return Coefficients(
        c_0=get("c_0"), c_m1=get("c_m1"), c_m2=get("c_m2"), c_m3=get("c_m3"),
        c_hw=get("c_hw"), c_nft_1=get("c_nft_1"), c_nft_2=get("c_nft_2"),
        c_nm=get("c_nm"), c_rev=get("c_rev"), c_gs1=get("c_gs1"), c_zt=get("c_zt"),
        c_vs=c_vs, gs_break=gs_break, gs_exp=gs_exp, zt_break=zt_break,
        attn_mode=attn_mode, Q_0=Q_0, Q_exp=Q_exp, c_attn=c_attn,
    )


def compute_deltaWS(
    data_dict: Dict[str, Any],
    site_values: Dict[str, Any],
    dataset_region: str = "wus",
    const: Optional[ModelConstants] = None,
):
    """
    Single-station residuals (deltaWS = Y - median) computed from the
    *location* (posterior mean) of every coefficient and random effect
    -- i.e. the fitted model's point estimate, not a resampled draw.

    Unlike getting `deltaWS` as a `numpyro.deterministic` site from a
    guide trace (which uses one stochastic sample of deltaB/deltaS/
    deltaB_attn), this reuses `calculate_median_training` directly on
    the real training records with `loc_deltaB` etc. from
    `resolve_svi_site_values` -- deterministic given the fit, and
    consistent with everything else built from `site_values` so far
    (`coefficients_from_site_values`, `scenario_predict`).

    Parameters
    ----------
    data_dict : dict
        The same F / X_rec / X_eq / X_stat / X_id / Y (and, for
        dataset_region='global', global_dict) used to fit the model.
    site_values : dict
        Resolved SVI results (`resolve_svi_site_values` /
        `write_svi_results`) -- coefficient values plus the location of
        every random effect, already under the site's own name (e.g.
        `site_values['deltaB']` is `loc_deltaB`, not a sample).
    dataset_region : {'wus', 'global'}
    const : ModelConstants, optional

    Returns
    -------
    Array, shape (n_records, n_freq)
        Y - median. NaN wherever Y is NaN (unobserved record/frequency
        combinations).

    Limitation
    ----------
    Same as `scenario_predict`: doesn't support `dist_cell` attenuation
    mode (`coefficients_from_site_values` raises for it). Unlike
    scenario prediction, this specific function *could* support it --
    the real per-record `dist_cell` array is available in `data_dict`,
    unlike for an arbitrary scenario grid -- ask if you need that.
    """
    const = const if const is not None else ModelConstants()

    if dataset_region == "global":
        X_rec = data_dict["global_dict"]["X_rec"]
        X_eq = data_dict["global_dict"]["X_eq"]
        X_stat = data_dict["global_dict"]["X_stat"]
        X_id = data_dict["global_dict"]["X_id"]
        Y = data_dict["global_dict"]["Y"]
    else:
        X_rec = data_dict["X_rec"]
        X_eq = data_dict["X_eq"]
        X_stat = data_dict["X_stat"]
        X_id = data_dict["X_id"]
        Y = data_dict["Y"]

    F = jnp.asarray(data_dict["F"])

    R, Rjb, Rx, Ry0 = jnp.asarray(X_rec).T
    M_eq, Zt_eq, Fnm_eq, Frev_eq, FW_eq, Dip_eq, M_sd = jnp.asarray(X_eq).T
    VS_stat, VS_stat_sd, vs_meas_id = jnp.asarray(X_stat).T
    vs_measured_id = vs_meas_id.astype(int)

    if dataset_region == "wus":
        eq_id, stat_id, subregion_id, basin_id = jnp.asarray(X_id).T.astype(int)
    else:
        eq_id, stat_id, _, _ = jnp.asarray(X_id).T.astype(int)
        subregion_id = jnp.zeros_like(eq_id)
        basin_id = jnp.zeros_like(eq_id)

    R_scaled = R / 100.0
    Zt_eq_scaled = Zt_eq / 10.0

    # Magnitude uncertainty: use the fitted M_model location if present
    # (dataset_region='wus' and include_mag_unc was on), else raw M_eq --
    # matches model.py, which never applies magnitude uncertainty to global.
    if dataset_region == "wus" and "M_model" in site_values:
        M_model = jnp.asarray(site_values["M_model"])
    else:
        M_model = M_eq

    evt_by_eq = EventParams(
        M_model=M_model, Dip_eq=Dip_eq, FW_eq=FW_eq,
        Zt_eq=Zt_eq, Zt_eq_scaled=Zt_eq_scaled, Fnm_eq=Fnm_eq, Frev_eq=Frev_eq,
    )
    site_by_stat = SiteParams(
        VS_stat=VS_stat, lnVS=jnp.log(VS_stat) - jnp.log(800.0),
        vs_measured_id=vs_measured_id,
    )
    idx = RecordIndex(eq_id=eq_id, stat_id=stat_id, subregion_id=subregion_id, basin_id=basin_id)

    coef = coefficients_from_site_values(site_values, dataset_region=dataset_region)

    gl_suffix = "_gl" if dataset_region == "global" else ""
    deltaB = jnp.asarray(site_values[f"deltaB{gl_suffix}"])
    deltaS = jnp.asarray(site_values[f"deltaS{gl_suffix}"])
    deltaB_attn_key = f"deltaB_attn{gl_suffix}"
    deltaB_attn = (jnp.asarray(site_values[deltaB_attn_key])
                   if deltaB_attn_key in site_values else jnp.zeros_like(deltaB))
    kappa_adj = (jnp.asarray(site_values["kappa_adj"])
                 if dataset_region == "wus" and "kappa_adj" in site_values
                 else None)

    if dataset_region == "global":
        c_basin = jnp.zeros((1, len(F)))
        c_subregion = jnp.zeros((1, len(F)))
    else:
        c_basin = jnp.asarray(site_values["c_basin"])
        c_subregion = jnp.asarray(site_values["c_region"])

    median, _ = calculate_median_training(
        R=R, Rx=Rx, Ry0=Ry0, F=F, R_scaled=R_scaled,
        dist_cell=data_dict.get("dist_cell"),
        idx=idx, evt_by_eq=evt_by_eq, site_by_stat=site_by_stat,
        coef=coef, const=const,
        func_gs_scaling=data_dict.get("func_gs_scaling", "stafford"),
        nl_model_dict=data_dict["nl_model_dict"],
        deltaB=deltaB, deltaS=deltaS, deltaB_attn=deltaB_attn,
        c_basin=c_basin, c_subregion=c_subregion,
        kappa_adj=kappa_adj,
    )

    Y = jnp.asarray(Y)
    return jnp.where(jnp.isnan(Y), jnp.nan, Y - median)

def _predict_from_dataframe(
    df_scenarios: pd.DataFrame,
    site_values: Dict[str, Any],
    F: np.ndarray,
    nl_model_dict: Optional[dict],
    dataset_region: str,
    func_gs_scaling: str,
    const: ModelConstants,
) -> jnp.ndarray:
    """
    Shared core of `scenario_predict` / `sample_scenarios`: given a
    fully-populated `df_scenarios` (every DEFAULTS column present),
    compute ln_median. No grid-building or sampling here.
    """
    coef = coefficients_from_site_values(site_values, dataset_region=dataset_region)

    if dataset_region == "global":
        c_basin_table = jnp.zeros((1, len(F)))
        c_subregion_table = jnp.zeros((1, len(F)))
        basin_id = jnp.zeros(len(df_scenarios), dtype=int)
        subregion_id = jnp.zeros(len(df_scenarios), dtype=int)
    else:
        c_basin_table = jnp.asarray(site_values["c_basin"])
        c_subregion_table = jnp.asarray(site_values["c_region"])
        basin_id = jnp.asarray(df_scenarios["basin_id"].values, dtype=int)
        subregion_id = jnp.asarray(df_scenarios["subregion_id"].values, dtype=int)

    kappa_adj_table = (jnp.asarray(site_values["kappa_region_table"])
                       if dataset_region == "wus" and "kappa_region_table" in site_values
                       else None)

    R = jnp.asarray(df_scenarios["R"].values)
    Rx = jnp.asarray(df_scenarios["Rx"].values)
    Ry0 = jnp.asarray(df_scenarios["Ry0"].values)
    R_scaled = R / 100.0

    evt = EventParams(
        M_model=jnp.asarray(df_scenarios["M"].values),
        Dip_eq=jnp.asarray(df_scenarios["Dip"].values),
        FW_eq=jnp.asarray(df_scenarios["FW"].values),
        Zt_eq=jnp.asarray(df_scenarios["Z"].values),
        Zt_eq_scaled=jnp.asarray(df_scenarios["Z"].values) / 10.0,
        Fnm_eq=jnp.asarray(df_scenarios["Fnm"].values),
        Frev_eq=jnp.asarray(df_scenarios["Frev"].values),
    )
    VS = jnp.asarray(df_scenarios["VS"].values)
    site = SiteParams(
        VS_stat=VS,
        lnVS=jnp.log(VS) - jnp.log(800.0),
        vs_measured_id=jnp.asarray(df_scenarios["vsmeas_id"].values, dtype=int),
    )

    ln_median, _ = predict_median_categorical(
        R, Rx, Ry0, jnp.asarray(F), R_scaled,
        evt, site, coef, const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        c_basin_table=c_basin_table, basin_id=basin_id,
        c_subregion_table=c_subregion_table, subregion_id=subregion_id,
        kappa_adj_table=kappa_adj_table,
    )
    return ln_median


def scenario_predict(
    site_values: Dict[str, Any],
    F: np.ndarray,
    nl_model_dict: Optional[dict],
    dataset_region: str = "wus",
    func_gs_scaling: str = "stafford",
    const: Optional[ModelConstants] = None,
    **kwargs,
):
    """
    Generate GMM median predictions for all combinations of the supplied
    scenario variables.

    Parameters
    ----------
    site_values : dict
        Resolved SVI results (e.g. from `resolve_svi_site_values` /
        `write_svi_results`).
    F : array
        Frequencies (Hz) matching the coefficients in `site_values`.
    nl_model_dict : dict
        Interpolated nonlinear site amplification coefficients, same
        structure as passed to `calculate_median_training`.
    dataset_region : {'wus', 'global'}
        Which dataset's coefficients to use.
    func_gs_scaling : {'stafford', other}
    const : ModelConstants, optional
    **kwargs
        Arrays for any subset of scenario variables: M, R, Rx, Ry0, Z,
        VS, Frev, Fnm, Dip, FW, subregion_id, basin_id, vsmeas_id.
        Unspecified variables take their DEFAULTS value.

    Returns
    -------
    df_scenarios : DataFrame, shape (n_combos, n_vars)
    ln_median : Array, shape (n_combos, n_freq)

    Example
    -------
        df_sc, ln_median = scenario_predict(
            site_values, F, nl_model_dict=nl_model_dict,
            M=np.linspace(4, 8, 40), R=np.array([10., 30., 100.]),
        )
    """
    const = const if const is not None else ModelConstants()

    varied_keys = list(kwargs.keys())
    varied_vals = [np.atleast_1d(kwargs[k]) for k in varied_keys]
    grid = list(itertools.product(*varied_vals))
    df_scenarios = pd.DataFrame(grid, columns=varied_keys)

    for col, default in DEFAULTS.items():
        if col not in df_scenarios.columns:
            df_scenarios[col] = default

    ln_median = _predict_from_dataframe(
        df_scenarios, site_values, F, nl_model_dict,
        dataset_region, func_gs_scaling, const,
    )

    freq_cols = {f"f{f:.3f}": np.asarray(ln_median)[:, i] for i, f in enumerate(F)}
    df_pred = pd.concat([df_scenarios, pd.DataFrame(freq_cols)], axis=1)

    return df_pred, ln_median


def sample_scenarios(
    site_values: Dict[str, Any],
    F: np.ndarray,
    nl_model_dict: Optional[dict],
    n_sample: int = 100_000,
    seed: int = 1701,
    dataset_region: str = "wus",
    func_gs_scaling: str = "stafford",
    const: Optional[ModelConstants] = None,
    bounds: Optional[Dict[str, tuple]] = None,
    fixed: Optional[Dict[str, Any]] = None,
    fault_type_logits: Optional[jnp.ndarray] = None,
):
    """
    Randomly sample `n_sample` scenario predictor combinations and
    compute the GMM median EAS for each -- the random-sampling
    counterpart to `scenario_predict`'s Cartesian grid.

    Sampled: M, R, Z (Ztor), VS ~ Uniform(bounds), fault type (informs
    Frev/Fnm), subregion_id, basin_id, vsmeas_id ~ categorical
    (uniform over however many categories `site_values` has fitted).
    Geometry beyond M/R/Z/VS is fixed to `FOOTWALL_GEOMETRY` (footwall
    only -- see module note). Anything else takes its `DEFAULTS` value.

    Parameters
    ----------
    site_values, F, nl_model_dict, dataset_region, func_gs_scaling, const
        Same as `scenario_predict`.
    n_sample : int
    seed : int
    bounds : dict, optional
        Overrides merged into `SAMPLE_BOUNDS`, e.g. {'M': (5.0, 7.5)}.
    fixed : dict, optional
        Overrides merged into `FOOTWALL_GEOMETRY` (Dip, FW, Rx, Ry0),
        or any other scalar that should be held constant instead of
        sampled/defaulted.
    fault_type_logits : array, shape (3,), optional
        Logits for [strike-slip, reverse, normal]; defaults to uniform.

    Returns
    -------
    df_scenarios : DataFrame, shape (n_sample, n_vars)
        Sampled predictor values, one row per scenario.
    ln_median : Array, shape (n_sample, n_freq)
        ln(median EAS) for each sampled scenario.
    """
    const = const if const is not None else ModelConstants()
    bounds_ = {**SAMPLE_BOUNDS, **(bounds or {})}
    fixed_ = {**FOOTWALL_GEOMETRY, **(fixed or {})}

    if dataset_region == "global":
        n_region, n_basin = 1, 1
    else:
        n_region = jnp.asarray(site_values["c_region"]).shape[0]
        n_basin = jnp.asarray(site_values["c_basin"]).shape[0]
    n_vsmeas = 2  # measured vs. estimated -- see coefficients_from_site_values

    if fault_type_logits is None:
        fault_type_logits = jnp.ones(3)

    rng_key = random.key(seed)
    (key_m, key_r, key_z, key_vs, key_ft,
     key_region, key_basin, key_vsmeas) = random.split(rng_key, 8)

    M = random.uniform(key_m, (n_sample,), minval=bounds_["M"][0], maxval=bounds_["M"][1])
    R = random.uniform(key_r, (n_sample,), minval=bounds_["R"][0], maxval=bounds_["R"][1])
    Z = random.uniform(key_z, (n_sample,), minval=bounds_["Z"][0], maxval=bounds_["Z"][1])
    VS = random.uniform(key_vs, (n_sample,), minval=bounds_["VS"][0], maxval=bounds_["VS"][1])

    fault_type = random.categorical(key_ft, fault_type_logits, shape=(n_sample,))
    Frev = jnp.where(fault_type == 1, 1.0, 0.0)
    Fnm = jnp.where(fault_type == 2, 1.0, 0.0)

    subregion_id = random.categorical(key_region, jnp.ones(n_region), shape=(n_sample,))
    basin_id = random.categorical(key_basin, jnp.ones(n_basin), shape=(n_sample,))
    vsmeas_id = random.categorical(key_vsmeas, jnp.ones(n_vsmeas), shape=(n_sample,))

    df_scenarios = pd.DataFrame({
        "M": np.asarray(M), "R": np.asarray(R), "Z": np.asarray(Z), "VS": np.asarray(VS),
        "Frev": np.asarray(Frev), "Fnm": np.asarray(Fnm),
        "subregion_id": np.asarray(subregion_id),
        "basin_id": np.asarray(basin_id),
        "vsmeas_id": np.asarray(vsmeas_id),
    })
    for col, value in fixed_.items():
        df_scenarios[col] = value
    for col, default in DEFAULTS.items():
        if col not in df_scenarios.columns:
            df_scenarios[col] = default

    ln_median = _predict_from_dataframe(
        df_scenarios, site_values, F, nl_model_dict,
        dataset_region, func_gs_scaling, const,
    )
    return df_scenarios, ln_median
