"""
Median ground-motion calculation, split into two layers:

Layer 1 (physics, `calculate_median_core`)
    Pure math: given a fully-resolved `Coefficients` object (plain
    per-freq values -- no per-record/region indexing left) and optional
    random-effect terms (default zero), compute the median. Knows
    nothing about event/station IDs or subregions.

Layer 2 (`calculate_median_training`)
    Only needed for fitting. Gathers the genuinely per-record quantities
    -- event/station random effects, basin term, subregion term, and
    which Vs30-measured category a station falls in -- using
    `RecordIndex`, then calls Layer 1. `coef` itself needs no gathering:
    it's already a complete, single-dataset set of values.

Prediction (`predict_median`)
    A thin wrapper around Layer 1 for new scenarios: no RecordIndex, no
    random effects (point prediction), just a `Coefficients` object.

Terminology
------------
Two different things have historically both been called "region" here --
worth keeping distinct:

- "data region" -- WUS vs. global (or more datasets, if added later).
  These are entirely separate calls: `numpyro_models.py` builds two independent
  `Coefficients` objects (reusing the same numpyro sample site for
  whichever coefficients should be shared between them, and separate
  sample sites for whichever should be dataset-specific) and calls
  `calculate_median_training` once per dataset. Nothing in this file
  needs to know that split exists -- by the time `Coefficients` reaches
  this module, "shared vs. dataset-specific" has already been resolved
  into a single set of plain values.
- "subregion" -- geology-based regions, estimated only within WUS as a
  random effect (`c_subregion`), always zero for the global dataset
  (which has no subregions). This *is* real per-record indexing, via
  `RecordIndex.subregion_id`, and is the only genuine "local" gather
  left in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax.numpy as jnp

from ngaw3_kuehnetal27.utils import (
    calculate_hw_scaling,
    func_gs,
    func_gs_lh,
    func_mag_gs,
    func_mag_gs_lh,
    logistic_hinge,
    smooth_trilinear_centered,
    smooth_trilinear_ramp,
)
from ngaw3_kuehnetal27.site_amplification.nl_models import (
    compute_ln_amplification_hashash_new,
    compute_ln_amplification_mahdi,
)

Array = jnp.ndarray


@dataclass
class Coefficients:
    """
    A complete, already-resolved set of median-scaling coefficients for
    one dataset (e.g. WUS, or global). Every field is a plain per-freq
    value -- shape (n_freq,), or (n_vs_categories, n_freq) for `c_vs`.

    There is no "shared vs. local" distinction inside this dataclass:
    whether a value was sampled once and reused across datasets, or
    sampled separately per dataset, is decided in `numpyro_models.py` when this
    object is constructed -- not here.

    Attenuation: exactly one of the three representations is active,
    selected by `attn_mode` ('dist_cell', 'Q', or 'c_attn').
    """
    c_0: Array
    c_m1: Array
    c_m2: Array
    c_m3: Array
    c_hw: Array
    c_nft_1: Array
    c_nft_2: Array
    c_nm: Array
    c_rev: Array
    c_gs1: Array
    c_zt: Array
    c_vs: Array          # (n_vs_categories, n_freq) -- gathered by vs_measured_id
    gs_break: Array
    gs_exp: Array
    zt_break: Array
    Q_0: Optional[Array] = None
    Q_exp: Optional[Array] = None
    c_attn: Optional[Array] = None
    attn_mode: str = "c_attn"


@dataclass
class ModelConstants:
    """Fixed constants, not estimated. Shared across all datasets."""
    c_Q: float = 3.5
    delta_zt: float = 0.1
    c_zt2: float = 0.0
    mb1: float = 5.3
    mb2: float = 6.7
    c_gs2: float = -0.5
    xi: float = 10.0
    delta_gs: float = 0.1


@dataclass
class RecordIndex:
    """
    Per-record indices into training-data arrays, all length n_records
    for one dataset (WUS or global -- each dataset gets its own
    RecordIndex with its own local numbering).
    """
    eq_id: Array
    stat_id: Array
    subregion_id: Array  # geology subregion; always 0 for global (zero-effect)
    basin_id: Array


@dataclass
class EventParams:
    M_model: Array
    Dip_eq: Array
    FW_eq: Array
    Zt_eq: Array
    Zt_eq_scaled: Array
    Fnm_eq: Array
    Frev_eq: Array


@dataclass
class SiteParams:
    VS_stat: Array
    lnVS: Array
    vs_measured_id: Array  # category (measured vs. estimated Vs30), per station


def calculate_median_core(
    R: Array,                 # (n_records,)
    Rx: Array,                # (n_records,)
    Ry0: Array,               # (n_records,)
    F: Array,                 # (n_freq,)
    R_scaled: Array,          # (n_records,)
    dist_cell: Optional[Array],  # (n_records, n_cell) or None
    evt: EventParams,
    site: SiteParams,
    coef: Coefficients,
    const: ModelConstants,
    *,
    func_gs_scaling: str = "stafford",
    nl_model_dict: dict,
    deltaB: Optional[Array] = None,        # (n_records, n_freq) or None
    deltaS: Optional[Array] = None,        # (n_records, n_freq) or None
    deltaB_attn: Optional[Array] = None,   # (n_records, n_freq) or None
    c_basin: Optional[Array] = None,       # (n_records, n_freq) or None
    c_subregion_adj: Optional[Array] = None,  # (n_records, n_freq) or None
) -> Tuple[Array, Array]:
    """
    Compute ln(median) for a set of records within one dataset.

    Pure physics: `coef` is a single, fully-resolved `Coefficients`
    object (see module docstring -- the WUS-vs-global / shared-vs-local
    decision has already happened before this is called). Random-effect
    arguments all default to zero, so calling this with none of them set
    gives a point prediction with no event-, station-, or
    subregion-specific adjustment.

    Parameters
    ----------
    R, Rx, Ry0, R_scaled : Array, shape (n_records,)
        Distance measures.
    F : Array, shape (n_freq,)
        Frequencies.
    dist_cell : Array, shape (n_records, n_cell), optional
        Only used if coef.attn_mode == 'dist_cell'.
    evt : EventParams
        Per-record event parameters (already gathered by eq_id if fitting;
        directly supplied per-scenario if predicting).
    site : SiteParams
        Per-record site parameters (already gathered by stat_id if fitting).
    coef : Coefficients
    const : ModelConstants
    func_gs_scaling : {'stafford', 'other'}
    nl_model_dict : dict
        {'model': 'mahdi'|'hashash_new', plus that model's interpolated
        coefficient dicts}.
    deltaB, deltaS, deltaB_attn, c_basin, c_subregion_adj : Array,
    shape (n_records, n_freq), optional
        Random effects / categorical adjustments, already gathered to
        per-record shape. None (the default) is treated as zero -- this
        is what makes point prediction for a new scenario just a call
        with these omitted.

    Returns
    -------
    median : Array, shape (n_records, n_freq)
    f_nl : Array, shape (n_records, n_freq)
        Nonlinear site amplification term, returned separately for
        diagnostics.
    """
    zero = 0.0
    deltaB = zero if deltaB is None else deltaB
    deltaS = zero if deltaS is None else deltaS
    deltaB_attn = zero if deltaB_attn is None else deltaB_attn
    c_basin = zero if c_basin is None else c_basin
    c_subregion_adj = zero if c_subregion_adj is None else c_subregion_adj

    # --- magnitude scaling ramp ---
    f_flt = smooth_trilinear_ramp(evt.M_model, 1.0, 4.5, 5.5)

    # --- hanging wall (uses M_model with smooth thresholds) ---
    hw_term = calculate_hw_scaling(
        evt.M_model, evt.Dip_eq, evt.FW_eq, Rx, evt.Zt_eq, Ry0,
    )

    # --- base median ---
    median = (
        coef.c_0
        + logistic_hinge(
            evt.Zt_eq_scaled[:, jnp.newaxis],
            coef.c_zt, const.c_zt2, coef.zt_break, const.delta_zt,
        )
        + coef.c_nm * f_flt[:, jnp.newaxis] * evt.Fnm_eq[:, jnp.newaxis]
        + coef.c_rev * f_flt[:, jnp.newaxis] * evt.Frev_eq[:, jnp.newaxis]
        + coef.c_hw * hw_term[:, jnp.newaxis]
        + c_subregion_adj
        + deltaB
        + deltaS
    )

    # --- magnitude and geometric spreading ---
    if func_gs_scaling == "stafford":
        median = median + func_mag_gs(
            evt.M_model[:, jnp.newaxis], R[:, jnp.newaxis],
            coef.c_m1, coef.c_m2, coef.c_m3,
            coef.c_gs1, const.c_gs2,
            coef.c_nft_1, coef.c_nft_2,
            const.mb1, const.mb2,
            const.delta_gs, coef.gs_break, const.xi,
        )
    else:
        median = median + func_mag_gs_lh(
            evt.M_model[:, jnp.newaxis], R[:, jnp.newaxis],
            coef.c_m1, coef.c_m2, coef.c_m3,
            coef.c_gs1, const.c_gs2,
            coef.c_nft_1, coef.c_nft_2,
            const.mb1, const.mb2,
            coef.gs_break, coef.gs_exp,
        )

    # --- anelastic attenuation ---
    if coef.attn_mode == "dist_cell":
        qterm = jnp.dot(dist_cell, 1 / coef.Q_0)
        median = median - jnp.pi * F[jnp.newaxis, :] ** coef.Q_exp * qterm[:, jnp.newaxis] / const.c_Q
    elif coef.attn_mode == "Q":
        median = median - jnp.pi * F[jnp.newaxis, :] ** coef.Q_exp * R[:, jnp.newaxis] / (coef.Q_0 * const.c_Q)
    else:  # 'c_attn'
        median = median - coef.c_attn * R_scaled[:, jnp.newaxis]

    # --- event-specific distance slope (zero unless deltaB_attn passed) ---
    median = median + deltaB_attn * R_scaled[:, jnp.newaxis]

    # --- nonlinear site amplification ---
    nl_model_name = nl_model_dict["model"]
    if nl_model_name == "mahdi":
        f_nl = compute_ln_amplification_mahdi(
            jnp.exp(median), site.VS_stat, evt.M_model,
            nl_model_dict["interp_params"], nl_model_dict["interp_nonlin"],
        )
    elif nl_model_name == "hashash_new":
        f_nl = compute_ln_amplification_hashash_new(
            jnp.exp(median), site.VS_stat, nl_model_dict["interp_hashash"],
        )
    else:
        raise ValueError(f"Unknown nl_model: {nl_model_name!r}")

    # --- vs30 scaling ---
    # coef.c_vs must already be resolved to (n_records, n_freq) by
    # vs_measured_id (see calculate_median_training).
    median = median + f_nl + c_basin + coef.c_vs * site.lnVS[:, jnp.newaxis]

    return median, f_nl


def calculate_median_training(
    R: Array, Rx: Array, Ry0: Array, F: Array, R_scaled: Array,
    dist_cell: Optional[Array],
    idx: RecordIndex,
    evt_by_eq: EventParams,      # indexed by eq (n_eq,), not yet gathered
    site_by_stat: SiteParams,    # indexed by station (n_stat,), not yet gathered
    coef: Coefficients,          # fully resolved for this dataset -- see module docstring
    const: ModelConstants,
    *,
    func_gs_scaling: str = "stafford",
    nl_model_dict: dict,
    deltaB: Array,          # (n_eq, n_freq)
    deltaS: Array,          # (n_stat, n_freq)
    deltaB_attn: Array,     # (n_eq, n_freq)
    c_basin: Array,         # (n_basin, n_freq)
    c_subregion: Array,     # (n_subregion, n_freq) -- zeros(1, n_freq) for global
) -> Tuple[Array, Array]:
    """
    Fitting-time wrapper for ONE dataset (call once for WUS, once for
    global if `global_dict` is not None -- that orchestration lives in
    `numpyro_models.py`, not here). Gathers per-record values by index, then
    calls `calculate_median_core`.

    `coef` needs no gathering at all -- it's already the right values
    for this dataset. Only the genuinely per-record/per-category things
    (random effects, basin, subregion, vs_measured_id) get indexed here.
    """
    vs_measured_id_per_record = site_by_stat.vs_measured_id[idx.stat_id]
    resolved_coef = replace(coef, c_vs=coef.c_vs[vs_measured_id_per_record])

    evt = EventParams(
        M_model=evt_by_eq.M_model[idx.eq_id],
        Dip_eq=evt_by_eq.Dip_eq[idx.eq_id],
        FW_eq=evt_by_eq.FW_eq[idx.eq_id],
        Zt_eq=evt_by_eq.Zt_eq[idx.eq_id],
        Zt_eq_scaled=evt_by_eq.Zt_eq_scaled[idx.eq_id],
        Fnm_eq=evt_by_eq.Fnm_eq[idx.eq_id],
        Frev_eq=evt_by_eq.Frev_eq[idx.eq_id],
    )
    site = SiteParams(
        VS_stat=site_by_stat.VS_stat[idx.stat_id],
        lnVS=site_by_stat.lnVS[idx.stat_id],
        vs_measured_id=vs_measured_id_per_record,
    )

    return calculate_median_core(
        R, Rx, Ry0, F, R_scaled, dist_cell,
        evt, site, resolved_coef, const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        deltaB=deltaB[idx.eq_id],
        deltaS=deltaS[idx.stat_id],
        deltaB_attn=deltaB_attn[idx.eq_id],
        c_basin=c_basin[idx.basin_id],
        c_subregion_adj=c_subregion[idx.subregion_id],
    )


def predict_median(
    R: Array, Rx: Array, Ry0: Array, F: Array, R_scaled: Array,
    evt: EventParams,     # per-scenario, no gathering needed
    site: SiteParams,     # per-scenario
    coef: Coefficients,   # single scenario -- c_vs already reduced to (n_freq,)
    const: ModelConstants,
    *,
    func_gs_scaling: str = "stafford",
    nl_model_dict: dict,
    dist_cell: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Point prediction for new scenarios (e.g. hazard-consistent ground
    motion generation): no event/station IDs, no random effects, no
    subregion adjustment.

    This is a thin wrapper: `calculate_median_core` never needed to
    change to support prediction, since random effects there already
    default to zero.
    """
    return calculate_median_core(
        R, Rx, Ry0, F, R_scaled, dist_cell,
        evt, site, coef, const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        # deltaB, deltaS, c_subregion_adj all default to None -> zero
    )


def predict_median_categorical(
    R: Array, Rx: Array, Ry0: Array, F: Array, R_scaled: Array,
    evt: EventParams,     # per-scenario, no gathering needed
    site: SiteParams,     # per-scenario; site.vs_measured_id selects the Vs30 category
    coef: Coefficients,   # coef.c_vs shape (n_vs_categories, n_freq) -- not yet gathered
    const: ModelConstants,
    *,
    func_gs_scaling: str = "stafford",
    nl_model_dict: dict,
    dist_cell: Optional[Array] = None,
    c_basin_table: Optional[Array] = None,      # (n_basin, n_freq)
    basin_id: Optional[Array] = None,           # (n_scenarios,) -- required if c_basin_table given
    c_subregion_table: Optional[Array] = None,  # (n_subregion, n_freq)
    subregion_id: Optional[Array] = None,       # (n_scenarios,) -- required if c_subregion_table given
) -> Tuple[Array, Array]:
    """
    Like `predict_median`, but for scenarios that need to select a
    category rather than assume a single one -- e.g. a scenario grid
    that varies Vs30-measured-vs-estimated, basin, or geology subregion
    across rows. Still a point prediction: no event/station random
    effects (always zero), same as `predict_median`.

    Parameters
    ----------
    coef : Coefficients
        `coef.c_vs` has shape (n_vs_categories, n_freq) -- gathered here
        by `site.vs_measured_id`, same as in `calculate_median_training`.
    c_basin_table, c_subregion_table : Array, optional
        Full per-category tables; pass together with the matching
        `basin_id` / `subregion_id` (length n_scenarios) to look up a
        value per scenario row. Omit both (the default) for no
        basin/subregion adjustment.
    """
    resolved_coef = replace(coef, c_vs=coef.c_vs[site.vs_measured_id])

    c_basin = c_basin_table[basin_id] if c_basin_table is not None else None
    c_subregion_adj = c_subregion_table[subregion_id] if c_subregion_table is not None else None

    return calculate_median_core(
        R, Rx, Ry0, F, R_scaled, dist_cell,
        evt, site, resolved_coef, const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        c_basin=c_basin,
        c_subregion_adj=c_subregion_adj,
    )
