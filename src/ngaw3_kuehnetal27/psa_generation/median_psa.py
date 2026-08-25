"""
Median PSA calculation, built on top of `calculate_median_core` (the EAS
model's own physics layer) rather than duplicating it.

Why this works: `calculate_median_core` only ever uses `coef.c_vs` and
`coef.c_attn` in expressions like `coef.c_vs * site.lnVS[:, newaxis]` and
`coef.c_attn * R_scaled[:, newaxis]`. Neither expression cares whether
`coef.c_vs`/`coef.c_attn` has shape (n_freq,) (one value shared by every
record -- the EAS model's case) or (n_records, n_freq) (a different
value per record -- what magnitude-dependent Vs30/attenuation needs).
So instead of modifying `calculate_median_core`, this module resolves
`c_vs`/`c_attn` to (n_records, n_freq) *before* constructing a
`Coefficients` object, then calls `calculate_median_core` directly.

Also unlike the EAS model, PSA samples have no event/station structure
(each sampled scenario is already a fully-resolved record -- see
`sample_scenarios`), so there's no `calculate_median_training`-style
gather-by-eq_id/stat_id step. `calculate_median_psa` below is used
identically during fitting (called once per training sample inside the
numpyro model) and during scenario prediction (called once per
scenario-grid row) -- unlike the EAS side, PSA doesn't need two separate
entry points for those two cases.

Regions are fixed effects here (not a random effect, unlike EAS's
`c_region`) -- intentional: PSA samples aren't data-limited per region
the way real recordings are, so there's no need for partial pooling.
Identifiability uses a centered (sum-to-zero) parameterization --
`center_region_coefficients` -- rather than a reference category, since
there's no natural "reference region" for simulated scenarios.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import jax.numpy as jnp

from ngaw3_kuehnetal27.median_core import (
    Coefficients,
    EventParams,
    ModelConstants,
    SiteParams,
    calculate_median_core,
)
from ngaw3_kuehnetal27.utils import logistic_hinge

Array = jnp.ndarray

V_REF = 800.0


def center_region_coefficients(c_region_raw: Array) -> Array:
    """
    Sum-to-zero centering for a fixed-effect regional term: subtract the
    across-region mean at each frequency, so `c_region` has exactly
    `n_region` rows (no inserted reference row) and is identifiable
    against `c_0` without picking an arbitrary reference category.

    Parameters
    ----------
    c_region_raw : Array, shape (n_region, n_freq)

    Returns
    -------
    Array, shape (n_region, n_freq)
    """
    return c_region_raw - jnp.mean(c_region_raw, axis=0, keepdims=True)


def resolve_vs_coefficient(
    vsmag: str,
    vs_measured_id: Array,          # (n_records,)
    magnitudes: Array,              # (n_records,)
    mag_bin: Optional[Array] = None,        # (n_records,), required if vsmag == "magbin"
    c_vs_magbin: Optional[Array] = None,    # (2, n_magbin, n_freq), required if vsmag == "magbin"
    c_vs_meas_ic: Optional[Array] = None,   # (n_freq,), required if vsmag == "continuous"
    c_vs_meas_sl1: Optional[Array] = None,
    c_vs_meas_sl2: Optional[Array] = None,
    c_vs_meas_br: Optional[Array] = None,   # scalar
    c_vs_est_ic: Optional[Array] = None,
    c_vs_est_sl1: Optional[Array] = None,
    c_vs_est_sl2: Optional[Array] = None,
    c_vs_est_br: Optional[Array] = None,    # scalar
    delta: float = 0.2,
) -> Array:
    """
    Resolve the Vs30-scaling coefficient (to be multiplied by lnVS
    downstream, in `calculate_median_core`) to shape (n_records, n_freq),
    letting it depend on magnitude either through discrete bins or a
    continuous logistic-hinge break.

    Parameters
    ----------
    vsmag : {'magbin', 'continuous'}
    vs_measured_id : Array, int, shape (n_records,)
        0 = measured, 1 = estimated.

    'magbin' mode
        c_vs_magbin[vs_measured_id, mag_bin] is gathered directly.
    'continuous' mode
        c_meas = c_vs_meas_ic + logistic_hinge(M, c_vs_meas_sl1,
                 c_vs_meas_sl2, c_vs_meas_br, delta), and likewise for
        c_est; then selected per record by vs_measured_id.

    Returns
    -------
    Array, shape (n_records, n_freq)
    """
    if vsmag == "magbin":
        if c_vs_magbin is None or mag_bin is None:
            raise ValueError("vsmag='magbin' requires c_vs_magbin and mag_bin")
        return c_vs_magbin[vs_measured_id, mag_bin]

    elif vsmag == "continuous":
        required = (c_vs_meas_ic, c_vs_meas_sl1, c_vs_meas_sl2, c_vs_meas_br,
                    c_vs_est_ic, c_vs_est_sl1, c_vs_est_sl2, c_vs_est_br)
        if any(v is None for v in required):
            raise ValueError("vsmag='continuous' requires all c_vs_meas_*/c_vs_est_* args")

        M = magnitudes[:, jnp.newaxis]
        c_meas = c_vs_meas_ic + logistic_hinge(M, c_vs_meas_sl1, c_vs_meas_sl2, c_vs_meas_br, delta=delta)
        c_est = c_vs_est_ic + logistic_hinge(M, c_vs_est_sl1, c_vs_est_sl2, c_vs_est_br, delta=delta)
        return jnp.where(vs_measured_id[:, jnp.newaxis] == 1, c_est, c_meas)

    else:
        raise ValueError(f"Unknown vsmag mode: {vsmag!r}")


def resolve_attn_coefficient(
    attnmag: bool,
    mag_bin: Optional[Array] = None,        # (n_records,), required if attnmag
    c_attn_magbin: Optional[Array] = None,  # (n_magbin, n_freq), required if attnmag
    c_attn: Optional[Array] = None,         # (n_freq,), required if not attnmag
) -> Array:
    """
    Resolve the anelastic-attenuation coefficient (multiplied by
    R_scaled downstream) to either (n_records, n_freq) if magnitude-
    dependent, or (n_freq,) (broadcasts the same way) if not.
    """
    if attnmag:
        if c_attn_magbin is None or mag_bin is None:
            raise ValueError("attnmag=True requires c_attn_magbin and mag_bin")
        return c_attn_magbin[mag_bin]
    else:
        if c_attn is None:
            raise ValueError("attnmag=False requires c_attn")
        return c_attn


def calculate_median_psa(
    M: Array, R: Array, Zt: Array, VS: Array,
    Frev: Array, Fnm: Array, Dip: Array, FW: Array, Rx: Array, Ry0: Array,
    F: Array,
    vs_measured_id: Array,
    coef: Coefficients,      # coef.c_vs, coef.c_attn already resolved -- see resolve_*_coefficient
    const: ModelConstants,
    *,
    func_gs_scaling: str = "stafford",
    nl_model_dict: Optional[dict] = None,
    basin_id: Optional[Array] = None,
    c_basin_table: Optional[Array] = None,      # (n_basin, n_freq)
    subregion_id: Optional[Array] = None,
    c_region_table: Optional[Array] = None,     # (n_region, n_freq) -- already centered
):
    """
    Median ln(PSA) for a set of fully-resolved scenario records (no
    event/station gathering -- every row is already its own scenario;
    see module docstring). Used identically for fitting (called once
    per training sample, inside the numpyro model) and for scenario
    prediction (called once per scenario-grid row).

    Parameters
    ----------
    M, R, Zt, VS, Frev, Fnm, Dip, FW, Rx, Ry0 : Array, shape (n_records,)
    F : Array, shape (n_freq,)
    vs_measured_id : Array, int, shape (n_records,)
    coef : Coefficients
        c_vs and c_attn must already be resolved (see
        `resolve_vs_coefficient`/`resolve_attn_coefficient`) -- shape
        (n_records, n_freq), or (n_freq,) if not magnitude-dependent.
        Every other field is a plain (n_freq,) value, same as the EAS
        model.
    nl_model_dict : dict or None, default None
        Passed straight through to `calculate_median_core`. Default is
        no PSA-specific nonlinear site amplification (matches how the
        RVT-converted samples were generated); pass a real PSA NL model
        dict here if/when one exists.
    const : ModelConstants
        Pass `dataclasses.replace(ModelConstants(), c_zt2=<estimated>)`
        for `estimate_zt2=True`; the fixed default (c_zt2=0.0) otherwise.
    basin_id, c_basin_table : optional
        Pass together for a basin adjustment; omit both for none.
    subregion_id, c_region_table : optional
        Pass together for a (centered, fixed-effect) regional
        adjustment; omit both for none. `c_region_table` should already
        be centered -- see `center_region_coefficients`.

    Returns
    -------
    median : Array, shape (n_records, n_freq)
    f_nl : Array, shape (n_records, n_freq)
        Zero unless `nl_model_dict` is given -- returned for the same
        signature as `calculate_median_core`.
    """
    R_scaled = R / 100.0
    lnVS = jnp.log(VS) - jnp.log(V_REF)

    evt = EventParams(
        M_model=M, Dip_eq=Dip, FW_eq=FW, Zt_eq=Zt, Zt_eq_scaled=Zt / 10.0,
        Fnm_eq=Fnm, Frev_eq=Frev,
    )
    site = SiteParams(VS_stat=VS, lnVS=lnVS, vs_measured_id=vs_measured_id)

    c_basin = c_basin_table[basin_id] if c_basin_table is not None else None
    c_subregion_adj = c_region_table[subregion_id] if c_region_table is not None else None

    return calculate_median_core(
        R, Rx, Ry0, F, R_scaled, dist_cell=None,
        evt=evt, site=site, coef=coef, const=const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        c_basin=c_basin,
        c_subregion_adj=c_subregion_adj,
    )
