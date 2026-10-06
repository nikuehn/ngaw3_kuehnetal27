"""
Calibration ("partitioning") of an already-fitted PSA model -- parametric
or NN -- against observed PSA data: estimates event terms (deltaB), site
terms (deltaS), an optional regional random effect (c_region), an
overall per-frequency additive correction (c_add), and the associated
variance components (tau, phi_ss, phi_s2s, tau_attn), plus the model's
own residual misfit (nu_rec, a StudentT df).

Design: split the median into a FIXED part and a PER-ITERATION part.

FIXED (computed once, outside the SVI loop -- never depends on anything
sampled here, only on already-fitted coefficients/weights and the data):
    median_ref       -- median at Vs30=800, ref_basin_id, no random
                         effects, no NL amplification, no hanging-wall
                         term (see below).
    site_adjustment  -- median_site - median_ref, holding M/R/... fixed
                         at their actual per-record values -- the
                         combined Vs30 + basin correction.

PER-ITERATION (must be recomputed every SVI step, since it depends on
sampled random effects):
    c_add, c_hw, deltaB, deltaS, deltaB_attn, c_region, and f_nl
    (nonlinear site amplification depends on the random-effect-adjusted
    median).

Hanging wall (c_hw) is estimated fresh here, NOT taken from the fitted
PSA model's own coefficients -- `sample_scenarios` (EAS side) only ever
generates footwall geometry (see `FOOTWALL_GEOMETRY`), so neither the
parametric PSA model nor the NN ever saw real hanging-wall variation;
whatever "c_hw" the PSA fit itself produced is confounded with c_0, not
a real effect. Real observed data does include hanging-wall records, so
c_hw needs a genuine estimate here, on top of median_ref/site_adjustment
for either precompute path.

`compute_median_ref_and_site_adjustment_parametric` and `..._nn` both
just produce `(median_ref, site_adjustment)` -- `model_psa_calibration`
doesn't know or care which one produced them, since for the NN there's
no separable Vs30/basin term to keep apart (two NN evaluations and
their difference stand in for the parametric model's vs_term +
c_basin_gathered sum).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from patsy import dmatrix

from ngaw3_kuehnetal27.spline_coeff import make_spline_coeff
from ngaw3_kuehnetal27.spline_coeff_guide import make_spline_coeff_guide
from ngaw3_kuehnetal27.median_core import (
    Coefficients, EventParams, ModelConstants, SiteParams, predict_median,
)
from ngaw3_kuehnetal27.utils import smooth_trilinear_ramp_repar, calculate_hw_scaling
from ngaw3_kuehnetal27.site_amplification.nl_models import (
    compute_ln_amplification_hashash_new,
    compute_ln_amplification_mahdi,
)
from ngaw3_kuehnetal27.psa_generation.median_psa import resolve_vs_coefficient
from ngaw3_kuehnetal27.psa_generation.nn_psa_model import predict as predict_nn

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Precompute: parametric model
# ---------------------------------------------------------------------------

def compute_median_ref_and_site_adjustment_parametric(
    R: Array, Rx: Array, Ry0: Array, F: Array,
    evt_by_eq: EventParams,      # (n_eq,) -- M_model here IS M_eq, no mag. uncertainty
    site_by_stat: SiteParams,    # (n_stat,)
    eq_id: Array, stat_id: Array, basin_id: Array,  # (n_records,)
    coef: Coefficients,            # c_0, c_m1..c_m3, c_zt, c_gs1, c_nm, c_rev, c_attn,
                                    # c_nft*, gs_break/zt_break/gs_exp -- from summary_stats
                                    # (fixed). c_vs AND c_hw are ignored (zeroed internally):
                                    # c_hw is excluded from median_ref -- see module docstring.
    const: ModelConstants,
    vs_hinge_params: dict,        # c_vs_meas_ic/sl1/sl2/br, c_vs_est_ic/sl1/sl2/br
    c_basin_table: Array,         # (n_basin, n_freq)
    ref_vs30: float = 800.0,
    func_gs_scaling: str = "stafford",
):
    """
    Returns
    -------
    median_ref : Array, shape (n_records, n_freq)
    site_adjustment : Array, shape (n_records, n_freq)
        vs_term (continuous, magnitude-dependent -- see
        `resolve_vs_coefficient`) + c_basin_table[basin_id].
    """
    n_records = eq_id.shape[0]

    evt = EventParams(
        M_model=evt_by_eq.M_model[eq_id],
        Dip_eq=evt_by_eq.Dip_eq[eq_id],
        FW_eq=evt_by_eq.FW_eq[eq_id],
        Zt_eq=evt_by_eq.Zt_eq[eq_id],
        Zt_eq_scaled=evt_by_eq.Zt_eq_scaled[eq_id],
        Fnm_eq=evt_by_eq.Fnm_eq[eq_id],
        Frev_eq=evt_by_eq.Frev_eq[eq_id],
    )
    vs_measured_id_rec = site_by_stat.vs_measured_id[stat_id]

    # lnVS=0 at the reference Vs30 -- coef.c_vs * lnVS is zero here no
    # matter what coef.c_vs is, so a (n_freq,) zero placeholder is fine.
    site_ref = SiteParams(
        VS_stat=jnp.full(n_records, ref_vs30),
        lnVS=jnp.zeros(n_records),
        vs_measured_id=vs_measured_id_rec,
    )
    R_scaled = R / 100.0

    # c_hw is deliberately excluded here -- estimated fresh in
    # model_psa_calibration instead (see module docstring).
    coef = replace(coef, c_hw=jnp.zeros_like(coef.c_hw))

    median_ref, _ = predict_median(
        R, Rx, Ry0, F, R_scaled, evt, site_ref, coef, const,
        func_gs_scaling=func_gs_scaling, nl_model_dict=None,
    )

    vs_term = resolve_vs_coefficient(
        "continuous", vs_measured_id_rec, evt.M_model, **vs_hinge_params,
    )
    site_adjustment = vs_term + c_basin_table[basin_id]

    return median_ref, site_adjustment


# ---------------------------------------------------------------------------
# Precompute: NN model
# ---------------------------------------------------------------------------

def compute_median_ref_and_site_adjustment_nn(
    model, scaler, df_records: pd.DataFrame,
    ref_vs30: float = 800.0, ref_basin_id: int = 1,
):
    """
    NN counterpart to `compute_median_ref_and_site_adjustment_parametric`.
    `df_records` needs the columns `nn_psa_model.build_features` expects
    (M, R, Z, VS, Frev, Fnm, vsmeas_id, basin_id, and subregion_id if
    `model.include_region`).

    The NN has no separable Vs30/basin term, so `site_adjustment` here
    is a finite difference (site prediction minus reference prediction,
    holding every other predictor fixed at its actual value) rather than
    a sum of two named terms -- but `model_psa_calibration` only ever
    needs the sum, so this is a drop-in equivalent of the parametric
    version above.

    Returns
    -------
    median_ref : Array, shape (n_records, n_freq)
    site_adjustment : Array, shape (n_records, n_freq)
    """
    df_ref = df_records.copy()
    df_ref["VS"] = ref_vs30
    df_ref["basin_id"] = ref_basin_id

    median_ref = predict_nn(model, scaler, df_ref)
    median_site = predict_nn(model, scaler, df_records)

    return jnp.asarray(median_ref), jnp.asarray(median_site - median_ref)


# ---------------------------------------------------------------------------
# Calibration model / guide
# ---------------------------------------------------------------------------

def _resolve_f_nl(median_before_nl, VS_stat_rec, M_rec, nl_model_dict):
    nl_model_name = nl_model_dict["model"]
    if nl_model_name == "mahdi":
        return compute_ln_amplification_mahdi(
            jnp.exp(median_before_nl), VS_stat_rec, M_rec,
            nl_model_dict["interp_params"], nl_model_dict["interp_nonlin"],
        )
    elif nl_model_name == "hashash_new":
        return compute_ln_amplification_hashash_new(
            jnp.exp(median_before_nl), VS_stat_rec, nl_model_dict["interp_hashash"],
        )
    else:
        raise ValueError(f"Unknown nl_model: {nl_model_name!r}")


def model_psa_calibration(
    F, eq_id, stat_id, subregion_id, R, Rx, Ry0, VS_stat, vs_measured_id_stat,
    M_eq, Dip_eq, FW_eq, Zt_eq,
    median_ref, site_adjustment,
    nl_model_dict,
    Y=None,
    attn_eq=True,
    include_region=True,
    spline_degree=3, spline_df=7,
    calc_dWS=False, save_f_nl=False, calc_log_lik=False,
    L_freq=None,
):
    """
    Parameters
    ----------
    F : array, shape (n_freq,)
    eq_id, stat_id, subregion_id, R, Rx, Ry0 : array, shape (n_records,)
        `subregion_id` is ignored if `include_region=False`. Rx, Ry0 are
        record-level (station-to-fault geometry) -- unlike M/Dip/FW/Zt,
        which are per-event.
    VS_stat, vs_measured_id_stat : array, shape (n_stat,)
        `vs_measured_id_stat` (0=measured, 1=estimated) selects the
        phi_s2s category per station.
    M_eq, Dip_eq, FW_eq, Zt_eq : array, shape (n_eq,)
        Used only for the hanging-wall term (`c_hw`, estimated fresh
        here -- see module docstring) and, for M_eq, the NL model input
        and tau/phi_ss magnitude dependence.
    median_ref, site_adjustment : array, shape (n_records, n_freq)
        See `compute_median_ref_and_site_adjustment_parametric`/`..._nn`.
    nl_model_dict : dict
        {'model': 'mahdi'|'hashash_new', plus that model's interpolated
        coefficient dicts} -- required here (unlike the fitting models,
        calibration always needs a real NL model to add back on top of
        the linear-site median_ref/site_adjustment).
    Y : array, shape (n_records, n_freq), optional
    include_region : bool
        Whether to estimate a regional random effect (c_region) on top
        of event/site terms. Only meaningful if the underlying model
        (sampling and/or the PSA fit) was itself built with regions;
        keeping that consistent is the caller's responsibility.
    """
    n_records = eq_id.shape[0]
    n_eq = int(np.max(eq_id)) + 1
    n_stat = int(np.max(stat_id)) + 1
    n_freq = len(F)
    R_scaled = R / 100.0

    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    spline_basis = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )

    const = ModelConstants()
    mu_freq = jnp.zeros(n_freq)
    L_freq = L_freq if L_freq is not None else jnp.eye(n_freq)

    c_add = make_spline_coeff(spline_basis, "c_add", mu_loc=0.0, mu_scale=5.0)
    c_hw = make_spline_coeff(spline_basis, "c_hw", mu_loc=0.5, mu_scale=0.5)

    # --- standard deviations / random effects ---
    phi_s2s_meas = make_spline_coeff(spline_basis, "phi_s2s_meas", mu_loc=-0.7, mu_scale=0.5,
                                      positive=True, transform="softplus")
    phi_s2s_est = make_spline_coeff(spline_basis, "phi_s2s_est", mu_loc=-0.7, mu_scale=0.5,
                                     positive=True, transform="softplus")
    phi_s2s = jnp.stack([phi_s2s_meas, phi_s2s_est])

    if attn_eq:
        tau_attn = make_spline_coeff(spline_basis, "tau_attn", mu_loc=-0.7, mu_scale=0.5,
                                      positive=True, transform="softplus")

    phi_ss_0 = make_spline_coeff(spline_basis, "phi_ss_0", mu_loc=-0.7, mu_scale=0.5)
    phi_ss_1 = make_spline_coeff(spline_basis, "phi_ss_1", mu_loc=-0.7, mu_scale=0.5)
    phi_ss = jnp.exp(smooth_trilinear_ramp_repar(
        M_eq[eq_id][:, jnp.newaxis], phi_ss_0, phi_ss_1, const.mb1, const.mb2, delta=0.2,
    ))

    mag_sd = jnp.linspace(4, 7.5, 15)
    numpyro.deterministic("phi_ss_mag", jnp.exp(smooth_trilinear_ramp_repar(
        mag_sd[:, jnp.newaxis], phi_ss_0, phi_ss_1, const.mb1, const.mb2, delta=0.2,
    )))

    tau_0 = make_spline_coeff(spline_basis, "tau_0", mu_loc=-0.7, mu_scale=0.5)
    tau_1 = make_spline_coeff(spline_basis, "tau_1", mu_loc=-0.7, mu_scale=0.5)
    tau = jnp.exp(smooth_trilinear_ramp_repar(
        M_eq[:, jnp.newaxis], tau_0, tau_1, const.mb1, const.mb2, delta=0.2,
    ))
    numpyro.deterministic("tau_mag", jnp.exp(smooth_trilinear_ramp_repar(
        mag_sd[:, jnp.newaxis], tau_0, tau_1, const.mb1, const.mb2, delta=0.2,
    )))

    # --- regional random effect (optional -- see docstring) ---
    if include_region:
        n_region = int(np.max(subregion_id)) + 1
        sigma_region = make_spline_coeff(spline_basis, "sigma_region", mu_loc=-0.7, mu_scale=0.5,
                                          positive=True, transform="softplus")
        L_region = sigma_region[..., None] * L_freq
        with numpyro.plate("plate_region", n_region, dim=-1):
            c_region = numpyro.sample("c_region", dist.MultivariateNormal(loc=mu_freq, scale_tril=L_region))

    with numpyro.plate("plate_freq", n_freq, dim=-1):
        nu_rec = numpyro.sample("nu_rec", dist.Gamma(2, 0.1))

        with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
            deltaS = numpyro.sample("deltaS", dist.TransformedDistribution(
                dist.Normal(0, 1), dist.transforms.AffineTransform(0, phi_s2s[vs_measured_id_stat]),
            ))

        with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
            deltaB = numpyro.sample("deltaB", dist.TransformedDistribution(
                dist.Normal(0, 1), dist.transforms.AffineTransform(0, tau),
            ))
            if attn_eq:
                deltaB_attn = numpyro.sample("deltaB_attn", dist.TransformedDistribution(
                    dist.Normal(0, 1), dist.transforms.AffineTransform(0, tau_attn),
                ))
            else:
                deltaB_attn = jnp.zeros((n_eq, n_freq))

    # --- per-iteration median assembly ---
    M_rec = M_eq[eq_id]
    hw_term = calculate_hw_scaling(M_rec, Dip_eq[eq_id], FW_eq[eq_id], Rx, Zt_eq[eq_id], Ry0)

    median_before_nl = median_ref + c_add + c_hw * hw_term[:, jnp.newaxis] + deltaB[eq_id] + deltaS[stat_id]
    if include_region:
        median_before_nl = median_before_nl + c_region[subregion_id]
    if attn_eq:
        median_before_nl = median_before_nl + deltaB_attn[eq_id] * R_scaled[:, jnp.newaxis]

    VS_stat_rec = VS_stat[stat_id]
    f_nl = _resolve_f_nl(median_before_nl, VS_stat_rec, M_rec, nl_model_dict)

    median = median_before_nl + f_nl + site_adjustment

    if save_f_nl:
        numpyro.deterministic("f_nl", f_nl)
    if calc_dWS:
        numpyro.deterministic("deltaWS", Y - median)

    if Y is not None:
        obs_mask = ~np.isnan(Y)
        Y_obs = Y[obs_mask]
        median_obs = median[obs_mask]
        scale_obs = phi_ss[obs_mask]
        df_obs = (jnp.ones((n_records, n_freq)) * nu_rec)[obs_mask]

        if calc_log_lik:
            with numpyro.plate("data", n_records):
                numpyro.deterministic(
                    "obs_log_lik",
                    dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs).log_prob(Y_obs),
                )
    else:
        Y_obs = None
        median_obs = median
        scale_obs = phi_ss
        df_obs = jnp.ones((n_records, n_freq)) * nu_rec

    numpyro.sample("obs", dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs), obs=Y_obs)


def guide_psa_calibration(
    F, eq_id, stat_id, subregion_id, R, Rx, Ry0, VS_stat, vs_measured_id_stat,
    M_eq, Dip_eq, FW_eq, Zt_eq,
    median_ref, site_adjustment,
    nl_model_dict,
    Y=None,
    attn_eq=True,
    include_region=True,
    spline_degree=3, spline_df=7,
    calc_dWS=False, save_f_nl=False, calc_log_lik=False,
    L_freq=None,
):
    """Guide for `model_psa_calibration` -- same arguments, same rules
    (must match the model's attn_eq/include_region on every call)."""
    n_eq = int(np.max(eq_id)) + 1
    n_stat = int(np.max(stat_id)) + 1
    n_freq = len(F)

    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    spline_basis = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )

    make_spline_coeff_guide(spline_basis, "c_add", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_hw", monotonic=None, init_mu=0.5)

    make_spline_coeff_guide(spline_basis, "phi_s2s_meas", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "phi_s2s_est", monotonic=None, init_mu=-0.7)

    if attn_eq:
        make_spline_coeff_guide(spline_basis, "tau_attn", monotonic=None, init_mu=-0.7)

    make_spline_coeff_guide(spline_basis, "phi_ss_0", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "phi_ss_1", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "tau_0", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "tau_1", monotonic=None, init_mu=-0.7)

    if include_region:
        n_region = int(np.max(subregion_id)) + 1
        make_spline_coeff_guide(spline_basis, "sigma_region", monotonic=None, init_mu=-0.7)
        with numpyro.plate("plate_region", n_region, dim=-1):
            numpyro.sample(
                "c_region",
                dist.MultivariateNormal(
                    loc=numpyro.param("loc_c_region", jnp.zeros((n_region, n_freq))),
                    scale_tril=numpyro.param(
                        "scale_tril_c_region",
                        jnp.tile(jnp.eye(n_freq) * 0.2, (n_region, 1, 1)),
                        constraint=dist.constraints.lower_cholesky,
                    ),
                ),
            )

    with numpyro.plate("plate_freq", n_freq, dim=-1):
        numpyro.sample("nu_rec", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_nu_rec", 6.0 * jnp.ones(n_freq))),
            transforms=dist.transforms.ExpTransform(),
        ))

        with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
            numpyro.sample("deltaS", dist.Normal(
                loc=numpyro.param("loc_deltaS", jnp.zeros((n_stat, n_freq))),
                scale=numpyro.param("scale_deltaS", 0.2 * jnp.ones((n_stat, n_freq)),
                                     constraint=dist.constraints.positive),
            ))

        with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
            numpyro.sample("deltaB", dist.Normal(
                loc=numpyro.param("loc_deltaB", jnp.zeros((n_eq, n_freq))),
                scale=numpyro.param("scale_deltaB", 0.2 * jnp.ones((n_eq, n_freq)),
                                     constraint=dist.constraints.positive),
            ))
            if attn_eq:
                numpyro.sample("deltaB_attn", dist.Normal(
                    loc=numpyro.param("loc_deltaB_attn", jnp.zeros((n_eq, n_freq))),
                    scale=numpyro.param("scale_deltaB_attn", 0.2 * jnp.ones((n_eq, n_freq)),
                                         constraint=dist.constraints.positive),
                ))
