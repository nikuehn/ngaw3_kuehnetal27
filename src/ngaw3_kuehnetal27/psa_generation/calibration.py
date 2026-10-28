"""
Calibration ("partitioning") of an already-fitted PSA model -- parametric
or NN -- against observed PSA data: estimates event terms (deltaB), site
terms (deltaS), an overall per-frequency additive correction (c_add), the
associated variance components (tau, phi_ss, phi_s2s, tau_attn), the
model's own residual misfit (nu_rec, a StudentT df) and, optionally, a
hanging-wall term (c_hw) and a regional random effect (c_region).

The model is parameterized exactly like `model_eas` (numpyro_models.py):
the random effects are sampled on the "raw" scale (deltaB_raw,
deltaS_raw, deltaB_attn_raw, c_region_raw ~ Normal(0, 1), or
ZeroSumNormal(1) with `zerosumnormal=True`) and multiplied by their
standard deviations afterwards. It takes the same input matrices
(X_rec, X_eq, X_stat, X_id) as `model_eas`, as built by
`model_inputs.extract_model_arrays` (unpacked with
`model_inputs.unpack_model_inputs`), so `svi_fitting.write_svi_results`
works for both models.

Design: split the median into a FIXED part and a PER-ITERATION part.

FIXED (computed once, outside the SVI loop -- never depends on anything
sampled here, only on already-fitted coefficients/weights and the data):
    median_ref       -- median at Vs30=800, ref_basin_id, no random
                         effects, no NL amplification (and, if
                         `include_hw`, no hanging-wall term).
    site_adjustment  -- median_site - median_ref, holding M/R/... fixed
                         at their actual per-record values -- the
                         combined Vs30 + basin correction.

PER-ITERATION (must be recomputed every SVI step, since it depends on
sampled random effects):
    c_add, deltaB, deltaS, deltaB_attn, optionally c_hw and c_region, and
    f_nl (nonlinear site amplification depends on the random-effect-
    adjusted median).

What `include_hw` and `include_region` mean
-------------------------------------------
Both flags say "estimate this term HERE, on top of the fixed median".
Set them to False when the term is already part of the PSA model's
predictions -- estimating it again would only soak up whatever misfit is
left, and for c_hw the prior (mean 0.5) is only appropriate when the
median has no hanging-wall term yet.

    * Sampling (`sample_scenarios`, incl. hanging-wall geometry) and the
      PSA fit/NN both contain the term -> include_hw=False.
    * The sampling/NN used `subregion_id` (regional EAS terms are in the
      predictions) -> include_region=False. With True, c_region would
      only estimate the *remaining* regional misfit (e.g. region-
      dependent RVT bias); it is not needed for the calibration.
    * The PSA model does NOT contain the term (e.g. NN trained without
      subregion_id, or without hanging-wall predictors) -> True, so the
      term is estimated from the observed data and not left in the
      event/site terms.

For the parametric precompute use the same `include_hw` value in
`compute_median_ref_and_site_adjustment_parametric` and in
`model_psa_calibration`. The NN precompute cannot remove a hanging-wall
dependence the NN may have; with include_hw=True the NN must not use
hanging-wall predictors.

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

from ngaw3_kuehnetal27.model_inputs import unpack_model_inputs
from ngaw3_kuehnetal27.spline_coeff import make_spline_coeff
from ngaw3_kuehnetal27.spline_coeff_guide import make_spline_coeff_guide
from ngaw3_kuehnetal27.numpyro_guides import _zsn_guide_dist
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


def _spline_basis(F, spline_degree, spline_df):
    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    return dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )


# ---------------------------------------------------------------------------
# Precompute: parametric model
# ---------------------------------------------------------------------------

def compute_median_ref_and_site_adjustment_parametric(
    F: Array, X_rec: Array, X_eq: Array, X_stat: Array, X_id: Array,
    coef: Coefficients,            # c_0, c_m1..c_m3, c_zt, c_gs1, c_nm, c_rev, c_attn,
                                    # c_nft*, gs_break/zt_break/gs_exp -- from summary_stats
                                    # (fixed). c_vs is ignored (lnVS=0 at the reference).
    const: ModelConstants,
    vs_hinge_params: dict,        # c_vs_meas_ic/sl1/sl2/br, c_vs_est_ic/sl1/sl2/br
    c_basin_table: Array,         # (n_basin, n_freq)
    ref_vs30: float = 800.0,
    func_gs_scaling: str = "stafford",
    include_hw: bool = False,
):
    """
    Same input matrices as `model_eas` / `model_psa_calibration`.

    include_hw : bool
        Same flag as in `model_psa_calibration`. True: the calibration
        estimates c_hw, so the PSA model's own c_hw is zeroed in
        `median_ref`. False (default): hanging wall is already in the PSA
        model (sampling included hanging-wall geometry), so its c_hw is
        kept in `median_ref`.

    Returns
    -------
    median_ref : Array, shape (n_records, n_freq)
    site_adjustment : Array, shape (n_records, n_freq)
        vs_term (continuous, magnitude-dependent -- see
        `resolve_vs_coefficient`) + c_basin_table[basin_id].
    """
    inp = unpack_model_inputs(X_rec, X_eq, X_stat, X_id)
    eq_id, stat_id = inp.eq_id, inp.stat_id

    evt = EventParams(
        M_model=inp.M_eq[eq_id],       # M_model IS M_eq, no mag. uncertainty
        Dip_eq=inp.Dip_eq[eq_id],
        FW_eq=inp.FW_eq[eq_id],
        Zt_eq=inp.Zt_eq[eq_id],
        Zt_eq_scaled=inp.Zt_eq[eq_id] / 10.0,
        Fnm_eq=inp.Fnm_eq[eq_id],
        Frev_eq=inp.Frev_eq[eq_id],
    )
    vs_measured_id_rec = inp.vs_measured_id[stat_id]

    # lnVS=0 at the reference Vs30 -- coef.c_vs * lnVS is zero here no
    # matter what coef.c_vs is, so a (n_freq,) zero placeholder is fine.
    site_ref = SiteParams(
        VS_stat=jnp.full(inp.n_rec, ref_vs30),
        lnVS=jnp.zeros(inp.n_rec),
        vs_measured_id=vs_measured_id_rec,
    )
    R_scaled = inp.R / 100.0

    if include_hw:
        coef = replace(coef, c_hw=jnp.zeros_like(coef.c_hw))

    median_ref, _ = predict_median(
        inp.R, inp.Rx, inp.Ry0, F, R_scaled, evt, site_ref, coef, const,
        func_gs_scaling=func_gs_scaling, nl_model_dict=None,
    )

    vs_term = resolve_vs_coefficient(
        "continuous", vs_measured_id_rec, evt.M_model, **vs_hinge_params,
    )
    site_adjustment = vs_term + c_basin_table[inp.basin_id]

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
    F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
    median_ref, site_adjustment,
    Y=None,
    attn_eq=True,
    include_region=False,
    include_hw=False,
    spline_degree=3, spline_df=7,
    calc_dWS=False, save_f_nl=False, calc_log_lik=False, save_ranef=True,
    L_freq=None, zerosumnormal=False,
):
    """
    Parameters
    ----------
    F : array, shape (n_freq,)
    X_rec, X_eq, X_stat, X_id : arrays
        Same matrices (and column layout) as `model_eas`, from
        `model_inputs.extract_model_arrays`. Only the columns the model
        needs are used (R, Rx, Ry0; M_eq, and for `include_hw` Zt/FW/Dip;
        VS_stat, vs_measured_id; the four ids).
    nl_model_dict : dict
        {'model': 'mahdi'|'hashash_new', plus that model's interpolated
        coefficient dicts} -- required here (unlike the fitting models,
        calibration always needs a real NL model to add back on top of
        the linear-site median_ref/site_adjustment).
    median_ref, site_adjustment : array, shape (n_records, n_freq)
        See `compute_median_ref_and_site_adjustment_parametric`/`..._nn`.
    Y : array, shape (n_records, n_freq), optional
    attn_eq : bool
        Event-specific attenuation term (deltaB_attn), as in `model_eas`.
    include_region : bool, default False
        Estimate a regional random effect (c_region) on top of the event
        and site terms. Use False when the sampling/PSA model already
        contains regional terms (e.g. NN trained with `subregion_id`):
        the regional variation is then already in `median_ref`. Use True
        when it does not and the observed data should determine it (also
        see the module docstring).
    include_hw : bool, default False
        Estimate a hanging-wall term (c_hw) here. Use False when the
        sampling and the PSA model already include hanging-wall scaling;
        True when they do not (then c_hw must also be zeroed in
        `median_ref` -- same flag in the parametric precompute).
    L_freq : array, shape (n_freq, n_freq), optional
        Cholesky factor of the frequency correlation of the regional
        effect; identity if None.
    zerosumnormal : bool
        Raw random effects ZeroSumNormal(1) instead of Normal(0, 1), as
        in `model_eas`. Must match `guide_psa_calibration`.
    save_ranef : bool
        Save deltaS, deltaB, deltaB_attn and c_region as deterministic
        sites (as `model_eas` does).
    """
    inp = unpack_model_inputs(X_rec, X_eq, X_stat, X_id)
    eq_id, stat_id, subregion_id = inp.eq_id, inp.stat_id, inp.subregion_id
    n_rec, n_eq, n_stat, n_subregion = inp.n_rec, inp.n_eq, inp.n_stat, inp.n_subregion
    M_eq = inp.M_eq
    vs_measured_id = inp.vs_measured_id
    n_freq = len(F)
    R_scaled = inp.R / 100.0

    spline_basis = _spline_basis(F, spline_degree, spline_df)

    const = ModelConstants()
    mu_freq = jnp.zeros(n_freq)
    L_freq = L_freq if L_freq is not None else jnp.eye(n_freq)

    c_add = make_spline_coeff(spline_basis, "c_add", mu_loc=0.0, mu_scale=5.0)
    if include_hw:
        c_hw = make_spline_coeff(spline_basis, "c_hw", mu_loc=0.5, mu_scale=0.5)

    # --- standard deviations ---
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

    # --- raw random effects (same structure as model_eas) ---
    with numpyro.plate("plate_freq", n_freq, dim=-1):
        nu_rec = numpyro.sample("nu_rec", dist.Gamma(2, 0.1))

        if zerosumnormal:
            # event axis last inside the frequency plate -> (n_freq, n)
            if include_region:
                c_region_raw = numpyro.sample(
                    "c_region_raw", dist.ZeroSumNormal(1.0, event_shape=(n_subregion,)))
            deltaS_raw = numpyro.sample(
                "deltaS_raw", dist.ZeroSumNormal(1.0, event_shape=(n_stat,)))
            deltaB_raw = numpyro.sample(
                "deltaB_raw", dist.ZeroSumNormal(1.0, event_shape=(n_eq,)))
            if attn_eq:
                deltaB_attn_raw = numpyro.sample(
                    "deltaB_attn_raw", dist.ZeroSumNormal(1.0, event_shape=(n_eq,)))
        else:
            if include_region:
                with numpyro.plate("plate_freq_region", n_subregion, dim=-2):
                    c_region_raw = numpyro.sample("c_region_raw", dist.Normal(0.0, 1.0))

            with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
                deltaS_raw = numpyro.sample("deltaS_raw", dist.Normal(0, 1))

            with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
                deltaB_raw = numpyro.sample("deltaB_raw", dist.Normal(0, 1))
                if attn_eq:
                    deltaB_attn_raw = numpyro.sample("deltaB_attn_raw", dist.Normal(0, 1))

    if zerosumnormal:
        # back to (n, n_freq), as in the Normal parameterization
        deltaS_raw = deltaS_raw.T
        deltaB_raw = deltaB_raw.T
        if attn_eq:
            deltaB_attn_raw = deltaB_attn_raw.T
        if include_region:
            c_region_raw = c_region_raw.T

    deltaS = deltaS_raw * phi_s2s[vs_measured_id]
    deltaB = deltaB_raw * tau
    if attn_eq:
        deltaB_attn = deltaB_attn_raw * tau_attn
    else:
        deltaB_attn = jnp.zeros((n_eq, n_freq))

    # --- regional random effect (optional -- see docstring) ---
    if include_region:
        sigma_subregion = make_spline_coeff(spline_basis, "sigma_region", mu_loc=-0.7, mu_scale=0.5,
                                            positive=True, transform="softplus")
        L_subregion = sigma_subregion[..., None] * L_freq
        c_subregion = mu_freq[jnp.newaxis, :] + c_region_raw @ L_subregion.T
    else:
        c_subregion = jnp.zeros((n_subregion, n_freq))

    if save_ranef:
        numpyro.deterministic("deltaS", deltaS)
        numpyro.deterministic("deltaB", deltaB)
        numpyro.deterministic("deltaB_attn", deltaB_attn)
        numpyro.deterministic("c_region", c_subregion)

    # --- per-iteration median assembly ---
    M_rec = M_eq[eq_id]

    median_before_nl = median_ref + c_add + deltaB[eq_id] + deltaS[stat_id]
    if include_hw:
        hw_term = calculate_hw_scaling(
            M_rec, inp.Dip_eq[eq_id], inp.FW_eq[eq_id], inp.Rx, inp.Zt_eq[eq_id], inp.Ry0)
        median_before_nl = median_before_nl + c_hw * hw_term[:, jnp.newaxis]
    if include_region:
        median_before_nl = median_before_nl + c_subregion[subregion_id]
    if attn_eq:
        median_before_nl = median_before_nl + deltaB_attn[eq_id] * R_scaled[:, jnp.newaxis]

    VS_stat_rec = inp.VS_stat[stat_id]
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
        df_obs = (jnp.ones((n_rec, n_freq)) * nu_rec)[obs_mask]

        if calc_log_lik:
            with numpyro.plate("data", n_rec):
                numpyro.deterministic(
                    "obs_log_lik",
                    dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs).log_prob(Y_obs),
                )
    else:
        Y_obs = None
        median_obs = median
        scale_obs = phi_ss
        df_obs = jnp.ones((n_rec, n_freq)) * nu_rec

    numpyro.sample("obs", dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs), obs=Y_obs)


def guide_psa_calibration(
    F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
    median_ref, site_adjustment,
    Y=None,
    attn_eq=True,
    include_region=False,
    include_hw=False,
    spline_degree=3, spline_df=7,
    calc_dWS=False, save_f_nl=False, calc_log_lik=False, save_ranef=True,
    L_freq=None, zerosumnormal=False,
):
    """Guide for `model_psa_calibration` -- same arguments, same rules
    (must match the model's attn_eq/include_region/include_hw/zerosumnormal
    on every call). Site and param names follow `guide_eas`
    (loc_/scale_ of the *_raw sites)."""
    n_eq = np.asarray(X_eq).shape[0]
    n_stat = np.asarray(X_stat).shape[0]
    n_subregion = int(np.max(np.asarray(X_id)[:, 2])) + 1
    n_freq = len(F)

    spline_basis = _spline_basis(F, spline_degree, spline_df)

    make_spline_coeff_guide(spline_basis, "c_add", monotonic=None, init_mu=0.0)
    if include_hw:
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
        make_spline_coeff_guide(spline_basis, "sigma_region", monotonic=None, init_mu=-0.7)

    with numpyro.plate("plate_freq", n_freq, dim=-1):
        numpyro.sample("nu_rec", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_nu_rec", 6.0 * jnp.ones(n_freq))),
            transforms=dist.transforms.ExpTransform(),
        ))

        if zerosumnormal:
            if include_region:
                numpyro.sample("c_region_raw",
                               _zsn_guide_dist("c_region_raw", (n_freq,), n_subregion))
            numpyro.sample("deltaS_raw", _zsn_guide_dist("deltaS_raw", (n_freq,), n_stat))
            numpyro.sample("deltaB_raw", _zsn_guide_dist("deltaB_raw", (n_freq,), n_eq))
            if attn_eq:
                numpyro.sample("deltaB_attn_raw",
                               _zsn_guide_dist("deltaB_attn_raw", (n_freq,), n_eq))
        else:
            if include_region:
                with numpyro.plate("plate_freq_region", n_subregion, dim=-2):
                    numpyro.sample("c_region_raw", dist.Normal(
                        loc=numpyro.param("loc_c_region_raw", jnp.zeros((n_subregion, n_freq))),
                        scale=numpyro.param("scale_c_region_raw", 0.2 * jnp.ones((n_subregion, n_freq)),
                                            constraint=dist.constraints.positive),
                    ))

            with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
                numpyro.sample("deltaS_raw", dist.Normal(
                    loc=numpyro.param("loc_deltaS_raw", jnp.zeros((n_stat, n_freq))),
                    scale=numpyro.param("scale_deltaS_raw", 0.2 * jnp.ones((n_stat, n_freq)),
                                         constraint=dist.constraints.positive),
                ))

            with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
                numpyro.sample("deltaB_raw", dist.Normal(
                    loc=numpyro.param("loc_deltaB_raw", jnp.zeros((n_eq, n_freq))),
                    scale=numpyro.param("scale_deltaB_raw", 0.2 * jnp.ones((n_eq, n_freq)),
                                         constraint=dist.constraints.positive),
                ))
                if attn_eq:
                    numpyro.sample("deltaB_attn_raw", dist.Normal(
                        loc=numpyro.param("loc_deltaB_attn_raw", jnp.zeros((n_eq, n_freq))),
                        scale=numpyro.param("scale_deltaB_attn_raw", 0.2 * jnp.ones((n_eq, n_freq)),
                                             constraint=dist.constraints.positive),
                    ))
