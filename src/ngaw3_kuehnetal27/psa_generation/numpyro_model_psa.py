"""
Numpyro model for the parametric PSA GMM, fit to RVT-converted
sampled-scenario data (see `sample_scenarios`, `duration_pinilla_ramos_
d575`, and the RVT-conversion step). Reuses the EAS model's own median
physics via `calculate_median_psa` (see that module's docstring) rather
than duplicating `calculate_median_core`.

Differences from `numpyro_models.py`'s `model_eas`, all because each
sampled scenario is its own independent record (no shared events or
stations):
  - No random effects (deltaB/deltaS/deltaB_attn) -- nothing to pool.
  - No dist_cell/Q attenuation modes -- always the `c_attn` spline form,
    optionally magnitude-binned (`attnmag`).
  - No magnitude uncertainty, no kappa term.
  - `sigma` (HalfNormal per frequency) is the parametric form's own fit
    residual against the RVT-simulated samples -- NOT the same thing as
    the EAS model's tau/phi_ss/phi_s2s; those describe real recording
    variability and get estimated later, against observations, in the
    calibration step.
  - Regional term (`c_region`) is a fixed effect with sum-to-zero
    centering (`center_region_coefficients`), not a random effect --
    intentional: simulated scenarios aren't data-limited per region the
    way real recordings are, so there's no need for partial pooling.

X_id column order: subregion_id, basin_id, vs_measured_id, mag_bin.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from patsy import dmatrix

from ngaw3_kuehnetal27.spline_coeff import make_spline_coeff
from ngaw3_kuehnetal27.median_core import Coefficients, ModelConstants
from ngaw3_kuehnetal27.utils import (
    insert_zero_row,
    compute_magnitude_gradients,
    compute_magnitude_gradients_lh,
)
from ngaw3_kuehnetal27.psa_generation.median_psa import (
    calculate_median_psa,
    center_region_coefficients,
    resolve_attn_coefficient,
    resolve_vs_coefficient,
)


def model_psa(
    F, X_inp, X_id,
    Y=None,
    ref_basin_id=1,
    regularize_grad=False, regularization_sigma=0.1,
    calc_nft="ya14",
    estimate_gs_break=False, estimate_zt_break=False,
    freq_id_grad=(0, 6, 8, 10, 16, 20, 23),
    spline_degree=3, spline_df=7,
    func_gs_scaling="stafford", estimate_gs_exp="fixed",
    vsmag="magbin", attnmag=False,
    include_region=True, estimate_zt2=False,
    nl_model_dict=None,
):
    """
    Parameters
    ----------
    F : array, shape (n_freq,)
    X_inp : array, shape (n_rec, 10)
        Columns: M, R, Zt, VS, Frev, Fnm, Dip, FW, Rx, Ry0.
    X_id : array, shape (n_rec, 4)
        Columns: subregion_id, basin_id, vs_measured_id, mag_bin.
    Y : array, shape (n_rec, n_freq), optional
        ln(PSA); omit for prior-predictive / prediction use.
    vsmag : {'magbin', 'continuous'}
        See `median_psa.resolve_vs_coefficient`.
    attnmag : bool
        See `median_psa.resolve_attn_coefficient`.
    include_region : bool
        Whether to add the centered regional fixed effect.
    estimate_zt2 : bool
        If True, estimates a second (upper) Ztor-scaling slope
        (`c_zt2`, spline) instead of the EAS model's fixed 0.0.
    nl_model_dict : dict or None
        Passed through to `calculate_median_psa` -- None (default)
        matches how the RVT-converted training samples were generated
        (no PSA-specific nonlinear site amplification).

    n_region/n_basin/n_magbin are all inferred from the data
    (`np.max(id) + 1`), not hardcoded.
    """
    M, R, Zt, VS, Frev, Fnm, Dip, FW, Rx, Ry0 = X_inp.T
    subregion_id, basin_id, vs_measured_id, mag_bin = X_id.T
    vs_measured_id = vs_measured_id.astype("int")
    mag_bin = mag_bin.astype("int")
    subregion_id = subregion_id.astype("int")
    basin_id = basin_id.astype("int")

    n_rec = X_inp.shape[0]
    n_subregion = np.max(subregion_id) + 1
    n_basin = np.max(basin_id) + 1
    n_magbin = np.max(mag_bin) + 1
    n_freq = len(F)

    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    spline_basis = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )

    const = ModelConstants()

    # --- geometric spreading break ---
    if estimate_gs_break:
        gs_break = numpyro.sample("gs_break", dist.Gamma(94.8, 1.9))
    else:
        gs_break = 50.0

    # --- zt_break, and optionally a second (upper) Ztor slope ---
    if estimate_zt_break:
        zt_break = numpyro.sample("zt_break", dist.Gamma(6.94, 5.49))
    else:
        zt_break = 1.5

    if estimate_zt2:
        c_zt2 = make_spline_coeff(spline_basis, "c_zt2", mu_loc=0.0, mu_scale=0.5)
        const = replace(const, c_zt2=c_zt2)
    # else: const.c_zt2 stays at its default (0.0), matching the EAS model.

    c_m3 = numpyro.sample("c_m3", dist.LogNormal(-0.45, 0.8))

    # --- near-fault term ---
    if calc_nft == "ya14":
        c_nft_1 = -1.72 * jnp.log(10) + 0.43 * jnp.log(10) * 4.5
        c_nft_2 = 0.43 * jnp.log(10)
    elif calc_nft == "coeff":
        c_nft_1 = numpyro.sample("c_nft_1", dist.Normal(0.5, 0.5))
        c_nft_2 = numpyro.sample("c_nft_2", dist.LogNormal(0, 0.2))
    elif calc_nft == "freq":
        c_nft_1 = make_spline_coeff(spline_basis, "c_nft_1", mu_loc=0.5, mu_scale=0.5,
                                     monotonic="decreasing")
        c_nft_2 = make_spline_coeff(spline_basis, "c_nft_2", mu_loc=0.0, mu_scale=0.2,
                                     positive=True, transform="softplus", monotonic="increasing")
    else:
        raise ValueError(f"Unknown calc_nft: {calc_nft!r}")

    # --- base median-scaling coefficients ---
    c_0 = make_spline_coeff(spline_basis, "c_0", mu_loc=0.0, mu_scale=5.0)
    c_gs1 = make_spline_coeff(spline_basis, "c_gs1", mu_loc=0.5, mu_scale=0.5,
                               positive=True, transform="softplus")
    c_m1 = make_spline_coeff(spline_basis, "c_m1", mu_loc=1.5, mu_scale=1.0,
                              positive=True, transform="softplus")
    c_m2 = make_spline_coeff(spline_basis, "c_m2", mu_loc=1.0, mu_scale=1.0,
                              positive=True, transform="softplus")
    c_zt = make_spline_coeff(spline_basis, "c_zt", mu_loc=0.0, mu_scale=0.5)
    c_nm = make_spline_coeff(spline_basis, "c_nm", mu_loc=0.0, mu_scale=0.5)
    c_rev = make_spline_coeff(spline_basis, "c_rev", mu_loc=0.0, mu_scale=0.5)
    c_hw = make_spline_coeff(spline_basis, "c_hw", mu_loc=0.5, mu_scale=0.5)

    # --- anelastic attenuation: single spline, or one per magnitude bin ---
    if attnmag:
        c_attn_magbin = jnp.stack([
            make_spline_coeff(spline_basis, f"c_attn_{i}", mu_loc=-1.0, mu_scale=1.0,
                               positive=True, transform="softplus")
            for i in range(n_magbin)
        ])
        c_attn_resolved = resolve_attn_coefficient(True, mag_bin=mag_bin, c_attn_magbin=c_attn_magbin)
    else:
        c_attn = make_spline_coeff(spline_basis, "c_attn", mu_loc=-1.0, mu_scale=1.0,
                                    positive=True, transform="softplus")
        c_attn_resolved = resolve_attn_coefficient(False, c_attn=c_attn)

    # --- vs30 scaling: magnitude-bin categories, or continuous logistic-hinge ---
    if vsmag == "magbin":
        c_vs_meas = jnp.stack([
            make_spline_coeff(spline_basis, f"c_vs_meas_{i}", mu_loc=0.0, mu_scale=1.0)
            for i in range(n_magbin)
        ])
        c_vs_est = jnp.stack([
            make_spline_coeff(spline_basis, f"c_vs_est_{i}", mu_loc=0.0, mu_scale=1.0)
            for i in range(n_magbin)
        ])
        c_vs_magbin = jnp.stack([c_vs_meas, c_vs_est])
        c_vs_resolved = resolve_vs_coefficient(
            "magbin", vs_measured_id, M, mag_bin=mag_bin, c_vs_magbin=c_vs_magbin,
        )
    elif vsmag == "continuous":
        c_vs_meas_ic = make_spline_coeff(spline_basis, "c_vs_meas_ic", mu_loc=-0.5, mu_scale=1.0)
        c_vs_meas_sl1 = make_spline_coeff(spline_basis, "c_vs_meas_sl1", mu_loc=0.0, mu_scale=1.0)
        c_vs_meas_sl2 = make_spline_coeff(spline_basis, "c_vs_meas_sl2", mu_loc=0.0, mu_scale=1.0)
        c_vs_meas_br = numpyro.sample("c_vs_meas_br", dist.Normal(6.0, 0.5))

        c_vs_est_ic = make_spline_coeff(spline_basis, "c_vs_est_ic", mu_loc=-0.5, mu_scale=1.0)
        c_vs_est_sl1 = make_spline_coeff(spline_basis, "c_vs_est_sl1", mu_loc=0.0, mu_scale=1.0)
        c_vs_est_sl2 = make_spline_coeff(spline_basis, "c_vs_est_sl2", mu_loc=0.0, mu_scale=1.0)
        c_vs_est_br = numpyro.sample("c_vs_est_br", dist.Normal(6.0, 0.5))

        c_vs_resolved = resolve_vs_coefficient(
            "continuous", vs_measured_id, M,
            c_vs_meas_ic=c_vs_meas_ic, c_vs_meas_sl1=c_vs_meas_sl1,
            c_vs_meas_sl2=c_vs_meas_sl2, c_vs_meas_br=c_vs_meas_br,
            c_vs_est_ic=c_vs_est_ic, c_vs_est_sl1=c_vs_est_sl1,
            c_vs_est_sl2=c_vs_est_sl2, c_vs_est_br=c_vs_est_br,
        )
    else:
        raise ValueError(f"Unknown vsmag: {vsmag!r}")

    # --- basin term: fixed effect, one reference category zeroed out ---
    c_basin_sample = jnp.stack([
        make_spline_coeff(spline_basis, f"c_b_{i}", mu_loc=0.0, mu_scale=0.5)
        for i in range(n_basin - 1)
    ])
    c_basin = numpyro.deterministic("c_basin", insert_zero_row(ref_basin_id, c_basin_sample))

    # --- regional term: fixed effect, sum-to-zero centered ---
    if include_region:
        c_region_raw = jnp.stack([
            make_spline_coeff(spline_basis, f"c_region_{i}", mu_loc=0.0, mu_scale=0.5)
            for i in range(n_subregion)
        ])
        c_region = numpyro.deterministic("c_region", center_region_coefficients(c_region_raw))
    else:
        c_region = None

    # --- geometric spreading exponent (logistic-hinge form only) ---
    if func_gs_scaling == "stafford":
        gs_exp = 2.0
    else:
        if estimate_gs_exp == "fixed":
            gs_exp = 2.0
        elif estimate_gs_exp == "coeff":
            gs_exp = numpyro.sample("gs_exp", dist.InverseGamma(6.18, 7.91))
        elif estimate_gs_exp == "freq":
            gs_exp = make_spline_coeff(spline_basis, "gs_exp", mu_loc=1.2, mu_scale=0.5,
                                        positive=True, transform="softplus")
        else:
            raise ValueError(f"Unknown estimate_gs_exp: {estimate_gs_exp!r}")

    # --- observation noise: model misfit to the RVT-simulated samples,
    # NOT the real-data tau/phi_ss/phi_s2s (see module docstring) ---
    with numpyro.plate("plate_freq", n_freq, dim=-1):
        nu_rec = numpyro.sample("nu_rec", dist.Gamma(2, 0.1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))

    # --- monotonicity regularization on magnitude scaling ---
    numpyro.factor("reg_c_m2", jnp.where(
        c_m2 > c_m1,
        dist.Normal(0, regularization_sigma).log_prob(c_m2 - c_m1),
        dist.Normal(0, regularization_sigma).log_prob(0),
    ))
    numpyro.factor("reg_c_m3", jnp.where(
        c_m3 > c_m2,
        dist.Normal(0, regularization_sigma).log_prob(c_m3 - c_m2),
        dist.Normal(0, regularization_sigma).log_prob(0),
    ))

    coef = Coefficients(
        c_0=c_0, c_m1=c_m1, c_m2=c_m2, c_m3=c_m3, c_hw=c_hw,
        c_nft_1=c_nft_1, c_nft_2=c_nft_2, c_nm=c_nm, c_rev=c_rev,
        c_gs1=c_gs1, c_zt=c_zt, c_vs=c_vs_resolved,
        gs_break=gs_break, gs_exp=gs_exp, zt_break=zt_break,
        attn_mode="c_attn", c_attn=c_attn_resolved,
    )

    median, f_nl = calculate_median_psa(
        M, R, Zt, VS, Frev, Fnm, Dip, FW, Rx, Ry0, F,
        vs_measured_id, coef, const,
        func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        basin_id=basin_id, c_basin_table=c_basin,
        subregion_id=subregion_id if include_region else None,
        c_region_table=c_region if include_region else None,
    )

    if regularize_grad:
        magnitudes = jnp.linspace(6, 8, 9)
        c_nft_1_arr = c_nft_1[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_1)
        c_nft_2_arr = c_nft_2[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_2)

        if func_gs_scaling == "stafford":
            grad_r1 = compute_magnitude_gradients(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], const.c_gs2, c_nft_1_arr, c_nft_2_arr,
                const.mb1, const.mb2, const.delta_gs, gs_break, const.xi,
            )
        else:
            gs_exp_array = gs_exp[freq_id_grad] if estimate_gs_exp == "freq" else jnp.full(len(freq_id_grad), gs_exp)
            grad_r1 = compute_magnitude_gradients_lh(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], const.c_gs2, c_nft_1_arr, c_nft_2_arr,
                const.mb1, const.mb2, gs_break, gs_exp_array,
            )
        numpyro.factor("grad_r1", jnp.where(
            grad_r1 < 0,
            dist.Normal(0, regularization_sigma).log_prob(grad_r1),
            dist.Normal(0, regularization_sigma).log_prob(0),
        ))

    numpyro.sample("obs", dist.StudentT(loc=median, scale=sigma, df=nu_rec), obs=Y)
