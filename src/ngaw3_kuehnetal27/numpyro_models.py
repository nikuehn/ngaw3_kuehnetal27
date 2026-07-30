"""
NGAW3 empirical PSA/EAS GMM -- numpyro model.

Assembles WUS (and, if `global_dict` is given, global) coefficients via
`coefficient_sharing.py` (config-driven shared vs. dataset-local
sampling, always same prior *structure* for both datasets), then calls
`calculate_median_training` once per dataset.
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from patsy import dmatrix

from ngaw3_kuehnetal27.spline_coeff import make_spline_coeff
from ngaw3_kuehnetal27.coefficient_sharing import (
    DEFAULT_COEFFICIENT_SHARING,
    sample_median_coefficient,
    sample_c0_coefficient,
    sample_cgs1_coefficient,
    sample_coefficient_separate,
)
from ngaw3_kuehnetal27.median_core import (
    Coefficients,
    ModelConstants,
    RecordIndex,
    EventParams,
    SiteParams,
    calculate_median_training,
)
from ngaw3_kuehnetal27.utils import (
    smooth_trilinear_ramp_repar,
    compute_magnitude_gradients,
    compute_magnitude_gradients_lh,
    insert_zero_row,
)


def model_eas(F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
              Y=None, dist_cell=None, attn_eq=True,
              ref_basin_id=1,
              regularize_grad=False, regularization_sigma=0.1,
              calc_nft="ya14", include_mag_unc="None", attn_Q=True,
              estimate_gs_break=False, estimate_zt_break=False,
              freq_id_grad=(0, 6, 8, 10, 16, 20, 23),
              spline_degree=3, spline_df=7,
              c0_parametric=False, cgs1_parametric=False,
              calc_dWS=False, save_f_nl=False,
              func_gs_scaling="stafford", estimate_gs_exp="fixed",
              L_freq=None, global_dict=None, calc_log_lik=False,
              sharing_config=None, estimate_kappa=None):

    sharing_config = sharing_config if sharing_config is not None else DEFAULT_COEFFICIENT_SHARING

    # ------------------------------------------------------------------
    # WUS data
    # ------------------------------------------------------------------
    R, Rjb, Rx, Ry0 = X_rec.T
    M_eq, Zt_eq, Fnm_eq, Frev_eq, FW_eq, Dip_eq, M_sd = X_eq.T
    VS_stat, VS_stat_sd, vs_meas_id = X_stat.T
    eq_id, stat_id, subregion_id, basin_id = X_id.T
    vs_measured_id = vs_meas_id.astype("int")

    Zt_eq_scaled = Zt_eq / 10.0
    R_scaled = R / 100.0

    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    spline_basis = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )

    n_stat = X_stat.shape[0]
    n_eq = X_eq.shape[0]
    n_rec = X_rec.shape[0]
    n_subregion = np.max(subregion_id) + 1
    n_basin = np.max(basin_id) + 1
    n_freq = len(F)

    mu_freq = jnp.zeros(n_freq)
    if L_freq is None:
        L_freq = jnp.eye(n_freq)

    const = ModelConstants()
    mb1, mb2, c_gs2, xi, delta_gs = const.mb1, const.mb2, const.c_gs2, const.xi, const.delta_gs

    V_ref = 800.0
    log_V_ref = jnp.log(V_ref)
    lnVS = jnp.log(VS_stat) - log_V_ref

    # --- geometric spreading break: WUS-only estimate; global always fixed ---
    if estimate_gs_break:
        gs_break = numpyro.sample("gs_break", dist.Gamma(94.8, 1.9))
    else:
        gs_break = 50.0

    # --- magnitude uncertainty (WUS only, as in original) ---
    if include_mag_unc == "Classical":
        mean_M = numpyro.sample("mean_M", dist.Normal(5, 1))
        sigma_M = numpyro.sample("sigma_M", dist.LogNormal(-0.14, 0.236))
        M_model = numpyro.sample("M_model", dist.Normal(mean_M, sigma_M).expand([n_eq]))
        numpyro.sample("M_obs", dist.Normal(M_model, M_sd), obs=M_eq)
    elif include_mag_unc == "Berkson":
        M_model = numpyro.sample("M_model", dist.Normal(M_eq, M_sd))
    else:
        M_model = M_eq

    # --- attenuation ---
    # dist_cell / Q modes are WUS-only: global has no per-path distance-cell
    # or Q covariates, so it always falls back to the simple c_attn spline,
    # regardless of what WUS uses (see notes at the end of this file).
    if dist_cell is not None:
        attn_mode = "dist_cell"
        n_cell = dist_cell.shape[1]
        mu_Q_0 = numpyro.sample("mu_Q_0", dist.Gamma(4.65, 0.01))
        Q_exp = numpyro.sample("Q_exp", dist.Gamma(4.65, 11.16))
        sigma_Q_0 = numpyro.sample("sigma_Q_0", dist.HalfNormal(10))
        with numpyro.plate("plate_cell", n_cell, dim=-1):
            Q_0 = numpyro.sample("Q_0", dist.TruncatedNormal(mu_Q_0, sigma_Q_0, low=0))
        c_attn = None
    elif attn_Q:
        attn_mode = "Q"
        Q_0 = numpyro.sample("Q_0", dist.Gamma(4.65, 0.01))
        Q_exp = numpyro.sample("Q_exp", dist.Gamma(4.65, 11.16))
        c_attn = None
    else:
        attn_mode = "c_attn"
        Q_0 = None
        Q_exp = None

    if attn_mode == "c_attn":
        c_attn, c_attn_gl = sample_median_coefficient(
            "c_attn", spline_basis, sharing_config,
            mu_loc=-1.0, mu_scale=1.0, positive=True, transform="softplus",
        )
    elif global_dict is not None:
        # No WUS c_attn to (potentially) share -- global samples its own,
        # same spline structure as the 'c_attn' branch above would use.
        c_attn_gl = make_spline_coeff(
            spline_basis, "c_attn_gl", mu_loc=-1.0, mu_scale=1.0,
            positive=True, transform="softplus",
        )
    else:
        c_attn_gl = None

    # --- zt_break: WUS-only estimate; global always fixed ---
    if estimate_zt_break:
        zt_break = numpyro.sample("zt_break", dist.Gamma(6.94, 5.49))
    else:
        zt_break = 1.5

    # c_m3: single scalar (not frequency-dependent), always shared as in
    # the original -- there's no dataset_local path implemented for this
    # one yet, so sharing_config["c_m3"] is currently a no-op placeholder.
    c_m3 = numpyro.sample("c_m3", dist.LogNormal(-0.45, 0.8))
    c_m3_gl = c_m3

    # c_nft_1/c_nft_2: only the 'freq' mode is a spline; 'ya14'/'coeff' are
    # scalars. Same as c_m3, always shared -- no dataset_local path exists
    # for these yet.
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
    c_nft_1_gl, c_nft_2_gl = c_nft_1, c_nft_2

    # --- c_0, c_gs1: parametric-vs-spline switch, applied consistently ---
    c_0, c_0_gl, c_0_kappa_star = sample_c0_coefficient(
        spline_basis, ln_F, F, sharing_config, parametric=c0_parametric,
    )
    c_gs1, c_gs1_gl = sample_cgs1_coefficient(spline_basis, ln_F, sharing_config, parametric=cgs1_parametric)

    # --- remaining spline coefficients, via the generic sampler ---
    c_m1, c_m1_gl = sample_median_coefficient(
        "c_m1", spline_basis, sharing_config,
        mu_loc=1.5, mu_scale=1.0, positive=True, transform="softplus", monotonic="decreasing",
    )
    c_m2, c_m2_gl = sample_median_coefficient(
        "c_m2", spline_basis, sharing_config,
        mu_loc=1.0, mu_scale=1.0, positive=True, transform="softplus", monotonic="decreasing",
    )
    c_zt, c_zt_gl = sample_median_coefficient(
        "c_zt", spline_basis, sharing_config, mu_loc=0.0, mu_scale=0.5,
    )
    c_nm, c_nm_gl = sample_median_coefficient(
        "c_nm", spline_basis, sharing_config, mu_loc=0.0, mu_scale=0.5,
    )
    c_rev, c_rev_gl = sample_median_coefficient(
        "c_rev", spline_basis, sharing_config, mu_loc=0.0, mu_scale=0.5,
    )
    c_hw, c_hw_gl = sample_median_coefficient(
        "c_hw", spline_basis, sharing_config, mu_loc=0.5, mu_scale=0.5,
    )

    # --- Vs30 categories ---
    # WUS distinguishes measured vs. estimated Vs30 (2 categories); global
    # has no such distinction (1 category). Per the consistency principle,
    # global's single category is now a spline too (previously a flat
    # per-frequency Normal) -- see notes below.
    c_vs_meas = make_spline_coeff(spline_basis, "c_vs_meas", mu_loc=0.0, mu_scale=1.0)
    c_vs_est = make_spline_coeff(spline_basis, "c_vs_est", mu_loc=0.0, mu_scale=1.0)
    c_vs = jnp.stack([c_vs_meas, c_vs_est])
    if global_dict is not None:
        c_vs_gl_single = make_spline_coeff(spline_basis, "c_vs_gl", mu_loc=0.0, mu_scale=1.0)
        c_vs_gl = jnp.stack([c_vs_gl_single])

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

    # --- standard deviations / random effects (WUS) -- unchanged, out of scope ---
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
    phi_ss = jnp.exp(smooth_trilinear_ramp_repar(M_model[eq_id, jnp.newaxis],
                                                  phi_ss_0, phi_ss_1, mb1, mb2, delta=0.2))

    tau_0 = make_spline_coeff(spline_basis, "tau_0", mu_loc=-0.7, mu_scale=0.5)
    tau_1 = make_spline_coeff(spline_basis, "tau_1", mu_loc=-0.7, mu_scale=0.5)
    tau = jnp.exp(smooth_trilinear_ramp_repar(M_model[:, jnp.newaxis],
                                               tau_0, tau_1, mb1, mb2, delta=0.2))

    # --- geology subregion random effect (WUS only) ---
    sigma_subregion = make_spline_coeff(spline_basis, "sigma_region", mu_loc=-0.7, mu_scale=0.5,
                                         positive=True, transform="softplus")
    L_subregion = sigma_subregion[..., None] * L_freq
    with numpyro.plate("plate_region", n_subregion, dim=-1):
        c_subregion = numpyro.sample("c_region", dist.MultivariateNormal(loc=mu_freq, scale_tril=L_subregion))

    # --- kappa random effect (WUS only) ---
    estimate_region_kappa = estimate_kappa in ("regional", "hierarchical")
    estimate_station_kappa = estimate_kappa in ("station", "hierarchical")
    estimate_any_kappa = estimate_region_kappa or estimate_station_kappa

    if estimate_any_kappa:
        if not c0_parametric:
            c_0_kappa_star = numpyro.sample("c_0_kappa_star", dist.HalfNormal(0.3))
        # else: c_0_kappa_star already sampled inside sample_c0_coefficient above

        if estimate_region_kappa:
            sigma_ln_kappa_region = numpyro.sample("sigma_ln_kappa_region", dist.Exponential(10.0))
            with numpyro.plate("plate_region", n_subregion, dim=-1):
                ln_kappa_region_raw = numpyro.sample("ln_kappa_region_raw", dist.Normal(0, 1))
            m_region = numpyro.deterministic("m_region", jnp.exp(ln_kappa_region_raw * sigma_ln_kappa_region))
            m_region_id = m_region[subregion_id]

            if c0_parametric:
                kappa_region_table = numpyro.deterministic("kappa_region_table", c_0_kappa_star * (m_region - 1.0))
            else:
                kappa_region_table = numpyro.deterministic("kappa_region_table", c_0_kappa_star * m_region)
        else:
            m_region_id = 1.0

        if estimate_station_kappa:
            sigma_ln_kappa_station = numpyro.sample("sigma_ln_kappa_station", dist.Exponential(10.0))
            with numpyro.plate("plate_kappa_stat", n_stat, dim=-1):
                ln_kappa_station_raw = numpyro.sample("ln_kappa_station_raw", dist.Normal(0, 1))
            m_station = numpyro.deterministic("m_station", jnp.exp(ln_kappa_station_raw * sigma_ln_kappa_station))
            m_station_id = m_station[stat_id]

            if c0_parametric:
                kappa_station_table = numpyro.deterministic("kappa_station_table", c_0_kappa_star * (m_station - 1.0))
            else:
                kappa_station_table = numpyro.deterministic("kappa_station_table", c_0_kappa_star * m_station)
        else:
            m_station_id = 1.0

        if c0_parametric:
            # c_0 already contains -c_0_kappa_star * F; this is a zero-mean deviation on top
            kappa_adj = numpyro.deterministic(
                "kappa_adj", c_0_kappa_star * (m_region_id * m_station_id - 1.0)
            )
        else:
            # c_0 (spline) has no kappa term at all -- this IS the whole thing
            kappa_adj = numpyro.deterministic(
                "kappa_adj", c_0_kappa_star * m_region_id * m_station_id
            )
    else:
        kappa_adj = None

    # --- basin term (WUS only) ---
    c_basin_sample = jnp.stack([
        make_spline_coeff(spline_basis, f"c_b_{i}", mu_loc=0, mu_scale=0.5)
        for i in np.arange(n_basin - 1)
    ])
    c_basin = numpyro.deterministic("c_basin", insert_zero_row(ref_basin_id, c_basin_sample))

    with numpyro.plate("plate_freq", n_freq, dim=-1):
        nu_rec = numpyro.sample("nu_rec", dist.Gamma(2, 0.1))

        with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
            deltaS = numpyro.sample("deltaS", dist.TransformedDistribution(
                dist.Normal(0, 1),
                dist.transforms.AffineTransform(0, phi_s2s[vs_measured_id]),
            ))

        with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
            deltaB = numpyro.sample("deltaB", dist.TransformedDistribution(
                dist.Normal(0, 1),
                dist.transforms.AffineTransform(0, tau),
            ))
            if attn_eq:
                deltaB_attn = numpyro.sample("deltaB_attn", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, tau_attn),
                ))
            else:
                deltaB_attn = jnp.zeros((n_eq, n_freq))

    # --- monotonicity regularization on magnitude scaling (WUS only) ---
    numpyro.factor("reg_c_m2", jnp.where(c_m2 > c_m1,
                                          dist.Normal(0, regularization_sigma).log_prob(c_m2 - c_m1),
                                          dist.Normal(0, regularization_sigma).log_prob(0)))
    numpyro.factor("reg_c_m3", jnp.where(c_m3 > c_m2,
                                          dist.Normal(0, regularization_sigma).log_prob(c_m3 - c_m2),
                                          dist.Normal(0, regularization_sigma).log_prob(0)))

    # --- assemble WUS and call the median function ---
    ev_params = EventParams(
        M_model=M_model, Zt_eq=Zt_eq, Dip_eq=Dip_eq, FW_eq=FW_eq,
        Zt_eq_scaled=Zt_eq_scaled, Fnm_eq=Fnm_eq, Frev_eq=Frev_eq,
    )
    site_params = SiteParams(VS_stat=VS_stat, lnVS=lnVS, vs_measured_id=vs_measured_id)
    idx = RecordIndex(eq_id=eq_id, stat_id=stat_id, subregion_id=subregion_id, basin_id=basin_id)

    coef_wus = Coefficients(
        c_0=c_0, c_m1=c_m1, c_m2=c_m2, c_m3=c_m3, c_hw=c_hw,
        c_nft_1=c_nft_1, c_nft_2=c_nft_2, c_nm=c_nm, c_rev=c_rev,
        c_gs1=c_gs1, c_zt=c_zt, c_vs=c_vs,
        gs_break=gs_break, gs_exp=gs_exp, zt_break=zt_break,
        attn_mode=attn_mode, Q_0=Q_0, Q_exp=Q_exp, c_attn=c_attn,
    )

    median, f_nl = calculate_median_training(
        R=R, Rx=Rx, Ry0=Ry0, F=F, R_scaled=R_scaled, dist_cell=dist_cell,
        idx=idx, evt_by_eq=ev_params, site_by_stat=site_params,
        coef=coef_wus, const=const, func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        deltaB=deltaB, deltaS=deltaS, deltaB_attn=deltaB_attn,
        c_basin=c_basin, c_subregion=c_subregion,
        kappa_adj=kappa_adj,
    )

    if save_f_nl:
        numpyro.deterministic("f_nl", f_nl)

    if calc_dWS:
        numpyro.deterministic("deltaWS", Y - median)

    if regularize_grad:
        magnitudes = jnp.linspace(6, 8, 9)
        c_nft_1_arr = c_nft_1[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_1)
        c_nft_2_arr = c_nft_2[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_2)

        if func_gs_scaling == "stafford":
            grad_r1 = compute_magnitude_gradients(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], c_gs2, c_nft_1_arr, c_nft_2_arr,
                mb1, mb2, delta_gs, gs_break, xi,
            )
        else:
            gs_exp_array = gs_exp[freq_id_grad] if estimate_gs_exp == "freq" else jnp.full(len(freq_id_grad), gs_exp)
            grad_r1 = compute_magnitude_gradients_lh(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], c_gs2, c_nft_1_arr, c_nft_2_arr,
                mb1, mb2, gs_break, gs_exp_array,
            )
        numpyro.factor("grad_r1", jnp.where(
            grad_r1 < 0,
            dist.Normal(0, regularization_sigma).log_prob(grad_r1),
            dist.Normal(0, regularization_sigma).log_prob(0),
        ))

    if Y is not None:
        obs_mask = ~np.isnan(Y)
        Y_obs = Y[obs_mask]
        median_obs = median[obs_mask]
        scale_obs = (jnp.ones((n_rec, n_freq)) * phi_ss)[obs_mask]
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
        scale_obs = jnp.ones((n_rec, n_freq)) * phi_ss
        df_obs = jnp.ones((n_rec, n_freq)) * nu_rec

    numpyro.sample("obs", dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs), obs=Y_obs)

    # ------------------------------------------------------------------
    # global data (optional)
    # ------------------------------------------------------------------
    if global_dict is not None:
        R_gl, Rjb_gl, Rx_gl, Ry0_gl = global_dict["X_rec"].T
        M_eq_gl, Zt_eq_gl, Fnm_eq_gl, Frev_eq_gl, FW_eq_gl, Dip_eq_gl, M_sd_gl = global_dict["X_eq"].T
        VS_stat_gl, VS_stat_sd_gl, vs_meas_id_gl = global_dict["X_stat"].T
        eq_id_gl, stat_id_gl, _, _ = global_dict["X_id"].T

        Y_gl = global_dict["Y"]

        n_stat_gl = VS_stat_gl.shape[0]
        n_eq_gl = M_eq_gl.shape[0]
        n_rec_gl = Y_gl.shape[0]

        subregion_id_gl = jnp.zeros(n_rec_gl, dtype=int)
        basin_id_gl = jnp.zeros(n_rec_gl, dtype=int)
        vs_measured_id_gl = jnp.zeros(n_stat_gl, dtype=int)

        Zt_eq_scaled_gl = Zt_eq_gl / 10.0
        R_scaled_gl = R_gl / 100.0
        lnVS_gl = jnp.log(VS_stat_gl) - log_V_ref

        with numpyro.plate("plate_freq_gl", n_freq, dim=-1):
            nu_rec_gl = numpyro.sample("nu_rec_gl", dist.Gamma(2, 0.1))
            phi_ss_gl = numpyro.sample("phi_ss_gl", dist.HalfNormal(0.5))
            tau_gl = numpyro.sample("tau_gl", dist.HalfNormal(0.5))
            phi_s2s_gl = numpyro.sample("phi_s2s_gl", dist.HalfNormal(0.5))

            if attn_eq:
                tau_attn_gl = numpyro.sample("tau_attn_gl", dist.HalfNormal(0.5))

            with numpyro.plate("plate_freq_stat_gl", n_stat_gl, dim=-2):
                deltaS_gl = numpyro.sample("deltaS_gl", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, phi_s2s_gl),
                ))

            with numpyro.plate("plate_freq_eq_gl", n_eq_gl, dim=-2):
                deltaB_gl = numpyro.sample("deltaB_gl", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, tau_gl),
                ))
                if attn_eq:
                    deltaB_attn_gl = numpyro.sample("deltaB_attn_gl", dist.TransformedDistribution(
                        dist.Normal(0, 1),
                        dist.transforms.AffineTransform(0, tau_attn_gl),
                    ))
                else:
                    deltaB_attn_gl = jnp.zeros((n_eq_gl, n_freq))

        c_basin_gl = jnp.zeros((1, n_freq))
        c_subregion_gl = jnp.zeros((1, n_freq))

        coef_gl = Coefficients(
            c_0=c_0_gl, c_m1=c_m1_gl, c_m2=c_m2_gl, c_m3=c_m3_gl, c_hw=c_hw_gl,
            c_nft_1=c_nft_1_gl, c_nft_2=c_nft_2_gl, c_nm=c_nm_gl, c_rev=c_rev_gl,
            c_gs1=c_gs1_gl, c_zt=c_zt_gl, c_vs=c_vs_gl,
            gs_break=50.0, gs_exp=2.0, zt_break=1.5,
            attn_mode="c_attn", Q_0=None, Q_exp=None, c_attn=c_attn_gl,
        )

        ev_params_gl = EventParams(
            M_model=M_eq_gl, Zt_eq=Zt_eq_gl, Dip_eq=Dip_eq_gl, FW_eq=FW_eq_gl,
            Zt_eq_scaled=Zt_eq_scaled_gl, Fnm_eq=Fnm_eq_gl, Frev_eq=Frev_eq_gl,
        )
        site_params_gl = SiteParams(VS_stat=VS_stat_gl, lnVS=lnVS_gl, vs_measured_id=vs_measured_id_gl)
        idx_gl = RecordIndex(eq_id=eq_id_gl, stat_id=stat_id_gl,
                              subregion_id=subregion_id_gl, basin_id=basin_id_gl)

        median_gl, _ = calculate_median_training(
            R=R_gl, Rx=Rx_gl, Ry0=Ry0_gl, F=F, R_scaled=R_scaled_gl, dist_cell=None,
            idx=idx_gl, evt_by_eq=ev_params_gl, site_by_stat=site_params_gl,
            coef=coef_gl, const=const, func_gs_scaling=func_gs_scaling,
            nl_model_dict=nl_model_dict,
            deltaB=deltaB_gl, deltaS=deltaS_gl, deltaB_attn=deltaB_attn_gl,
            c_basin=c_basin_gl, c_subregion=c_subregion_gl,
        )

        obs_mask_gl = ~np.isnan(Y_gl)
        Y_obs_gl = Y_gl[obs_mask_gl]
        median_obs_gl = median_gl[obs_mask_gl]
        scale_obs_gl = (jnp.ones((n_rec_gl, n_freq)) * phi_ss_gl)[obs_mask_gl]
        df_obs_gl = (jnp.ones((n_rec_gl, n_freq)) * nu_rec_gl)[obs_mask_gl]

        numpyro.sample("obs_gl", dist.StudentT(loc=median_obs_gl, scale=scale_obs_gl, df=df_obs_gl),
                        obs=Y_obs_gl)

# ----------------------------------------------------------------------
# Judgment calls made while assembling this from model_full.py -- please
# review before running:
#
# 1. c_attn_gl is now a spline (`make_spline_coeff`), not the original
#    flat per-frequency LogNormal(-0.4, 0.9) -- matches your consistency
#    request, but is a real prior change worth a sanity check against a
#    known fit.
# 2. Global always gets its own attenuation spline (c_attn_gl), even when
#    WUS uses attn_mode 'Q' or 'dist_cell' -- global has no per-path
#    distance-cell/Q covariates to support those forms, so it falls back
#    to c_attn regardless. This preserves the original guarantee (global
#    always had SOME attenuation term) but you didn't explicitly confirm
#    this fallback -- flag if that's wrong.
# 3. c_vs_gl is now a single spline coefficient, not a flat per-frequency
#    Normal(0, 1) -- same reasoning as c_attn_gl.
# 4. c_m3 and c_nft_1/c_nft_2 (in 'ya14'/'coeff' modes) are single scalars,
#    not frequency-dependent, and were never structured as shared-vs-local
#    in the original -- they stay exactly as before. sharing_config
#    entries for these don't currently do anything; only add a
#    dataset_local path for them if you actually need one.
# 5. fault_term and delta_c_vs removed entirely, per your last request.
# 6. c_m1_parametric / c_m2_parametric were in the original signature but
#    never used in the body -- dropped as dead parameters.
# ----------------------------------------------------------------------




def model_separate(F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
                    Y=None, dist_cell=None, attn_eq=True,
                    ref_basin_id=1,
                    regularize_grad=False, regularization_sigma=0.1,
                    calc_nft="ya14", include_mag_unc="None", attn_Q=True,
                    estimate_gs_break=False, estimate_zt_break=False,
                    freq_id_grad=(0, 6, 8, 10, 16, 20, 23),
                    calc_dWS=False, save_f_nl=False,
                    func_gs_scaling="stafford", estimate_gs_exp="fixed",
                    global_dict=None, calc_log_lik=False,
                    sharing_config=None):
    """
    Same functional form as `model`, but every frequency-dependent
    coefficient (median-scaling and standard-deviation) is modeled
    independently at each frequency instead of via a spline or
    parametric form -- exactly the same idea as `nu_rec` already used:
    one free parameter per frequency, no smoothness assumption across
    frequency.

    Consequences of "no splines":
    - No `c0_parametric`/`cgs1_parametric` switch, no `spline_degree`/
      `spline_df` -- there's no spline basis to build at all.
    - Any monotonicity a spline enforced (e.g. c_m1/c_m2 decreasing with
      frequency, via `monotonic='decreasing'`) is NOT enforced here --
      each frequency's value is a free, independent parameter.
    - The subregion effect (c_region) loses its correlation-across-
      frequency structure (previously a MultivariateNormal with
      `L_freq`); it's now an independent Normal per (subregion,
      frequency) pair, with its own per-frequency-independent SD
      (sigma_region) instead of a single spline-smoothed SD.
    - gs_break/zt_break were never spline-modeled in `model` either
      (single scalars) -- unaffected here.
    - Global's standard-deviation components (phi_ss_gl, tau_gl,
      phi_s2s_gl, tau_attn_gl) were already per-frequency-independent in
      `model` -- unaffected here, same code.

    Shared-vs-dataset-local routing (`sharing_config`) is unchanged --
    it's an orthogonal concern to spline vs. per-frequency-independent.
    """
    sharing_config = sharing_config if sharing_config is not None else DEFAULT_COEFFICIENT_SHARING

    # ------------------------------------------------------------------
    # WUS data
    # ------------------------------------------------------------------
    R, Rjb, Rx, Ry0 = X_rec.T
    M_eq, Zt_eq, Fnm_eq, Frev_eq, FW_eq, Dip_eq, M_sd = X_eq.T
    VS_stat, VS_stat_sd, vs_meas_id = X_stat.T
    eq_id, stat_id, subregion_id, basin_id = X_id.T
    vs_measured_id = vs_meas_id.astype("int")

    Zt_eq_scaled = Zt_eq / 10.0
    R_scaled = R / 100.0

    n_stat = X_stat.shape[0]
    n_eq = X_eq.shape[0]
    n_rec = X_rec.shape[0]
    n_subregion = np.max(subregion_id) + 1
    n_basin = np.max(basin_id) + 1
    n_freq = len(F)

    const = ModelConstants()
    mb1, mb2, c_gs2, xi, delta_gs = const.mb1, const.mb2, const.c_gs2, const.xi, const.delta_gs

    V_ref = 800.0
    log_V_ref = jnp.log(V_ref)
    lnVS = jnp.log(VS_stat) - log_V_ref

    # --- geometric spreading break: never frequency-dependent, unaffected ---
    if estimate_gs_break:
        gs_break = numpyro.sample("gs_break", dist.Gamma(94.8, 1.9))
    else:
        gs_break = 50.0

    # --- magnitude uncertainty (WUS only, as in model) ---
    if include_mag_unc == "Classical":
        mean_M = numpyro.sample("mean_M", dist.Normal(5, 1))
        sigma_M = numpyro.sample("sigma_M", dist.LogNormal(-0.14, 0.236))
        M_model = numpyro.sample("M_model", dist.Normal(mean_M, sigma_M).expand([n_eq]))
        numpyro.sample("M_obs", dist.Normal(M_model, M_sd), obs=M_eq)
    elif include_mag_unc == "Berkson":
        M_model = numpyro.sample("M_model", dist.Normal(M_eq, M_sd))
    else:
        M_model = M_eq

    # --- attenuation mode selection: Q_0/Q_exp are never frequency-dependent
    # (single scalars, or per-cell for dist_cell), so this happens before
    # the frequency plate, same position as in `model`. ---
    if dist_cell is not None:
        attn_mode = "dist_cell"
        n_cell = dist_cell.shape[1]
        mu_Q_0 = numpyro.sample("mu_Q_0", dist.Gamma(4.65, 0.01))
        Q_exp = numpyro.sample("Q_exp", dist.Gamma(4.65, 11.16))
        sigma_Q_0 = numpyro.sample("sigma_Q_0", dist.HalfNormal(10))
        with numpyro.plate("plate_cell", n_cell, dim=-1):
            Q_0 = numpyro.sample("Q_0", dist.TruncatedNormal(mu_Q_0, sigma_Q_0, low=0))
        c_attn = None
    elif attn_Q:
        attn_mode = "Q"
        Q_0 = numpyro.sample("Q_0", dist.Gamma(4.65, 0.01))
        Q_exp = numpyro.sample("Q_exp", dist.Gamma(4.65, 11.16))
        c_attn = None
    else:
        attn_mode = "c_attn"
        Q_0 = None
        Q_exp = None

    # --- zt_break: never frequency-dependent, unaffected ---
    if estimate_zt_break:
        zt_break = numpyro.sample("zt_break", dist.Gamma(6.94, 5.49))
    else:
        zt_break = 1.5

    # c_m3: single scalar, never frequency-dependent, unaffected.
    c_m3 = numpyro.sample("c_m3", dist.LogNormal(-0.45, 0.8))
    c_m3_gl = c_m3

    if calc_nft == "ya14":
        c_nft_1 = -1.72 * jnp.log(10) + 0.43 * jnp.log(10) * 4.5
        c_nft_2 = 0.43 * jnp.log(10)
        c_nft_1_gl, c_nft_2_gl = c_nft_1, c_nft_2
    elif calc_nft == "coeff":
        c_nft_1 = numpyro.sample("c_nft_1", dist.Normal(0.5, 0.5))
        c_nft_2 = numpyro.sample("c_nft_2", dist.LogNormal(0, 0.2))
        c_nft_1_gl, c_nft_2_gl = c_nft_1, c_nft_2

    # ------------------------------------------------------------------
    # Everything frequency-dependent: one plate, matching where `nu_rec`
    # already lived in `model`. Order mirrors `model`'s structure --
    # coefficients, then std-dev components, then subregion/basin, then
    # tau/phi_ss (computed from the just-sampled per-freq values), then
    # deltaS/deltaB/deltaB_attn nested inside the same plate.
    # ------------------------------------------------------------------
    with numpyro.plate("plate_freq", n_freq, dim=-1):
        nu_rec = numpyro.sample("nu_rec", dist.Gamma(2, 0.1))

        if calc_nft == "freq":
            c_nft_1 = numpyro.sample("c_nft_1", dist.Normal(0.5, 0.5))
            c_nft_2 = numpyro.sample("c_nft_2", dist.LogNormal(0.0, 0.2))
            c_nft_1_gl, c_nft_2_gl = c_nft_1, c_nft_2

        # --- c_0, c_gs1: no parametric option here -- always per-frequency-independent ---
        c_0, c_0_gl = sample_coefficient_separate(
            "c_0", sharing_config, mu_loc=-5.0, mu_scale=5.0,
        )
        c_gs1, c_gs1_gl = sample_coefficient_separate(
            "c_gs1", sharing_config, mu_loc=0.5, mu_scale=0.5, positive=True, transform="softplus",
        )

        # --- remaining median coefficients ---
        c_m1, c_m1_gl = sample_coefficient_separate(
            "c_m1", sharing_config, mu_loc=1.5, mu_scale=1.0, positive=True, transform="softplus",
        )
        c_m2, c_m2_gl = sample_coefficient_separate(
            "c_m2", sharing_config, mu_loc=1.0, mu_scale=1.0, positive=True, transform="softplus",
        )
        c_zt, c_zt_gl = sample_coefficient_separate("c_zt", sharing_config, mu_loc=0.0, mu_scale=0.5)
        c_nm, c_nm_gl = sample_coefficient_separate("c_nm", sharing_config, mu_loc=0.0, mu_scale=0.5)
        c_rev, c_rev_gl = sample_coefficient_separate("c_rev", sharing_config, mu_loc=0.0, mu_scale=0.5)
        c_hw, c_hw_gl = sample_coefficient_separate("c_hw", sharing_config, mu_loc=0.5, mu_scale=0.5)

        # --- attenuation (c_attn branch only; Q_0/Q_exp handled above, not frequency-dependent) ---
        if attn_mode == "c_attn":
            c_attn, c_attn_gl = sample_coefficient_separate(
                "c_attn", sharing_config, mu_loc=-1.0, mu_scale=1.0, positive=True, transform="softplus",
            )
        elif global_dict is not None:
            # No WUS c_attn to (potentially) share -- global samples its own.
            c_attn_gl = numpyro.sample("c_attn_gl", dist.TransformedDistribution(
                dist.Normal(-1.0, 1.0), dist.transforms.SoftplusTransform(),
            ))
        else:
            c_attn_gl = None

        # --- Vs30 categories ---
        c_vs_meas = numpyro.sample("c_vs_meas", dist.Normal(0.0, 1.0))
        c_vs_est = numpyro.sample("c_vs_est", dist.Normal(0.0, 1.0))
        c_vs = jnp.stack([c_vs_meas, c_vs_est])
        if global_dict is not None:
            c_vs_gl_single = numpyro.sample("c_vs_gl", dist.Normal(0.0, 1.0))
            c_vs_gl = jnp.stack([c_vs_gl_single])

        # --- geometric spreading exponent ('freq' mode only) ---
        if func_gs_scaling == "stafford":
            gs_exp = 2.0
        elif estimate_gs_exp == "freq":
            gs_exp = numpyro.sample("gs_exp", dist.TransformedDistribution(
                dist.Normal(1.2, 0.5), dist.transforms.SoftplusTransform(),
            ))
        elif estimate_gs_exp == "coeff":
            pass  # sampled below, outside the plate -- not frequency-dependent
        else:
            gs_exp = 2.0

        # --- standard deviations (WUS) -- per-frequency-independent ---
        phi_s2s_meas = numpyro.sample("phi_s2s_meas", dist.TransformedDistribution(
            dist.Normal(-0.7, 0.5), dist.transforms.SoftplusTransform(),
        ))
        phi_s2s_est = numpyro.sample("phi_s2s_est", dist.TransformedDistribution(
            dist.Normal(-0.7, 0.5), dist.transforms.SoftplusTransform(),
        ))
        phi_s2s = jnp.stack([phi_s2s_meas, phi_s2s_est])

        if attn_eq:
            tau_attn = numpyro.sample("tau_attn", dist.TransformedDistribution(
                dist.Normal(-0.7, 0.5), dist.transforms.SoftplusTransform(),
            ))

        phi_ss_0 = numpyro.sample("phi_ss_0", dist.Normal(-0.7, 0.5))
        phi_ss_1 = numpyro.sample("phi_ss_1", dist.Normal(-0.7, 0.5))
        tau_0 = numpyro.sample("tau_0", dist.Normal(-0.7, 0.5))
        tau_1 = numpyro.sample("tau_1", dist.Normal(-0.7, 0.5))

        # --- geology subregion effect: independent per (subregion, frequency),
        # no correlation-across-frequency structure ---
        sigma_region = numpyro.sample("sigma_region", dist.TransformedDistribution(
            dist.Normal(-0.7, 0.5), dist.transforms.SoftplusTransform(),
        ))
        with numpyro.plate("plate_region", n_subregion, dim=-2):
            c_subregion = numpyro.sample("c_region", dist.Normal(0.0, sigma_region))

        # --- basin term (WUS only) ---
        with numpyro.plate("plate_basin", n_basin - 1, dim=-2):
            c_basin_sample = numpyro.sample("c_basin_coef", dist.Normal(0.0, 0.5))
        c_basin = numpyro.deterministic("c_basin", insert_zero_row(ref_basin_id, c_basin_sample))

        # phi_ss / tau: magnitude-dependent ramp, using the per-frequency
        # phi_ss_0/phi_ss_1/tau_0/tau_1 just sampled above -- computed here
        # (a plain jnp computation, not a new sample site) so deltaS/deltaB
        # below can use the correct, fully-resolved scale.
        phi_ss = jnp.exp(smooth_trilinear_ramp_repar(M_model[eq_id, jnp.newaxis],
                                                      phi_ss_0, phi_ss_1, mb1, mb2, delta=0.2))
        tau = jnp.exp(smooth_trilinear_ramp_repar(M_model[:, jnp.newaxis],
                                                   tau_0, tau_1, mb1, mb2, delta=0.2))

        with numpyro.plate("plate_freq_stat", n_stat, dim=-2):
            deltaS = numpyro.sample("deltaS", dist.TransformedDistribution(
                dist.Normal(0, 1),
                dist.transforms.AffineTransform(0, phi_s2s[vs_measured_id]),
            ))

        with numpyro.plate("plate_freq_eq", n_eq, dim=-2):
            deltaB = numpyro.sample("deltaB", dist.TransformedDistribution(
                dist.Normal(0, 1),
                dist.transforms.AffineTransform(0, tau),
            ))
            if attn_eq:
                deltaB_attn = numpyro.sample("deltaB_attn", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, tau_attn),
                ))
            else:
                deltaB_attn = jnp.zeros((n_eq, n_freq))

    # --- gs_exp under 'coeff' mode: single scalar, not frequency-dependent ---
    if func_gs_scaling != "stafford" and estimate_gs_exp == "coeff":
        gs_exp = numpyro.sample("gs_exp", dist.TransformedDistribution(
            dist.Normal(0.7, 0.3), dist.transforms.SoftplusTransform(),
        ))

    # --- monotonicity regularization on magnitude scaling (WUS only) ---
    numpyro.factor("reg_c_m2", jnp.where(c_m2 > c_m1,
                                          dist.Normal(0, regularization_sigma).log_prob(c_m2 - c_m1),
                                          dist.Normal(0, regularization_sigma).log_prob(0)))
    numpyro.factor("reg_c_m3", jnp.where(c_m3 > c_m2,
                                          dist.Normal(0, regularization_sigma).log_prob(c_m3 - c_m2),
                                          dist.Normal(0, regularization_sigma).log_prob(0)))

    # --- assemble WUS and call the median function ---
    ev_params = EventParams(
        M_model=M_model, Zt_eq=Zt_eq, Dip_eq=Dip_eq, FW_eq=FW_eq,
        Zt_eq_scaled=Zt_eq_scaled, Fnm_eq=Fnm_eq, Frev_eq=Frev_eq,
    )
    site_params = SiteParams(VS_stat=VS_stat, lnVS=lnVS, vs_measured_id=vs_measured_id)
    idx = RecordIndex(eq_id=eq_id, stat_id=stat_id, subregion_id=subregion_id, basin_id=basin_id)

    coef_wus = Coefficients(
        c_0=c_0, c_m1=c_m1, c_m2=c_m2, c_m3=c_m3, c_hw=c_hw,
        c_nft_1=c_nft_1, c_nft_2=c_nft_2, c_nm=c_nm, c_rev=c_rev,
        c_gs1=c_gs1, c_zt=c_zt, c_vs=c_vs,
        gs_break=gs_break, gs_exp=gs_exp, zt_break=zt_break,
        attn_mode=attn_mode, Q_0=Q_0, Q_exp=Q_exp, c_attn=c_attn,
    )

    median, f_nl = calculate_median_training(
        R=R, Rx=Rx, Ry0=Ry0, F=F, R_scaled=R_scaled, dist_cell=dist_cell,
        idx=idx, evt_by_eq=ev_params, site_by_stat=site_params,
        coef=coef_wus, const=const, func_gs_scaling=func_gs_scaling,
        nl_model_dict=nl_model_dict,
        deltaB=deltaB, deltaS=deltaS, deltaB_attn=deltaB_attn,
        c_basin=c_basin, c_subregion=c_subregion,
    )

    if save_f_nl:
        numpyro.deterministic("f_nl", f_nl)

    if calc_dWS:
        numpyro.deterministic("deltaWS", Y - median)

    if regularize_grad:
        magnitudes = jnp.linspace(6, 8, 9)
        c_nft_1_arr = c_nft_1[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_1)
        c_nft_2_arr = c_nft_2[freq_id_grad] if calc_nft == "freq" else jnp.full(len(freq_id_grad), c_nft_2)

        if func_gs_scaling == "stafford":
            grad_r1 = compute_magnitude_gradients(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], c_gs2, c_nft_1_arr, c_nft_2_arr,
                mb1, mb2, delta_gs, gs_break, xi,
            )
        else:
            gs_exp_array = gs_exp[freq_id_grad] if estimate_gs_exp == "freq" else jnp.full(len(freq_id_grad), gs_exp)
            grad_r1 = compute_magnitude_gradients_lh(
                magnitudes, 1.0,
                c_m1[freq_id_grad], c_m2[freq_id_grad], c_m3,
                c_gs1[freq_id_grad], c_gs2, c_nft_1_arr, c_nft_2_arr,
                mb1, mb2, gs_break, gs_exp_array,
            )
        numpyro.factor("grad_r1", jnp.where(
            grad_r1 < 0,
            dist.Normal(0, regularization_sigma).log_prob(grad_r1),
            dist.Normal(0, regularization_sigma).log_prob(0),
        ))

    if Y is not None:
        obs_mask = ~np.isnan(Y)
        Y_obs = Y[obs_mask]
        median_obs = median[obs_mask]
        scale_obs = (jnp.ones((n_rec, n_freq)) * phi_ss)[obs_mask]
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
        scale_obs = jnp.ones((n_rec, n_freq)) * phi_ss
        df_obs = jnp.ones((n_rec, n_freq)) * nu_rec

    numpyro.sample("obs", dist.StudentT(loc=median_obs, scale=scale_obs, df=df_obs), obs=Y_obs)

    # ------------------------------------------------------------------
    # global data (optional) -- identical to `model`: global's std-dev
    # components were already per-frequency-independent there, and
    # c_vs_gl/c_attn_gl were handled above in the shared plate.
    # ------------------------------------------------------------------
    if global_dict is not None:
        R_gl, Rjb_gl, Rx_gl, Ry0_gl = global_dict["X_rec"].T
        M_eq_gl, Zt_eq_gl, Fnm_eq_gl, Frev_eq_gl, FW_eq_gl, Dip_eq_gl, M_sd_gl = global_dict["X_eq"].T
        VS_stat_gl, VS_stat_sd_gl, vs_meas_id_gl = global_dict["X_stat"].T
        eq_id_gl, stat_id_gl, _, _ = global_dict["X_id"].T

        Y_gl = global_dict["Y"]

        n_stat_gl = VS_stat_gl.shape[0]
        n_eq_gl = M_eq_gl.shape[0]
        n_rec_gl = Y_gl.shape[0]

        subregion_id_gl = jnp.zeros(n_rec_gl, dtype=int)
        basin_id_gl = jnp.zeros(n_rec_gl, dtype=int)
        vs_measured_id_gl = jnp.zeros(n_stat_gl, dtype=int)

        Zt_eq_scaled_gl = Zt_eq_gl / 10.0
        R_scaled_gl = R_gl / 100.0
        lnVS_gl = jnp.log(VS_stat_gl) - log_V_ref

        with numpyro.plate("plate_freq_gl", n_freq, dim=-1):
            nu_rec_gl = numpyro.sample("nu_rec_gl", dist.Gamma(2, 0.1))
            phi_ss_gl = numpyro.sample("phi_ss_gl", dist.HalfNormal(0.5))
            tau_gl = numpyro.sample("tau_gl", dist.HalfNormal(0.5))
            phi_s2s_gl = numpyro.sample("phi_s2s_gl", dist.HalfNormal(0.5))

            if attn_eq:
                tau_attn_gl = numpyro.sample("tau_attn_gl", dist.HalfNormal(0.5))

            with numpyro.plate("plate_freq_stat_gl", n_stat_gl, dim=-2):
                deltaS_gl = numpyro.sample("deltaS_gl", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, phi_s2s_gl),
                ))

            with numpyro.plate("plate_freq_eq_gl", n_eq_gl, dim=-2):
                deltaB_gl = numpyro.sample("deltaB_gl", dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(0, tau_gl),
                ))
                if attn_eq:
                    deltaB_attn_gl = numpyro.sample("deltaB_attn_gl", dist.TransformedDistribution(
                        dist.Normal(0, 1),
                        dist.transforms.AffineTransform(0, tau_attn_gl),
                    ))
                else:
                    deltaB_attn_gl = jnp.zeros((n_eq_gl, n_freq))

        c_basin_gl = jnp.zeros((1, n_freq))
        c_subregion_gl = jnp.zeros((1, n_freq))

        coef_gl = Coefficients(
            c_0=c_0_gl, c_m1=c_m1_gl, c_m2=c_m2_gl, c_m3=c_m3_gl, c_hw=c_hw_gl,
            c_nft_1=c_nft_1_gl, c_nft_2=c_nft_2_gl, c_nm=c_nm_gl, c_rev=c_rev_gl,
            c_gs1=c_gs1_gl, c_zt=c_zt_gl, c_vs=c_vs_gl,
            gs_break=50.0, gs_exp=2.0, zt_break=1.5,
            attn_mode="c_attn", Q_0=None, Q_exp=None, c_attn=c_attn_gl,
        )

        ev_params_gl = EventParams(
            M_model=M_eq_gl, Zt_eq=Zt_eq_gl, Dip_eq=Dip_eq_gl, FW_eq=FW_eq_gl,
            Zt_eq_scaled=Zt_eq_scaled_gl, Fnm_eq=Fnm_eq_gl, Frev_eq=Frev_eq_gl,
        )
        site_params_gl = SiteParams(VS_stat=VS_stat_gl, lnVS=lnVS_gl, vs_measured_id=vs_measured_id_gl)
        idx_gl = RecordIndex(eq_id=eq_id_gl, stat_id=stat_id_gl,
                              subregion_id=subregion_id_gl, basin_id=basin_id_gl)

        median_gl, _ = calculate_median_training(
            R=R_gl, Rx=Rx_gl, Ry0=Ry0_gl, F=F, R_scaled=R_scaled_gl, dist_cell=None,
            idx=idx_gl, evt_by_eq=ev_params_gl, site_by_stat=site_params_gl,
            coef=coef_gl, const=const, func_gs_scaling=func_gs_scaling,
            nl_model_dict=nl_model_dict,
            deltaB=deltaB_gl, deltaS=deltaS_gl, deltaB_attn=deltaB_attn_gl,
            c_basin=c_basin_gl, c_subregion=c_subregion_gl,
        )

        obs_mask_gl = ~np.isnan(Y_gl)
        Y_obs_gl = Y_gl[obs_mask_gl]
        median_obs_gl = median_gl[obs_mask_gl]
        scale_obs_gl = (jnp.ones((n_rec_gl, n_freq)) * phi_ss_gl)[obs_mask_gl]
        df_obs_gl = (jnp.ones((n_rec_gl, n_freq)) * nu_rec_gl)[obs_mask_gl]

        numpyro.sample("obs_gl", dist.StudentT(loc=median_obs_gl, scale=scale_obs_gl, df=df_obs_gl),
                        obs=Y_obs_gl)
