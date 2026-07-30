"""
Custom SVI guide for the NGAW3 GMM in numpyro_models.py.

Coefficients (splines / parametric forms) are point masses (Delta) --
this is a MAP-style estimate for them. Random effects (deltaB, deltaS,
deltaB_attn) are mean-field Normal, to integrate them out. The geology
subregion term (c_region) is MultivariateNormal with a learned Cholesky
factor, matching the model's correlated-across-frequency prior.

Must be called with the SAME sharing_config (and the same
c0_parametric / cgs1_parametric / attn_Q / dist_cell / estimate_gs_break
/ estimate_zt_break / calc_nft / attn_eq / global_dict-is-None-or-not /
func_gs_scaling / estimate_gs_exp) as `model()` in every SVI run -- any
mismatch produces a different site set and numpyro will error.
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from patsy import dmatrix

from ngaw3_kuehnetal27.spline_coeff_guide import make_spline_coeff_guide
from ngaw3_kuehnetal27.coefficient_sharing import DEFAULT_COEFFICIENT_SHARING
from ngaw3_kuehnetal27.coefficient_sharing_guide import (
    sample_median_coefficient_guide,
    sample_c0_coefficient_guide,
    sample_cgs1_coefficient_guide,
    sample_coefficient_separate_guide,
)
from ngaw3_kuehnetal27.median_core import ModelConstants


def guide_eas(F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
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

    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    spline_basis = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )

    n_stat = X_stat.shape[0]
    n_eq = X_eq.shape[0]
    n_subregion = np.max(subregion_id) + 1
    n_basin = np.max(basin_id) + 1
    n_freq = len(F)

    # --- geometric spreading break ---
    if estimate_gs_break:
        numpyro.sample("gs_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_gs_break", 3.9)),
            transforms=dist.transforms.ExpTransform(),
        ))

    # --- magnitude uncertainty ---
    if include_mag_unc == "Classical":
        numpyro.sample("mean_M", dist.Delta(v=numpyro.param("loc_mean_M", 5.0)))
        numpyro.sample("sigma_M", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma_M", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("M_model", dist.Normal(
            loc=numpyro.param("loc_M_model", M_eq),
            scale=numpyro.param("scale_M_model", 0.2 * jnp.ones(n_eq),
                                 constraint=dist.constraints.positive),
        ))
    elif include_mag_unc == "Berkson":
        numpyro.sample("M_model", dist.Normal(loc=M_eq, scale=M_sd))
    # "None": M_model isn't sampled in the model either -- no guide site.

    # --- attenuation ---
    if dist_cell is not None:
        n_cell = dist_cell.shape[1]
        numpyro.sample("mu_Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_mu_Q_0", 5.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("Q_exp", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_exp", -0.7)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("sigma_Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma_Q_0", 1.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        with numpyro.plate("plate_cell", n_cell, dim=-1):
            numpyro.sample("Q_0", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_Q_0_cell", 5.0 * jnp.ones(n_cell))),
                transforms=dist.transforms.ExpTransform(),
            ))
        attn_mode = "dist_cell"
    elif attn_Q:
        numpyro.sample("Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_0", 5.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("Q_exp", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_exp", -0.7)),
            transforms=dist.transforms.ExpTransform(),
        ))
        attn_mode = "Q"
    else:
        attn_mode = "c_attn"

    if attn_mode == "c_attn":
        sample_median_coefficient_guide(
            "c_attn", spline_basis, sharing_config, init_mu=-1.0,
        )
    elif global_dict is not None:
        # No WUS c_attn to share -- global gets its own spline guide site.
        make_spline_coeff_guide(spline_basis, "c_attn_gl", monotonic=None, init_mu=-1.0)

    # --- zt_break ---
    if estimate_zt_break:
        numpyro.sample("zt_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_zt_break", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))

    # --- c_m3: single scalar, always shared -- no dataset_local path yet ---
    numpyro.sample("c_m3", dist.TransformedDistribution(
        dist.Delta(v=numpyro.param("loc_log_c_m3", -0.45)),
        transforms=dist.transforms.ExpTransform(),
    ))

    # --- near-fault term ---
    if calc_nft == "coeff":
        numpyro.sample("c_nft_1", dist.Delta(v=numpyro.param("loc_c_nft_1", 0.5)))
        numpyro.sample("c_nft_2", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_c_nft_2", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
    elif calc_nft == "freq":
        make_spline_coeff_guide(spline_basis, "c_nft_1", monotonic="decreasing", init_mu=0.5)
        make_spline_coeff_guide(spline_basis, "c_nft_2", monotonic="increasing", init_mu=0.0)
    # "ya14": fixed, not sampled -- no guide site.

    # --- c_0, c_gs1: parametric-vs-spline, shared/dataset_local per config ---
    sample_c0_coefficient_guide(spline_basis, sharing_config, parametric=c0_parametric)
    sample_cgs1_coefficient_guide(spline_basis, sharing_config, parametric=cgs1_parametric)

    # --- remaining spline coefficients ---
    sample_median_coefficient_guide("c_m1", spline_basis, sharing_config,
                                     monotonic="decreasing", init_mu=1.5)
    sample_median_coefficient_guide("c_m2", spline_basis, sharing_config,
                                     monotonic="decreasing", init_mu=1.0)
    sample_median_coefficient_guide("c_zt", spline_basis, sharing_config, init_mu=0.0)
    sample_median_coefficient_guide("c_nm", spline_basis, sharing_config, init_mu=0.0)
    sample_median_coefficient_guide("c_rev", spline_basis, sharing_config, init_mu=0.0)
    sample_median_coefficient_guide("c_hw", spline_basis, sharing_config, init_mu=0.5)

    # --- Vs30 categories: WUS has 2 (measured/estimated), global has 1 ---
    make_spline_coeff_guide(spline_basis, "c_vs_meas", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_vs_est", monotonic=None, init_mu=0.0)
    if global_dict is not None:
        make_spline_coeff_guide(spline_basis, "c_vs_gl", monotonic=None, init_mu=0.0)

    # --- basin term (WUS only) ---
    for i in np.arange(n_basin - 1):
        make_spline_coeff_guide(spline_basis, f"c_b_{i}", monotonic=None, init_mu=0.0)

    # --- geometric spreading exponent ---
    if func_gs_scaling != "stafford":
        if estimate_gs_exp == "coeff":
            numpyro.sample("gs_exp", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_gs_exp", 0.7)),
                transforms=dist.transforms.ExpTransform(),
            ))
        elif estimate_gs_exp == "freq":
            make_spline_coeff_guide(spline_basis, "gs_exp", monotonic=None, init_mu=0.0)

    # --- standard deviations / random effects (WUS) -- unchanged, out of scope ---
    make_spline_coeff_guide(spline_basis, "phi_s2s_meas", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "phi_s2s_est", monotonic=None, init_mu=-0.7)

    if attn_eq:
        make_spline_coeff_guide(spline_basis, "tau_attn", monotonic=None, init_mu=-0.7)

    make_spline_coeff_guide(spline_basis, "phi_ss_0", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "phi_ss_1", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "tau_0", monotonic=None, init_mu=-0.7)
    make_spline_coeff_guide(spline_basis, "tau_1", monotonic=None, init_mu=-0.7)

    # --- geology subregion random effect (WUS only) ---
    make_spline_coeff_guide(spline_basis, "sigma_region", monotonic=None, init_mu=-0.7)
    with numpyro.plate("plate_region", n_subregion, dim=-1):
        numpyro.sample(
            "c_region",
            dist.MultivariateNormal(
                loc=numpyro.param("loc_c_region", jnp.zeros((n_subregion, n_freq))),
                scale_tril=numpyro.param(
                    "scale_tril_c_region",
                    jnp.tile(jnp.eye(n_freq) * 0.2, (n_subregion, 1, 1)),
                    constraint=dist.constraints.lower_cholesky,
                ),
            ),
        )

    estimate_region_kappa = estimate_kappa in ("regional", "hierarchical")
    estimate_station_kappa = estimate_kappa in ("station", "hierarchical")
    estimate_any_kappa = estimate_region_kappa or estimate_station_kappa

    if estimate_any_kappa:
        if not c0_parametric:
            numpyro.sample("c_0_kappa_star", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_c_0_kappa_star", jnp.log(0.1))),
                transforms=dist.transforms.ExpTransform(),
            ))

        if estimate_region_kappa:
            numpyro.sample("sigma_ln_kappa_region", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_sigma_ln_kappa_region", -2.0)),
                transforms=dist.transforms.ExpTransform(),
            ))
            with numpyro.plate("plate_region", n_subregion, dim=-1):
                numpyro.sample("ln_kappa_region_raw", dist.Normal(
                    loc=numpyro.param("loc_ln_kappa_region_raw", jnp.zeros(n_subregion)),
                    scale=numpyro.param("scale_ln_kappa_region_raw", 0.2 * jnp.ones(n_subregion),
                                        constraint=dist.constraints.positive),
                ))

        if estimate_station_kappa:
            numpyro.sample("sigma_ln_kappa_station", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_sigma_ln_kappa_station", -2.0)),
                transforms=dist.transforms.ExpTransform(),
            ))
            with numpyro.plate("plate_kappa_stat", n_stat, dim=-1):
                numpyro.sample("ln_kappa_station_raw", dist.Normal(
                    loc=numpyro.param("loc_ln_kappa_station_raw", jnp.zeros(n_stat)),
                    scale=numpyro.param("scale_ln_kappa_station_raw", 0.2 * jnp.ones(n_stat),
                                        constraint=dist.constraints.positive),
                ))

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

    # ------------------------------------------------------------------
    # global data (optional)
    # ------------------------------------------------------------------
    if global_dict is not None:
        VS_stat_gl, VS_stat_sd_gl, vs_meas_id_gl = global_dict["X_stat"].T
        M_eq_gl = global_dict["X_eq"].T[0]
        eq_id_gl, stat_id_gl, _, _ = global_dict["X_id"].T
        Y_gl = global_dict["Y"]

        n_stat_gl = VS_stat_gl.shape[0]
        n_eq_gl = M_eq_gl.shape[0]

        with numpyro.plate("plate_freq_gl", n_freq, dim=-1):
            numpyro.sample("nu_rec_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_nu_rec_gl", 6.0 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("phi_ss_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_phi_ss_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("phi_s2s_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_phi_s2s_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("tau_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_tau_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            if attn_eq:
                numpyro.sample("tau_attn_gl", dist.TransformedDistribution(
                    dist.Delta(v=numpyro.param("loc_log_tau_attn_gl", 0.7 * jnp.ones(n_freq))),
                    transforms=dist.transforms.ExpTransform(),
                ))

            with numpyro.plate("plate_freq_stat_gl", n_stat_gl, dim=-2):
                numpyro.sample("deltaS_gl", dist.Normal(
                    loc=numpyro.param("loc_deltaS_gl", jnp.zeros((n_stat_gl, n_freq))),
                    scale=numpyro.param("scale_deltaS_gl", 0.2 * jnp.ones((n_stat_gl, n_freq)),
                                         constraint=dist.constraints.positive),
                ))

            with numpyro.plate("plate_freq_eq_gl", n_eq_gl, dim=-2):
                numpyro.sample("deltaB_gl", dist.Normal(
                    loc=numpyro.param("loc_deltaB_gl", jnp.zeros((n_eq_gl, n_freq))),
                    scale=numpyro.param("scale_deltaB_gl", 0.2 * jnp.ones((n_eq_gl, n_freq)),
                                         constraint=dist.constraints.positive),
                ))
                if attn_eq:
                    numpyro.sample("deltaB_attn_gl", dist.Normal(
                        loc=numpyro.param("loc_deltaB_attn_gl", jnp.zeros((n_eq_gl, n_freq))),
                        scale=numpyro.param("scale_deltaB_attn_gl", 0.2 * jnp.ones((n_eq_gl, n_freq)),
                                             constraint=dist.constraints.positive),
                    ))


def guide_separate(F, X_rec, X_eq, X_stat, X_id, nl_model_dict,
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
    Guide for `model_separate`. Same site set as `guide`, except every
    coefficient that's per-frequency-independent in `model_separate`
    (no spline, no parametric form) gets a Delta point mass at every
    frequency here too, via `sample_coefficient_separate_guide` --
    matching how `nu_rec`'s guide site already worked.

    Must be called with the SAME sharing_config (and the same
    attn_Q / dist_cell / estimate_gs_break / estimate_zt_break /
    calc_nft / attn_eq / global_dict-is-None-or-not / func_gs_scaling /
    estimate_gs_exp) as `model_separate` in every SVI run.
    """
    sharing_config = sharing_config if sharing_config is not None else DEFAULT_COEFFICIENT_SHARING

    R, Rjb, Rx, Ry0 = X_rec.T
    M_eq, Zt_eq, Fnm_eq, Frev_eq, FW_eq, Dip_eq, M_sd = X_eq.T
    VS_stat, VS_stat_sd, vs_meas_id = X_stat.T
    eq_id, stat_id, subregion_id, basin_id = X_id.T
    vs_measured_id = vs_meas_id.astype("int")

    n_stat = X_stat.shape[0]
    n_eq = X_eq.shape[0]
    n_subregion = np.max(subregion_id) + 1
    n_basin = np.max(basin_id) + 1
    n_freq = len(F)

    # --- geometric spreading break: never frequency-dependent, unaffected ---
    if estimate_gs_break:
        numpyro.sample("gs_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_gs_break", 3.9)),
            transforms=dist.transforms.ExpTransform(),
        ))

    # --- magnitude uncertainty ---
    if include_mag_unc == "Classical":
        numpyro.sample("mean_M", dist.Delta(v=numpyro.param("loc_mean_M", 5.0)))
        numpyro.sample("sigma_M", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma_M", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("M_model", dist.Normal(
            loc=numpyro.param("loc_M_model", M_eq),
            scale=numpyro.param("scale_M_model", 0.2 * jnp.ones(n_eq),
                                 constraint=dist.constraints.positive),
        ))
    elif include_mag_unc == "Berkson":
        numpyro.sample("M_model", dist.Normal(loc=M_eq, scale=M_sd))

    # --- attenuation mode selection: not frequency-dependent, unaffected ---
    if dist_cell is not None:
        n_cell = dist_cell.shape[1]
        numpyro.sample("mu_Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_mu_Q_0", 5.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("Q_exp", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_exp", -0.7)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("sigma_Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma_Q_0", 1.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        with numpyro.plate("plate_cell", n_cell, dim=-1):
            numpyro.sample("Q_0", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_Q_0_cell", 5.0 * jnp.ones(n_cell))),
                transforms=dist.transforms.ExpTransform(),
            ))
        attn_mode = "dist_cell"
    elif attn_Q:
        numpyro.sample("Q_0", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_0", 5.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("Q_exp", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_Q_exp", -0.7)),
            transforms=dist.transforms.ExpTransform(),
        ))
        attn_mode = "Q"
    else:
        attn_mode = "c_attn"

    # --- zt_break: never frequency-dependent, unaffected ---
    if estimate_zt_break:
        numpyro.sample("zt_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_zt_break", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))

    # --- c_m3: single scalar, unaffected ---
    numpyro.sample("c_m3", dist.TransformedDistribution(
        dist.Delta(v=numpyro.param("loc_log_c_m3", -0.45)),
        transforms=dist.transforms.ExpTransform(),
    ))

    if calc_nft == "coeff":
        numpyro.sample("c_nft_1", dist.Delta(v=numpyro.param("loc_c_nft_1", 0.5)))
        numpyro.sample("c_nft_2", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_c_nft_2", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))
    # "ya14": fixed, not sampled -- no guide site.
    # "freq": handled inside plate_freq below.

    # ------------------------------------------------------------------
    # Everything frequency-dependent: one plate, matching model_separate.
    # ------------------------------------------------------------------
    with numpyro.plate("plate_freq", n_freq, dim=-1):
        numpyro.sample("nu_rec", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_nu_rec", 6.0 * jnp.ones(n_freq))),
            transforms=dist.transforms.ExpTransform(),
        ))

        if calc_nft == "freq":
            sample_coefficient_separate_guide("c_nft_1", n_freq, {"c_nft_1": "shared"}, init_mu=0.5)
            sample_coefficient_separate_guide("c_nft_2", n_freq, {"c_nft_2": "shared"},
                                               positive=True, transform="exp", init_mu=0.0)

        sample_coefficient_separate_guide("c_0", n_freq, sharing_config, init_mu=-5.0)
        sample_coefficient_separate_guide("c_gs1", n_freq, sharing_config,
                                           positive=True, transform="softplus", init_mu=0.5)
        sample_coefficient_separate_guide("c_m1", n_freq, sharing_config,
                                           positive=True, transform="softplus", init_mu=1.5)
        sample_coefficient_separate_guide("c_m2", n_freq, sharing_config,
                                           positive=True, transform="softplus", init_mu=1.0)
        sample_coefficient_separate_guide("c_zt", n_freq, sharing_config, init_mu=0.0)
        sample_coefficient_separate_guide("c_nm", n_freq, sharing_config, init_mu=0.0)
        sample_coefficient_separate_guide("c_rev", n_freq, sharing_config, init_mu=0.0)
        sample_coefficient_separate_guide("c_hw", n_freq, sharing_config, init_mu=0.5)

        if attn_mode == "c_attn":
            sample_coefficient_separate_guide("c_attn", n_freq, sharing_config,
                                               positive=True, transform="softplus", init_mu=-1.0)
        elif global_dict is not None:
            numpyro.sample("c_attn_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_c_attn_gl", -1.0 * jnp.ones(n_freq))),
                transforms=dist.transforms.SoftplusTransform(),
            ))

        numpyro.sample("c_vs_meas", dist.Delta(v=numpyro.param("loc_c_vs_meas", jnp.zeros(n_freq))))
        numpyro.sample("c_vs_est", dist.Delta(v=numpyro.param("loc_c_vs_est", jnp.zeros(n_freq))))
        if global_dict is not None:
            numpyro.sample("c_vs_gl", dist.Delta(v=numpyro.param("loc_c_vs_gl", jnp.zeros(n_freq))))

        if func_gs_scaling != "stafford" and estimate_gs_exp == "freq":
            numpyro.sample("gs_exp", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_gs_exp", jnp.zeros(n_freq))),
                transforms=dist.transforms.SoftplusTransform(),
            ))

        numpyro.sample("phi_s2s_meas", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_phi_s2s_meas", -0.7 * jnp.ones(n_freq))),
            transforms=dist.transforms.SoftplusTransform(),
        ))
        numpyro.sample("phi_s2s_est", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_phi_s2s_est", -0.7 * jnp.ones(n_freq))),
            transforms=dist.transforms.SoftplusTransform(),
        ))

        if attn_eq:
            numpyro.sample("tau_attn", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_tau_attn", -0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.SoftplusTransform(),
            ))

        numpyro.sample("phi_ss_0", dist.Delta(v=numpyro.param("loc_phi_ss_0", -0.7 * jnp.ones(n_freq))))
        numpyro.sample("phi_ss_1", dist.Delta(v=numpyro.param("loc_phi_ss_1", -0.7 * jnp.ones(n_freq))))
        numpyro.sample("tau_0", dist.Delta(v=numpyro.param("loc_tau_0", -0.7 * jnp.ones(n_freq))))
        numpyro.sample("tau_1", dist.Delta(v=numpyro.param("loc_tau_1", -0.7 * jnp.ones(n_freq))))

        numpyro.sample("sigma_region", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma_region", -0.7 * jnp.ones(n_freq))),
            transforms=dist.transforms.SoftplusTransform(),
        ))
        with numpyro.plate("plate_region", n_subregion, dim=-2):
            numpyro.sample("c_region", dist.Delta(
                v=numpyro.param("loc_c_region", jnp.zeros((n_subregion, n_freq))),
            ))

        with numpyro.plate("plate_basin", n_basin - 1, dim=-2):
            numpyro.sample("c_basin_coef", dist.Delta(
                v=numpyro.param("loc_c_basin_coef", jnp.zeros((n_basin - 1, n_freq))),
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

    # --- gs_exp under 'coeff' mode: single scalar, unaffected ---
    if func_gs_scaling != "stafford" and estimate_gs_exp == "coeff":
        numpyro.sample("gs_exp", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_gs_exp_scalar", 0.7)),
            transforms=dist.transforms.SoftplusTransform(),
        ))

    # ------------------------------------------------------------------
    # global data (optional) -- identical to `guide`.
    # ------------------------------------------------------------------
    if global_dict is not None:
        VS_stat_gl, VS_stat_sd_gl, vs_meas_id_gl = global_dict["X_stat"].T
        M_eq_gl = global_dict["X_eq"].T[0]
        eq_id_gl, stat_id_gl, _, _ = global_dict["X_id"].T
        Y_gl = global_dict["Y"]

        n_stat_gl = VS_stat_gl.shape[0]
        n_eq_gl = M_eq_gl.shape[0]

        with numpyro.plate("plate_freq_gl", n_freq, dim=-1):
            numpyro.sample("nu_rec_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_nu_rec_gl", 6.0 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("phi_ss_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_phi_ss_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("phi_s2s_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_phi_s2s_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample("tau_gl", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_tau_gl", 0.7 * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
            if attn_eq:
                numpyro.sample("tau_attn_gl", dist.TransformedDistribution(
                    dist.Delta(v=numpyro.param("loc_log_tau_attn_gl", 0.7 * jnp.ones(n_freq))),
                    transforms=dist.transforms.ExpTransform(),
                ))

            with numpyro.plate("plate_freq_stat_gl", n_stat_gl, dim=-2):
                numpyro.sample("deltaS_gl", dist.Normal(
                    loc=numpyro.param("loc_deltaS_gl", jnp.zeros((n_stat_gl, n_freq))),
                    scale=numpyro.param("scale_deltaS_gl", 0.2 * jnp.ones((n_stat_gl, n_freq)),
                                         constraint=dist.constraints.positive),
                ))

            with numpyro.plate("plate_freq_eq_gl", n_eq_gl, dim=-2):
                numpyro.sample("deltaB_gl", dist.Normal(
                    loc=numpyro.param("loc_deltaB_gl", jnp.zeros((n_eq_gl, n_freq))),
                    scale=numpyro.param("scale_deltaB_gl", 0.2 * jnp.ones((n_eq_gl, n_freq)),
                                         constraint=dist.constraints.positive),
                ))
                if attn_eq:
                    numpyro.sample("deltaB_attn_gl", dist.Normal(
                        loc=numpyro.param("loc_deltaB_attn_gl", jnp.zeros((n_eq_gl, n_freq))),
                        scale=numpyro.param("scale_deltaB_attn_gl", 0.2 * jnp.ones((n_eq_gl, n_freq)),
                                             constraint=dist.constraints.positive),
                    ))
