"""
Custom SVI guide for `model_psa` in `numpyro_model_psa.py`.

All coefficients (splines / parametric forms) are point masses (Delta)
-- MAP-style, same as `guide_eas` uses for its coefficients. Unlike
`guide_eas`, there are no random effects to integrate out (deltaB/
deltaS/deltaB_attn don't exist here -- see `numpyro_model_psa.py`'s
module docstring), so this guide is entirely Delta sites; no
MultivariateNormal block is needed even for the regional term, since
`c_region` is a fixed effect here, not a random effect.

Must be called with the SAME vsmag / attnmag / include_region /
estimate_zt2 / calc_nft / estimate_gs_break / estimate_zt_break /
func_gs_scaling / estimate_gs_exp arguments as `model_psa` in every SVI
run -- any mismatch produces a different site set and numpyro will
error.

X_id column order: subregion_id, basin_id, vs_measured_id, mag_bin.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from patsy import dmatrix

from ngaw3_kuehnetal27.spline_coeff_guide import make_spline_coeff_guide


def guide_psa(
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
    subregion_id, basin_id, vs_measured_id, mag_bin = X_id.T
    vs_measured_id = vs_measured_id.astype("int")
    mag_bin = mag_bin.astype("int")
    subregion_id = subregion_id.astype("int")
    basin_id = basin_id.astype("int")

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

    if estimate_gs_break:
        numpyro.sample("gs_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_gs_break", 3.9)),
            transforms=dist.transforms.ExpTransform(),
        ))

    if estimate_zt_break:
        numpyro.sample("zt_break", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_zt_break", 0.0)),
            transforms=dist.transforms.ExpTransform(),
        ))

    if estimate_zt2:
        make_spline_coeff_guide(spline_basis, "c_zt2", monotonic=None, init_mu=0.0)

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
    elif calc_nft == "freq":
        make_spline_coeff_guide(spline_basis, "c_nft_1", monotonic="decreasing", init_mu=0.5)
        make_spline_coeff_guide(spline_basis, "c_nft_2", monotonic="increasing", init_mu=0.0)
    # "ya14": fixed, not sampled -- no guide site.

    make_spline_coeff_guide(spline_basis, "c_0", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_gs1", monotonic=None, init_mu=-1.0)
    make_spline_coeff_guide(spline_basis, "c_m1", monotonic=None, init_mu=1.5)
    make_spline_coeff_guide(spline_basis, "c_m2", monotonic=None, init_mu=1.0)
    make_spline_coeff_guide(spline_basis, "c_zt", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_nm", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_rev", monotonic=None, init_mu=0.0)
    make_spline_coeff_guide(spline_basis, "c_hw", monotonic=None, init_mu=0.5)

    if attnmag:
        for i in range(n_magbin):
            make_spline_coeff_guide(spline_basis, f"c_attn_{i}", monotonic=None, init_mu=-1.0)
    else:
        make_spline_coeff_guide(spline_basis, "c_attn", monotonic=None, init_mu=-1.0)

    if vsmag == "magbin":
        for i in range(n_magbin):
            make_spline_coeff_guide(spline_basis, f"c_vs_meas_{i}", monotonic=None, init_mu=0.0)
            make_spline_coeff_guide(spline_basis, f"c_vs_est_{i}", monotonic=None, init_mu=0.0)
    elif vsmag == "continuous":
        make_spline_coeff_guide(spline_basis, "c_vs_meas_ic", monotonic=None, init_mu=-0.5)
        make_spline_coeff_guide(spline_basis, "c_vs_meas_sl1", monotonic=None, init_mu=0.0)
        make_spline_coeff_guide(spline_basis, "c_vs_meas_sl2", monotonic=None, init_mu=0.0)
        numpyro.sample("c_vs_meas_br", dist.Delta(v=numpyro.param("loc_c_vs_meas_br", 6.0)))

        make_spline_coeff_guide(spline_basis, "c_vs_est_ic", monotonic=None, init_mu=-0.5)
        make_spline_coeff_guide(spline_basis, "c_vs_est_sl1", monotonic=None, init_mu=0.0)
        make_spline_coeff_guide(spline_basis, "c_vs_est_sl2", monotonic=None, init_mu=0.0)
        numpyro.sample("c_vs_est_br", dist.Delta(v=numpyro.param("loc_c_vs_est_br", 6.0)))

    for i in range(n_basin - 1):
        make_spline_coeff_guide(spline_basis, f"c_b_{i}", monotonic=None, init_mu=0.0)

    if include_region:
        for i in range(n_subregion):
            make_spline_coeff_guide(spline_basis, f"c_region_{i}", monotonic=None, init_mu=0.0)

    if func_gs_scaling != "stafford":
        if estimate_gs_exp == "coeff":
            numpyro.sample("gs_exp", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param("loc_log_gs_exp", 0.7)),
                transforms=dist.transforms.ExpTransform(),
            ))
        elif estimate_gs_exp == "freq":
            make_spline_coeff_guide(spline_basis, "gs_exp", monotonic=None, init_mu=0.0)
        # "fixed": not sampled -- no guide site.

    with numpyro.plate("plate_freq", n_freq, dim=-1):
        numpyro.sample("nu_rec", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_nu_rec", 6.0 * jnp.ones(n_freq))),
            transforms=dist.transforms.ExpTransform(),
        ))
        numpyro.sample("sigma", dist.TransformedDistribution(
            dist.Delta(v=numpyro.param("loc_log_sigma", -0.7 * jnp.ones(n_freq))),
            transforms=dist.transforms.ExpTransform(),
        ))
