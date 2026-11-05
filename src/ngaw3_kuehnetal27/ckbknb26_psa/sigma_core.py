"""
Total standard deviation of ln(PSA) for the NGA-West3 CKBKNB26
(campbelletal27) GMM, ported from the "CALCULATE STANDARD DEVIATIONS" part
of `PSA_NGAW3_GMM.R` (function `NGAW3medianPSA`).

Only the four quantities a user normally wants are returned (one per call,
selected with `single_site`/`include_nl`):

    include_nl=False, single_site=False  ->  Sig_tot_lin
    include_nl=False, single_site=True   ->  Sig_ss_lin
    include_nl=True,  single_site=False  ->  Sig_tot_nln
    include_nl=True,  single_site=True   ->  Sig_ss_nln

The intermediate components the R function also tabulates (the nonlinear
Tau_0/Tau_r/Tau_path/Tau/Phi_s2s/Phi_ss, which do not enter Sig_*_nln) and
the lowess smoothing over period are not ported.

The nonlinear versions need more than magnitude, distance and the Vs30
measurement flag: the nonlinear site term's derivative (`Alpha`) and its own
standard deviation (`Phi_nln`) both depend on the rock-reference PSA
(`Ir = min(PSAr, 1 g)`) and on Vs30, and PSAr in turn depends on the whole
rupture/site scenario. `calculate_sigma_campbelletal27` therefore also
accepts the remaining scenario variables (keyword-only, same names and
NaN-means-back-fill conventions as
`median_core.calculate_ln_median_psa_campbelletal27`); they are ignored when
`include_nl=False`.

The linear-model sigma coefficients (Tau_0/Phi_ss at M5.3 and M6.7, Tau_r,
Phi_s2s for measured/proxy Vs30, Phi_lin) are read from `CoefMean.csv` along
with the median coefficients, so they come with `Coefficients`.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from .median_core import (
    Coefficients,
    _check_same_periods,
    _prepare_scenario,
    _rock_psa,
)
from .nonlinear_site import (
    IR_CAP,
    Ilhan26Coefficients,
    Ilhan26StdCoefficients,
    compute_alpha_ilhan26,
    compute_sig_fnln_ilhan26,
)

__all__ = ["calculate_sigma_campbelletal27"]

# Smoothness of the magnitude transition in `GenSigmoid` (R: delta=0.1).
_SIGMOID_DELTA = 0.1


def _gen_sigmoid(x, ln_s1, ln_s2, mb1, mb2, delta):
    """
    Generalized sigmoid of the R code (`GenSigmoid`): a smooth, log-linear
    transition from ``exp(ln_s1)`` below magnitude `mb1` to ``exp(ln_s2)``
    above `mb2`. ``log(1 + exp(z))`` is evaluated with `logaddexp` so large
    arguments cannot overflow.
    """
    a1 = ln_s1
    a2 = (ln_s2 - ln_s1) / (mb2 - mb1)
    return jnp.exp(
        a1
        + a2 * delta * jnp.logaddexp(0.0, (x - mb1) / delta)
        - a2 * delta * jnp.logaddexp(0.0, (x - mb2) / delta)
    )


def calculate_sigma_campbelletal27(
    coef: Coefficients,
    M,
    Rrup,
    Vs30_meas,
    single_site: bool = False,
    include_nl: bool = True,
    nl_coef: Optional[Ilhan26Coefficients] = None,
    nl_std_coef: Optional[Ilhan26StdCoefficients] = None,
    *,
    Vs30=760.0,
    Rjb=jnp.nan,
    Rx=jnp.nan,
    Frv=0.0,
    Fnm=0.0,
    W=jnp.nan,
    Dip=90.0,
    Ztor=jnp.nan,
    Zhyp=jnp.nan,
    Zbot=jnp.nan,
    Z25=jnp.nan,
    HW=0.0,
) -> jnp.ndarray:
    """
    Standard deviation of ln(PSA) for every (scenario, period) combination.

    Every scenario argument is broadcast to a common shape ``(n,)``, as in
    `calculate_ln_median_psa_campbelletal27`.

    Parameters
    ----------
    coef : Coefficients
        From `median_core.load_coefficients`; defines the periods.
    M : moment magnitude
    Rrup : closest distance to rupture (km); enters through the path term of
        the between-event standard deviation (distances beyond `Rcross`).
    Vs30_meas : 0 if Vs30 is measured, 1 if it is a proxy/estimate; selects
        `Phi_s2s_meas` or `Phi_s2s_prxy`. Has no effect when
        ``single_site=True``.
    single_site : bool, default False
        False returns the total standard deviation, which includes the
        site-to-site term `Phi_s2s` (R: `Sig_tot_*`). True returns the
        single-site standard deviation, without it (R: `Sig_ss_*`).
    include_nl : bool, default True
        True returns the nonlinear standard deviation (R: `Sig_*_nln`), i.e.
        the linear one rescaled by ``(1 + Alpha)`` plus the variability of
        the nonlinear amplification itself. False returns the linear model's
        (R: `Sig_*_lin`).
    nl_coef : Ilhan26Coefficients
        Required if `include_nl` (for `Alpha`). Load with
        `nonlinear_site.load_ilhan26_coefficients`, same `periods` as `coef`.
    nl_std_coef : Ilhan26StdCoefficients
        Required if `include_nl` (for `Phi_nln`). Load with
        `nonlinear_site.load_ilhan26_std_coefficients`, same `periods` as
        `coef`.
    Vs30, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot, Z25, HW :
        Remaining scenario variables, keyword-only, used only when
        `include_nl` is True (they determine the rock-reference PSAr that
        drives the nonlinear terms). Same meaning, defaults and
        NaN-means-back-fill behavior as in
        `calculate_ln_median_psa_campbelletal27`; pass the same values you
        used for the median so the two are consistent.

    Returns
    -------
    jnp.ndarray, shape (n, n_periods)
        Standard deviation in natural-log units.

    Notes
    -----
    `single_site` and `include_nl` are plain Python bools: under `jax.jit`
    mark them static (e.g. ``static_argnames=("single_site", "include_nl")``).
    """
    if include_nl and (nl_coef is None or nl_std_coef is None):
        raise ValueError(
            "include_nl=True needs both nl_coef and nl_std_coef (load them with "
            "load_ilhan26_coefficients / load_ilhan26_std_coefficients, same "
            "`periods` as coef); pass include_nl=False for the linear-model sigma."
        )

    if include_nl:
        _check_same_periods(coef, nl_coef, "nl_coef")
        _check_same_periods(coef, nl_std_coef, "nl_std_coef")
        sc = _prepare_scenario(M, Rrup, Rjb, Rx, Frv, Fnm, W, Dip, Ztor, Zhyp, Zbot,
                               Vs30, Vs30_meas, Z25, HW)
        M_, Rrup_, Vs30_meas_ = sc.M, sc.Rrup, sc.Vs30_meas
    else:
        M_, Rrup_, Vs30_meas_ = (
            v[:, None] for v in jnp.broadcast_arrays(*[
                jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)) for x in (M, Rrup, Vs30_meas)
            ])
        )

    c = coef

    # Linear standard deviations
    tau_0 = _gen_sigmoid(M_, jnp.log(c.Tau_0_M53), jnp.log(c.Tau_0_M67), c.m1, c.m2, _SIGMOID_DELTA)
    tau_path = c.Tau_r * (Rrup_ - c.Rcross) / 100.0 * (Rrup_ > c.Rcross)
    tau = tau_0 + tau_path
    phi_ss = _gen_sigmoid(M_, jnp.log(c.Phi_ss_M53), jnp.log(c.Phi_ss_M67), c.m1, c.m2, _SIGMOID_DELTA)

    var_lin = tau ** 2 + phi_ss ** 2
    if not single_site:
        phi_s2s = jnp.where(Vs30_meas_ == 0, c.Phi_s2s_meas, c.Phi_s2s_prxy)
        var_lin = var_lin + phi_s2s ** 2

    if not include_nl:
        return jnp.sqrt(var_lin)

    # Nonlinear adjustment
    ir = jnp.minimum(_rock_psa(coef, sc), IR_CAP)
    alpha = compute_alpha_ilhan26(sc.Vs30, ir, nl_coef)
    phi_nln = compute_sig_fnln_ilhan26(sc.Vs30, ir, nl_std_coef)

    return jnp.sqrt((var_lin - c.Phi_lin ** 2) * (1.0 + alpha) ** 2 + c.Phi_lin ** 2 + phi_nln ** 2)
