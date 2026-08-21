"""
Ground-motion variance-component evaluation over sampled scenarios.

Used both for EAS (RVT input, mean-vs-median correction) and later for
PSA (calibration step) -- same functional form, different fitted
summary_stats values passed in. Other shared variability-related
helpers can live here too.
"""
from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from ngaw3_kuehnetal27.utils import smooth_trilinear_ramp_repar

Array = jnp.ndarray

MB1 = 5.3
MB2 = 6.7
DELTA = 0.2


def calculate_variance(
    magnitudes: Array,
    rrups: Array,
    vsmeas_id: Array,
    summary_stats: Dict[str, Any],
    single_site: bool = False,
) -> Array:
    """
    Total variance (tau^2 + phi_ss^2 + attenuation-related tau_attn
    contribution, plus phi_s2s^2 unless `single_site`) for each sampled
    scenario, at every frequency in `summary_stats`.

    Parameters
    ----------
    magnitudes, rrups : Array, shape (n_sample,)
    vsmeas_id : Array, shape (n_sample,)
        0 = measured Vs30, 1 = estimated -- selects phi_s2s_meas vs.
        phi_s2s_est per scenario (same convention as `coefficients_
        from_site_values`'s `c_vs = [c_vs_meas, c_vs_est]` stacking).
        Ignored when `single_site=True`.
    summary_stats : dict
        Resolved SVI site values (same object as `site_values` used
        elsewhere -- `summary_stats` is just this function's name for
        it). Must contain, each shape (n_freq,): tau_0, tau_1,
        phi_ss_0, phi_ss_1, tau_attn, and (unless `single_site`)
        phi_s2s_meas, phi_s2s_est.
    single_site : bool, default False
        If True, excludes phi_s2s (site-to-site variability) from the
        total -- i.e. the variance of ground motion at a single,
        already-known site, rather than the full population variance
        across sites. If False (default), phi_s2s is included.

    Returns
    -------
    var : Array, shape (n_sample, n_freq)
        Total variance (not std. dev.) at each frequency.

    Notes
    -----
    tau and phi_ss are magnitude-dependent via `smooth_trilinear_ramp_
    repar`, breakpoints at M 5.3/6.7 (delta=0.2) -- matching the fitting
    model's own parameterization. tau_attn scales with rrup/100 (an
    attenuation-related between-event term, not magnitude-dependent).
    phi_s2s does not depend on magnitude or distance, only site
    (measured vs. estimated Vs30).
    """
    magnitudes = jnp.asarray(magnitudes)
    rrups = jnp.asarray(rrups)

    tau = jnp.exp(smooth_trilinear_ramp_repar(
        magnitudes[:, jnp.newaxis],
        summary_stats["tau_0"], summary_stats["tau_1"],
        MB1, MB2, delta=DELTA,
    ))
    phi_ss = jnp.exp(smooth_trilinear_ramp_repar(
        magnitudes[:, jnp.newaxis],
        summary_stats["phi_ss_0"], summary_stats["phi_ss_1"],
        MB1, MB2, delta=DELTA,
    ))

    var = (
        tau ** 2
        + phi_ss ** 2
        + summary_stats["tau_attn"] ** 2 * (rrups[:, jnp.newaxis] / 100) ** 2
    )

    if not single_site:
        vsmeas_id = jnp.asarray(vsmeas_id)
        phi_s2s_meas = jnp.asarray(summary_stats["phi_s2s_meas"])[jnp.newaxis, :]
        phi_s2s_est = jnp.asarray(summary_stats["phi_s2s_est"])[jnp.newaxis, :]
        phi_s2s = jnp.where(vsmeas_id[:, jnp.newaxis] == 0, phi_s2s_meas, phi_s2s_est)
        var = var + phi_s2s ** 2

    return var
