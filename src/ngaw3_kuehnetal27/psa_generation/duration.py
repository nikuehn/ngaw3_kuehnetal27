"""
Significant duration (D5-75) for use as RVT input.

Vectorized JAX port of `pygmm.pinilla_ramos_et_al_2023.PinillaRamosEtAl2023`
(Pinilla-Ramos et al., 2023), median D5-75 branch only (energy=0.75) --
coefficients copied directly from that source. No fault-mechanism
dependence: the fitted model doesn't use it, despite `pygmm`'s API
accepting a `mechanism` scenario field.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray


class Duration(NamedTuple):
    median: Array
    plus_sigma: Array
    minus_sigma: Array


def duration_pinilla_ramos_d575(mag: Array, rrup: Array, vs30: Array) -> Duration:
    """
    Median D5-75 significant duration and its +-1 sigma bounds.

    Parameters
    ----------
    mag, rrup, vs30 : Array, shape (n,)

    Returns
    -------
    Duration
        `.median`, `.plus_sigma`, `.minus_sigma`, each shape (n,), in
        seconds. `plus_sigma`/`minus_sigma` follow the model's own
        power-mean parameterization, (median**n2 +- sigma)**(1/n2) --
        not a lognormal sigma multiplier. `minus_sigma` is not clamped
        to be non-negative (matching pygmm); check for NaN/negative
        values downstream for scenarios where sigma > median**n2 (short
        distance, small magnitude).
    """
    mag = jnp.asarray(mag, dtype=float)
    rrup = jnp.asarray(rrup, dtype=float)
    vs30 = jnp.asarray(vs30, dtype=float)

    n2 = 0.3

    # Magnitude scaling coefficients
    c1 = 3.655
    c21, c22, c23, c24, c2base = 0.41, 0.455, 0.54, 0.575, 0.515
    r1_c2, r2_c2, r3_c2 = 10, 42, 200

    # Distance scaling coefficients
    c31, c312, c32 = 0.063, 0.034, 0.083
    c3_base = 0.041

    # Site scaling coefficients (median)
    c4 = -0.619
    phi, phi_max = 0.565, 1.11
    v1, v2, v3 = 200, 275, 2000
    s1 = 0.278
    mth = 6.75

    c2 = jnp.where(
        rrup <= r1_c2, c21 + (c22 - c21) * rrup / r1_c2,
        jnp.where(
            rrup <= r2_c2, c22 + (c23 - c22) * (rrup - r1_c2) / (r2_c2 - r1_c2),
            c23 + (c24 - c23) * (rrup - r2_c2) / (r3_c2 - r2_c2),
        ),
    )

    source_term = jnp.where(
        mag < mth,
        c1 * 10 ** ((mag - 6.75) * c2base),
        c1 * 10 ** ((mag - 6.75) * c2),
    )

    r1, r2 = 44, 130
    path_term = (
        c3_base * rrup
        + (rrup < r1) * (c31 * rrup)
        + (rrup >= r1) * (rrup < r2) * (c31 * r1 + c312 * (rrup - r1))
        + (rrup >= r2) * (c31 * r1 + c312 * (r2 - r1) + c32 * (rrup - r2))
    )

    sigma_vs30 = (
        (vs30 < v3) * (vs30 >= v3 * 0.95) * phi
        * (jnp.log(v3) - jnp.log(vs30)) / (jnp.log(v3) - jnp.log(v3 * 0.95))
        + (vs30 < v3 * 0.95) * (vs30 >= v2) * phi
        + (vs30 < v2) * (vs30 >= v1)
        * (phi + (phi_max - phi) * (jnp.log(v2) - jnp.log(vs30)) / (jnp.log(v2) - jnp.log(v1)))
        + (vs30 < v1) * phi_max
    )
    site_term = c4 * jnp.log(vs30 / v3) * jnp.exp(s1 * sigma_vs30)

    median = source_term + path_term + site_term

    # Standard deviation coefficients
    a0, a1, a2, b1, b2, d1, d2, d3, v4 = (
        0.537, -0.093, 0.0278, -0.0372, 0.00179, 0.0206, 2.401, 0.0419, 200,
    )

    sigma_vs30_std = jnp.minimum(d1 * (v4 / vs30) ** d2, d3)
    sigma = (
        a0
        + a1 * (rrup / 100)
        + a2 * (rrup / 100) ** 2
        + b1 * mag
        + b2 * mag ** 2
        + sigma_vs30_std
    )

    plus_sigma = (median ** n2 + sigma) ** (1 / n2)
    minus_sigma = (median ** n2 - sigma) ** (1 / n2)

    return Duration(median=median, plus_sigma=plus_sigma, minus_sigma=minus_sigma)
