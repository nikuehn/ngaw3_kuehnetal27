"""
Spline-based coefficient sampling for numpyro models.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def make_spline_coeff(spline_basis, name, mu_loc=0, mu_scale=2,
                       positive=False, monotonic=None, transform="exp"):
    """
    Model a function of frequency as a spline with a fixed number of
    knots, as a numpyro prior.

    Parameters
    ----------
    spline_basis : Array, shape (n_freq, n_spline_coefs)
        Precomputed spline basis matrix (e.g. from `dmatrix(...)` in
        model.py). Evaluate it against your frequency/log-frequency grid
        before calling this function -- it's no longer computed here.
    name : str
        Name of the coefficient; used to construct numpyro sample-site
        names (`mu_{name}`, `sigma_spline_{name}`, etc.) and the final
        `numpyro.deterministic(name, ...)` site.
    mu_loc : float or array_like
        Prior mean for the overall mean of the spline.
    mu_scale : float or array_like
        Prior standard deviation for the overall mean of the spline.
    positive : bool
        Whether to constrain the coefficient to be positive.
    monotonic : {'increasing', 'decreasing', None}
        If set, enforces monotonicity in frequency via a cumulative sum
        of positive increments. If None, no constraint.
    transform : {'exp', 'softplus', 'square'}
        Transformation to use if positive=True.

    Returns
    -------
    Array, shape (n_freq,)
        The spline evaluated at the frequencies implied by `spline_basis`.
    """
    mu = numpyro.sample(f"mu_{name}", dist.Normal(mu_loc, mu_scale))

    n_spline_coefs = spline_basis.shape[1]

    sigma_spline = numpyro.sample(f"sigma_spline_{name}", dist.HalfNormal(1))
    if monotonic == "increasing":
        spline_coefs_raw = numpyro.sample(
            f"spline_coefs_raw_{name}",
            dist.HalfNormal(jnp.ones(n_spline_coefs)),
        )
        spline_coefs = jnp.cumsum(sigma_spline * spline_coefs_raw)
    elif monotonic == "decreasing":
        spline_coefs_raw = numpyro.sample(
            f"spline_coefs_raw_{name}",
            dist.HalfNormal(jnp.ones(n_spline_coefs)),
        )
        spline_coefs = -jnp.cumsum(sigma_spline * spline_coefs_raw)
    else:
        spline_coefs_raw = numpyro.sample(
            f"spline_coefs_raw_{name}",
            dist.ZeroSumNormal(1.0, event_shape=(n_spline_coefs,)),
        )
        spline_coefs = sigma_spline * spline_coefs_raw

    spline_term = mu + jnp.dot(spline_basis, spline_coefs)

    if positive:
        if transform == "exp":
            coeff = jnp.exp(spline_term)
        elif transform == "softplus":
            coeff = jax.nn.softplus(spline_term)
        elif transform == "square":
            coeff = spline_term ** 2
        else:
            raise ValueError(f"Unknown transform: {transform}")
        return numpyro.deterministic(name, coeff)
    else:
        return numpyro.deterministic(name, spline_term)
