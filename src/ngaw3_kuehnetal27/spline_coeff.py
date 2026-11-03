"""
Spline-based coefficient sampling for numpyro models.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from patsy import dmatrix


def make_spline_basis(F, spline_degree=3, spline_df=7):
    """
    B-spline basis in ln(frequency), shared by every model and guide that
    models a coefficient as a smooth function of frequency
    (`make_spline_coeff`, `make_spline_coeff_guide`).

    `spline_df` equally spaced points between min and max of ln(F) are
    generated and the end points are dropped twice, leaving
    `spline_df - 4` interior knots. With `include_intercept=True` the
    basis then has `spline_df - 4 + spline_degree + 1` columns, i.e.
    `spline_df` for cubic splines. Model and guide must be given the same
    F / spline_degree / spline_df so that both see the same basis.

    Parameters
    ----------
    F : array_like, shape (n_freq,)
        Frequencies (Hz), or 1/period for PSA models.
    spline_degree : int
    spline_df : int

    Returns
    -------
    patsy DesignMatrix, shape (n_freq, n_basis) -- pass it as
    `spline_basis` to `make_spline_coeff` / `make_spline_coeff_guide`.
    """
    ln_F = np.log(F)
    knot_list = np.linspace(np.min(ln_F), np.max(ln_F), spline_df)[1:-1]
    return dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True) - 1",
        {"x": ln_F, "knots": knot_list[1:-1], "degree": spline_degree},
    )


def make_spline_coeff(spline_basis, name, mu_loc=0, mu_scale=2,
                       positive=False, monotonic=None, transform="exp"):
    """
    Model a function of frequency as a spline with a fixed number of
    knots, as a numpyro prior.

    Parameters
    ----------
    spline_basis : Array, shape (n_freq, n_spline_coefs)
        Precomputed spline basis matrix from `make_spline_basis`. It is
        evaluated on the frequency grid beforehand, not inside this function.
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
