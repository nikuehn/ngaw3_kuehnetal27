"""
Guide (SVI) version of spline-coefficient sampling: every latent site is
a point mass (Delta), optionally transformed to enforce positivity --
this is the MAP-style guide component for coefficients. Random effects
and subregion terms use their own guide code directly in `guide.py`
(Normal / MultivariateNormal, to integrate them out rather than point-
estimate them).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def make_spline_coeff_guide(spline_basis, name, monotonic=None,
                             init_mu=0.0, init_sigma=-2.3):
    """
    Guide component matching `make_spline_coeff` in spline_coeff.py --
    same site names (`mu_{name}`, `sigma_spline_{name}`,
    `spline_coefs_{name}` or `spline_coefs_raw_{name}`), each a Delta
    (point mass) at a learned numpyro.param, so this behaves as a MAP
    estimate for that coefficient.

    Parameters
    ----------
    spline_basis : Array, shape (n_freq, n_spline_coefs)
    name : str
        Must match the coefficient name used in the model.
    monotonic : {'increasing', 'decreasing', None}
        Must match the model's setting for this coefficient.
    init_mu : float
        Initial value for the overall mean.
    init_sigma : float
        Initial value (in log space) for the spline smoothness scale.
    """
    n_spline_coefs = spline_basis.shape[1]

    numpyro.sample(
        f"mu_{name}",
        dist.Delta(v=numpyro.param(f"loc_mu_{name}", init_value=init_mu)),
    )

    numpyro.sample(
        f"sigma_spline_{name}",
        dist.TransformedDistribution(
            dist.Delta(v=numpyro.param(f"loc_log_sigma_spline_{name}", init_value=init_sigma)),
            transforms=dist.transforms.ExpTransform(),
        ),
    )

    if monotonic in ("increasing", "decreasing"):
        numpyro.sample(
            f"spline_coefs_raw_{name}",
            dist.TransformedDistribution(
                dist.Delta(
                    v=numpyro.param(
                        f"loc_log_spline_coefs_raw_{name}",
                        init_value=jnp.zeros(n_spline_coefs),
                    )
                ),
                transforms=dist.transforms.ExpTransform(),
            ),
        )
    else:
        numpyro.sample(
            f"spline_coefs_raw_{name}",
            dist.Delta(
                v=numpyro.param(
                    f"loc_spline_coefs_{name}",
                    init_value=jnp.zeros(n_spline_coefs),
                    constraint=dist.constraints.zero_sum(1),
                )
            ),
        )
