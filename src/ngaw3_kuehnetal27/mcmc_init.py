"""
Initial values for MCMC (NUTS) from a fitted SVI model.
"""
from __future__ import annotations

import gzip
import json
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.distributions.transforms import biject_to

from ngaw3_kuehnetal27.svi_fitting import to_jax_arrays


def load_svi_params(path) -> Dict[str, Any]:
    """Read the `results_orig_{filestem}.json.gz` file written by
    `write_svi_results` back into a dict of jax arrays (usable as the
    `svi_params` of `init_values_from_svi`)."""
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return to_jax_arrays(json.load(f))


def init_values_from_svi(
    model: Callable,
    guide: Callable,
    svi_params: Dict[str, Any],
    data_dict: Dict[str, Any],
    *,
    rng_key=None,
    noise_scale: float = 0.05,
    seed: int = 1701,
) -> Dict[str, Any]:
    """
    Constrained-scale initial values for every sample site of `model`,
    taken from a fitted SVI guide; pass the result to
    `numpyro.infer.init_to_value(values=...)`.

    Parameters
    ----------
    model, guide : Callable
        The model/guide pair used for the SVI fit.
    svi_params : dict
        Fitted `numpyro.param` values (`SVIFitResult.params`, or
        `load_svi_params(".../results_orig_{filestem}.json.gz")`).
    data_dict : dict
        The same arguments used for the SVI fit (including e.g.
        `zerosumnormal`).
    rng_key : jax PRNG key, optional
        If given (and noise_scale > 0), noise is added to every site -- use a
        different key per chain.
    noise_scale : float
        Standard deviation of the Gaussian noise added on the *unconstrained*
        scale of each site (log scale for positive sites, free coordinates for
        zero-sum sites), so the perturbed values always stay inside the
        site's support. 0.05 on a log-scale parameter is ~5 % multiplicative.
    seed : int
        Only used to trace the model and guide.

    How the values are obtained
    ---------------------------
    The guide is run with the fitted params, so site names and constrained
    values come out right for every guide component (Delta sites, `loc_log_*`
    positive sites, zero-sum spline coefficients, ...) without parsing param
    names. Sites with a stochastic guide (Normal / ZeroSumNormal /
    MultivariateNormal random effects) are set to their fitted location
    `loc_{site}` instead of a random draw.
    """
    key = jax.random.key(seed)

    guide_trace = handlers.trace(
        handlers.substitute(handlers.seed(guide, key), data=svi_params)
    ).get_trace(**data_dict)
    values = {
        name: site["value"] for name, site in guide_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    for name in values:
        if f"loc_{name}" in svi_params and (
            f"scale_{name}" in svi_params or f"scale_tril_{name}" in svi_params
        ):
            values[name] = svi_params[f"loc_{name}"]

    model_trace = handlers.trace(handlers.seed(model, key)).get_trace(**data_dict)
    sites = {
        name: site for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"] and name in values
    }

    init = {}
    names = sorted(sites)
    keys = (jax.random.split(rng_key, len(names))
            if rng_key is not None and noise_scale > 0 and names else [None] * len(names))
    for name, k in zip(names, keys):
        site = sites[name]
        value = jnp.asarray(values[name])
        if value.shape != site["value"].shape:
            raise ValueError(
                f"Site '{name}': shape from the guide/SVI params {value.shape} does not "
                f"match the model site {site['value'].shape} -- model/guide/data_dict "
                "arguments differ from the ones used for the SVI fit?"
            )
        if k is not None:
            transform = biject_to(site["fn"].support)
            u = transform.inv(value)
            u = u + noise_scale * jax.random.normal(k, jnp.shape(u))
            value = transform(u)
        init[name] = value
    return init
