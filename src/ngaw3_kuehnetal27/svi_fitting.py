"""
SVI fitting helper, usable for any model/guide pair (EAS, PSA, ...).
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import numpy as np
import numpyro
import optax
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO


@dataclass
class SVIFitResult:
    """
    params : Dict[str, Any]
        Fitted numpyro.param values -- what you need for prediction,
        further analysis, or `write_svi_results`.
    losses : Array
        Per-step ELBO loss, for a convergence check (e.g. plot on a log
        scale).
    svi_result : Any
        The raw numpyro SVIRunResult, in case you need something beyond
        params/losses.
    """
    params: Dict[str, Any]
    losses: Any
    svi_result: Any


def run_svi(
    model: Callable,
    guide: Callable,
    data_dict: Dict[str, Any],
    *,
    num_steps: int = 2000,
    seed: int = 1701,
    learning_rate: float = 1e-4,
    clip_norm: float = 1.0,
    optimizer=None,
    loss=None,
) -> SVIFitResult:
    """
    Run SVI for a given model/guide pair and data dict.

    Parameters
    ----------
    model, guide : Callable
        numpyro model and guide functions -- pass explicitly so this
        works for any model (e.g. model_eas/guide_eas today, a PSA
        model/guide later).
    data_dict : dict
        All keyword arguments model/guide need (F, X_rec, X_eq, ...).
    num_steps : int
        Number of SVI optimization steps.
    seed : int
        RNG seed.
    learning_rate : float
        Adam learning rate. Ignored if `optimizer` is given.
    clip_norm : float
        Global-norm gradient clipping threshold. Ignored if `optimizer`
        is given.
    optimizer : numpyro optimizer, optional
        Overrides the default clip_by_global_norm + adam chain.
    loss : numpyro ELBO loss, optional
        Defaults to `Trace_ELBO()`.

    Returns
    -------
    SVIFitResult

    Example
    -------
        result = run_svi(model_eas, guide_eas, data_dict, num_steps=2000)
        result.params   # fitted params
        result.losses   # for a convergence plot
    """
    rng_key = jax.random.key(seed)

    if optimizer is None:
        optimizer = numpyro.optim.optax_to_numpyro(
            optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate))
        )
    if loss is None:
        loss = Trace_ELBO()

    svi = SVI(model, guide, optimizer, loss=loss)
    svi_result = svi.run(rng_key, num_steps, **data_dict)

    return SVIFitResult(params=svi_result.params, losses=svi_result.losses, svi_result=svi_result)


# ---------------------------------------------------------------------------
# Writing out SVI results
# ---------------------------------------------------------------------------

def convert_to_json_serializable(obj):
    """Recursively convert JAX/numpy arrays (and containers of them) to
    plain Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "tolist"):  # JAX/numpy arrays
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(i) for i in obj]
    else:
        return obj


def resolve_svi_site_values(
    model: Callable,
    guide: Callable,
    svi_params: Dict[str, Any],
    data_dict: Dict[str, Any],
    *,
    seed: int = 1701,
) -> Dict[str, Any]:
    rng_key = jax.random.key(seed)

    guide_trace = handlers.trace(
        handlers.substitute(handlers.seed(guide, rng_key), data=svi_params)
    ).get_trace(**data_dict)

    site_values = {
        name: site["value"] for name, site in guide_trace.items() if site["type"] == "sample"
    }

    # Swap in each stochastic site's fitted location (and attach its
    # fitted spread) BEFORE recomputing model-side deterministics --
    # detected generically from svi_params, exactly as before. This
    # ordering is what makes m_region/kappa_adj/c_region (anything
    # downstream of a random-effect sample) get built from the
    # location/mode rather than one arbitrary draw.
    for param_name, value in svi_params.items():
        if not param_name.startswith("loc_"):
            continue
        site_name = param_name[len("loc_"):]
        scale_key = f"scale_{site_name}"
        scale_tril_key = f"scale_tril_{site_name}"
        if scale_key in svi_params:
            site_values[site_name] = value
            site_values[scale_key] = svi_params[scale_key]
        elif scale_tril_key in svi_params:
            site_values[site_name] = value
            site_values[scale_tril_key] = svi_params[scale_tril_key]

    model_trace = handlers.trace(
        handlers.substitute(model, data=site_values)
    ).get_trace(**data_dict)

    deterministics = {
        name: site["value"] for name, site in model_trace.items() if site["type"] == "deterministic"
    }
    site_values.update(deterministics)

    
    n_freq = len(data_dict['F'])
    if 'L_freq' not in data_dict or data_dict['L_freq'] is None:
        L_freq = jnp.eye(n_freq)
    else:
        L_freq = jnp.asarray(data_dict['L_freq'])

    L_subregion = site_values["sigma_region"][..., None] * L_freq
    var_c_region = (L_subregion**2) @ (svi_params["scale_c_region_raw"]**2).T
    scale_c_region = jnp.sqrt(var_c_region).T
    site_values["scale_c_region"] = scale_c_region

    if "scale_ln_kappa_region_raw" in svi_params:
        site_values["scale_ln_kappa_region"] = site_values["sigma_ln_kappa_region"] * svi_params["scale_ln_kappa_region_raw"]
    if "scale_ln_kappa_station_raw" in svi_params:
        site_values["scale_ln_kappa_station"] = site_values["sigma_ln_kappa_station"] * svi_params["scale_ln_kappa_station_raw"]

    return site_values


def write_svi_results(
    model: Callable,
    guide: Callable,
    svi_params: Dict[str, Any],
    data_dict: Dict[str, Any],
    results_dir,
    filestem: str,
    *,
    frequencies=None,
    seed: int = 1701,
) -> None:
    """
    Write out fitted SVI results as three JSON files:

    - `results_orig_{filestem}.json` -- raw fitted numpyro.param values.
    - `results_{filestem}.json` -- resolved site values: model
      deterministics (actual coefficients) plus random-effect / uncertain
      site locations and scales (see `resolve_svi_site_values`).
    - `data_{filestem}.json` -- the data_dict used for this run, so you
      know which options produced these results.

    Parameters
    ----------
    model, guide : Callable
    svi_params : dict
        `SVIFitResult.params` from `run_svi`.
    data_dict : dict
    results_dir : str or Path
        Directory to write into (created if it doesn't exist).
    filestem : str
        Used to build all three output filenames.
    frequencies : array-like, optional
        If given, added to the resolved results under 'frequency'.
    seed : int
    """
    results_dir = str(results_dir)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"results_dir does not exist: {results_dir!r}. Create it first -- "
            "not auto-created, to avoid silently writing into an unintended folder."
        )

    with gzip.open(os.path.join(results_dir, f"results_orig_{filestem}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(convert_to_json_serializable(svi_params), f, indent=2)

    site_values = resolve_svi_site_values(model, guide, svi_params, data_dict, seed=seed)
    if frequencies is not None:
        site_values["frequency"] = frequencies

    with gzip.open(os.path.join(results_dir, f"results_{filestem}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(convert_to_json_serializable(site_values), f, indent=2)

    with gzip.open(os.path.join(results_dir, f"data_{filestem}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(convert_to_json_serializable(data_dict), f, indent=2)
