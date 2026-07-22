"""
Guide-side counterpart to `coefficient_sharing.py`.

Every latent site the model declares needs a matching guide site with
the same name and shape. Since the model routes coefficient sampling
through `resolve_shared_coefficient` (shared -> one site; dataset_local
-> two), the guide must branch on the *same* `sharing_config` the same
way, or the site sets between model and guide won't match and SVI will
fail immediately.

IMPORTANT: call `model(..., sharing_config=cfg)` and
`guide(..., sharing_config=cfg)` with the identical `cfg` (and identical
c0_parametric / cgs1_parametric / attn_Q / dist_cell / estimate_gs_break
/ estimate_zt_break / calc_nft / attn_eq / global_dict-is-None-or-not /
func_gs_scaling / estimate_gs_exp) for every SVI run -- any mismatch in
these produces a different set of sample sites and numpyro will error
(or silently miss parameters) rather than warn about it.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpyro
import numpyro.distributions as dist

from ngaw3_kuehnetal27.spline_coeff_guide import make_spline_coeff_guide


def resolve_shared_coefficient_guide(
    base_name: str,
    make_guide_sample: Callable[[str], None],
    sharing: str,
    gl_suffix: str = "_gl",
) -> None:
    """
    Declare guide site(s) for one coefficient, mirroring
    `resolve_shared_coefficient` in the model.

    'shared'        -> one guide site (base_name) -- no separate global
                        site exists in the model, so none is declared here.
    'dataset_local' -> two guide sites (base_name, base_name + gl_suffix).
    """
    make_guide_sample(base_name)
    if sharing == "dataset_local":
        make_guide_sample(base_name + gl_suffix)
    elif sharing != "shared":
        raise ValueError(f"Unknown sharing mode {sharing!r} for {base_name!r}")


def sample_median_coefficient_guide(
    base_name: str,
    spline_basis,
    sharing_config: Dict[str, str],
    *,
    monotonic: str = None,
    init_mu: float = 0.0,
    init_sigma: float = -2.3,
    gl_suffix: str = "_gl",
) -> None:
    """Guide counterpart to `sample_median_coefficient`."""
    def make_guide_sample(name):
        make_spline_coeff_guide(
            spline_basis, name, monotonic=monotonic,
            init_mu=init_mu, init_sigma=init_sigma,
        )

    resolve_shared_coefficient_guide(
        base_name, make_guide_sample, sharing_config[base_name], gl_suffix=gl_suffix,
    )


def sample_c0_coefficient_guide(
    spline_basis,
    sharing_config: Dict[str, str],
    *,
    parametric: bool = False,
    gl_suffix: str = "_gl",
) -> None:
    """
    Guide counterpart to `sample_c0_coefficient`. Parametric form
    declares Delta guide sites for ic/slope1/slope2/fb/delta/kappa_star;
    spline form delegates to `make_spline_coeff_guide`. Applied to both
    WUS and global site names when dataset_local, matching the model.
    """
    if parametric:
        def make_guide_sample(name):
            numpyro.sample(f"{name}_ic", dist.Delta(v=numpyro.param(f"loc_{name}_ic", 0.0)))
            numpyro.sample(f"{name}_slope1", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_slope1", 0.7)),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample(f"{name}_slope2", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_slope2", -2.3)),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample(f"{name}_fb", dist.Delta(v=numpyro.param(f"loc_{name}_fb", 0.0)))
            numpyro.sample(f"{name}_delta", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_delta", -2.3)),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample(f"{name}_kappa_star", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_kappa_star", -2.3)),
                transforms=dist.transforms.ExpTransform(),
            ))
    else:
        def make_guide_sample(name):
            make_spline_coeff_guide(spline_basis, name, monotonic=None, init_mu=-5.0)

    resolve_shared_coefficient_guide(
        "c_0", make_guide_sample, sharing_config["c_0"], gl_suffix=gl_suffix,
    )


def sample_cgs1_coefficient_guide(
    spline_basis,
    sharing_config: Dict[str, str],
    *,
    parametric: bool = False,
    gl_suffix: str = "_gl",
) -> None:
    """
    Guide counterpart to `sample_cgs1_coefficient`. Parametric form
    declares ic/slope1/fb/delta only (slope2 is fixed at 0 in the model,
    not sampled).
    """
    if parametric:
        def make_guide_sample(name):
            numpyro.sample(f"{name}_ic", dist.Delta(v=numpyro.param(f"loc_{name}_ic", 1.3)))
            numpyro.sample(f"{name}_slope1", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_slope1", -0.7)),
                transforms=dist.transforms.ExpTransform(),
            ))
            numpyro.sample(f"{name}_fb", dist.Delta(v=numpyro.param(f"loc_{name}_fb", 0.0)))
            numpyro.sample(f"{name}_delta", dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}_delta", -2.3)),
                transforms=dist.transforms.ExpTransform(),
            ))
    else:
        def make_guide_sample(name):
            make_spline_coeff_guide(spline_basis, name, monotonic=None, init_mu=-5.0)

    resolve_shared_coefficient_guide(
        "c_gs1", make_guide_sample, sharing_config["c_gs1"], gl_suffix=gl_suffix,
    )


def sample_coefficient_separate_guide(
    base_name: str,
    n_freq: int,
    sharing_config: Dict[str, str],
    *,
    positive: bool = False,
    transform: str = "exp",
    init_mu: float = 0.0,
    gl_suffix: str = "_gl",
) -> None:
    """
    Guide counterpart to `sample_coefficient_separate`: a Delta point
    mass at every frequency independently, shape (n_freq,) -- same
    idea as `nu_rec`'s guide site. Should be called inside the same
    `with numpyro.plate('plate_freq', n_freq, dim=-1):` block the model
    uses, for consistency (Delta guides don't strictly require matching
    plates for ELBO correctness, but it keeps guide and model in the
    same shape/structure).

    Parameters
    ----------
    base_name : str
    n_freq : int
    sharing_config : Dict[str, str]
    positive : bool
    transform : {'exp', 'softplus'}
    init_mu : float
        Initial value (in log space if positive=True) for every frequency.
    gl_suffix : str
    """
    import jax.numpy as jnp

    def make_guide_sample(name):
        if not positive:
            numpyro.sample(name, dist.Delta(
                v=numpyro.param(f"loc_{name}", init_mu * jnp.ones(n_freq)),
            ))
        elif transform == "exp":
            numpyro.sample(name, dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}", init_mu * jnp.ones(n_freq))),
                transforms=dist.transforms.ExpTransform(),
            ))
        elif transform == "softplus":
            numpyro.sample(name, dist.TransformedDistribution(
                dist.Delta(v=numpyro.param(f"loc_log_{name}", init_mu * jnp.ones(n_freq))),
                transforms=dist.transforms.SoftplusTransform(),
            ))
        else:
            raise ValueError(f"Unknown transform: {transform!r}")

    resolve_shared_coefficient_guide(
        base_name, make_guide_sample, sharing_config[base_name], gl_suffix=gl_suffix,
    )
