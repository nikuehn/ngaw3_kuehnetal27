"""
Config-driven sampling of "shared vs. dataset-local" coefficients.

This is the piece that answers: "should c_m2 be one value shared between
WUS and global, or two independently-estimated values (c_m2, c_m2_gl)?"
-- driven by a `sharing_config` dict passed in by the caller (typically
`numpyro_models.py`, which can build its own dict, or a copy of
`DEFAULT_COEFFICIENT_SHARING` with a few entries overridden, and pass it
straight through) -- never hardcoded here, so it's editable right up to
the point you call the model, with no package code changes.

WUS and global always use the *same* prior structure for a given
coefficient (spline for spline, parametric-function for
parametric-function) -- only the sampled value differs when
'dataset_local'.

Coefficients that are inherently dataset-specific in a way that isn't
just "shared vs. separate sample site" -- c_basin/c_subregion (WUS
only, global has none), gs_break/gs_exp/zt_break (global uses fixed
literals, not a sample site), deltaB/deltaS/deltaB_attn (always
independent per dataset, never shared), and standard-deviation
components (phi_ss, tau, phi_s2s -- out of scope for now) -- stay
hand-written in `numpyro_models.py`.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ngaw3_kuehnetal27.spline_coeff import make_spline_coeff
from ngaw3_kuehnetal27.utils import logistic_hinge

Array = jnp.ndarray


# Default sharing config. Pass your own dict (e.g. a copy of this with a
# few entries flipped) as `sharing_config` to any function below --
# nothing in this module reads this constant directly.
DEFAULT_COEFFICIENT_SHARING: Dict[str, str] = {
    "c_0": "dataset_local",
    "c_m1": "dataset_local",
    "c_m2": "shared",
    "c_m3": "shared",
    "c_hw": "shared",
    "c_nft_1": "shared",
    "c_nft_2": "shared",
    "c_nm": "shared",
    "c_rev": "shared",
    "c_gs1": "dataset_local",
    "c_zt": "dataset_local",
    "c_vs": "dataset_local",   # always dataset_local in practice -- global has no measured/estimated split
    "c_attn": "dataset_local",
}


def resolve_shared_coefficient(
    base_name: str,
    make_sample: Callable[[str], Array],
    sharing: str,
    gl_suffix: str = "_gl",
) -> Tuple[Array, Array]:
    """
    Sample a coefficient once or twice depending on `sharing`, and
    return values for (wus, global).

    Parameters
    ----------
    base_name : str
        Coefficient name, e.g. 'c_m2'. Used as the WUS sample site name
        (and, if shared, the global site reuses the same value).
    make_sample : Callable[[str], Array]
        Given a site name, samples and returns that coefficient. Called
        twice (once per site name) when `sharing == 'dataset_local''`.
    sharing : {'shared', 'dataset_local'}
        Typically `sharing_config[base_name]`.
    gl_suffix : str
        Suffix appended to `base_name` for the global-specific sample
        site, when `sharing == 'dataset_local'`.

    Returns
    -------
    (value_wus, value_gl)
        `value_gl is value_wus` when `sharing == 'shared'`.
    """
    value_wus = make_sample(base_name)
    if sharing == "shared":
        return value_wus, value_wus
    elif sharing == "dataset_local":
        value_gl = make_sample(base_name + gl_suffix)
        return value_wus, value_gl
    else:
        raise ValueError(f"Unknown sharing mode {sharing!r} for {base_name!r}")


def sample_median_coefficient(
    base_name: str,
    spline_basis: Array,
    sharing_config: Dict[str, str],
    *,
    mu_loc=0, mu_scale=2,
    positive: bool = False,
    monotonic: str = None,
    transform: str = "exp",
    gl_suffix: str = "_gl",
) -> Tuple[Array, Array]:
    """
    Generic spline-based median coefficient, shared or dataset-local
    per `sharing_config[base_name]`. Covers every coefficient except
    `c_0` and `c_gs1` when their parametric form is active -- use
    `sample_c0_coefficient` / `sample_cgs1_coefficient` for those.

    Parameters
    ----------
    base_name : str
        Coefficient name, e.g. 'c_m1', 'c_zt', 'c_nm'.
    spline_basis : Array, shape (n_freq, n_spline_coefs)
        Same basis for WUS and global (same frequency grid).
    sharing_config : Dict[str, str]
        e.g. a copy of `DEFAULT_COEFFICIENT_SHARING`.
    mu_loc, mu_scale, positive, monotonic, transform :
        Passed straight through to `make_spline_coeff`.
    gl_suffix : str
        Suffix for the global-specific site when dataset_local.

    Returns
    -------
    (value_wus, value_gl)

    Example
    -------
        c_m1, c_m1_gl = sample_median_coefficient(
            "c_m1", spline_basis, sharing_config,
            mu_loc=1.5, mu_scale=1.0, positive=True,
            transform="softplus", monotonic="decreasing",
        )
    """
    def make_sample(name):
        return make_spline_coeff(
            spline_basis, name, mu_loc=mu_loc, mu_scale=mu_scale,
            positive=positive, transform=transform, monotonic=monotonic,
        )

    return resolve_shared_coefficient(
        base_name, make_sample, sharing_config[base_name], gl_suffix=gl_suffix,
    )


def sample_c0_coefficient(
    spline_basis: Array,
    ln_F: Array,
    F: Array,
    sharing_config: Dict[str, str],
    *,
    parametric: bool = False,
    gl_suffix: str = "_gl",
) -> Tuple[Array, Array, Array | None]:
    """
    ...same docstring...

    Returns
    -------
    (value_wus, value_gl, kappa_star_wus)
        `kappa_star_wus` is the WUS-side `c_0_kappa_star` sample when
        `parametric=True`, else `None`. (The global-side kappa_star, if
        dataset_local, is sampled but not returned -- nothing downstream
        currently needs it.)
    """
    kappa_star_wus = {"value": None}

    if parametric:
        def make_sample(name):
            ic = numpyro.sample(f"{name}_ic", dist.Normal(0, 2.0))
            slope1 = numpyro.sample(f"{name}_slope1", dist.LogNormal(0.35, 0.45))
            slope2 = numpyro.sample(f"{name}_slope2", dist.HalfNormal(0.2))
            fb = numpyro.sample(f"{name}_fb", dist.Normal(0, 1))
            delta = numpyro.sample(f"{name}_delta", dist.HalfNormal(0.2))
            kappa_star = numpyro.sample(f"{name}_kappa_star", dist.HalfNormal(0.3))
            if name == "c_0":
                kappa_star_wus["value"] = kappa_star
            return numpyro.deterministic(
                name,
                ic + logistic_hinge(ln_F, slope1, slope2, fb, delta) - kappa_star * F,
            )
    else:
        def make_sample(name):
            return make_spline_coeff(spline_basis, name, mu_loc=-5.0, mu_scale=5.0)

    value_wus, value_gl = resolve_shared_coefficient(
        "c_0", make_sample, sharing_config["c_0"], gl_suffix=gl_suffix,
    )
    return value_wus, value_gl, kappa_star_wus["value"]


def sample_cgs1_coefficient(
    spline_basis: Array,
    ln_F: Array,
    sharing_config: Dict[str, str],
    *,
    parametric: bool = False,
    gl_suffix: str = "_gl",
) -> Tuple[Array, Array]:
    """
    c_gs1 (near-source geometric spreading): dataset_local by default,
    with a parametric-vs-spline switch, applied identically to WUS and
    global.

    Spline form is constrained positive (softplus), matching how c_gs1
    is used downstream with a minus sign in the geometric-spreading term
    (physically, spreading should always reduce amplitude with
    distance).

    Parametric form: softplus(c_gs1_ic + logistic_hinge(ln_F, slope1,
    0.0, fb, delta)) -- slope2 is fixed at 0 (flat at high frequency,
    like the spline form's asymptotic behavior), and the whole
    logistic-hinge output is passed through softplus.

    Without the softplus wrap, the pre-transform value has slope1 > 0
    in log-frequency space at low frequency and 0 at high frequency --
    meaning it's unbounded *below* as frequency decreases, and can go
    negative (physically wrong sign, unlike the always-positive spline
    form). softplus fixes this: it leaves the high-frequency asymptote
    close to unchanged (softplus(x) ~= x for x well above 0, and
    c_gs1_ic's prior is centered such that this holds), while the
    low-frequency tail smoothly rolls over toward (but never below)
    zero instead of diverging.
    """
    if parametric:
        def make_sample(name):
            ic = numpyro.sample(f"{name}_ic", dist.Normal(1, 0.5))
            slope1 = numpyro.sample(f"{name}_slope1", dist.LogNormal(-0.8, 0.65))
            slope2 = 0.0
            fb = numpyro.sample(f"{name}_fb", dist.Normal(0, 1))
            delta = numpyro.sample(f"{name}_delta", dist.HalfNormal(0.2))
            raw = ic + logistic_hinge(ln_F, slope1, slope2, fb, delta)
            return numpyro.deterministic(name, jax.nn.softplus(raw))
    else:
        def make_sample(name):
            return make_spline_coeff(
                spline_basis, name, mu_loc=0.5, mu_scale=0.5,
                positive=True, transform="softplus",
            )

    return resolve_shared_coefficient(
        "c_gs1", make_sample, sharing_config["c_gs1"], gl_suffix=gl_suffix,
    )


def sample_coefficient_separate(
    base_name: str,
    sharing_config: Dict[str, str],
    *,
    mu_loc: float = 0.0,
    mu_scale: float = 2.0,
    positive: bool = False,
    transform: str = "exp",
    gl_suffix: str = "_gl",
) -> Tuple[Array, Array]:
    """
    Sample a coefficient independently at every frequency -- no spline,
    no parametric form, no smoothness assumption across frequency.
    Exactly the same idea as `nu_rec`: one independent draw per
    frequency. Same shared/dataset_local routing as
    `sample_median_coefficient` -- that's an orthogonal concern to
    whether a coefficient is spline-smoothed or frequency-independent.

    MUST be called inside `with numpyro.plate('plate_freq', n_freq,
    dim=-1):` -- the plate is what gives each frequency its own
    independent draw; this function doesn't declare the plate itself,
    since it's typically shared across many coefficients sampled
    together (see `numpyro_models.model_separate`).

    Parameters
    ----------
    base_name : str
    sharing_config : Dict[str, str]
    mu_loc, mu_scale : float
        Prior location/scale (in log space if positive=True and
        transform='exp', i.e. LogNormal(mu_loc, mu_scale)).
    positive : bool
    transform : {'exp', 'softplus'}
    gl_suffix : str

    Returns
    -------
    (value_wus, value_gl), each shape (n_freq,)
    """
    def make_sample(name):
        if not positive:
            return numpyro.sample(name, dist.Normal(mu_loc, mu_scale))
        if transform == "exp":
            return numpyro.sample(name, dist.LogNormal(mu_loc, mu_scale))
        elif transform == "softplus":
            return numpyro.sample(name, dist.TransformedDistribution(
                dist.Normal(mu_loc, mu_scale), dist.transforms.SoftplusTransform(),
            ))
        else:
            raise ValueError(f"Unknown transform: {transform!r}")

    return resolve_shared_coefficient(
        base_name, make_sample, sharing_config[base_name], gl_suffix=gl_suffix,
    )
