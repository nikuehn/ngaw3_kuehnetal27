"""
Parametric extrapolation of the EAS spectrum beyond the fitted frequency
band, as an alternative to the "legacy" extrapolation inside
`rvt.rvt_psa_batch` (omega-square slope of 2 below, linear-in-ln(f)
interpolation to a kappa-filtered end point above).

Idea
----
Outside the fitted band every spline-modelled coefficient is held at its
value at the nearest band edge. Only the terms of `calculate_median_core`
with an explicit, parametric frequency dependence keep changing:

    c_0(f)      = ic + logistic_hinge(ln f; slope1, slope2, fb, delta)
                  - kappa_star * f                      (c0_parametric=True)
    c_gs1(f)    = softplus(ic + logistic_hinge(ln f; slope1, 0, fb, delta))
                                                        (cgs1_parametric=True)
    regional kappa deviation:   - kappa_adj * f
    anelastic attenuation (attn_mode "Q" or "dist_cell"):
                  - pi * f**Q_exp * path / c_Q,  path = R/Q_0  or  dist_cell @ (1/Q_0)

Because everything else cancels in the difference to the edge value,

    ln_eas(f) = ln_eas(f_edge)
                + [c_0(f) - c_0(f_edge)]
                + [c_gs1(f) - c_gs1(f_edge)] * G_rec
                - kappa_adj_rec * (f - f_edge)
                - pi * path_rec * (f**Q_exp - f_edge**Q_exp) / c_Q

with G_rec = d(func_mag_gs)/d(c_gs1) for each record (the geometric-
spreading term is linear in c_gs1). `parametric_extrap_delta` returns the
additive terms; `rvt.rvt_psa_batch(..., extrapolation="parametric",
extrap_delta=...)` adds them to the edge predictions. Inside the band the
log-linear interpolation of the legacy scheme is unchanged.

Neglected on purpose (held at the edge value): regional/basin terms, the
nonlinear site term f_nl, c_attn (spline attenuation), 1D amplification.

Units: kappa enters as `kappa_star` (the same quantity as in the median,
exp(-kappa_star * f)), i.e. no division by pi as for `site_atten`.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

from ngaw3_kuehnetal27.utils import func_mag_gs, func_mag_gs_lh, logistic_hinge

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Parametric coefficient functions (must mirror coefficient_sharing.py)
# ---------------------------------------------------------------------------

def make_c0_fn(ic, slope1, slope2, fb, delta, kappa_star) -> Callable[[Array], Array]:
    """f (Hz) -> c_0(f); same formula as `sample_c0_coefficient`."""
    def c0(F):
        F = jnp.asarray(F)
        return ic + logistic_hinge(jnp.log(F), slope1, slope2, fb, delta) - kappa_star * F
    return c0


def make_cgs1_fn(ic, slope1, fb, delta) -> Callable[[Array], Array]:
    """f (Hz) -> c_gs1(f); same formula as `sample_cgs1_coefficient`
    (slope2 fixed at 0, softplus-wrapped)."""
    def cgs1(F):
        F = jnp.asarray(F)
        return jax.nn.softplus(ic + logistic_hinge(jnp.log(F), slope1, 0.0, fb, delta))
    return cgs1


def c0_fn_from_params(params: Mapping, prefix: str = "c_0") -> Callable[[Array], Array]:
    """Build c_0(f) from one posterior draw / point estimate, using the
    sample-site names `{prefix}_ic, _slope1, _slope2, _fb, _delta, _kappa_star`."""
    return make_c0_fn(*(params[f"{prefix}_{k}"]
                        for k in ("ic", "slope1", "slope2", "fb", "delta", "kappa_star")))


def cgs1_fn_from_params(params: Mapping, prefix: str = "c_gs1") -> Callable[[Array], Array]:
    """Build c_gs1(f) from one posterior draw, using `{prefix}_ic, _slope1, _fb, _delta`."""
    return make_cgs1_fn(*(params[f"{prefix}_{k}"] for k in ("ic", "slope1", "fb", "delta")))


# ---------------------------------------------------------------------------
# Per-record geometric-spreading factor
# ---------------------------------------------------------------------------

def spreading_factor(
    M: Array, R: Array, *, const, c_nft_1, c_nft_2, gs_break, gs_exp=2.0,
    func_gs_scaling: str = "stafford",
) -> Array:
    """
    G_rec = d(func_mag_gs)/d(c_gs1) at fixed other coefficients, shape (n_rec,).

    The coefficient arguments are the (scalar) values at ONE frequency --
    the band edge the extrapolation starts from (call once per side if
    `c_nft_*` or `gs_exp` vary with frequency). `const` is `ModelConstants`.
    Works for both spreading forms since the term is linear in c_gs1.
    """
    Mc, Rc = M[:, jnp.newaxis], R[:, jnp.newaxis]
    z = 0.0
    if func_gs_scaling == "stafford":
        def term(c):
            return func_mag_gs(Mc, Rc, z, z, z, c, const.c_gs2, c_nft_1, c_nft_2,
                               const.mb1, const.mb2, const.delta_gs, gs_break, const.xi)
    else:
        def term(c):
            return func_mag_gs_lh(Mc, Rc, z, z, z, c, const.c_gs2, c_nft_1, c_nft_2,
                                  const.mb1, const.mb2, gs_break, gs_exp)
    return (term(1.0) - term(0.0))[:, 0]


# ---------------------------------------------------------------------------
# Extrapolation terms
# ---------------------------------------------------------------------------

def parametric_extrap_delta(
    ops, *,
    c0_fn: Callable[[Array], Array],
    cgs1_fn: Callable[[Array], Array],
    G_lo: Array, G_hi: Array,
    kappa_adj: Optional[Array] = None,
    attn_path: Optional[Array] = None,
    Q_exp: Optional[Array] = None,
    c_Q: float = 3.5,
) -> Tuple[Array, Array]:
    """
    Additive extrapolation terms relative to the band-edge prediction.

    Parameters
    ----------
    ops : RvtSea25Operators (provides the out-of-band frequencies and edges)
    c0_fn, cgs1_fn : f -> coefficient (see `make_c0_fn`, `make_cgs1_fn`)
    G_lo, G_hi : (n_rec,) spreading factor at the lowest / highest EAS
        frequency (`spreading_factor`).
    kappa_adj : (n_rec,) per-record regional kappa deviation as used in the
        median (`median -= kappa_adj * F`), or None.
    attn_path : (n_rec,) R / Q_0 (attn_mode "Q") or dist_cell @ (1/Q_0)
        ("dist_cell"); None for attn_mode "c_attn" (held at the edge).
    Q_exp, c_Q : attenuation exponent and constant (needed with attn_path).

    Returns
    -------
    delta_lo : (n_rec, n_lo), delta_hi : (n_rec, n_hi)
    """
    n_rec = G_lo.shape[0]

    def side(f, fe, G):
        f = jnp.asarray(f)
        if f.shape[0] == 0:
            return jnp.zeros((n_rec, 0))
        delta = (c0_fn(f) - c0_fn(fe))[jnp.newaxis, :] \
            + G[:, jnp.newaxis] * (cgs1_fn(f) - cgs1_fn(fe))[jnp.newaxis, :]
        if kappa_adj is not None:
            delta = delta - kappa_adj[:, jnp.newaxis] * (f - fe)[jnp.newaxis, :]
        if attn_path is not None:
            delta = delta - jnp.pi * attn_path[:, jnp.newaxis] \
                * (f ** Q_exp - fe ** Q_exp)[jnp.newaxis, :] / c_Q
        return delta

    return (side(ops.freqs_lo, ops.f_edge_lo, G_lo),
            side(ops.freqs_hi, ops.f_edge_hi, G_hi))
