"""
Assembling numpyro model inputs (X_rec, X_eq, X_stat, X_id, Y,
global_dict, nl_model_dict, L_freq) from the tables `prepare_data`
produces.

The final `data_dict = {...}` that gets passed to `model()`/`guide()` as
**kwargs isn't wrapped in a function here -- it's just gathering these
pieces plus the model's own configuration flags, and a wrapper would
have to mirror model()'s entire parameter list, which is more
maintenance risk than it saves. Build it as a plain dict in your script;
see the example at the bottom of this file's docstring.

Example
-------
    data_selected, frequencies_used = load_data(eas_path, site_path, region="WUS")
    data_used, data_eq, data_stat = prepare_data(frequencies_used, data_selected)
    X_rec, X_eq, X_stat, X_id, Y = extract_model_arrays(data_used, data_eq, data_stat, frequencies_used)

    data_selected_gl, _ = load_data(eas_path, site_path, region="Global", frequencies=frequencies_used)
    data_used_gl, data_eq_gl, data_stat_gl = prepare_data(frequencies_used, data_selected_gl)
    global_dict = build_global_dict(data_used_gl, data_eq_gl, data_stat_gl, frequencies_used)

    nl_model_dict = setup_nl_model("hashash_new", jnp.array(frequencies_used))
    L_freq = build_frequency_cholesky(frequencies_used)

    data_dict = {
        "F": jnp.array(frequencies_used),
        "X_rec": X_rec, "X_eq": X_eq, "X_stat": X_stat, "X_id": X_id, "Y": Y,
        "nl_model_dict": nl_model_dict, "L_freq": L_freq, "global_dict": global_dict,
        # ... plus model configuration flags (attn_eq, calc_nft, c0_parametric, ...)
    }
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np


def extract_model_arrays(data_used, data_eq, data_stat, frequencies_used):
    """
    Build the X_rec, X_eq, X_stat, X_id, Y arrays the numpyro model
    expects, from the tables `prepare_data` produces.

    Uses `data_eq['mag_sd']` as already computed by `prepare_data` --
    your original bottom script recomputed the same magnitude-dependent
    taper (0.1 / 0.05 / 0.01) a second time here, which was redundant
    (and, for the global dataset, was actually being *overridden* with a
    flat 0.1 afterwards -- see `build_global_dict`).
    """
    X_id = np.array(data_used[['eq', 'stat', 'regional', 'basin']])
    X_eq = np.c_[
        np.array(data_eq[['magnitude', 'ztor', 'F_nm', 'F_rev', 'fault_width', 'dip']]),
        data_eq['mag_sd'].values,
    ]
    X_stat = np.array(data_stat[['vs30', 'vs30_lnstd', 'vs30_measured']])
    X_rec = np.array(data_used[['rrup', 'rjb', 'rx', 'ry0']])
    Y = data_used[frequencies_used].values
    return X_rec, X_eq, X_stat, X_id, Y


class ModelInputs(NamedTuple):
    """The columns of X_rec / X_eq / X_stat / X_id, named (see `unpack_model_inputs`)."""
    R: Array; Rjb: Array; Rx: Array; Ry0: Array                  # (n_rec,)
    M_eq: Array; Zt_eq: Array; Fnm_eq: Array; Frev_eq: Array     # (n_eq,)
    FW_eq: Array; Dip_eq: Array; M_sd: Array
    VS_stat: Array; VS_stat_sd: Array; vs_measured_id: Array     # (n_stat,)
    eq_id: np.ndarray; stat_id: np.ndarray                       # (n_rec,), int
    subregion_id: np.ndarray; basin_id: np.ndarray
    n_rec: int; n_eq: int; n_stat: int; n_subregion: int


def unpack_model_inputs(X_rec, X_eq, X_stat, X_id) -> ModelInputs:
    """
    Unpack the model input matrices (as built by `extract_model_arrays`)
    into named arrays, with the same column layout `model_eas` uses:
    X_rec = (R, Rjb, Rx, Ry0), X_eq = (M, Ztor, Fnm, Frev, FW, Dip, M_sd),
    X_stat = (Vs30, Vs30_sd, vs30_measured), X_id = (eq, stat, subregion,
    basin). The id columns are returned as numpy int arrays (usable for
    indexing and static shapes); the other columns as jax arrays.
    Used by the PSA calibration (`psa_generation.calibration`).
    """
    R, Rjb, Rx, Ry0 = np.asarray(X_rec).T
    M_eq, Zt_eq, Fnm_eq, Frev_eq, FW_eq, Dip_eq, M_sd = np.asarray(X_eq).T
    VS_stat, VS_stat_sd, vs_meas_id = np.asarray(X_stat).T
    eq_id, stat_id, subregion_id, basin_id = np.asarray(X_id).T.astype(int)
    return ModelInputs(
        R=jnp.asarray(R), Rjb=jnp.asarray(Rjb), Rx=jnp.asarray(Rx), Ry0=jnp.asarray(Ry0),
        M_eq=jnp.asarray(M_eq), Zt_eq=jnp.asarray(Zt_eq), Fnm_eq=jnp.asarray(Fnm_eq),
        Frev_eq=jnp.asarray(Frev_eq), FW_eq=jnp.asarray(FW_eq), Dip_eq=jnp.asarray(Dip_eq),
        M_sd=jnp.asarray(M_sd),
        VS_stat=jnp.asarray(VS_stat), VS_stat_sd=jnp.asarray(VS_stat_sd),
        vs_measured_id=np.asarray(vs_meas_id).astype(int),
        eq_id=eq_id, stat_id=stat_id, subregion_id=subregion_id, basin_id=basin_id,
        n_rec=len(eq_id), n_eq=np.asarray(X_eq).shape[0], n_stat=np.asarray(X_stat).shape[0],
        n_subregion=int(np.max(subregion_id)) + 1,
    )


def build_global_dict(data_used_gl, data_eq_gl, data_stat_gl, frequencies_used):
    """
    Build the `global_dict` passed to model()/guide().

    NOTE: your original script overrode the global events' `mag_sd` with
    a flat 0.1 for every event, rather than the same magnitude-dependent
    taper used for WUS (0.1 / 0.05 / 0.01 by magnitude, from
    `prepare_data`). This version uses the same tapered `mag_sd` for
    both datasets, for consistency -- flag if the flat 0.1 for global
    was actually intentional (e.g. because global magnitudes are less
    reliable) and I'll revert this one spot.
    """
    X_rec_gl, X_eq_gl, X_stat_gl, X_id_gl, Y_gl = extract_model_arrays(
        data_used_gl, data_eq_gl, data_stat_gl, frequencies_used,
    )
    return {
        'X_id': X_id_gl,
        'X_rec': X_rec_gl,
        'X_stat': X_stat_gl,
        'X_eq': X_eq_gl,
        'Y': Y_gl,
    }


def setup_nl_model(nl_model_name: str, frequencies_used_array: jnp.ndarray) -> Dict:
    """
    Build the `nl_model_dict` passed to model()/guide(), interpolating
    the chosen nonlinear site amplification model's coefficients to
    `frequencies_used_array`.

    Parameters
    ----------
    nl_model_name : {'hashash_new', 'mahdi'}
    frequencies_used_array : Array

    Returns
    -------
    dict
    """
    from ngaw3_kuehnetal27.site_amplification import nl_models

    if nl_model_name == 'hashash_new':
        interp_hashash, interp_sd = nl_models.setup_hashash_new_coefficients(frequencies_used_array)
        return {'model': 'hashash_new', 'interp_hashash': interp_hashash}
    elif nl_model_name == 'mahdi':
        interp_params, interp_nonlin = nl_models.setup_mahdi_coefficients(frequencies_used_array)
        return {'model': 'mahdi', 'interp_params': interp_params, 'interp_nonlin': interp_nonlin}
    else:
        raise ValueError(f"Unknown nl_model_name: {nl_model_name!r}. Expected 'hashash_new' or 'mahdi'.")


def build_frequency_cholesky(
    frequencies_used: Sequence[float],
    ell_gp: float = 1.2,
    var_gp: float = 1.0,
    jitter: float = 1e-9,
    kernel=None,
) -> jnp.ndarray:
    """
    Build the Cholesky factor (`L_freq`) of a frequency-correlation
    kernel, used for the subregion random effect's correlation-across-
    frequency prior.

    Parameters
    ----------
    frequencies_used : array-like
    ell_gp, var_gp : float
        Length scale and variance of the kernel.
    jitter : float
        Added to the diagonal before the Cholesky decomposition, for
        numerical stability.
    kernel : Callable, optional
        Defaults to `ngaw3_kuehnetal27.utils.kernel_matern52`. Pass e.g.
        `ngaw3_kuehnetal27.utils.kernel_sqexp` for the squared-exponential kernel.

    Returns
    -------
    Array, shape (n_freq, n_freq)
    """
    from ngaw3_kuehnetal27.utils import kernel_matern52

    kernel = kernel if kernel is not None else kernel_matern52
    log_f = jnp.log(jnp.array(frequencies_used))
    K = kernel(log_f, log_f, var_gp, ell_gp)
    K = K + jitter * jnp.eye(K.shape[0])
    return jnp.linalg.cholesky(K)
