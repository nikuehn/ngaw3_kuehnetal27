"""
JAX implementations of nonlinear site amplification models for numpyro.

Design notes
------------
- Coefficients are pre-interpolated to fixed frequencies (see
  `coefficients.py`). No interpolation happens during model execution.
- Every model exposes a `*_single` function that computes ln(amplification)
  for one record (shape (n_freq,) in, shape (n_freq,) out). Batched
  versions are produced by vmapping that *same* function -- see
  `batch_over_records` below -- rather than by hand-writing a second copy
  of the computation. This means there is exactly one implementation of
  each model's math to keep correct.

Usage
-----
    frequencies = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    interp_param, interp_nonlin = setup_mahdi_coefficients(frequencies)

    # Inside numpyro model, vs30/magnitude vary per record:
    ln_amp = compute_ln_amplification_mahdi(
        eas_ref, vs30, magnitude, interp_param, interp_nonlin
    )  # shape (n_records, n_freq)

    # Or when vs30/magnitude are shared scalars across all records:
    ln_amp = compute_ln_amplification_mahdi_shared(
        eas_ref, vs30, magnitude, interp_param, interp_nonlin
    )  # shape (n_records, n_freq)
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import vmap

from ngaw3_kuehnetal27.site_amplification.coefficients import (
    interpolate_coeffs,
    load_coefficients,
)
from ngaw3_kuehnetal27.utils import package_data_path

Array = jnp.ndarray


def batch_over_records(single_fn: Callable, in_axes: Tuple) -> Callable:
    """
    Vectorize a `*_single` record function over a batch of records.

    Passing `0` for an argument's axis means "this varies per record"
    (its leading axis is the record axis); passing `None` means "this is
    shared/broadcast across all records". This one helper produces both
    the "everything varies per record" batched function and the "vs30 /
    magnitude shared across records" variant -- previously these were
    two independently hand-written functions per model; they're
    mathematically the same function, just vmapped differently.

    Parameters
    ----------
    single_fn : Callable
        A `*_single` function with signature (eas_ref, *scalar_args,
        *coeff_dicts) -> Array, operating on one record.
    in_axes : Tuple
        vmap in_axes matching `single_fn`'s signature, e.g.
        (0, 0, 0, None, None) for "eas_ref, vs30, magnitude vary;
        coefficient dicts are shared", or (0, None, None, None, None)
        for "vs30 and magnitude are shared scalars".

    Returns
    -------
    Callable
        Batched function taking eas_ref with shape (n_records, n_freq)
        and returning ln(amplification) with shape (n_records, n_freq).
    """
    return vmap(single_fn, in_axes=in_axes, out_axes=0)


# ---------------------------------------------------------------------------
# Mahdi model
# ---------------------------------------------------------------------------

MAHDI_PARAM_COLUMNS = {
    "beta0": "intercept",
    "beta1": "coeff_M",
    "c1": "slope_Vs30_low",
    "c2": "slope_Vs30_mid",
    "c3": "slope_Vs30_high",
}

MAHDI_NONLIN_COLUMNS = {
    name: name
    for name in (
        "a1_f2", "a2_f2", "a3_f2", "a4_f2", "a5_f2", "a6_f2",
        "a1_f4", "a2_f4", "a3_f4", "a4_f4", "a5_f4", "a6_f4",
    )
}

REFERENCE_VS30 = 800.0  # m/s
VS30_HINGE_LOW = 600.0  # m/s
VS30_HINGE_HIGH = 1000.0  # m/s


def setup_mahdi_coefficients(
    frequencies: Array,
    data_dir=None,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """
    Load and interpolate Mahdi model coefficients to target frequencies.

    Call once in preprocessing, before running MCMC.

    Parameters
    ----------
    frequencies : Array
        Target frequencies (Hz) for the GMM.
    data_dir : str or Path, optional
        Directory containing the coefficient CSVs. Defaults to the
        package's bundled `data/` directory.

    Returns
    -------
    interp_param : Dict[str, Array]
        Interpolated parameter coefficients (beta0, beta1, c1, c2, c3).
    interp_nonlin : Dict[str, Array]
        Interpolated nonlinearity coefficients (a1_f2..a6_f2, a1_f4..a6_f4).
    """
    param_path = (data_dir / "regression_parameters_trilinear_all.csv"
                  if data_dir is not None
                  else package_data_path("regression_parameters_trilinear_all.csv"))
    nonlin_path = (data_dir / "nn_parametric_coefficients_all.csv"
                   if data_dir is not None
                   else package_data_path("nn_parametric_coefficients_all.csv"))

    param_coeffs = load_coefficients(param_path)
    nonlin_coeffs = load_coefficients(nonlin_path)

    interp_param = interpolate_coeffs(frequencies, param_coeffs, MAHDI_PARAM_COLUMNS)
    interp_nonlin = interpolate_coeffs(frequencies, nonlin_coeffs, MAHDI_NONLIN_COLUMNS)

    return interp_param, interp_nonlin


def compute_trilinear_vs30_terms(vs30: float) -> Tuple[Array, Array, Array]:
    """
    Compute trilinear Vs30 terms F1, F2, F3.

    Parameters
    ----------
    vs30 : float
        Vs30 in m/s (scalar or works under vmap).

    Returns
    -------
    Tuple[Array, Array, Array]
        F1, F2, F3 terms.
    """
    ln_vs30_600 = jnp.log(vs30 / VS30_HINGE_LOW)
    ln_1000_600 = jnp.log(VS30_HINGE_HIGH / VS30_HINGE_LOW)
    ln_vs30_1000 = jnp.log(vs30 / VS30_HINGE_HIGH)

    F1 = jnp.minimum(ln_vs30_600, 0.0)
    F2 = jnp.maximum(0.0, jnp.minimum(ln_vs30_600, ln_1000_600))
    F3 = jnp.maximum(0.0, ln_vs30_1000)

    return F1, F2, F3


def compute_ln_imt_ref_pred(
    vs30: float,
    magnitude: float,
    interp_param: Dict[str, Array],
) -> Array:
    """
    Compute predicted reference IMT in log scale.

    ln(IMT_ref,pred) = beta0 + beta1*(M-6) + c1*F1 + c2*F2 + c3*F3

    Parameters
    ----------
    vs30 : float
        Vs30 in m/s (scalar).
    magnitude : float
        Earthquake magnitude (scalar).
    interp_param : Dict[str, Array]
        Pre-interpolated parameter coefficients, each shape (n_freq,).

    Returns
    -------
    Array
        ln(IMT_ref,pred) with shape (n_freq,).
    """
    F1, F2, F3 = compute_trilinear_vs30_terms(vs30)

    return (
        interp_param["beta0"]
        + interp_param["beta1"] * (magnitude - 6.0)
        + interp_param["c1"] * F1
        + interp_param["c2"] * F2
        + interp_param["c3"] * F3
    )


def compute_ln_nonlinearity_mahdi(
    vs30: float,
    imt_ref_n: Array,
    interp_nonlin: Dict[str, Array],
) -> Array:
    """
    Compute the Mahdi nonlinearity term in log scale.

    Parameters
    ----------
    vs30 : float
        Vs30 in m/s (scalar).
    imt_ref_n : Array
        Normalized reference IMT with shape (n_freq,).
    interp_nonlin : Dict[str, Array]
        Pre-interpolated nonlinearity coefficients, each shape (n_freq,).

    Returns
    -------
    Array
        ln(nonlinearity) with shape (n_freq,).
    """
    sigmoid_f2 = (
        interp_nonlin["a1_f2"]
        / (1.0 + jnp.exp(interp_nonlin["a2_f2"] * (vs30 - interp_nonlin["a3_f2"])))
    )
    gaussian_f2 = interp_nonlin["a4_f2"] * jnp.exp(
        -((vs30 - interp_nonlin["a5_f2"]) ** 2) / (2.0 * interp_nonlin["a6_f2"] ** 2)
    )
    f2 = sigmoid_f2 + gaussian_f2

    sigmoid_f4 = (
        interp_nonlin["a1_f4"]
        / (1.0 + jnp.exp(interp_nonlin["a2_f4"] * (vs30 - interp_nonlin["a3_f4"])))
    )
    gaussian_f4 = interp_nonlin["a4_f4"] * jnp.exp(
        -((vs30 - interp_nonlin["a5_f4"]) ** 2) / (2.0 * interp_nonlin["a6_f4"] ** 2)
    )
    f4 = sigmoid_f4 + gaussian_f4

    return (
        f2 * jnp.log(imt_ref_n + 1.0)
        - f4 * jnp.log(jnp.sqrt(imt_ref_n ** 2 + 1.0))
    )


def compute_ln_amplification_mahdi_single(
    eas_ref: Array,
    vs30: float,
    magnitude: float,
    interp_param: Dict[str, Array],
    interp_nonlin: Dict[str, Array],
) -> Array:
    """
    Compute ln(amplification) for a single record using the Mahdi model.

    Parameters
    ----------
    eas_ref : Array
        Reference EAS at Vs30=800 m/s, shape (n_freq,).
    vs30 : float
        Vs30 in m/s (scalar).
    magnitude : float
        Earthquake magnitude (scalar).
    interp_param, interp_nonlin : Dict[str, Array]
        Pre-interpolated coefficients.

    Returns
    -------
    Array
        ln(amplification) with shape (n_freq,).
    """
    ln_imt_ref_pred = compute_ln_imt_ref_pred(vs30, magnitude, interp_param)
    imt_ref_n = eas_ref / jnp.exp(ln_imt_ref_pred)
    return compute_ln_nonlinearity_mahdi(vs30, imt_ref_n, interp_nonlin)


# Records vary in eas_ref, vs30, and magnitude; coefficient dicts are shared.
compute_ln_amplification_mahdi = batch_over_records(
    compute_ln_amplification_mahdi_single,
    in_axes=(0, 0, 0, None, None),
)
compute_ln_amplification_mahdi.__doc__ = """
Compute ln(amplification) for multiple records using the Mahdi model,
where vs30 and magnitude both vary per record.

Parameters
----------
eas_ref : Array, shape (n_records, n_freq)
vs30 : Array, shape (n_records,)
magnitude : Array, shape (n_records,)
interp_param, interp_nonlin : Dict[str, Array], each value shape (n_freq,)

Returns
-------
Array, shape (n_records, n_freq)
"""

# Records vary only in eas_ref; vs30 and magnitude are shared scalars.
compute_ln_amplification_mahdi_shared = batch_over_records(
    compute_ln_amplification_mahdi_single,
    in_axes=(0, None, None, None, None),
)
compute_ln_amplification_mahdi_shared.__doc__ = """
Compute ln(amplification) for multiple records using the Mahdi model,
where vs30 and magnitude are single scalars shared across all records
(e.g. all records from one site / one event).

Parameters
----------
eas_ref : Array, shape (n_records, n_freq)
vs30 : float
magnitude : float
interp_param, interp_nonlin : Dict[str, Array], each value shape (n_freq,)

Returns
-------
Array, shape (n_records, n_freq)
"""


# ---------------------------------------------------------------------------
# Hashash model (new form)
# ---------------------------------------------------------------------------

HASHASH_NEW_COLUMNS = {name: name for name in ("f3", "f4", "f5", "Vf", "Vg")}
HASHASH_NEW_SD_COLUMNS = {name: name for name in ("sigma_c", "V1", "V2", "sigma_f")}

V_REF_HASHASH = 800.0   # Reference Vs30, m/s
V_REF2_HASHASH = 360.0  # Second reference velocity, m/s
IR_LIM = 0.3            # g-sec
FREQ_TAPER_CENTER = 4.998  # Hz, center of cosine taper for W_f


def setup_hashash_new_coefficients(
    frequencies: Array,
    target: str = "EAS",
    data_dir=None,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """
    Load and interpolate coefficients for the new Hashash model.

    Call once in preprocessing, before running MCMC.

    Parameters
    ----------
    frequencies : Array
        Target frequencies (Hz) for EAS, or periods (s) for PSA -- must
        match the units of the `target` table's reference column.
    target : {'EAS', 'PSA'}
        Which coefficient table to load.
    data_dir : str or Path, optional
        Directory containing the coefficient CSVs. Defaults to the
        package's bundled `data/` directory.

    Returns
    -------
    interp_param : Dict[str, Array]
        Interpolated coefficients (f3, f4, f5, Vf, Vg), each shape (n_freq,).
    interp_sd : Dict[str, Array]
        Interpolated standard-deviation coefficients (sigma_c, V1, V2,
        sigma_f), each shape (n_freq,).
    """
    if target == "EAS":
        param_file, sd_file, ref_col = "N2_EAS_ay26.csv", "N2_EAS_ay26-STDEV.csv", "Frequency"
    elif target == "PSA":
        param_file, sd_file, ref_col = "N2_RS_ay26.csv", "N2_RS_ay26-STDEV.csv", "Period"
    else:
        raise ValueError(f"Unknown target: {target!r}. Expected 'EAS' or 'PSA'.")

    param_path = (data_dir / param_file if data_dir is not None
                  else package_data_path(param_file))
    sd_path = (data_dir / sd_file if data_dir is not None
               else package_data_path(sd_file))

    param_coeffs = load_coefficients(param_path)
    param_coeffs["freq"] = param_coeffs.pop(ref_col)

    sd_coeffs = load_coefficients(sd_path)
    sd_coeffs["freq"] = sd_coeffs.pop(ref_col)

    interp_param = interpolate_coeffs(frequencies, param_coeffs, HASHASH_NEW_COLUMNS)
    interp_param["freq"] = frequencies
    interp_sd = interpolate_coeffs(frequencies, sd_coeffs, HASHASH_NEW_SD_COLUMNS)
    interp_sd["freq"] = frequencies

    return interp_param, interp_sd


def compute_ln_nonlinearity_hashash_new(
    vs30: float,
    eas_ref: Array,
    interp_hashash: Dict[str, Array],
) -> Array:
    """
    Compute the new-form Hashash nonlinearity term in log scale.

    Based on N2_EAS_New_Model_Form_Coeffs_v2, including a transition term
    TEM_NL and frequency-dependent weight W_f.

    Parameters
    ----------
    vs30 : float
        Vs30 in m/s (scalar).
    eas_ref : Array
        Reference EAS at Vs30=800 m/s (linear scale, g-sec), shape (n_freq,).
    interp_hashash : Dict[str, Array]
        Pre-interpolated coefficients with keys 'f3', 'f4', 'f5', 'Vf',
        'Vg', 'freq', each shape (n_freq,).

    Returns
    -------
    Array
        ln(nonlinearity) with shape (n_freq,).
    """
    f3 = interp_hashash["f3"]
    f4 = interp_hashash["f4"]
    f5 = interp_hashash["f5"]
    Vf = interp_hashash["Vf"]
    Vg = interp_hashash["Vg"]
    freqs = interp_hashash["freq"]

    vs30_lim = jnp.clip(vs30, 200.0, V_REF_HASHASH)
    ir_lim = jnp.minimum(eas_ref, IR_LIM)

    tem_nl = jnp.clip(
        (V_REF_HASHASH - vs30_lim) / (V_REF_HASHASH - Vg), 0.0, 1.0
    )

    f2_base = f4 * (
        jnp.exp(f5 * (vs30_lim - V_REF2_HASHASH))
        - jnp.exp(f5 * (V_REF_HASHASH - V_REF2_HASHASH))
    )

    delta_f2 = f4 * (
        jnp.exp(f5 * (vs30_lim - V_REF2_HASHASH))
        - jnp.exp(f5 * (Vf - V_REF2_HASHASH))
    ) * tem_nl

    f_a = 0.9 * FREQ_TAPER_CENTER
    f_b = 1.1 * FREQ_TAPER_CENTER
    w_f_cos = 0.5 * (1.0 + jnp.cos(
        jnp.pi * (jnp.log(freqs) - jnp.log(f_a)) / (jnp.log(f_b) - jnp.log(f_a))
    ))
    w_f = jnp.where(freqs <= f_a, 1.0, jnp.where(freqs >= f_b, 0.0, w_f_cos))

    return (f2_base + delta_f2 * w_f) * jnp.log((ir_lim + f3) / f3)


def compute_ln_amplification_hashash_new_single(
    eas_ref: Array,
    vs30: float,
    interp_hashash: Dict[str, Array],
) -> Array:
    """
    Compute ln(amplification) for a single record using the new Hashash
    model.

    Parameters
    ----------
    eas_ref : Array
        Reference EAS at Vs30=800 m/s (g-sec), shape (n_freq,).
    vs30 : float
        Vs30 in m/s (scalar).
    interp_hashash : Dict[str, Array]
        Pre-interpolated coefficients (f3, f4, f5, Vf, Vg, freq).

    Returns
    -------
    Array
        ln(amplification) with shape (n_freq,).
    """
    return compute_ln_nonlinearity_hashash_new(vs30, eas_ref, interp_hashash)


# Records vary in eas_ref and vs30; coefficient dict is shared.
compute_ln_amplification_hashash_new = batch_over_records(
    compute_ln_amplification_hashash_new_single,
    in_axes=(0, 0, None),
)
compute_ln_amplification_hashash_new.__doc__ = """
Compute ln(amplification) for multiple records using the new Hashash
model, where vs30 varies per record.

Parameters
----------
eas_ref : Array, shape (n_records, n_freq)
vs30 : Array, shape (n_records,)
interp_hashash : Dict[str, Array], each value shape (n_freq,)

Returns
-------
Array, shape (n_records, n_freq)
"""

# Records vary only in eas_ref; vs30 is a shared scalar.
compute_ln_amplification_hashash_new_shared = batch_over_records(
    compute_ln_amplification_hashash_new_single,
    in_axes=(0, None, None),
)
compute_ln_amplification_hashash_new_shared.__doc__ = """
Compute ln(amplification) for multiple records using the new Hashash
model, where vs30 is a single scalar shared across all records.

Parameters
----------
eas_ref : Array, shape (n_records, n_freq)
vs30 : float
interp_hashash : Dict[str, Array], each value shape (n_freq,)

Returns
-------
Array, shape (n_records, n_freq)
"""
