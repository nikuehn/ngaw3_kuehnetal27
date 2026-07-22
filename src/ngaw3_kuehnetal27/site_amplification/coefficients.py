"""
Utilities for loading GMM coefficient tables and interpolating them onto a
set of target frequencies.

Every site amplification model needs the same two steps: (1) load a CSV of
coefficients vs. frequency/period, (2) log-frequency interpolate each
coefficient column onto the model's target frequencies. This module
provides that once, generically -- individual models just supply a
`column_map` describing which output key pulls from which input column.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import jax.numpy as jnp
import pandas as pd

Array = jnp.ndarray


def load_coefficients(filepath: Union[str, Path]) -> Dict[str, Array]:
    """
    Load a coefficient CSV file and convert each column to a JAX array.

    Parameters
    ----------
    filepath : str or Path
        Path to the coefficient CSV file.

    Returns
    -------
    Dict[str, Array]
        Dictionary mapping column names to JAX arrays.
    """
    df = pd.read_csv(filepath, comment="#")
    return {col: jnp.array(df[col].values) for col in df.columns}


def interpolate_coeffs(
    frequencies: Array,
    raw_coeffs: Dict[str, Array],
    column_map: Dict[str, str],
    freq_col: str = "freq",
) -> Dict[str, Array]:
    """
    Log-frequency interpolate raw coefficient columns onto target
    frequencies.

    This is the single interpolation routine used by every site
    amplification model. What differs between models is only *which*
    columns get pulled out and what they're called downstream -- that's
    what `column_map` specifies, so the interpolation math itself is
    never duplicated.

    Parameters
    ----------
    frequencies : Array
        Target frequencies (Hz) for the model.
    raw_coeffs : Dict[str, Array]
        Raw coefficients as returned by `load_coefficients`.
    column_map : Dict[str, str]
        Mapping from output key -> input column name in `raw_coeffs`,
        e.g. {'beta0': 'intercept', 'beta1': 'coeff_M'}. If a model's
        output keys already match the input column names, pass
        `{name: name for name in [...]}`.
    freq_col : str
        Key in `raw_coeffs` holding the reference frequencies (or
        periods -- caller is responsible for consistent units between
        `frequencies` and `raw_coeffs[freq_col]`).

    Returns
    -------
    Dict[str, Array]
        One interpolated array per key in `column_map`, each shape
        (n_freq,).
    """
    log_freq = jnp.log(frequencies)
    log_freq_ref = jnp.log(raw_coeffs[freq_col])

    return {
        out_key: jnp.interp(log_freq, log_freq_ref, raw_coeffs[in_col])
        for out_key, in_col in column_map.items()
    }
