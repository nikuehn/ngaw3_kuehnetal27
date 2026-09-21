"""
Precomputed 1D site amplification (from Vs30-indexed profiles).

Two use cases:

- Training: `log_amp` is computed once per station (outside this module,
  since it needs the real station Vs30 values from the data) and passed
  into `numpyro_models.model_eas` as `amp1d_dict={'log_amp': ...}`, which
  forwards it to `median_core.calculate_median_training` for per-record
  gathering by station.
- Scenario prediction: there are no station IDs, just a Vs30 value per
  scenario row, so `interp_amp1d` is called directly on that Vs30 array
  (see `scenario_prediction.py`).

Both paths go through the same interpolation function so a given Vs30
always maps to the same amplification, whether at training or predict
time.
"""
from __future__ import annotations

import jax.numpy as jnp
import xarray
from jax.scipy.ndimage import map_coordinates

from ngaw3_kuehnetal27.utils import package_data_path

Array = jnp.ndarray

DEFAULT_AMP1D_FILENAME = "amp-generic-profiles-vs30.nc"
DEFAULT_AMP1D_TARGET = "amp_predicted"


def load_amp1d_dataset(filename: str = DEFAULT_AMP1D_FILENAME) -> xarray.Dataset:
    """
    Load the precomputed 1D-profile amplification dataset (vs30 x freq
    grid) from the package's data directory (`amp_1d/<filename>` under
    `package_data_path`).

    NOTE: assumes `package_data_path(*parts)` joins its arguments onto
    the package data directory and returns the resulting path, the way
    `os.path.join` would -- adjust the call below if its actual
    signature differs.
    """
    return xarray.open_dataset(package_data_path(filename))


def interp_amp1d(
    ds: xarray.Dataset,
    new_vs30: Array,
    new_freq: Array,
    key: str = DEFAULT_AMP1D_TARGET,
    log_freq: bool = True,
    log_vs30: bool = True,
    log_target: bool = True,
) -> Array:
    """
    Bilinearly interpolate ds[key] (shape (vs30, freq)) at new_vs30 x new_freq.

    Interpolation is done in log-space for vs30, freq, and/or the target
    values themselves, per the flags below, then transformed back.
    `mode='nearest'` in the underlying `map_coordinates` call means
    Vs30/frequency values outside the grid are clamped to the nearest
    edge rather than extrapolated.

    Returns a jax array of shape (len(new_vs30), len(new_freq)) -- every
    vs30-freq combination. When `new_freq` is the full frequency array F
    shared by every scenario row, this is already exactly the
    (n_scenarios, n_freq) shape `amp1d_adj` needs (see
    `amp1d_adj_from_vs30` below).
    """
    dims = ds[key].dims  # e.g. ('vs30', 'freq') -- inferred, not hardcoded
    vs30_coords = jnp.asarray(ds[dims[0]].values)
    freq_coords = jnp.asarray(ds[dims[1]].values)
    data = jnp.asarray(ds[key].values)

    new_vs30 = jnp.atleast_1d(jnp.asarray(new_vs30, dtype=jnp.float32))
    new_freq = jnp.atleast_1d(jnp.asarray(new_freq, dtype=jnp.float32))

    if log_vs30:
        vs30_coords_t, new_vs30_t = jnp.log(vs30_coords), jnp.log(new_vs30)
    else:
        vs30_coords_t, new_vs30_t = vs30_coords, new_vs30

    if log_freq:
        freq_coords_t, new_freq_t = jnp.log(freq_coords), jnp.log(new_freq)
    else:
        freq_coords_t, new_freq_t = freq_coords, new_freq

    if log_target:
        data_t = jnp.log(data)
    else:
        data_t = data

    # map physical coordinates -> fractional grid-index space
    vs30_idx = jnp.interp(new_vs30_t, vs30_coords_t, jnp.arange(len(vs30_coords)))
    freq_idx = jnp.interp(new_freq_t, freq_coords_t, jnp.arange(len(freq_coords)))

    vs30_grid, freq_grid = jnp.meshgrid(vs30_idx, freq_idx, indexing="ij")
    coords = jnp.stack([vs30_grid.ravel(), freq_grid.ravel()])

    out = map_coordinates(data_t, coords, order=1, mode="nearest")
    out = out.reshape(len(new_vs30), len(new_freq))

    # Note: when log_target=True this is *not* exponentiated back -- the
    # function returns ln(amplification) directly, which is exactly the
    # additive, log-scale quantity `amp1d_adj`/`log_amp` needs to be.
    # Only set log_target=False if `ds[key]` is already stored in log
    # scale (in which case this returns it unchanged).
    return out


def amp1d_adj_from_vs30(
    ds: xarray.Dataset,
    vs30: Array,
    F: Array,
    key: str = DEFAULT_AMP1D_TARGET,
    **interp_kwargs,
) -> Array:
    """
    Interpolate `ds[key]` at each scenario row's own Vs30, across the
    full shared frequency array `F`. Thin, named wrapper around
    `interp_amp1d` for this specific call shape -- use this to build
    `amp1d_adj` for `predict_median` / `predict_median_categorical` (see
    `scenario_prediction._predict_from_dataframe`).

    Returns array of shape (len(vs30), len(F)).
    """
    return interp_amp1d(ds, new_vs30=vs30, new_freq=F, key=key, **interp_kwargs)
