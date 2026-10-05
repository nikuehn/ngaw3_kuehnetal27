"""
Loading observed PSA (RotD50) for a set of motions already selected via the
EAS workflow (see data_loading.py).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ngaw3_kuehnetal27.utils import convert_column_name_to_period


def _match_periods(
    requested: Sequence[float],
    available: Sequence[float],
    available_names: Sequence[str],
    tol: float = 1e-6,
) -> Tuple[List[float], List[str]]:
    """
    Match each requested period to the nearest available PSA column
    within `tol`.

    Raises
    ------
    ValueError
        If any requested period has no available match within `tol`.
    """
    available_arr = np.asarray(available)
    matched_values, matched_names = [], []
    for T in requested:
        idx = int(np.argmin(np.abs(available_arr - T)))
        diff = abs(available_arr[idx] - T)
        if diff > tol:
            raise ValueError(
                f"Requested period {T} has no match within tol={tol} "
                f"(closest available: {available_arr[idx]}, diff={diff})."
            )
        matched_values.append(available[idx])
        matched_names.append(available_names[idx])
    return matched_values, matched_names


def _min_usable_frequency(df: pd.DataFrame) -> np.ndarray:
    """
    Minimum usable frequency per record (same definition as in
    `load_data` for the EAS flatfile): the larger of the two high-pass
    corners, each multiplied by the usable-frequency factor.
    """
    factor = np.where(df['usable_frequency_factor'] == -999, 1.25, df['usable_frequency_factor'])
    v1 = np.where(df['hpass_fc_h1'] <= 0, 0.1, df['hpass_fc_h1']) * factor
    v2 = np.where(df['hpass_fc_h2'] <= 0, 0.1, df['hpass_fc_h2']) * factor
    return np.where(v1 > v2, v1, v2)


def load_psa_data(
    psa_flatfile_path: str,
    motion_ids: Sequence[int],
    periods: Sequence[float],
    period_tolerance: float = 1e-6,
    on_missing: str = "raise",
) -> Tuple[pd.DataFrame, List[float], List[str]]:
    """
    Select observed PSA for a given set of motions from the NGA West3
    PSA (RotD50) flatfile.

    All record-level selection (region, magnitude, distance, outliers,
    ...) is assumed to have been done already when building the EAS
    data, so only `motion_ids` is used to select records here. Values
    are set to NaN where the period is beyond the usable range, i.e.
    where the minimum usable frequency exceeds 1/T, or where PSA <= 0.

    Parameters
    ----------
    psa_flatfile_path : str or Path
        Full path to the NGA West3 PSA RotD50 flatfile CSV.
    motion_ids : array-like of int
        Motion IDs to select (e.g. `data_used['motion_id']` from
        `prepare_data`). Must be unique. The output rows follow this
        order, so they line up with the EAS record table.
    periods : array-like of float
        Target PSA periods (s). Matched to the flatfile's own
        `psa_rotd50(...)` columns within `period_tolerance`. Raises if
        any period has no match.
    period_tolerance : float
        Maximum allowed |requested - actual| when matching `periods`.
    on_missing : {'raise', 'drop'}
        What to do if some `motion_ids` are not in the PSA flatfile:
        raise a ValueError, or drop them (the returned `motion_id`
        column then shows which motions were found).

    Returns
    -------
    data_psa : DataFrame
        Column `motion_id` plus one column per period (linear PSA, in
        the flatfile's units, named like the flatfile columns).
    periods_used : list[float]
    names_periods_used : list[str]
    """
    if on_missing not in ("raise", "drop"):
        raise ValueError("on_missing must be 'raise' or 'drop'")

    motion_ids = np.asarray(motion_ids)
    if pd.Series(motion_ids).duplicated().any():
        raise ValueError("motion_ids contains duplicates.")

    data_psa = pd.read_csv(psa_flatfile_path)

    names_all = [col for col in data_psa.columns if col.startswith('psa_rotd50')]
    periods_all = [convert_column_name_to_period(col) for col in names_all]
    periods_used, names_periods_used = _match_periods(
        periods, periods_all, names_all, tol=period_tolerance,
    )

    data_psa = data_psa[data_psa['motion_id'].isin(motion_ids)]
    missing = np.setdiff1d(motion_ids, data_psa['motion_id'].to_numpy())
    if len(missing) > 0:
        if on_missing == "raise":
            raise ValueError(
                f"{len(missing)} motion_ids not found in PSA flatfile "
                f"(first few: {missing[:10].tolist()})."
            )
        motion_ids = motion_ids[~np.isin(motion_ids, missing)]

    # same order as motion_ids
    data_psa = data_psa.set_index('motion_id').loc[motion_ids]

    min_freq = _min_usable_frequency(data_psa)
    periods_arr = np.asarray(periods_used, dtype=float)
    psa = data_psa[names_periods_used].to_numpy(dtype=float)

    unusable = min_freq[:, None] > 1.0 / periods_arr[None, :]
    with np.errstate(invalid='ignore'):
        unusable |= psa <= 0
    psa[unusable] = np.nan

    out = pd.DataFrame(psa, columns=names_periods_used)
    out.insert(0, 'motion_id', motion_ids)

    return out, periods_used, names_periods_used
