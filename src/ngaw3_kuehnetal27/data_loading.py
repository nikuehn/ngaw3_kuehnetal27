"""
Loading and filtering the NGA West3 EAS flatfile.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ngaw3_kuehnetal27.utils import convert_column_name_to_frequency, convert_frequency_to_name, package_data_path


def _match_frequencies(
    requested: Sequence[float],
    available: Sequence[float],
    available_names: Sequence[str],
    tol: float = 1e-6,
) -> Tuple[List[float], List[str]]:
    """
    Match each requested frequency to the nearest available column
    frequency, within `tol`. Column-derived frequencies carry ~10
    decimal places of floating-point noise (e.g. 5.011872000), so exact
    equality isn't usable here.

    Raises
    ------
    ValueError
        If any requested frequency has no available match within `tol`.
    """
    available_arr = np.asarray(available)
    matched_values, matched_names = [], []
    for f in requested:
        idx = int(np.argmin(np.abs(available_arr - f)))
        diff = abs(available_arr[idx] - f)
        if diff > tol:
            raise ValueError(
                f"Requested frequency {f} has no match within tol={tol} "
                f"(closest available: {available_arr[idx]}, diff={diff})."
            )
        matched_values.append(available[idx])
        matched_names.append(available_names[idx])
    return matched_values, matched_names


def load_data(
    eas_flatfile_path: str,
    site_labels_path: Optional[str] = None,
    frequencies: Optional[Sequence[float]] = None,
    minimum_frequency: float = 0.099999,
    maximum_frequency: float = 21,
    step_frequency: int = 10,
    freq_tolerance: float = 1e-6,
    region: str = "California",
    rrup_max: float = 300,
    outlier_ids: Optional[List[int]] = None,
):
    """
    Load and filter the NGA West3 EAS flatfile for one region.

    Parameters
    ----------
    eas_flatfile_path : str or Path
        Full path to the NGA West3 EAS flatfile CSV. Not shipped with
        the package (too large) -- point this at wherever you keep it
        locally.
    site_labels_path : str or Path, optional
        Full path to the site labels CSV (basin/region/bouguer).
        Defaults to the copy shipped with the package
        (ngaw3_kuehnetal27/data/site_data_assigned_apr26.csv) -- pass an
        explicit path only if you need to override it.
    frequencies : array-like of float, optional
        Specific target frequencies (Hz) to select. Matched to the
        flatfile's own frequency columns within `freq_tolerance` (see
        `_match_frequencies`). If given, `minimum_frequency` /
        `maximum_frequency` / `step_frequency` are ignored. Raises if
        any requested frequency has no match within tolerance.
    minimum_frequency, maximum_frequency, step_frequency :
        Used only if `frequencies` is not given -- selects an index
        range + stride over the flatfile's available frequency columns,
        same as before.
    freq_tolerance : float
        Maximum allowed |requested - actual| when matching `frequencies`.
    region : {'California', 'WUS', 'Global'}
    rrup_max : float
    outlier_ids : list[int], optional
        Motion IDs to exclude as outliers. Defaults to an empty list --
        no records excluded on this basis.

    Returns
    -------
    data_selected3 : DataFrame
    frequencies_used : list[float]
    """
    if site_labels_path is None:
        site_labels_path = package_data_path("site_data_assigned_apr26.csv")

    data_ngaw3 = pd.read_csv(eas_flatfile_path)
    site_labels = pd.read_csv(site_labels_path)
    site_region_unknown = 0

    offshore_station_ids = [15331, 15342, 15343, 15344, 15345, 15346, 15347, 15348,
                            15349, 15350, 15351, 15353, 15354, 15355, 15356, 15357,
                            15358, 15359, 15360, 15362, 15366, 15367, 15368, 15369,
                            15370, 15371, 15372, 15373, 15374, 15375, 15376, 15377,
                            15378, 15379, 15380, 15381, 15382, 15383, 15384, 15385,
                            15386, 15387, 15388, 15389, 15390, 15391, 15393, 15394,
                            15395, 15396, 15397]

    offshore_event_ids = [3, 5, 11, 13, 22, 26, 3275, 3276, 3277, 3280, 3332, 3333,
                          3336, 3337, 3339, 3340, 3341, 3344, 3345, 3349, 3350, 3351,
                          3352, 3353, 3354, 3355, 3356, 3358, 3359, 3360, 3361, 3362,
                          3363, 3364, 3365, 3366, 3367, 3368, 3369, 3371, 3375, 3376,
                          3377, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387,
                          3388, 3389, 3391, 3392, 3393, 3394, 3395, 3396, 3398, 3399,
                          3400, 3401, 3402, 3403, 3407, 3408, 3409, 3410, 3411, 3418,
                          3420, 3422, 3425, 3431, 3434, 9196]
    outlier_ids = outlier_ids if outlier_ids is not None else []

    names_frequencies = list(filter(lambda col: col.startswith('eas'), data_ngaw3.columns))
    all_frequencies = [convert_column_name_to_frequency(col) for col in names_frequencies]

    if region == 'California':
        reg_condition = (data_ngaw3['event_subdivision'].isin(['California']))
    elif region == 'WUS':
        reg_condition = (data_ngaw3['event_country'].isin(['United States of America']))
    elif region == 'Global':
        reg_condition = (data_ngaw3['event_country'].isin(["Mexico", "Taiwan", "Greece",
                                                            "Italy", "New Zealand",
                                                            "Japan", "Turkey"]))
    else:
        raise ValueError("Region not recognized")

    data_ngaw3 = (
        data_ngaw3
        .assign(
            v1=np.where(data_ngaw3['hpass_fc_h1'] <= 0, 0.1, data_ngaw3['hpass_fc_h1']) *
               np.where(data_ngaw3['usable_frequency_factor'] == -999, 1.25, data_ngaw3['usable_frequency_factor']),
            v2=np.where(data_ngaw3['hpass_fc_h2'] <= 0, 0.1, data_ngaw3['hpass_fc_h2']) *
               np.where(data_ngaw3['usable_frequency_factor'] == -999, 1.25, data_ngaw3['usable_frequency_factor']),
            v1a=np.where(data_ngaw3['lpass_fc_h1'] <= 0, 30, data_ngaw3['lpass_fc_h1']) /
                np.where(data_ngaw3['usable_frequency_factor'] == -999, 1.25, data_ngaw3['usable_frequency_factor']),
            v2a=np.where(data_ngaw3['lpass_fc_h2'] <= 0, 30, data_ngaw3['lpass_fc_h2']) /
                np.where(data_ngaw3['usable_frequency_factor'] == -999, 1.25, data_ngaw3['usable_frequency_factor']),
        )
        .assign(
            min_freq=lambda x: np.where(x['v1'] > x['v2'], x['v1'], x['v2']),
            max_freq=lambda x: np.where(x['v1a'] < x['v2a'], x['v1a'], x['v2a']),
        )
    )

    condition = (data_ngaw3['magnitude_type'] == 'Mw') & \
        (data_ngaw3['magnitude'].between(4, 9)) & \
        (data_ngaw3['rrup'].between(0, rrup_max)) & \
        (data_ngaw3['hypocentral_distance'] > 0) & \
        (data_ngaw3['ztor'].between(0, 30)) & \
        (data_ngaw3['hypocenter_depth'].between(-10, 30)) & \
        (data_ngaw3['dip'] > 0) & \
        (data_ngaw3['vs30'] > 0) & \
        (data_ngaw3['sensor_depth'] <= 5) & \
        (~data_ngaw3['cosmos_station_type'].isin([7, 9, 10, 11, 12, 13, 14, 15, 20, 26, 50, 51, 52])) & \
        (~data_ngaw3['station_id'].isin(offshore_station_ids)) & \
        (~data_ngaw3['event_id'].isin(offshore_event_ids)) & \
        (~data_ngaw3['motion_id'].isin(outlier_ids)) & \
        reg_condition & \
        (data_ngaw3['eas(1p000000000Hz)'] > 0) & \
        (data_ngaw3['eas(5p011872000Hz)'] > 0) & \
        (data_ngaw3['eas(2p511886300Hz)'] > 1e-7) & \
        (data_ngaw3['eas(10p000000000Hz)'].between(1e-11, 10)) & \
        (data_ngaw3['max_freq'] > data_ngaw3['min_freq']) & \
        (data_ngaw3['max_freq'] > 5.1) & \
        (data_ngaw3['min_freq'] < 4.0)

    data_selected = data_ngaw3[condition]

    if frequencies is not None:
        frequencies_used, names_frequencies_used = _match_frequencies(
            frequencies, all_frequencies, names_frequencies, tol=freq_tolerance,
        )
    else:
        min_index = next((i for i, val in enumerate(all_frequencies) if val > minimum_frequency), None)
        max_index = next((len(all_frequencies) - 1 - i for i, val in enumerate(all_frequencies[::-1])
                           if val < maximum_frequency), None)
        frequencies_used = all_frequencies[min_index:max_index:step_frequency]
        names_frequencies_used = names_frequencies[min_index:max_index:step_frequency]

    column_mapping = {col: convert_column_name_to_frequency(col) for col in names_frequencies}
    data_selected2 = data_selected.rename(columns=column_mapping)

    def set_values_to_nan(row, numerical_value):
        if row['min_freq'] > numerical_value or row['max_freq'] < numerical_value:
            return np.nan
        return row[numerical_value]

    for freq in frequencies_used:
        data_selected2[freq] = data_selected2.apply(lambda row: set_values_to_nan(row, freq), axis=1)

    data_selected3 = pd.merge(
        data_selected2,
        site_labels[['station_id', 'basin_label', 'region_id', 'bouguer']],
        on='station_id', how='left',
    )

    data_selected3['regional_label'] = data_selected3['region_id'].fillna(site_region_unknown)
    data_selected3['basin_label'] = data_selected3['basin_label'].fillna(1)
    data_selected3['bouguer'] = data_selected3['bouguer'].fillna(-9999)
   
    column_mapping = {col: convert_frequency_to_name(col) for col in all_frequencies}
    data_selected3 = data_selected3.rename(columns=column_mapping)
    frequencies_used_names = [convert_frequency_to_name(col) for col in frequencies_used]

    return data_selected3, frequencies_used, frequencies_used_names

def prepare_data(frequencies_used, data_selected3, n_rec_eq=20,
                  mag_bins=(4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.)):
    """
    Turn the filtered flatfile into record/event/station tables ready
    for `extract_model_arrays` (see model_inputs.py).

    Same logic as your original `prepare_data` -- fixed only to `.copy()`
    the record-level slice before mutating it in place (avoids a pandas
    SettingWithCopyWarning).
    """
    selected_columns = ['magnitude', 'rrup', 'rjb', 'rx', 'ry0',
                        'ztor', 'hypocenter_depth', 'vs30',
                        'mechanism_based_on_Rake',
                        'fault_width', 'dip',
                        'event_id', 'station_id', 'motion_id', 'ravg',
                        'station_latitude', 'station_longitude',
                        'hypocenter_latitude', 'hypocenter_longitude',
                        'closest_point_latitude', 'closest_point_longitude',
                        'closest_point_depth',
                        'z1p0_preferred', 'z1p0_preferred_lnstd', 'z1p0_code_id',
                        'z2p5_preferred', 'z2p5_preferred_lnstd', 'z2p5_code_id',
                        'event_country', 'event_subdivision',
                        'site_country', 'site_subdivision',
                        'vs30_code_id', 'vs30_lnstd',
                        'basin_label', 'regional_label', 'bouguer']

    df = data_selected3[selected_columns + frequencies_used].copy()
    df[frequencies_used] = df[frequencies_used].apply(np.log)

    event_counts = df.groupby('event_id').size().reset_index(name='event_count')
    df_with_counts = df.merge(event_counts, on='event_id')

    def sample_up_to_n(group, n=20, random_state=1701):
        return group.sample(n=min(len(group), n), random_state=random_state)

    data_used = (df_with_counts.loc[
        (df_with_counts['event_country'] == "United States of America") |
        ((df_with_counts['event_country'] != "United States of America") &
         (df_with_counts['magnitude'] < 6) &
         (df_with_counts['event_count'] >= 10)) |
        ((df_with_counts['event_country'] != "United States of America") &
         (df_with_counts['magnitude'] >= 6) &
         (df_with_counts['event_count'] >= 3))
    ].set_index('event_id').groupby('event_id', group_keys=False)
     .apply(sample_up_to_n, n=n_rec_eq, random_state=1701)
     .sort_values(by='motion_id').copy())

    data_used['event_id'] = data_used.index
    data_used = data_used.copy().reset_index(drop=True)

    data_used['F_nm'] = 0
    data_used.loc[data_used['mechanism_based_on_Rake'].isin([1, 4]), 'F_nm'] = 1
    data_used['F_rev'] = 0
    data_used.loc[data_used['mechanism_based_on_Rake'].isin([2, 3]), 'F_rev'] = 1

    data_used['eq'] = pd.factorize(data_used['event_id'])[0]
    data_used['stat'] = pd.factorize(data_used['station_id'])[0]

    data_used['basin'] = data_used['basin_label'].astype(int)
    data_used['regional'] = data_used['regional_label'].astype(int)
    data_used['mag_bin'] = pd.cut(data_used['magnitude'], bins=list(mag_bins),
                                   labels=False, right=False, include_lowest=True)

    data_eq = data_used.groupby(['eq', 'magnitude', 'ztor', 'hypocenter_depth', 'mechanism_based_on_Rake',
                                 'fault_width', 'dip',
                                 'event_id', 'F_nm', 'F_rev',
                                 'hypocenter_latitude', 'hypocenter_longitude',
                                 'event_country', 'event_subdivision']
                                ).size().reset_index().rename(columns={0: 'count'})
    data_eq['minR'] = data_used.loc[data_used.groupby('eq').rrup.idxmin()].rrup.reset_index(drop=True)
    data_eq['maxR'] = data_used.loc[data_used.groupby('eq').rrup.idxmax()].rrup.reset_index(drop=True)
    data_eq['range'] = data_eq['maxR'] - data_eq['minR']

    mag_sd = np.repeat(0.1, data_eq.shape[0])
    mag_sd[data_eq.magnitude > 5] = 0.05
    mag_sd[data_eq.magnitude > 7] = 0.01
    data_eq['mag_sd'] = mag_sd

    data_stat = data_used.groupby(['stat', 'station_id', 'basin', 'regional',
                                   'vs30', 'station_latitude', 'station_longitude',
                                  'z1p0_preferred', 'z1p0_preferred_lnstd', 'z1p0_code_id',
                                  'z2p5_preferred', 'z2p5_preferred_lnstd', 'z2p5_code_id',
                                  'site_country', 'site_subdivision',
                                  'vs30_code_id', 'vs30_lnstd', 'bouguer']
                                  ).size().reset_index().rename(columns={0: 'count'})
    data_stat['vs30_measured'] = np.where(data_stat['vs30_code_id'].isin([0, 1, 2]), 0, 1)

    return data_used, data_eq, data_stat
