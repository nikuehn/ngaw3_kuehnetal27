"""
Documentation table summarizing the GMM's coefficients: how each depends
on frequency, whether it's shared between WUS/global or dataset-local,
and any constraints -- generated from the actual run configuration
(`data_dict` flags + `sharing_config`) rather than hand-maintained, so it
can't drift out of sync with a given fit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

COLUMNS = ["Coefficient", "Frequency dependence", "Regional", "Constraints"]

# LaTeX rendering of each coefficient name, in math mode.
LATEX_NAMES: Dict[str, str] = {
    "c_0": r"$c_0$",
    "c_m1": r"$c_{m1}$",
    "c_m2": r"$c_{m2}$",
    "c_m3": r"$c_{m3}$",
    "c_hw": r"$c_{hw}$",
    "c_nft_1": r"$c_{nft,1}$",
    "c_nft_2": r"$c_{nft,2}$",
    "c_nm": r"$c_{nm}$",
    "c_rev": r"$c_{rev}$",
    "c_gs1": r"$c_{gs1}$",
    "c_zt": r"$c_{zt}$",
    "c_vs": r"$c_{vs}$ (meas./est.)",
    "c_attn": r"$c_{attn}$",
    "Q_0": r"$Q_0$",
    "Q_exp": r"$Q_{exp}$",
    "gs_break": r"$R_{gs,break}$",
    "gs_exp": r"$n_{gs}$",
    "zt_break": r"$Z_{t,break}$",
    "c_basin": r"$c_{basin}$",
    "c_region": r"$c_{subregion}$",
}


def _regional_label(sharing_config: Dict[str, str], name: str, fallback: str) -> str:
    sharing = sharing_config.get(name)
    if sharing == "shared":
        return "Shared (WUS \\& global)"
    elif sharing == "dataset_local":
        return "Local (separate for WUS \\& global)"
    return fallback


def build_coefficient_table(
    data_dict: Dict[str, Any],
    sharing_config: Dict[str, str],
) -> pd.DataFrame:
    """
    Build a summary table of the model's coefficients from the actual
    run configuration.

    Parameters
    ----------
    data_dict : dict
        The same dict passed to model()/guide() as **kwargs -- reads
        c0_parametric, cgs1_parametric, calc_nft, estimate_gs_break,
        estimate_zt_break, func_gs_scaling, estimate_gs_exp, attn_Q,
        dist_cell.
    sharing_config : dict
        e.g. `DEFAULT_COEFFICIENT_SHARING` or your own copy -- used for
        the "Regional" column of every coefficient that goes through
        `resolve_shared_coefficient`.

    Returns
    -------
    DataFrame, columns: Coefficient, Frequency dependence, Regional, Constraints
    """
    rows: List[Tuple[str, str, str, str]] = []

    # --- c_0 ---
    if data_dict.get("c0_parametric", False):
        freq_dep = "Parametric (logistic hinge + linear high-frequency term)"
    else:
        freq_dep = "Spline"
    rows.append(("c_0", freq_dep, _regional_label(sharing_config, "c_0", "Local"), "None"))

    # --- c_m1, c_m2 ---
    for name in ("c_m1", "c_m2"):
        rows.append((
            name, "Spline",
            _regional_label(sharing_config, name, "Local"),
            "Positive, decreasing in frequency",
        ))

    # --- c_m3: single scalar, always shared (no dataset_local path implemented) ---
    rows.append(("c_m3", "Constant", "Shared (always identical for WUS \\& global)", "Positive"))

    # --- c_hw, c_nm, c_rev, c_zt ---
    for name in ("c_hw", "c_nm", "c_rev", "c_zt"):
        rows.append((
            name, "Spline",
            _regional_label(sharing_config, name, "Local"),
            "None",
        ))

    # --- c_nft_1, c_nft_2 ---
    calc_nft = data_dict.get("calc_nft", "ya14")
    if calc_nft == "ya14":
        rows.append(("c_nft_1", "Unmodeled (fixed, Yenier \\& Atkinson 2014)", "N/A", "N/A"))
        rows.append(("c_nft_2", "Unmodeled (fixed, Yenier \\& Atkinson 2014)", "N/A", "N/A"))
    elif calc_nft == "coeff":
        rows.append(("c_nft_1", "Constant (estimated)", "Shared (no dataset\\_local path implemented)", "None"))
        rows.append(("c_nft_2", "Constant (estimated)", "Shared (no dataset\\_local path implemented)", "Positive"))
    elif calc_nft == "freq":
        rows.append(("c_nft_1", "Spline", "Shared (no dataset\\_local path implemented)", "Decreasing"))
        rows.append(("c_nft_2", "Spline", "Shared (no dataset\\_local path implemented)", "Positive, increasing"))

    # --- c_gs1 ---
    if data_dict.get("cgs1_parametric", False):
        freq_dep = "Parametric (logistic hinge)"
    else:
        freq_dep = "Spline"
    rows.append(("c_gs1", freq_dep, _regional_label(sharing_config, "c_gs1", "Local"), "Positive"))

    # --- c_vs (measured / estimated categories) ---
    rows.append((
        "c_vs", "Spline",
        _regional_label(sharing_config, "c_vs", "Local"),
        "None",
    ))

    # --- attenuation ---
    if data_dict.get("dist_cell") is not None:
        rows.append(("Q_0", "Constant per attenuation cell (estimated)",
                     "WUS only (global always uses $c_{attn}$)", "Positive"))
        rows.append(("Q_exp", "Constant (estimated)",
                     "WUS only (global always uses $c_{attn}$)", "Positive"))
    elif data_dict.get("attn_Q", True):
        rows.append(("Q_0", "Constant (estimated)",
                     "WUS only (global always uses $c_{attn}$)", "Positive"))
        rows.append(("Q_exp", "Constant (estimated)",
                     "WUS only (global always uses $c_{attn}$)", "Positive"))
    else:
        rows.append(("c_attn", "Spline", _regional_label(sharing_config, "c_attn", "Local"), "Positive"))

    # --- gs_break, zt_break: WUS-only estimate; global always a fixed literal ---
    if data_dict.get("estimate_gs_break", False):
        rows.append(("gs_break", "Constant (estimated)",
                     "Local (WUS estimated; global fixed at 50)", "Positive"))
    else:
        rows.append(("gs_break", "Unmodeled (fixed at 50)",
                     "Shared (fixed constant for WUS \\& global)", "N/A"))

    if data_dict.get("estimate_zt_break", False):
        rows.append(("zt_break", "Constant (estimated)",
                     "Local (WUS estimated; global fixed at 1.5)", "Positive"))
    else:
        rows.append(("zt_break", "Unmodeled (fixed at 1.5)",
                     "Shared (fixed constant for WUS \\& global)", "N/A"))

    # --- gs_exp ---
    func_gs_scaling = data_dict.get("func_gs_scaling", "stafford")
    if func_gs_scaling == "stafford":
        rows.append(("gs_exp", "Unmodeled (fixed at 2, power-law hinge form)", "N/A", "N/A"))
    else:
        estimate_gs_exp = data_dict.get("estimate_gs_exp", "fixed")
        if estimate_gs_exp == "fixed":
            rows.append(("gs_exp", "Unmodeled (fixed at 2)",
                         "Shared (fixed constant for WUS \\& global)", "N/A"))
        elif estimate_gs_exp == "coeff":
            rows.append(("gs_exp", "Constant (estimated)",
                         "Local (WUS estimated; global fixed at 2)", "Positive"))
        elif estimate_gs_exp == "freq":
            rows.append(("gs_exp", "Spline",
                         "Local (WUS estimated; global fixed at 2)", "Positive"))

    # --- basin term and geology subregion effect: WUS only, no global counterpart ---
    rows.append(("c_basin", "Spline (per basin category)",
                "WUS only (fixed at 0 for global)", "One category fixed at 0 (reference basin)"))
    rows.append(("c_region", "Spline, correlated across frequency (MultivariateNormal)",
                "WUS only (fixed at 0 for global)", "None"))

    return pd.DataFrame(rows, columns=COLUMNS)


def coefficient_table_to_latex(
    df: pd.DataFrame,
    caption: str = "Summary of GMM median-scaling coefficients.",
    label: str = "tab:gmm_coefficients",
) -> str:
    """
    Render a coefficient table (from `build_coefficient_table`) as a
    standalone LaTeX `table` environment, ready to \\input or paste into
    a report.

    Coefficient names are rendered in math mode via `LATEX_NAMES`
    (falling back to the raw name, escaped, if not in that mapping).
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{llll}",
        r"\toprule",
        " & ".join(COLUMNS) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name = LATEX_NAMES.get(row["Coefficient"], row["Coefficient"].replace("_", r"\_"))
        cells = [name, row["Frequency dependence"], row["Regional"], row["Constraints"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
