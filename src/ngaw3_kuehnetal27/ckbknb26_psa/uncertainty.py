"""
Epistemic (parameter) uncertainty of the CKBKNB26 (campbelletal27) median
model: sample coefficient sets from the fixed-effect covariance matrices and
evaluate the median for every sample. Python port of the sampling in
`calc_uncertainty_sampling.R`.

Workflow
--------
1. Once, in R: `export_cov_fixed.R` converts `cov_fixed.rds` (one covariance
   matrix per period) to a plain long-format CSV.
2. `load_coefficient_covariances(csv)` reads it.
3. `sample_coefficients_campbelletal27(coef, covs, n_samples)` returns
   `n_samples` coefficient sets.
4. `calculate_ln_median_psa_samples_campbelletal27(coef_samples, **scenario)`
   gives ln(median PSA) for every sample, shape ``(n_samples, n, n_periods)``.

Sampling follows the R script exactly: for each period, draw deviations from
MVN(0, Sigma_period), independently across periods, and add them to the mean
coefficients of `CoefMean.csv`. A deviation is matched to a coefficient by
**name** (a covariance row name equal to a `Coefficients` field); coefficients
without a matching name stay at their mean, and covariance entries without a
matching coefficient are ignored. If a name occurs more than once in a
covariance matrix, ``reg=1`` uses the first occurrence and ``reg=2`` the last.

Only the median coefficients are perturbed: the standard deviations
(`sigma_core`) and the nonlinear-site coefficients (`nl_coef`) are unchanged.
"""
from __future__ import annotations

import warnings
from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .median_core import Coefficients, calculate_ln_median_psa_campbelletal27

__all__ = [
    "CoefficientCovariances",
    "load_coefficient_covariances",
    "sample_coefficients_campbelletal27",
    "calculate_ln_median_psa_samples_campbelletal27",
]

_PERIOD_RTOL = 1e-6   # periods are matched with a tolerance (float32-rounded `coef.Per`, see median_core)
_PERIOD_ATOL = 1e-9


class CoefficientCovariances(NamedTuple):
    """Per-period covariance matrices, in the order of the CSV."""
    periods: np.ndarray            # (n_cov,)
    names: List[List[str]]         # per period: parameter names (rows = columns), duplicates kept
    covs: List[np.ndarray]         # per period: (k, k) covariance matrix


def load_coefficient_covariances(csv_path: str) -> CoefficientCovariances:
    """
    Read the long-format CSV written by `export_cov_fixed.R`
    (columns ``period, i, j, name_i, name_j, cov``).

    Raises
    ------
    ValueError
        If a matrix is incomplete, not symmetric, or its row and column names
        differ.
    """
    df = pd.read_csv(
        csv_path,
        dtype={"name_i": str, "name_j": str},
        keep_default_na=False,
        na_values={"period": ["NA"], "i": ["NA"], "j": ["NA"], "cov": ["NA"]},
    )
    missing_cols = {"period", "i", "j", "name_i", "name_j", "cov"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing_cols)}")

    periods, names, covs = [], [], []
    for per, g in df.groupby("period", sort=False):
        k = int(g["i"].max())
        if int(g["j"].max()) != k or len(g) != k * k:
            raise ValueError(f"period {per}: expected a complete {k}x{k} matrix, found {len(g)} entries")
        mat = np.full((k, k), np.nan)
        mat[g["i"].to_numpy(int) - 1, g["j"].to_numpy(int) - 1] = g["cov"].to_numpy(float)
        if np.isnan(mat).any():
            raise ValueError(f"period {per}: covariance matrix has missing entries")
        if not np.allclose(mat, mat.T, rtol=1e-8, atol=1e-10 * np.abs(mat).max()):
            raise ValueError(f"period {per}: covariance matrix is not symmetric")

        row_names = g.drop_duplicates("i").sort_values("i")["name_i"].tolist()
        col_names = g.drop_duplicates("j").sort_values("j")["name_j"].tolist()
        if row_names != col_names:
            raise ValueError(f"period {per}: row names and column names differ")

        periods.append(float(per))
        names.append(row_names)
        covs.append(mat)

    if any(n != names[0] for n in names[1:]):
        warnings.warn(
            "Covariance parameter names differ between periods. Each period is matched with its "
            "own names here; calc_uncertainty_sampling.R applies the names of period 2 to all periods.",
            stacklevel=2,
        )
    return CoefficientCovariances(np.asarray(periods), names, covs)


def _sample_zero_mean_mvn(cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    n draws from MVN(0, cov), shape (n, k). Eigendecomposition like
    `mvtnorm::rmvnorm` (default method), so positive *semi*-definite matrices
    -- e.g. with a fixed or redundant parameter -- are fine; tiny negative
    eigenvalues are set to zero.
    """
    w, v = np.linalg.eigh(cov)
    if w.min() < -np.sqrt(np.finfo(float).eps) * abs(w.max()):
        warnings.warn("covariance matrix is numerically not positive semidefinite; "
                      "negative eigenvalues set to 0", stacklevel=3)
    factor = v * np.sqrt(np.clip(w, 0.0, None))      # factor @ factor.T == cov
    return rng.standard_normal((n, cov.shape[0])) @ factor.T


def _match_period(per: float, cov_periods: np.ndarray) -> int:
    hits = np.nonzero(np.isclose(cov_periods, per, rtol=_PERIOD_RTOL, atol=_PERIOD_ATOL))[0]
    if len(hits) == 0:
        raise ValueError(f"No covariance matrix for period {per:g} s "
                         f"(available: {np.array2string(cov_periods, precision=4)})")
    return int(hits[0])


def sample_coefficients_campbelletal27(
    coef: Coefficients,
    covs: CoefficientCovariances,
    n_samples: int,
    *,
    seed: int = 0,
    reg: int = 1,
) -> Coefficients:
    """
    Draw `n_samples` coefficient sets: mean coefficients plus MVN deviations
    from each period's covariance matrix (see module docstring).

    Parameters
    ----------
    coef : Coefficients
        Mean coefficients (`load_coefficients`); also fixes which periods are
        sampled. A subset of the covariance periods is fine.
    covs : CoefficientCovariances
        From `load_coefficient_covariances`.
    n_samples : int
    seed : int
        Reproducible. Each period gets its own random stream derived from
        (seed, position of its matrix in the CSV), so a period's draws do not
        depend on which other periods are requested.
    reg : {1, 2}
        Which occurrence to use for parameter names that appear more than once
        in a covariance matrix: 1 = first, 2 = last (as in the R script).

    Returns
    -------
    Coefficients
        Same fields as `coef`, but every field has shape ``(n_samples, n_periods)``
        (unperturbed fields are repeated). Not for `calculate_ln_median_psa_campbelletal27`
        directly: use `calculate_ln_median_psa_samples_campbelletal27`, or pick one
        sample with ``jax.tree_util.tree_map(lambda a: a[s], coef_samples)``.
    """
    if reg not in (1, 2):
        raise ValueError(f"reg must be 1 or 2, got {reg!r}")

    per = np.asarray(coef.Per, dtype=np.float64)
    fields = [f for f in Coefficients._fields if f != "Per"]
    out = {f: np.tile(np.asarray(getattr(coef, f), dtype=np.float64), (n_samples, 1)) for f in fields}

    for p, per_p in enumerate(per):
        ci = _match_period(per_p, covs.periods)
        dev = _sample_zero_mean_mvn(covs.covs[ci], n_samples, np.random.default_rng([seed, ci]))
        for f in fields:
            pos = [j for j, name in enumerate(covs.names[ci]) if name == f]
            if pos:
                out[f][:, p] += dev[:, pos[0] if reg == 1 else pos[-1]]

    return Coefficients(
        Per=jnp.asarray(np.tile(per, (n_samples, 1)), dtype=jnp.float64),
        **{f: jnp.asarray(out[f], dtype=jnp.float64) for f in fields},
    )


def calculate_ln_median_psa_samples_campbelletal27(
    coef_samples: Coefficients, nl_coef=None, **scenario
) -> jnp.ndarray:
    """
    ln(median PSA) for every coefficient sample.

    Parameters
    ----------
    coef_samples : Coefficients
        From `sample_coefficients_campbelletal27`, fields ``(n_samples, n_periods)``.
    nl_coef : Ilhan26Coefficients, optional
        As in `calculate_ln_median_psa_campbelletal27` (not perturbed).
    **scenario
        Scenario variables exactly as for `calculate_ln_median_psa_campbelletal27`
        (M, Rrup, Rx, Frv, ...): scalars or arrays of length n.

    Returns
    -------
    jnp.ndarray, shape (n_samples, n, n_periods)

    Notes
    -----
    Samples are processed one after another (`jax.lax.map`, compiled once), so
    memory is that of a single sample plus the output array.
    """
    per = coef_samples.Per[0]
    sampled = {f: getattr(coef_samples, f) for f in Coefficients._fields if f != "Per"}

    def one_sample(fields):
        c = Coefficients(Per=per, **fields)
        return calculate_ln_median_psa_campbelletal27(c, nl_coef=nl_coef, **scenario)

    return jax.lax.map(one_sample, sampled)
