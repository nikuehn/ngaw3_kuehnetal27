"""
Pure math helper functions for median ground-motion scaling: smooth hinge/
ramp functions, geometric spreading forms, hanging-wall scaling, and the
combined magnitude + geometric spreading functions.

Nothing here depends on Coefficients, RecordIndex, or any other
model-bookkeeping structure -- every function takes plain arrays and
returns plain arrays, so these are usable standalone (e.g. for plotting
a scaling curve) as well as from `median_core.py`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pyproj
from importlib import resources
from jax import vmap
from pathlib import Path

Array = jnp.ndarray


def nft_ya14(M: Array) -> Array:
    """Near-fault term (fictitious depth) per Yenier & Atkinson (2014)."""
    return 10 ** (-1.72 + 0.43 * M)


def logistic_hinge(x: Array, c2: Array, c3: Array, x_break: Array, delta: Array = 1) -> Array:
    """Smooth (logistic) hinge between two linear segments of slope c2, c3."""
    return c2 * (x - x_break) + (c3 - c2) * delta * jnp.logaddexp(0, (x - x_break) / delta)


def func_gs(R: Array, lnR: Array, c_gs1: Array, c_gs2: Array, gs_break: Array,
            delta_gs: Array, xi: Array) -> Array:
    """Geometric spreading, power-law hinge form."""
    return c_gs1 * lnR + (c_gs2 - c_gs1) * delta_gs * jnp.log(
        (R ** xi + gs_break ** xi) / (1 + gs_break ** xi)
    )


def func_gs_lh(lnR: Array, c_gs1: Array, c_gs2: Array, gs_break: Array, delta_gs: Array) -> Array:
    """Geometric spreading, logistic-hinge form."""
    return logistic_hinge(lnR, c_gs1, c_gs2, jnp.log(gs_break), delta_gs) + c_gs1 * jnp.log(gs_break)


def smooth_trilinear_ramp(x: Array, c_slope: Array, x_b1: Array, x_b2: Array, delta: Array = 0.1) -> Array:
    """Smooth ramp: 0 below x_b1, rises with slope c_slope, flat above x_b2."""
    ramp_up = c_slope * delta * jnp.logaddexp(0, (x - x_b1) / delta)
    ramp_down = c_slope * delta * jnp.logaddexp(0, (x - x_b2) / delta)
    return ramp_up - ramp_down


def smooth_trilinear_ramp_repar(M: Array, intercept: Array, high_val: Array,
                                 mb1: Array, mb2: Array, delta: Array = 0.2) -> Array:
    """`smooth_trilinear_ramp`, reparameterized by intercept/high-value instead of slope."""
    slope = (high_val - intercept) / (mb2 - mb1)
    return intercept + smooth_trilinear_ramp(M, slope, mb1, mb2, delta=delta)


def smooth_trilinear(mag: Array, c_m1: Array, c_m2: Array, c_m3: Array,
                      mb1: Array, mb2: Array, delta: Array = 0.1) -> Array:
    """
    Smooth trilinear magnitude scaling with differentiable transitions at
    mb1, mb2. Slopes c_m1 (below mb1), c_m2 (between), c_m3 (above mb2).
    """
    first_segment = c_m1 * mag
    transition1 = (c_m2 - c_m1) * delta * jnp.logaddexp(0, (mag - mb1) / delta)
    transition2 = (c_m3 - c_m2) * delta * jnp.logaddexp(0, (mag - mb2) / delta)
    return first_segment + transition1 + transition2


def smooth_trilinear_centered(mag: Array, c_m1: Array, c_m2: Array, c_m3: Array,
                               mb1: Array, mb2: Array, delta: Array = 0.1) -> Array:
    """`smooth_trilinear`, shifted so the function passes through 0 at mag=0."""
    first_segment = c_m1 * mag
    transition1 = (c_m2 - c_m1) * delta * jnp.logaddexp(0, (mag - mb1) / delta)
    transition2 = (c_m3 - c_m2) * delta * jnp.logaddexp(0, (mag - mb2) / delta)
    return first_segment + transition1 + transition2 - c_m1 * mb1


def calculate_hw_scaling(magnitude: Array, dip: Array, fault_width: Array,
                          rx_m: Array, ztor2: Array, ry0_m: Array) -> Array:
    """
    Hanging-wall (HW) scaling using the form from ASK14 GMM for PSA.

    Combines five taper terms for dip, magnitude, Rx, ZTOR, and Ry0.

    Parameters
    ----------
    magnitude, dip, fault_width, rx_m, ztor2, ry0_m : Array
        Earthquake magnitude, fault dip (degrees), fault width, Rx (m),
        ZTOR (km), Ry0 (m).

    Returns
    -------
    Array
        Hanging-wall term.
    """
    h1, h2, h3 = 0.25, 1.5, -0.75

    dip = jnp.asarray(dip)
    magnitude = jnp.asarray(magnitude)
    fault_width = jnp.asarray(fault_width)
    rx_m = jnp.asarray(rx_m)
    ztor2 = jnp.asarray(ztor2)
    ry0_m = jnp.asarray(ry0_m)

    # ---- dip taper ----
    Tmp1 = jnp.where(dip <= 30, 60 / 45, (90 - dip) / 45)

    # ---- magnitude taper (smooth trilinear-hinge reformulation) ----
    c1, c2, c3 = 0, 1.66, 0.2
    mb1, mb2 = 5.5, 6.04
    delta1, delta2 = 0.05, 0.17
    Tmp2 = (c2 - c1) * delta1 * jnp.logaddexp(0, (magnitude - mb1) / delta1) \
        + (c3 - c2) * delta2 * jnp.logaddexp(0, (magnitude - mb2) / delta2)

    # ---- Rx taper ----
    R1 = fault_width * jnp.cos(dip * jnp.pi / 180)
    R2 = 3 * R1
    Tmp3_cond1 = h1 + h2 * (rx_m / R1) + h3 * (rx_m / R1) ** 2
    Tmp3_cond2 = 1 - ((rx_m - R1) / (R2 - R1))
    Tmp3 = jnp.where(rx_m < R1, Tmp3_cond1,
                      jnp.where((rx_m >= R1) & (rx_m <= R2), Tmp3_cond2, 0))

    # ---- ZTOR taper ----
    Tmp4 = jnp.where((ztor2 <= 10) & (ztor2 >= 0), 1 - (ztor2 ** 2 / 100), 0)

    # ---- Ry0 taper ----
    Ry1 = rx_m * jnp.tan(20 * jnp.pi / 180)
    ry_diff = ry0_m - Ry1
    Tmp5 = jnp.where(ry_diff < 0, 1,
                      jnp.where((ry_diff < 5) & (ry_diff >= 0), 1 - (ry_diff / 5), 0))

    # ---- Rx > 0 gate ----
    Tmp6 = jnp.where(rx_m > 0, 1, 0)

    return Tmp1 * Tmp2 * Tmp3 * Tmp4 * Tmp5 * Tmp6


def calculate_hw_scaling_swus(M: Array, Dip: Array, W: Array, Rx: Array, Rjb: Array,
                               Rrup: Array, Ztor: Array, C1: Array, C2: Array,
                               C3: Array, C4: Array) -> dict:
    """
    Alternative hanging-wall model with period-dependent coefficients
    (SWUS form). Not currently called by `calculate_median_core` -- kept
    for comparison / earlier analyses. Confirm still needed before next
    cleanup pass.

    f(M,Dip,W,Rx,Rjb,Rrup,Ztor) =
        C1(T) * cos(dip) * (C2(T) + (1-C2(T)) * tanh(C3(T)*Rx / (W*cos(dip)))) *
        (1 + C4(T)*(M-7)) * Taper(Rrup,Rjb) * Taper(Ztor)

    Returns
    -------
    dict
        Component terms and the final 'hw_scaling' result.
    """
    M, Dip, W, Rx = jnp.asarray(M), jnp.asarray(Dip), jnp.asarray(W), jnp.asarray(Rx)
    Rjb, Rrup, Ztor = jnp.asarray(Rjb), jnp.asarray(Rrup), jnp.asarray(Ztor)
    C1, C2, C3, C4 = jnp.asarray(C1), jnp.asarray(C2), jnp.asarray(C3), jnp.asarray(C4)

    dip_rad = Dip * jnp.pi / 180.0
    cos_dip_term = jnp.cos(dip_rad)

    denominator = W * cos_dip_term
    denominator = jnp.where(denominator == 0, 1e-10, denominator)
    tanh_term = C2 + (1 - C2) * jnp.tanh(C3 * Rx / denominator)

    magnitude_term = 1 + C4 * (M - 7)
    rrup_rjb_taper = 1 - Rjb / (Rrup + 0.1)
    ztor_taper = jnp.maximum(0, 1 - Ztor / 12)

    hw_scaling = C1 * cos_dip_term * tanh_term * magnitude_term * rrup_rjb_taper * ztor_taper

    return {
        "cos_dip_term": cos_dip_term,
        "tanh_term": tanh_term,
        "magnitude_term": magnitude_term,
        "rrup_rjb_taper": rrup_rjb_taper,
        "ztor_taper": ztor_taper,
        "hw_scaling": hw_scaling,
    }


def func_mag_gs(mag: Array, rrup: Array, c_m1: Array, c_m2: Array, c_m3: Array,
                 c_gs1: Array, c_gs2: Array, c_nft_1: Array, c_nft_2: Array,
                 mb1: Array, mb2: Array, delta_gs: Array, gs_break: Array, xi: Array) -> Array:
    """Combined magnitude scaling + geometric spreading, power-law hinge form."""
    nft_p = jnp.exp(c_nft_1 + c_nft_2 * (mag - 4.5))
    lnR_p = jnp.log(jnp.sqrt(rrup ** 2 + nft_p ** 2))
    return (
        smooth_trilinear_centered(mag, c_m1, c_m2, c_m3, mb1, mb2, delta=0.2)
        + func_gs(rrup, lnR_p, -c_gs1, c_gs2, gs_break, delta_gs, xi)
    )


# Vectorized d(median)/d(magnitude) over mags and (per-frequency) coefficients.
_grad_mag_single = jax.grad(func_mag_gs, argnums=0)
_grad_mag_over_mags = vmap(
    _grad_mag_single,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None),
)
compute_magnitude_gradients = vmap(
    _grad_mag_over_mags,
    in_axes=(None, None, 0, 0, None, 0, None, 0, 0, None, None, None, None, None),
)


def func_mag_gs_lh(mag: Array, rrup: Array, c_m1: Array, c_m2: Array, c_m3: Array,
                    c_gs1: Array, c_gs2: Array, c_nft_1: Array, c_nft_2: Array,
                    mb1: Array, mb2: Array, gs_break: Array, gs_exp: Array) -> Array:
    """Combined magnitude scaling + geometric spreading, logistic-hinge form."""
    inv_gs_exp = 1.0 / gs_exp
    nft_p = jnp.exp(c_nft_1 + c_nft_2 * (mag - 4.5))
    lnR_p = jnp.log((rrup ** gs_exp + nft_p ** gs_exp) ** inv_gs_exp)
    lnR_break = jnp.log((gs_break ** gs_exp + nft_p ** gs_exp) ** inv_gs_exp)
    return (
        smooth_trilinear_centered(mag, c_m1, c_m2, c_m3, mb1, mb2, delta=0.2)
        + func_gs_lh(lnR_p, -c_gs1, c_gs2, jnp.exp(lnR_break), delta_gs=0.2)
    )


_grad_mag_single_lh = jax.grad(func_mag_gs_lh, argnums=0)
_grad_mag_over_mags_lh = vmap(
    _grad_mag_single_lh,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None),
)
compute_magnitude_gradients_lh = vmap(
    _grad_mag_over_mags_lh,
    in_axes=(None, None, 0, 0, None, 0, None, 0, 0, None, None, None, 0),
)


# ---------------------------------------------------------------------------
# Coordinate conversion (lat/lon <-> UTM)
# ---------------------------------------------------------------------------

wgs84 = pyproj.CRS('EPSG:4326')


def get_utm_crs(longitude):
    """Determine the UTM CRS (zone) for a given longitude."""
    utm_zone = int((longitude + 180) / 6) + 1
    return pyproj.CRS(f'EPSG:326{utm_zone}')


def latlon_to_utm(lat, lon):
    """Convert lat/lon to UTM (km), using the UTM zone for `lon`."""
    utm_crs = get_utm_crs(lon)
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x / 1000, utm_y / 1000


def latlon_to_utm_lon2(lat, lon, lon2):
    """Convert lat/lon to UTM (km), using the UTM zone for `lon2` instead of `lon`."""
    utm_crs = get_utm_crs(lon2)
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x / 1000, utm_y / 1000


def latlon_to_utm_with_utmcrs(lat, lon, utm_crs):
    """Convert lat/lon to UTM (km), using a caller-supplied UTM CRS."""
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x / 1000, utm_y / 1000


# ---------------------------------------------------------------------------
# Column-name parsing
# ---------------------------------------------------------------------------

def convert_column_name_to_frequency(col_name):
    """Parse a frequency value out of a coefficient column name."""
    relevant_part = col_name[4:-3]
    numeric_string = relevant_part.replace('p', '.')
    return float(numeric_string)

def convert_frequency_to_name(freq):
    """Parse a frequency name value out of a value."""
    return f'F{freq:.3f}'


def convert_column_name_to_period(col_name):
    """Parse a period value out of a coefficient column name."""
    relevant_part = col_name[11:-2]
    numeric_string = relevant_part.replace('p', '.')
    return float(numeric_string)


# ---------------------------------------------------------------------------
# Package data paths
# ---------------------------------------------------------------------------

def package_data_path(filename: str) -> Path:
    """
    Path to a data file shipped inside the ngaw3_kuehnetal27 package
    (ngaw3_kuehnetal27/data/) -- coefficient CSVs for the nonlinear site
    amplification models, the site labels file (basin/region/bouguer),
    or any other small reference table that's small enough to
    version-control alongside the code.

    For large project-level datasets -- e.g. the main ground motion
    database -- pass an explicit path instead; those don't belong in
    the package.
    """
    return resources.files("ngaw3_kuehnetal27") / "data" / filename

def results_data_path(filename: str) -> Path:
    """
    Path to a results file shipped inside the ngaw3_kuehnetal27 package
    (ngaw3_kuehnetal27/results/) -- the reference fitted-coefficient
    JSON, so anyone who `pip install`s the package can load it for
    prediction without needing the full repo checkout.

    `filename` can include subdirectories, e.g. "eas/results_wus_main.json".

    Like `package_data_path`, this only resolves files inside
    src/ngaw3_kuehnetal27/ -- it can't be pointed at a repo-root
    directory (results/ has to live inside the package to be shipped
    by pip at all).
    """
    path = resources.files("ngaw3_kuehnetal27") / "results"
    for part in filename.split("/"):
        path = path / part
    return path

# ---------------------------------------------------------------------------
# GP kernels
# ---------------------------------------------------------------------------

def kernel_sqexp(X, Z, var, length):
    """Squared-exponential (RBF) kernel."""
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    return var * jnp.exp(-0.5 * deltaXsq)


def kernel_matern52(X, Z, var, length):
    """
    Matern kernel with nu = 5/2.

    Parameters
    ----------
    X : Array, shape (n_samples_X, n_features)
    Z : Array, shape (n_samples_Z, n_features)
    var : float
        Variance parameter.
    length : float
        Length scale parameter.

    Returns
    -------
    Array, shape (n_samples_X, n_samples_Z)
    """
    r = jnp.abs((X[:, None] - Z) / length)
    sqrt5_r = jnp.sqrt(5.0) * r
    return var * (1.0 + sqrt5_r + (5.0 / 3.0) * jnp.power(r, 2.0)) * jnp.exp(-sqrt5_r)


# ---------------------------------------------------------------------------
# Array manipulation
# ---------------------------------------------------------------------------

def insert_zero(ref_idx, arr):
    """Insert a scalar zero at position `ref_idx` in a 1D array."""
    return jnp.concatenate([arr[:ref_idx], jnp.array([0.0]), arr[ref_idx:]])


def insert_zero_row(ref_idx, arr):
    """
    Insert a row of zeros at `ref_idx` in a 2D array.

    Parameters
    ----------
    ref_idx : int
        Index where to insert the zero row.
    arr : Array, shape (n1, n_freq)

    Returns
    -------
    Array, shape (n1 + 1, n_freq)
    """
    n1, n_freq = arr.shape
    zero_row = jnp.zeros((1, n_freq))
    return jnp.concatenate([arr[:ref_idx], zero_row, arr[ref_idx:]], axis=0)


# ---------------------------------------------------------------------------
# C-vine correlation parameterization
# ---------------------------------------------------------------------------

def cvine_theta_to_cholesky(theta: jnp.ndarray, K: int) -> jnp.ndarray:
    """
    Convert unconstrained C-vine parameters to a Cholesky factor of a
    correlation matrix.

    Parameters
    ----------
    theta : Array, length K*(K-1)/2
        Unconstrained parameters (real line).
    K : int
        Dimension of the correlation matrix.

    Returns
    -------
    Array, shape (K, K)
        Lower-triangular Cholesky factor of a correlation matrix.
    """
    # Build (row, col) index pairs in vine order (0-indexed), lower triangle.
    rows, cols = [], []
    for lev in range(K - 1):
        for i in range(lev + 1, K):
            rows.append(i)
            cols.append(lev)
    rows = jnp.array(rows)
    cols = jnp.array(cols)

    z = jnp.zeros((K, K))
    z = z.at[rows, cols].set(jnp.tanh(theta))

    # L[i, j] = z[i, j] * prod_{k=0}^{j-1} sqrt(1 - z[i,k]^2)
    # L[i, i] =           prod_{k=0}^{i-1} sqrt(1 - z[i,k]^2)
    safe_z = jnp.tril(z, k=-1)
    sqrt_terms = jnp.sqrt(1.0 - jnp.square(safe_z))

    # Exclusive cumulative product along columns: product of all terms
    # *before* column j.
    cumprod = jnp.cumprod(sqrt_terms, axis=1)
    excl_cumprod = jnp.concatenate([jnp.ones((K, 1)), cumprod[:, :-1]], axis=1)

    L = safe_z * excl_cumprod
    L = L + jnp.diag(jnp.diag(excl_cumprod))

    return L
