"""
Ground Motion Model Neural Network (Equinox / JAX)
===================================================
Predicts ln(PSA) at 21 periods from predictor variables.

Input features
--------------
Continuous:
    M            : moment magnitude                    -> raw
    R             : distance (km)                        -> ln(R), R
    VS           : Vs30 (m/s)                            -> ln(VS), ln(VS) * vsmeas
                     (the second term is an interaction: ln(VS) for
                     measured sites, 0 for estimated sites -- vsmeas_id
                     itself is not passed as a separate feature)
    Z             : depth (km)                           -> raw  [switch to log10 if Z spans >2 decades]

Binary:
    Frev, Fnm : fault mechanism flags                -> raw {0, 1}

Categorical (learned embedding):
    basin_id     : 0-4                                    -> Embedding(n_basin_cats, basin_emb_dim)
    subregion_id : 0-(n_region-1), optional         -> Embedding(n_region_cats, region_emb_dim),
                     only if include_region=True

Ignored here (invariant in this simulation):
    Dip, FW, Rx, Ry0

Output
------
    ln(PSA) at 21 periods  (shape: [batch, 21])

Region handling
----------------
Region is off by default (`include_region=False`), matching how the
EAS model this was trained against was originally used (regions
neglected). Pass `include_region=True` (to `GMMNet`, `train`, and
`predict` via the model's own stored config) to add `subregion_id` as
a second learned categorical embedding, concatenated alongside the
basin embedding.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    df_x: pd.DataFrame,
    include_region: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Continuous + binary features, shape (N, 8):
        M, ln(R), R, ln(VS), ln(VS) * vsmeas, Z, Frev, Fnm
    basin_id (and, if `include_region`, subregion_id) returned
    separately for embedding.

    Returns
    -------
    cont : ndarray, shape (N, 8), float32
    basin_id : ndarray, shape (N,), int32
    region_id : ndarray, shape (N,), int32, or None if not include_region
    """
    R = df_x["R"].values.clip(0.1)
    ln_VS = np.log(df_x["VS"].values.clip(1.0))
    vsmeas = df_x["vsmeas_id"].values.astype(np.float32)

    cont = np.column_stack([
        df_x["M"].values,
        np.log(R),          # geometric spreading proxy
        R,                   # anelastic attenuation proxy
        ln_VS,
        ln_VS * vsmeas,     # interaction: ln(VS) for measured, 0 for estimated
        df_x["Z"].values,
        df_x["Frev"].values,
        df_x["Fnm"].values,
    ]).astype(np.float32)

    basin_id = df_x["basin_id"].values.astype(np.int32)
    region_id = df_x["subregion_id"].values.astype(np.int32) if include_region else None
    return cont, basin_id, region_id


def build_targets(df_psa: pd.DataFrame) -> np.ndarray:
    """ln(PSA), shape (N, 21)."""
    return df_psa.values.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Model definition
# ---------------------------------------------------------------------------

class GMMNet(eqx.Module):
    """
    MLP for ground motion prediction.

    Parameters
    ----------
    hidden_dims     : sequence of ints, e.g. [128, 128, 64]
    n_periods       : output dimension (21)
    basin_emb_dim   : embedding size for basin_id (default 4)
    n_basin_cats    : number of basin categories
    n_cont          : number of continuous+binary input features (8)
    key             : JAX PRNGKey
    include_region  : if True, adds a second embedding for subregion_id
    region_emb_dim  : embedding size for subregion_id (only used if
                      include_region)
    n_region_cats   : number of subregion categories (required if
                      include_region)
    """
    basin_embedding: eqx.nn.Embedding
    region_embedding: Optional[eqx.nn.Embedding]
    layers: list
    n_cont: int = eqx.field(static=True)
    include_region: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_dims: Sequence[int],
        n_periods: int,
        basin_emb_dim: int,
        n_basin_cats: int,
        n_cont: int,
        key: jax.Array,
        include_region: bool = False,
        region_emb_dim: int = 4,
        n_region_cats: Optional[int] = None,
    ):
        if include_region and n_region_cats is None:
            raise ValueError("n_region_cats must be given when include_region=True")

        self.n_cont = n_cont
        self.include_region = include_region

        n_embed_keys = 2 if include_region else 1
        keys = jax.random.split(key, len(hidden_dims) + 1 + n_embed_keys)

        self.basin_embedding = eqx.nn.Embedding(n_basin_cats, basin_emb_dim, key=keys[0])
        emb_dim_total = basin_emb_dim

        if include_region:
            self.region_embedding = eqx.nn.Embedding(n_region_cats, region_emb_dim, key=keys[1])
            emb_dim_total += region_emb_dim
            layer_key_start = 2
        else:
            self.region_embedding = None
            layer_key_start = 1

        # MLP: input = n_cont + total embedding dim
        in_dim = n_cont + emb_dim_total
        dims = [in_dim] + list(hidden_dims) + [n_periods]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[layer_key_start + i])
            for i in range(len(dims) - 1)
        ]

    def __call__(
        self,
        x_cont: Array,
        basin_id: Array,
        region_id: Optional[Array] = None,
    ) -> Array:
        """
        x_cont    : (n_cont,)   float32
        basin_id  : ()           int32
        region_id : ()           int32, required iff self.include_region
        returns   : (n_periods,) float32, ln(PSA)
        """
        embs = [self.basin_embedding(basin_id)]
        if self.include_region:
            if region_id is None:
                raise ValueError("include_region=True but no region_id was provided")
            embs.append(self.region_embedding(region_id))

        x = jnp.concatenate([x_cont] + embs)

        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))          # SiLU works well for smooth spectra

        return self.layers[-1](x)               # linear output -> ln(PSA)


# Vectorise over a batch
def batched_forward(
    model: GMMNet,
    x_cont: Array,
    basin_ids: Array,
    region_ids: Optional[Array] = None,
) -> Array:
    if model.include_region:
        return jax.vmap(model)(x_cont, basin_ids, region_ids)
    return jax.vmap(model)(x_cont, basin_ids)


# ---------------------------------------------------------------------------
# 3. Loss & training step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def loss_fn(
    model: GMMNet,
    x_cont: Array,
    basin_ids: Array,
    y: Array,
    region_ids: Optional[Array] = None,
) -> Array:
    pred = batched_forward(model, x_cont, basin_ids, region_ids)
    return jnp.mean((pred - y) ** 2)            # MSE in ln space


@eqx.filter_jit
def train_step(
    model: GMMNet,
    opt_state,
    optimizer,
    x_cont: Array,
    basin_ids: Array,
    y: Array,
    region_ids: Optional[Array] = None,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_cont, basin_ids, y, region_ids)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ---------------------------------------------------------------------------
# 4. Data loader (simple numpy-based)
# ---------------------------------------------------------------------------

def iter_batches(
    x_cont: Array,
    basin_ids: Array,
    y: Array,
    batch_size: int,
    key: jax.Array,
    region_ids: Optional[Array] = None,
):
    """
    Yields (x_cont, basin_ids, y) batches, or (x_cont, basin_ids,
    region_ids, y) if `region_ids` is not None.
    """
    n = x_cont.shape[0]
    idx = jax.random.permutation(key, n)
    for start in range(0, n, batch_size):
        b = idx[start: start + batch_size]
        if region_ids is not None:
            yield x_cont[b], basin_ids[b], region_ids[b], y[b]
        else:
            yield x_cont[b], basin_ids[b], y[b]


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------

def train(
    df_x: pd.DataFrame,
    df_psa: pd.DataFrame,
    hidden_dims: Sequence[int] = (256, 256, 128, 64),
    basin_emb_dim: int = 4,
    include_region: bool = False,
    region_emb_dim: int = 4,
    n_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 0,
):
    """
    Train `GMMNet` on (df_x, df_psa).

    Returns
    -------
    model : GMMNet
    scaler : StandardScaler
        Fit on the continuous features of the training split only.
    config : dict
        Architecture config, sufficient (with `scaler`) to reconstruct
        and reload `model` later -- see `save_model`/`load_model`.
    """
    key = jax.random.PRNGKey(seed)

    # --- features & targets ---
    X_cont_np, X_basin_np, X_region_np = build_features(df_x, include_region=include_region)
    y_np = build_targets(df_psa)

    # --- train/test split ---
    idx = np.arange(len(y_np))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed)

    # --- standardise continuous features on training set only ---
    scaler = StandardScaler()
    X_cont_np[idx_tr] = scaler.fit_transform(X_cont_np[idx_tr])
    X_cont_np[idx_te] = scaler.transform(X_cont_np[idx_te])

    # --- to JAX arrays ---
    X_cont = jnp.array(X_cont_np)
    X_basin = jnp.array(X_basin_np)
    Y = jnp.array(y_np)
    X_region = jnp.array(X_region_np) if include_region else None

    X_cont_tr, X_basin_tr, Y_tr = X_cont[idx_tr], X_basin[idx_tr], Y[idx_tr]
    X_cont_te, X_basin_te, Y_te = X_cont[idx_te], X_basin[idx_te], Y[idx_te]
    X_region_tr = X_region[idx_tr] if include_region else None
    X_region_te = X_region[idx_te] if include_region else None

    n_cont = X_cont.shape[1]  # 8

    # --- model & optimiser ---
    key, subkey = jax.random.split(key)
    n_basin_cats = int(df_x["basin_id"].max()) + 1
    n_region_cats = int(df_x["subregion_id"].max()) + 1 if include_region else None
    model = GMMNet(
        hidden_dims=hidden_dims,
        n_periods=df_psa.shape[1],
        basin_emb_dim=basin_emb_dim,
        n_basin_cats=n_basin_cats,
        n_cont=n_cont,
        key=subkey,
        include_region=include_region,
        region_emb_dim=region_emb_dim,
        n_region_cats=n_region_cats,
    )

    # Cosine-decay schedule with warm restarts works well here
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=500,
        decay_steps=n_epochs * (len(idx_tr) // batch_size),
    )
    optimizer = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- loop ---
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        epoch_losses = []

        for batch in iter_batches(
            X_cont_tr, X_basin_tr, Y_tr, batch_size, subkey, region_ids=X_region_tr
        ):
            if include_region:
                xc, xb, xr, yb = batch
                model, opt_state, loss = train_step(
                    model, opt_state, optimizer, xc, xb, yb, xr
                )
            else:
                xc, xb, yb = batch
                model, opt_state, loss = train_step(model, opt_state, optimizer, xc, xb, yb)
            epoch_losses.append(float(loss))

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            val_loss = float(loss_fn(model, X_cont_te, X_basin_te, Y_te, X_region_te))
            print(f"Epoch {epoch:4d} | train MSE {np.mean(epoch_losses):.4f} | val MSE {val_loss:.4f}")

    config = dict(
        hidden_dims=list(hidden_dims),
        basin_emb_dim=basin_emb_dim,
        n_periods=df_psa.shape[1],
        n_basin_cats=n_basin_cats,
        n_cont=n_cont,
        include_region=include_region,
        region_emb_dim=region_emb_dim,
        n_region_cats=n_region_cats,
    )
    return model, scaler, config


# ---------------------------------------------------------------------------
# 6. Prediction helper
# ---------------------------------------------------------------------------

def predict(model: GMMNet, scaler: StandardScaler, df_x_new: pd.DataFrame) -> np.ndarray:
    """
    Returns ln(PSA) predictions, shape (N, 21). Reads `include_region`
    off `model` itself, so there's no way to accidentally mismatch it
    against how the model was trained.
    """
    X_cont_np, X_basin_np, X_region_np = build_features(
        df_x_new, include_region=model.include_region
    )
    X_cont_np = scaler.transform(X_cont_np)
    region_arr = jnp.array(X_region_np) if model.include_region else None
    pred = batched_forward(model, jnp.array(X_cont_np), jnp.array(X_basin_np), region_arr)
    return np.array(pred)


# ---------------------------------------------------------------------------
# 7. Save / load
# ---------------------------------------------------------------------------

def save_model(model: GMMNet, scaler: StandardScaler, config: dict, dir_results: str) -> None:
    os.makedirs(dir_results, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(dir_results, "gmm_nn.eqx"), model)
    joblib.dump(scaler, os.path.join(dir_results, "gmm_scaler.joblib"))
    with open(os.path.join(dir_results, "gmm_config.json"), "w") as f:
        json.dump(config, f)


def load_model(dir_results: str, key: Optional[jax.Array] = None):
    """
    Returns (model, scaler, config). `key` only seeds the template
    model's initial (soon-to-be-overwritten) weights before
    `eqx.tree_deserialise_leaves` fills in the trained values -- any
    key works.
    """
    with open(os.path.join(dir_results, "gmm_config.json")) as f:
        config = json.load(f)

    key = key if key is not None else jax.random.PRNGKey(0)
    model_template = GMMNet(
        hidden_dims=config["hidden_dims"],
        n_periods=config["n_periods"],
        basin_emb_dim=config["basin_emb_dim"],
        n_basin_cats=config["n_basin_cats"],
        n_cont=config["n_cont"],
        key=key,
        include_region=config.get("include_region", False),
        region_emb_dim=config.get("region_emb_dim", 4),
        n_region_cats=config.get("n_region_cats"),
    )
    model = eqx.tree_deserialise_leaves(os.path.join(dir_results, "gmm_nn.eqx"), model_template)
    scaler = joblib.load(os.path.join(dir_results, "gmm_scaler.joblib"))
    return model, scaler, config
