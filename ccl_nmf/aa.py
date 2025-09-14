#!/usr/bin/env python3
"""
Adversarial Autoencoder (AAE) — NO-SEX pipeline (Minimal deterministic setup)

Determinism kept (minimal & CPU-friendly):
- Seeds Python, NumPy, and TensorFlow with the same --seed.
- Uses seeded StratifiedShuffleSplit and seeded tf.data shuffle.
- Sets tf.data to deterministic ordering.

No environment variables, no GPU forcing, no thread pinning.

Other features:
- Stratified splits by equal-width age bins over the observed range (choose 4 or 5 bins).
- Cyclical Learning Rate (CLR): base_lr, max_lr, step_size, gamma.
- Decoder takes only z (no sex conditioning); inputs assumed sex-corrected.

Artifacts:
  outdir/
    best_encoder.h5, best_decoder.h5, best_discriminator.h5
    encoder.h5, decoder.h5, discriminator.h5
    scaler.joblib
    training_history.csv
    train_covariates.csv / val_covariates.csv / heldout_covariates.csv
    train_features.csv / val_features.csv / heldout_features.csv

Inference outputs (under models_dir/<name>/):
  normalized.csv, reconstruction.csv, encoded.csv, reconstruction_error.csv
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Project-local imports (must exist in your repo)
from models import (
    make_encoder_model_v1,
    make_decoder_model_v1,
    make_discriminator_model_v1,
)

# ------------------------- Minimal determinism helpers -------------------------
def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ------------------------- Convenience -------------------------
def set_tf_verbosity(quiet: bool = True) -> None:
    if not quiet:
        return
    import logging, warnings  # noqa
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    tf.get_logger().setLevel(logging.ERROR)


def feature_cols_from_csv(path: Path) -> list[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return sorted(c for c in cols if c != "participant_id")
    

def load_and_merge(features_csv: Path, covariates_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    covars = pd.read_csv(covariates_csv)
    df = covars.merge(features, on="participant_id").reset_index(drop=True)
    return df


def stratify_key(df: pd.DataFrame, n_bins: int) -> pd.Series:
    """
    Equal-width age bins over the observed [min(Age), max(Age)].
    Assumes 'Age' has no NaNs (drop earlier if needed).
    """
    ages = df["Age"].to_numpy()
    min_age = float(np.nanmin(ages))
    max_age = float(np.nanmax(ages))
    if not np.isfinite(min_age) or not np.isfinite(max_age):
        raise ValueError("Age contains no finite values for stratification.")
    if max_age <= min_age:
        return pd.Series(["bin0"] * len(df), index=df.index)

    # Equal-width bins
    edges = np.linspace(min_age, max_age, num=n_bins + 1)
    labels = [f"bin{i}" for i in range(n_bins)]
    out = pd.cut(df["Age"], bins=edges, labels=labels,
                 include_lowest=True, right=False, duplicates="drop")
    # If duplicates dropped (rare), fallback: label NaNs as last bin
    out = out.astype("object")
    mask_nan = pd.isna(out.values)
    if mask_nan.any():
        out[mask_nan] = labels[-1]
    return pd.Series(out)


# ------------------------- Config -------------------------
@dataclass
class TrainConfig:
    project_root: Path
    features_csv: Path
    covariates_csv: Path
    outdir: Path
    # Splits
    valheldout_size: float = 0.35
    heldout_frac_within_val: float = 0.40
    age_bins: int = 5  # choose 4 or 5
    # Training
    batch_size: int = 200
    epochs: int = 1000
    patience: int = 50
    z_dim: int = 20
    h_dim: Tuple[int, int] = (110, 110)
    seed: int = 0
    # CLR
    base_lr: float = 1e-4
    max_lr: float = 5e-3
    gamma: float = 0.98
    step_size: int | None = None


# ------------------------- Train -------------------------
def train(cfg: TrainConfig) -> Path:
    set_global_seeds(cfg.seed)
    set_tf_verbosity(quiet=True)

    # Load & prepare
    df = load_and_merge(cfg.features_csv, cfg.covariates_csv)
    feature_cols = feature_cols_from_csv(cfg.features_csv)

    # Drop rows with missing Age before stratification
    if df["Age"].isna().any():
        dropped = int(df["Age"].isna().sum())
        print(f"[INFO] Dropping {dropped} rows with missing Age before stratified split.")
        df = df[df["Age"].notna()].reset_index(drop=True)

    df_split = pd.DataFrame({"participant_id": df["participant_id"], "Age": df["Age"]})
    strat = stratify_key(df_split, cfg.age_bins)

    # Split TRAIN vs (VAL+HELDOUT)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.valheldout_size, random_state=cfg.seed)
    train_idx, valheld_idx = next(sss.split(df_split["participant_id"], strat))
    df_train = df.iloc[train_idx].copy()
    df_valheld = df.iloc[valheld_idx].copy()

    # Split (VAL+HELDOUT) into VAL and HELDOUT
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.heldout_frac_within_val, random_state=cfg.seed)
    strat2 = stratify_key(df_valheld[["Age"]].assign(participant_id=df_valheld["participant_id"]), cfg.age_bins)
    val_idx, held_idx = next(sss2.split(df_valheld["participant_id"], strat2))
    # Smaller subset becomes validation, remainder heldout (consistent naming)
    df_val = df_valheld.iloc[held_idx].copy()
    df_held = df_valheld.iloc[val_idx].copy()

    # Save split manifests
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    df_train[["participant_id"]].to_csv(cfg.outdir / "train_participants.csv", index=False)
    df_val[["participant_id"]].to_csv(cfg.outdir / "val_participants.csv", index=False)
    df_held[["participant_id"]].to_csv(cfg.outdir / "heldout_participants.csv", index=False)
    
    df_train[["participant_id"] + feature_cols].to_csv(cfg.outdir / "train_features.csv", index=False)
    df_val  [["participant_id"] + feature_cols].to_csv(cfg.outdir / "val_features.csv", index=False)
    df_held [["participant_id"] + feature_cols].to_csv(cfg.outdir / "heldout_features.csv", index=False)


    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values).astype("float32")
    X_val   = scaler.transform(df_val[feature_cols].values).astype("float32")

    # Build models
    n_features = X_train.shape[1]
    encoder = make_encoder_model_v1(n_features, list(cfg.h_dim), cfg.z_dim)
    decoder = make_decoder_model_v1(cfg.z_dim, n_features, list(cfg.h_dim)[::-1])
    discriminator = make_discriminator_model_v1(cfg.z_dim, list(cfg.h_dim)[::-1])

    # Losses
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    def d_loss(real_out, fake_out):
        return bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)

    def g_loss(fake_out):
        return bce(tf.ones_like(fake_out), fake_out)

    # Optimizers (CLR-updated per step)
    ae_opt = keras.optimizers.Adam(learning_rate=cfg.base_lr)
    d_opt = keras.optimizers.Adam(learning_rate=cfg.base_lr)
    g_opt = keras.optimizers.Adam(learning_rate=cfg.base_lr)

    # tf.data pipeline (deterministic ordering and seeded shuffle)
    ds = tf.data.Dataset.from_tensor_slices(X_train)
    ds = ds.shuffle(buffer_size=len(df_train), seed=cfg.seed, reshuffle_each_iteration=True).batch(cfg.batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    # ---- Cyclical Learning Rate (CLR) setup ----
    n_samples = X_train.shape[0]
    step_size = cfg.step_size if cfg.step_size is not None else int(2 * np.ceil(n_samples / cfg.batch_size))
    def scale_fn(cycle):
        return cfg.gamma ** cycle
    global_step = 0

    @tf.function
    def train_step(xb, ae_lr, d_lr, g_lr):
        ae_opt.learning_rate = ae_lr
        d_opt.learning_rate = d_lr
        g_opt.learning_rate = g_lr
        # Autoencoder
        with tf.GradientTape() as t_ae:
            z = encoder(xb, training=True)
            xhat = decoder(z, training=True)
            ae_loss = mse(xb, xhat)
        ae_grads = t_ae.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
        ae_opt.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

        # Discriminator
        with tf.GradientTape() as t_d:
            z_real = tf.random.normal([tf.shape(xb)[0], cfg.z_dim])
            z_fake = encoder(xb, training=True)
            d_real = discriminator(z_real, training=True)
            d_fake = discriminator(z_fake, training=True)
            loss_d = d_loss(d_real, d_fake)
        d_grads = t_d.gradient(loss_d, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        # Generator/Encoder
        with tf.GradientTape() as t_g:
            z_fake = encoder(xb, training=True)
            d_fake = discriminator(z_fake, training=True)
            loss_g = g_loss(d_fake)
        g_grads = t_g.gradient(loss_g, encoder.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, encoder.trainable_variables))

        return ae_loss, loss_d, loss_g

    # Training loop with EarlyStopping on val AE loss
    best_val = np.inf
    best_epoch = -1
    history = {"ae_loss": [], "d_loss": [], "g_loss": [], "val_ae_loss": []}

    for epoch in range(cfg.epochs):
        ae_epoch, d_epoch, g_epoch = [], [], []
        for xb in ds:
            global_step += 1
            cycle = np.floor(1 + global_step / (2 * step_size))
            x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
            clr = cfg.base_lr + (cfg.max_lr - cfg.base_lr) * max(0.0, 1.0 - x_lr) * scale_fn(cycle)
            ae_l, d_l, g_l = train_step(xb, clr, clr, clr)
            ae_epoch.append(float(ae_l.numpy()))
            d_epoch.append(float(d_l.numpy()))
            g_epoch.append(float(g_l.numpy()))

        # Validation AE loss
        z_val = encoder(X_val, training=False)
        xhat_val = decoder(z_val, training=False)
        val_loss = float(mse(X_val, xhat_val).numpy())

        history["ae_loss"].append(np.mean(ae_epoch))
        history["d_loss"].append(np.mean(d_epoch))
        history["g_loss"].append(np.mean(g_epoch))
        history["val_ae_loss"].append(val_loss)

        print(f"Epoch {epoch:04d} | AE {np.mean(ae_epoch):.4f} | D {np.mean(d_epoch):.4f} "
              f"| G {np.mean(g_epoch):.4f} | VAL {val_loss:.4f} | BEST {best_val:.4f} @ {best_epoch}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            encoder.save(cfg.outdir / "best_encoder.h5")
            decoder.save(cfg.outdir / "best_decoder.h5")
            discriminator.save(cfg.outdir / "best_discriminator.h5")

        if epoch - best_epoch >= cfg.patience:
            break

    # Save final artifacts
    encoder.save(cfg.outdir / "encoder.h5")
    decoder.save(cfg.outdir / "decoder.h5")
    discriminator.save(cfg.outdir / "discriminator.h5")

    joblib.dump(scaler, cfg.outdir / "scaler.joblib")
    pd.DataFrame(history).to_csv(cfg.outdir / "training_history.csv", index=False)

    print(f"Best VAL AE loss: {best_val:.6f} @ epoch {best_epoch}")
    return cfg.outdir


# ------------------------- Inference -------------------------
def infer(models_dir: Path, features_csv: Path, covariates_csv: Path, dataset_name: str = "test", seed: int = 0) -> Path:
    set_global_seeds(seed)
    set_tf_verbosity(quiet=True)

    outdir = Path(models_dir) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(features_csv, covariates_csv)
    feature_cols = feature_cols_from_csv(features_csv)
    
    # Optional safety check:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
    	raise ValueError(f"Missing expected feature columns: {missing}")
    X = df[feature_cols].values.astype("float32")

    encoder = keras.models.load_model(Path(models_dir) / "best_encoder.h5", compile=False)
    decoder = keras.models.load_model(Path(models_dir) / "best_decoder.h5", compile=False)
    scaler = joblib.load(Path(models_dir) / "scaler.joblib")

    Xn = scaler.transform(X).astype("float32")

    Z = encoder(Xn, training=False)
    Xhat = decoder(Z, training=False)

    # normalized inputs
    norm_df = pd.DataFrame({"participant_id": df["participant_id"]})
    for i, c in enumerate(feature_cols): 
        norm_df[c] = Xn[:, i]
    norm_df.to_csv(outdir / "normalized.csv", index=False)

    # reconstructions
    recon_df = pd.DataFrame({"participant_id": df["participant_id"]})
    xhat_np = Xhat.numpy()
    for i, c in enumerate(feature_cols):
        recon_df[c] = xhat_np[:, i]
    recon_df.to_csv(outdir / "reconstruction.csv", index=False)

    # latent codes
    enc_df = pd.DataFrame({"participant_id": df["participant_id"]})
    z_np = Z.numpy()
    for i in range(z_np.shape[1]):
        enc_df[f"z{i:02d}"] = z_np[:, i]
    enc_df.to_csv(outdir / "encoded.csv", index=False)

    # reconstruction error (per-subject MSE)
    rec_err = np.mean((Xn - xhat_np) ** 2, axis=1)
    pd.DataFrame({
        "participant_id": df["participant_id"],
        "Reconstruction error": rec_err}).to_csv(outdir / "reconstruction_error.csv", index=False)

    print(f"Wrote outputs to {outdir}")
    return outdir


# ------------------------- CLI -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AAE pipeline (train + infer) — NO-SEX condition (minimal deterministic)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train models")
    pt.add_argument("--features_csv", required=True, type=Path)
    pt.add_argument("--covariates_csv", required=True, type=Path)
    pt.add_argument("--outdir", required=True, type=Path)
    pt.add_argument("--z_dim", type=int, default=20)
    pt.add_argument("--h_dim", type=int, nargs=2, default=[110, 110])
    pt.add_argument("--epochs", type=int, default=1000)
    pt.add_argument("--patience", type=int, default=50)
    pt.add_argument("--batch_size", type=int, default=200)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--valheldout_size", type=float, default=0.35)
    pt.add_argument("--heldout_frac_within_val", type=float, default=0.40)
    # CLR
    pt.add_argument("--base_lr", type=float, default=1e-4)
    pt.add_argument("--max_lr", type=float, default=5e-3)
    pt.add_argument("--gamma", type=float, default=0.98)
    pt.add_argument("--step_size", type=int, default=0, help="0=auto (2 * ceil(N/batch))")
    # Age-bin choice
    pt.add_argument("--age_bins", type=int, choices=[4, 5], default=5)
    pi = sub.add_parser("infer", help="Run inference with trained models")
    pi.add_argument("--models_dir", required=True, type=Path)
    pi.add_argument("--features_csv", required=True, type=Path)
    pi.add_argument("--covariates_csv", required=True, type=Path)
    pi.add_argument("--name", default="test", help="Output subfolder name (e.g., val/heldout/test)")
    pi.add_argument("--seed", type=int, default=0, help="Seed for deterministic inference")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            project_root=Path.cwd(),
            features_csv=args.features_csv,
            covariates_csv=args.covariates_csv,
            outdir=args.outdir,
            z_dim=args.z_dim,
            h_dim=(args.h_dim[0], args.h_dim[1]),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
            valheldout_size=args.valheldout_size,
            heldout_frac_within_val=args.heldout_frac_within_val,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            gamma=args.gamma,
            step_size=(None if args.step_size == 0 else args.step_size),
            age_bins=args.age_bins)
        train(cfg)

    elif args.cmd == "infer":
        infer(
            models_dir=args.models_dir,
            features_csv=args.features_csv,
            covariates_csv=args.covariates_csv,
            dataset_name=args.name,
            seed=args.seed)


if __name__ == "__main__":
    main()

