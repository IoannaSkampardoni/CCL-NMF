import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize
from jointNMF import clNMF  # clNMF signature: (a, X1, X2, k, delta, num_iter, model_path, ...)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cross', required=True, help='Cross-sectional residuals CSV')
    p.add_argument('--betas', required=True, help='Beta maps CSV')
    p.add_argument('--output-dir', required=True, help='Output directory')
    p.add_argument('--num-components', type=int, default=10)
    p.add_argument('--num-iter', type=int, default=50000)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--delta', type=float, default=1e-16)
    p.add_argument('--seed', type=int, default=0, help='Random seed for init (optional)')
    p.add_argument('--a-cost', type=float, default=None,
                   help='Weight on X1 term. If omitted, uses len(betas)/len(cross).')
    return p.parse_args()


def get_gm_rois(columns):
    """Return columns that match ROI_*"""
    pat = re.compile(r'^ROI_\d+$')
    return [c for c in columns if pat.match(c)]


def l2_normalize_rows(df):
    return pd.DataFrame(
        normalize(df, axis=1, norm='l2'),
        columns=df.columns,
        index=df.index)


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    # ----------------------------- C-map -----------------------------
    df_cross0 = pd.read_csv(args.cross)
    GM_ROIS = get_gm_rois(df_cross0.columns)
    if not GM_ROIS:
        raise ValueError("No ROI columns found in --cross using pattern ^ROI_\\d+$")
    print(f"[INFO] Using {len(GM_ROIS)} GM ROIs")

    scaler_cross = MinMaxScaler()
    d0 = scaler_cross.fit_transform(df_cross0[GM_ROIS])
    df_cross_normalized = pd.DataFrame(d0, columns=GM_ROIS)
    Cross_normalized = pd.concat([df_cross0.drop(columns=GM_ROIS), df_cross_normalized], axis=1).fillna(0)

    # ------------------------------- L-map ---------------------------------
    df_betas0 = pd.read_csv(args.betas)
    df_betas0.columns = df_betas0.columns.str[2:] # preserve original behavior: strip first two chars from all column names
    df_betas0.rename(columns={'rticipant_id': 'participant_id'}, inplace=True) # fix potential typo column for participant id

    df_betas = df_betas0.copy()
    df_betas[GM_ROIS] = df_betas[GM_ROIS].where(df_betas[GM_ROIS] > 0, np.nan) # keep only non-positive betas (positive -> NaN), then flip sign
    df_betas[GM_ROIS] = -df_betas[GM_ROIS]

    scaler_beta = MinMaxScaler()
    d0_betas = scaler_beta.fit_transform(df_betas[GM_ROIS])
    df_betas_normalized = pd.DataFrame(d0_betas, columns=GM_ROIS)
    Betas_normalized = pd.concat([df_betas.drop(columns=GM_ROIS), df_betas_normalized], axis=1).fillna(0)

    # Matrices (shape: m x n -> transpose to ROI x samples)
    Cross = Cross_normalized[GM_ROIS].T.to_numpy()
    Long  = Betas_normalized[GM_ROIS].T.to_numpy()

    # ------------------------------- jointNMF -----------------------------------
    output_dir = args.output_dir
    num_component = args.num_components
    component_path = os.path.join(output_dir, f"{num_component}_components", "jointNMF")
    os.makedirs(component_path, exist_ok=True)

    # a_cost: external override or default ratio
    if args.a_cost is None:
        a_cost = len(df_betas0) / len(df_cross0)
        print(f"[INFO] a_cost not provided; using default len(betas)/len(cross) = {a_cost:.6f}")
    else:
        a_cost = float(args.a_cost)
        print(f"[INFO] Using externally provided a_cost = {a_cost:.6f}")

    # clNMF signature: (a, X1, X2, k, delta, num_iter, model_path, ...)
    W, H1, H2, loss_list = clNMF(
        a_cost,
        Cross, Long, num_component,
        args.delta, args.num_iter,
        component_path,
        early_stopping_epoch=args.patience,
        init_W=None, init_H1=None, init_H2=None,
        print_enabled=True)


    print("\n[INFO] Joint NMF finished successfully.")
    print(f"[INFO] Results saved in: {component_path}")
    print("[INFO] Files generated include:")
    print("  • W.npy              : ROI x Components (shared dictionary)")
    print("  • H1.npy             : Components x Subjects (C-map loadings)")
    print("  • H2.npy             : Components x Subjects (L-map loadings)")
    print("  • loss.npy           : Final reconstruction loss")
    print("  • loss_list.npy      : Loss values across iterations")
    print("  • sparsity_list.npy  : Sparsity values across iterations\n")
    
    # -------------------------- Save reconstruction stats -----------------------
    rec_error = float(loss_list[-1]) if len(loss_list) else np.nan
    n = W.size
    sparsity = (np.sqrt(n) - (np.sum(np.abs(W)) / np.sqrt(np.sum(W**2)))) / (np.sqrt(n) - 1)
    nmf_cols = [f"NMF_{i}" for i in range(1, num_component + 1)]

    pd.DataFrame({
        "num_component": [num_component],
        "a_cost": [a_cost],
        "reconstruction_error": [rec_error],
        "dictionary sparsity": [sparsity]}).to_csv(
        os.path.join(component_path, "reconstruction_error_and_sparsity.tsv"),
        index=False, sep="\t")


if __name__ == "__main__":
    main()
