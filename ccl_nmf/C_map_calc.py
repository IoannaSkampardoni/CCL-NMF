import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def process(df_in, df_rec, rois):
    # Align by participant_id
    df_in = df_in.set_index("participant_id")
    df_rec = df_rec.set_index("participant_id")
    df_in, df_rec = df_in.loc[df_rec.index], df_rec

    # Residuals
    residuals = df_rec[rois] - df_in[rois]

    # Hypertrophy → NaN
    residuals_posnan = residuals.mask(residuals < 0, np.nan)

    # Total error = nanmean of squared residuals
    totals = np.nanmean(residuals_posnan ** 2, axis=1)

    return residuals_posnan.reset_index(), pd.DataFrame({"participant_id": residuals_posnan.index, "ReconstructionError": totals})


def main():
    parser = argparse.ArgumentParser(description="Select test subjects with high reconstruction error")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory containing 'heldout/' and 'test/' subfolders with normalized.csv and reconstruction.csv")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Directory to save results (default: output_dir)")
    args = parser.parse_args()

    outdir = args.save_dir if args.save_dir else args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)


    # Paths
    heldout_norm = args.output_dir / "heldout/normalized.csv"
    heldout_rec  = args.output_dir / "heldout/reconstruction.csv"
    test_norm    = args.output_dir / "test/normalized.csv"
    test_rec     = args.output_dir / "test/reconstruction.csv"

    # Load data
    df_held_norm = pd.read_csv(heldout_norm)
    df_held_rec  = pd.read_csv(heldout_rec)
    df_test_norm = pd.read_csv(test_norm)
    df_test_rec  = pd.read_csv(test_rec)

    # ROI columns
    roi_cols = [c for c in df_held_norm.columns if c != "participant_id"]

    # Process heldout
    held_residuals, held_totals = process(df_held_norm, df_held_rec, roi_cols)
    held_residuals.to_csv(outdir / "heldout_residuals.csv", index=False)
    held_totals.to_csv(outdir / "heldout_totals.csv", index=False)

    # Process test
    test_residuals, test_totals = process(df_test_norm, df_test_rec, roi_cols)
    test_residuals.to_csv(outdir / "test_residuals.csv", index=False)
    test_totals.to_csv(outdir / "test_totals.csv", index=False)
	
    # Threshold = 75th percentile of heldout
    threshold = held_totals["ReconstructionError"].quantile(0.75)

    # Select test subjects above threshold
    PT_posnan = test_totals.loc[test_totals["ReconstructionError"] >= threshold].reset_index(drop=True)
    # Merge ROI-wise residuals (with hypertrophy masked → NaN)
    PT_posnan = PT_posnan.merge(test_residuals, on="participant_id")
    PT_posnan.to_csv(outdir / "C_map.csv", index=False)

    print(f"[Done] Heldout N={len(held_totals)}, Test N={len(test_totals)}")
    print(f"75th percentile cutoff = {threshold:.4f}")
    print(f"Selected {len(PT_posnan)} test subjects with error ≥ cutoff")


if __name__ == "__main__":
    main()

