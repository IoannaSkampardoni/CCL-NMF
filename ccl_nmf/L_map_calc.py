import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser(description="Create L-maps with fixed LME (no PTID/SITE/Study).")
    ap.add_argument("--residuals_csv", required=True, help="Residuals CSV; used only to intersect participant_id.")
    ap.add_argument("--rest_csv", required=True, help="Longitudinal CSV with ROI volumes and metadata.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--min_scans", type=int, default=3, help="Minimum scans per subject (default: 3).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==== Load ====
    resid = pd.read_csv(args.residuals_csv, low_memory=False)
    rest  = pd.read_csv(args.rest_csv, low_memory=False)

    id_col, age_col = "participant_id", "Age"

    # Type safety
    if id_col in rest.columns:
        rest[id_col] = rest[id_col].astype(str)
    if id_col in resid.columns:
        resid[id_col] = resid[id_col].astype(str)

    # ROI detection: prefer "ROI_" then fallback to "H_MUSE_Volume_"
    roi_cols = [c for c in rest.columns if c.startswith("ROI_")]

    # Numerics
    rest[age_col] = pd.to_numeric(rest[age_col], errors="coerce")
    rest[roi_cols] = rest[roi_cols].apply(pd.to_numeric, errors="coerce")

    # ==== Intersect + min scans ====
    keep_ids = set(resid[id_col].dropna().unique()) if id_col in resid.columns else set(rest[id_col].dropna().unique())
    rest = rest[rest[id_col].isin(keep_ids)].copy()

    # Order by subject
    rest = rest.sort_values([id_col, age_col], na_position="last").reset_index(drop=True)
    rest["scan_idx"] = rest.groupby(id_col).cumcount() + 1

    counts = rest.groupby(id_col)["scan_idx"].max().reset_index(name="n_scans")
    keep_ids2 = set(counts.loc[counts["n_scans"] >= args.min_scans, id_col])
    rest = rest[rest[id_col].isin(keep_ids2)].copy()
    rest = rest.dropna(subset=[age_col])

    # ==== Baseline/Max + Time ====
    rest["Baseline_age"] = rest.groupby(id_col)[age_col].transform("min")
    rest["Max_age"]      = rest.groupby(id_col)[age_col].transform("max")
    rest["Time"]         = rest["Baseline_age"] - rest["Max_age"]  # negative duration

    df_interest = rest[(rest[age_col] == rest["Baseline_age"]) | (rest[age_col] == rest["Max_age"])].copy()
    idx_b = df_interest.groupby(id_col)[age_col].idxmin()
    idx_m = df_interest.groupby(id_col)[age_col].idxmax()
    df_b = df_interest.loc[idx_b].copy()
    df_m = df_interest.loc[idx_m].copy()

    # Align 1:1
    df_m = df_m.set_index(id_col).reindex(df_b[id_col]).reset_index()
    assert (df_b[id_col].values == df_m[id_col].values).all(), "Participant alignment failed."

    # ==== Delta table (vectorized) ====
    meta = df_b[[id_col, "Baseline_age", "Max_age", "Time"]].reset_index(drop=True)
    D = df_b[roi_cols].to_numpy() - df_m[roi_cols].to_numpy()
    d_cols = ["d_" + c for c in roi_cols]
    d_df = pd.DataFrame(D, columns=d_cols)

    # Save provenance deltas
    delta_df = pd.concat([meta, d_df], axis=1, copy=False)
    delta_df.to_csv(out_dir / "L_delta.csv", index=False)

    # ==== Build modeling frame ====
    base_vols = df_b[roi_cols].reset_index(drop=True)
    model_df = pd.concat([meta, base_vols, d_df], axis=1, copy=False)

    participants = sorted(model_df[id_col].astype(str).unique())

    # ==== MixedLM per ROI ====
    fe_rows = []
    re_time = {pid: {} for pid in participants}

    for d_col, base_col in zip(d_cols, roi_cols):
        data = model_df[[id_col, "Baseline_age", "Time", base_col, d_col]].rename(
            columns={d_col: "Intensity", base_col: "BaselineVol"}
        ).dropna(subset=["Intensity", "Baseline_age", "BaselineVol", "Time"])

        if data.empty:
            fe_rows.append({"ROI": d_col, "Intercept": np.nan, "Baseline_age": np.nan,
                            "BaselineVol": np.nan, "Time": np.nan})
            for pid in participants:
                re_time[pid][d_col] = np.nan
            continue

        try:
            md = smf.mixedlm(
                formula="Intensity ~ Baseline_age + BaselineVol + Time",
                data=data,
                groups=data[id_col],
                re_formula="~Time"
            )
            mdf = md.fit(method="bfgs", disp=False)

            fe_row = {"ROI": d_col}
            for k in ["Intercept", "Baseline_age", "BaselineVol", "Time"]:
                val = mdf.params.get(k, np.nan)
                if np.isnan(val) and "Intercept" in mdf.params.index:
                    val = mdf.params.get("Intercept", np.nan)
                fe_row[k] = float(val) if val is not None else np.nan
            fe_rows.append(fe_row)

            fe_time = float(mdf.params.get("Time", np.nan))
            for pid in participants:
                slope = np.nan
                if pid in mdf.random_effects:
                    vec = mdf.random_effects[pid]
                    if hasattr(vec, "get"):
                        slope = float(vec.get("Time", np.nan))
                    else:
                        slope = float(vec[1]) if len(vec) > 1 else np.nan
                re_time[pid][d_col] = slope + fe_time

        except Exception:
            fe_rows.append({"ROI": d_col, "Intercept": np.nan, "Baseline_age": np.nan,
                            "BaselineVol": np.nan, "Time": np.nan})
            for pid in participants:
                re_time[pid][d_col] = np.nan

    # ==== Save outputs ====
    fe_df = pd.DataFrame(fe_rows)
    fe_df.to_csv(out_dir / "Fixed_effect.csv", index=False)

    lmaps_df = pd.DataFrame.from_dict(re_time, orient="index").sort_index()
    lmaps_df.index.name = id_col
    lmaps_df.reset_index(inplace=True)
    lmaps_df.to_csv(out_dir / "L_map.csv", index=False)

    # Also RE-only
    if not fe_df.empty and "Time" in fe_df.columns:
        re_only = lmaps_df.copy()
        fe_time_series = fe_df.set_index("ROI")["Time"]
        for d_col in [c for c in re_only.columns if c.startswith("d_")]:
            if d_col in fe_time_series.index:
                re_only[d_col] = re_only[d_col] - float(fe_time_series.loc[d_col])
        re_only.to_csv(out_dir / "Random_effect.csv", index=False)
    else:
        re_only = lmaps_df.copy()
        re_only.to_csv(out_dir / "Random_effect.csv", index=False)

    print("Wrote:")
    print(f"- {out_dir/'L_delta.csv'}")
    print(f"- {out_dir/'Fixed_effect.csv'}")
    print(f"- {out_dir/'Random_effect.csv'}")
    print(f"- {out_dir/'L_map.csv'}  (final)")


if __name__ == "__main__":
    main()

