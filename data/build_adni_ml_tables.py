import argparse
import os

import pandas as pd


def build_adni_tables(dx_path: str,
                      fs_path: str,
                      out_features: str = "ADNI_FEATURES_READY.csv",
                      out_labels: str = "ADNI_LABELS_READY.csv") -> None:
    """
    Build ML-ready feature and label tables from:
      - DXSUM_*.csv   (diagnostic summary)
      - UCSFFSX7_*.csv (FreeSurfer 7.x cross-sectional)
    """

    print(f"[INFO] Loading diagnostic file: {dx_path}")
    dx = pd.read_csv(dx_path, low_memory=False)

    print(f"[INFO] Loading FreeSurfer file: {fs_path}")
    fs = pd.read_csv(fs_path, low_memory=False)

    # --- Filter diagnostic table: ADNI1 + ADNIGO, baseline only, DX in {1,2,3} ---

    # Ensure consistent string types
    dx2 = dx.copy()
    dx2["PHASE"] = dx2["PHASE"].astype(str)
    dx2["VISCODE"] = dx2["VISCODE"].astype(str)

    # Keep only ADNI1 + GO and baseline visit (VISCODE == "bl")
    mask_phase = dx2["PHASE"].isin(["ADNI1", "ADNIGO"])
    mask_vis = dx2["VISCODE"].str.lower().eq("bl")

    # DIAGNOSIS codes: 1=CN, 2=MCI, 3=AD
    mask_diag = dx2["DIAGNOSIS"].isin([1.0, 2.0, 3.0])

    dx_filt = dx2[mask_phase & mask_vis & mask_diag].copy()
    print(f"[INFO] After phase+baseline+diagnosis filter: {dx_filt.shape[0]} rows")

    # Parse exam date, pick earliest per RID (should already be unique, but safe)
    dx_filt["EXAMDATE"] = pd.to_datetime(dx_filt["EXAMDATE"], errors="coerce")
    dx_filt = (
        dx_filt.sort_values(["RID", "EXAMDATE"])
        .groupby("RID", as_index=False)
        .first()
    )

    # Map diagnosis to CN/MCI/AD
    label_map_diag = {1.0: "CN", 2.0: "MCI", 3.0: "AD"}
    dx_filt["label"] = dx_filt["DIAGNOSIS"].map(label_map_diag)

    print("[INFO] Label counts (baseline ADNI1+GO):")
    print(dx_filt["label"].value_counts())

    # --- Prepare FreeSurfer table: earliest imaging per RID in ADNI1+GO ---

    fs2 = fs.copy()
    fs2["PHASE"] = fs2["PHASE"].astype(str)
    fs2["VISCODE"] = fs2["VISCODE"].astype(str)
    fs2["EXAMDATE"] = pd.to_datetime(fs2["EXAMDATE"], errors="coerce")

    # Only keep rows for the RIDs present in dx_filt, and phases ADNI1/GO
    fs2 = fs2[fs2["RID"].isin(dx_filt["RID"]) & fs2["PHASE"].isin(["ADNI1", "ADNIGO"])].copy()

    # Prefer VISCODE="bl" if there are multiple, then earliest date
    fs2["vis_rank"] = fs2["VISCODE"].str.lower().ne("bl").astype(int)
    fs2 = fs2.sort_values(["RID", "vis_rank", "EXAMDATE"])

    fs_sel = fs2.groupby("RID", as_index=False).first()
    print(f"[INFO] Selected one FreeSurfer row per RID: {fs_sel.shape[0]} rows")

    # --- Merge diagnostics with FreeSurfer on RID+PTID+PHASE ---

    merged = dx_filt.merge(
        fs_sel,
        on=["RID", "PTID", "PHASE"],
        how="inner",
        suffixes=("_dx", "_fs"),
    )

    print(f"[INFO] Merged table shape: {merged.shape}")
    print(merged[["PHASE", "PTID", "RID", "VISCODE_dx", "VISCODE_fs", "label"]].head())

    # --- Build feature + label DataFrames ---

    # Metadata columns that should NOT be used as ML features
    meta_drop = [
        "PHASE", "PTID", "RID",
        "VISCODE", "VISCODE2", "EXAMDATE",
        "RUNDATE", "FSVER", "OVERALLQC",
        "TEMPQC", "FRONTQC", "PARQC", "OCCQC",
        "CINGQC", "SUBCORTQC",
        "IMAGEUID", "FIELD_STRENGTH",
        "STATUS", "update_stamp",
        "vis_rank", "ID", "USERDATE", "USERDATE2",
    ]

    # Figure out which FreeSurfer columns are actual features
    fs_cols = [c for c in fs_sel.columns if c not in meta_drop]

    print(f"[INFO] Number of FreeSurfer feature columns: {len(fs_cols)}")
    # Just to sanity check a few example columns:
    print("[INFO] Example FS columns:", fs_cols[:10])

    # Feature matrix: subject_id + all FS features
    features_df = merged[["PTID"] + fs_cols].copy()
    features_df = features_df.rename(columns={"PTID": "subject_id"})

    # Labels: subject_id + label
    labels_df = merged[["PTID", "label"]].copy().rename(columns={"PTID": "subject_id"})

    print(f"[INFO] Final features shape: {features_df.shape}")
    print(f"[INFO] Final labels shape:   {labels_df.shape}")

    # --- Save to disk ---

    features_df.to_csv(out_features, index=False)
    labels_df.to_csv(out_labels, index=False)

    print(f"[DONE] Saved features to: {out_features}")
    print(f"[DONE] Saved labels to:   {out_labels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dx_csv",
        default="DXSUM_18Nov2025.csv",
        help="Path to ADNI diagnostic summary CSV (DXSUM_*.csv)",
    )
    ap.add_argument(
        "--fs_csv",
        default="UCSFFSX7_18Nov2025.csv",
        help="Path to UCSF FreeSurfer 7.x CSV (UCSFFSX7_*.csv)",
    )
    ap.add_argument(
        "--out_features",
        default="ADNI_FEATURES_READY.csv",
        help="Output path for features CSV",
    )
    ap.add_argument(
        "--out_labels",
        default="ADNI_LABELS_READY.csv",
        help="Output path for labels CSV",
    )
    args = ap.parse_args()

    build_adni_tables(
        dx_path=args.dx_csv,
        fs_path=args.fs_csv,
        out_features=args.out_features,
        out_labels=args.out_labels,
    )


if __name__ == "__main__":
    main()
