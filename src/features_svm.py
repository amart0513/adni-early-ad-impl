import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils import ensure_dir, save_json, plot_confusion_matrix

def load_data(features_csv, labels_csv):
    # Load features and labels
    feat_df = pd.read_csv(features_csv)
    lab_df  = pd.read_csv(labels_csv)

    # 1) Join on subject_id to be 100% sure rows are aligned
    df = feat_df.merge(lab_df, on="subject_id", how="inner")

    # 2) Extract labels and map CN/MCI/AD -> 0/1/2
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    y = df["label"].map(label_map).values

    # 3) Build feature matrix (drop id + label)
    X = df.drop(columns=["subject_id", "label"], errors="ignore")

    # 4) Force numeric dtype, non-numeric → NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # 5) Drop any columns that are entirely NaN
    all_nan_cols = X.columns[X.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print(f"[INFO] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)

    # 6) Mean-impute remaining NaNs column-wise
    X = X.fillna(X.mean())

    # 7) As extra safety, drop any rows that still contain NaN
    row_mask = ~X.isna().any(axis=1)
    if not row_mask.all():
        dropped = (~row_mask).sum()
        print(f"[INFO] Dropping {dropped} rows that still contain NaNs after imputation")
    X = X.loc[row_mask]
    y = y[row_mask.values]

    # 8) Optionally drop zero-variance columns (can cause scaler warnings)
    var = X.var(axis=0)
    nonzero_var = var > 0
    if not nonzero_var.all():
        dropped_zero = (~nonzero_var).sum()
        print(f"[INFO] Dropping {dropped_zero} zero-variance columns")
        X = X.loc[:, nonzero_var]

    # 9) Return as numpy arrays
    X = X.values.astype(np.float32)
    return X, y


def evaluate_svm(X, y, out_dir):
    ensure_dir(out_dir)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    y_pred = cross_val_predict(clf, Xs, y, cv=skf)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro")),
        "recall_macro": float(recall_score(y, y_pred, average="macro")),
        "f1_macro": float(f1_score(y, y_pred, average="macro"))
    }
    y_proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")
    try:
        auc_ovr = roc_auc_score(y, y_proba, multi_class="ovr")
        metrics["auc_ovr"] = float(auc_ovr)
    except Exception as e:
        metrics["auc_ovr_error"] = str(e)

    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, ["CN","MCI","AD"], os.path.join(out_dir, "svm_confusion.png"))
    save_json(metrics, os.path.join(out_dir, "metrics_svm.json"))
    print(json.dumps(metrics, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    X, y = load_data(args.features_csv, args.labels_csv)
    evaluate_svm(X, y, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
