import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# -----------------------------
# 1. Data loading (adapted from features_svm.py)
# -----------------------------

def load_data_with_names(features_csv, labels_csv):
    """
    Load ADNI tabular features + labels, align by subject_id,
    handle NaNs, drop bad columns, and return:
      - X: pandas DataFrame of numeric features
      - y: numpy array of class indices (0=CN, 1=MCI, 2=AD)
    """
    feat_df = pd.read_csv(features_csv)
    lab_df  = pd.read_csv(labels_csv)

    # Join explicitly on subject_id to ensure alignment
    df = feat_df.merge(lab_df, on="subject_id", how="inner")

    # Map labels
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    y = df["label"].map(label_map).values

    # Drop non-feature columns
    X = df.drop(columns=["subject_id", "label"], errors="ignore")

    # Force numeric, non-numeric -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN columns
    all_nan_cols = X.columns[X.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print(f"[INFO] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)

    # Mean-impute remaining NaNs
    X = X.fillna(X.mean())

    # Drop rows still containing NaN (extra safety)
    row_mask = ~X.isna().any(axis=1)
    if not row_mask.all():
        dropped = (~row_mask).sum()
        print(f"[INFO] Dropping {dropped} rows with NaNs after imputation")
    X = X.loc[row_mask]
    y = y[row_mask.values]

    # Drop zero-variance columns
    var = X.var(axis=0)
    nonzero_var = var > 0
    if not nonzero_var.all():
        dropped_zero = (~nonzero_var).sum()
        print(f"[INFO] Dropping {dropped_zero} zero-variance columns")
        X = X.loc[:, nonzero_var]

    return X, y


# -----------------------------
# 2. Fit SVM with CV and get y_pred, y_proba
# -----------------------------

def run_svm_cv(X_df, y):
    """
    Run 5-fold Stratified CV with RBF SVM, return:
      - y_pred: predicted labels (CV)
      - y_proba: predicted probabilities (CV)
      - Xs: standardized feature matrix (numpy)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df.values.astype(np.float32))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)

    print("[INFO] Running cross_val_predict for labels...")
    y_pred = cross_val_predict(clf, Xs, y, cv=skf)

    print("[INFO] Running cross_val_predict for probabilities...")
    y_proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")

    return Xs, y_pred, y_proba


# -----------------------------
# 3. Plots
# -----------------------------

def plot_confusion_normalized(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SVM Confusion Matrix (Normalized)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved normalized confusion matrix to {out_path}")


def plot_multiclass_roc(y_true, y_proba, class_names, out_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    n_classes = y_bin.shape[1]

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC curves to {out_path}")


def plot_linear_svm_importance(X_df, y, out_path, top_k=15):
    """
    Train a LinearSVC just for interpretability.
    Plots top_k features by mean |weight| across classes.
    """
    lin = LinearSVC(max_iter=5000)
    lin.fit(X_df.values, y)

    # coef_ shape: (n_classes, n_features)
    coef = lin.coef_
    importance = np.mean(np.abs(coef), axis=0)

    feature_names = np.array(X_df.columns)
    top_idx = np.argsort(importance)[::-1][:top_k]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), importance[top_idx])
    plt.xticks(range(top_k), feature_names[top_idx], rotation=45, ha="right")
    plt.ylabel("Mean |weight| (LinearSVC)")
    plt.title(f"Top {top_k} SVM Feature Importances")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved feature importance plot to {out_path}")


# -----------------------------
# 4. CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default="figures_svm",
        help="Directory to save SVM visualizations",
    )
    args = parser.parse_args()

    class_names = ["CN", "MCI", "AD"]

    print("[INFO] Loading data...")
    X_df, y = load_data_with_names(args.features_csv, args.labels_csv)
    print(f"[INFO] Data shape: X={X_df.shape}, y={y.shape}")

    print("[INFO] Running SVM with cross-validation...")
    Xs, y_pred, y_proba = run_svm_cv(X_df, y)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Normalized confusion matrix
    cm_path = os.path.join(args.out_dir, "svm_confusion_normalized.png")
    plot_confusion_normalized(y, y_pred, class_names, cm_path)

    # 2) Multi-class ROC curves
    roc_path = os.path.join(args.out_dir, "svm_roc_curves.png")
    plot_multiclass_roc(y, y_proba, class_names, roc_path)

    # 3) Feature importance (Linear SVM)
    fi_path = os.path.join(args.out_dir, "svm_feature_importance.png")
    plot_linear_svm_importance(X_df, y, fi_path, top_k=15)


if __name__ == "__main__":
    main()
