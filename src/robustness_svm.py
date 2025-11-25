import os, argparse, json, numpy as np, pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def eval_svm(
    features_csv,
    labels_csv,
    atrophy_levels=(0.0, 0.05, 0.10, 0.15),
    feature_keys=("hippocampus", "entorhinal", "temporal", "ventricle"),
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score

    # --- Load and align features + labels by subject_id ---
    X_raw = pd.read_csv(features_csv)
    ydf = pd.read_csv(labels_csv)

    if "subject_id" in X_raw.columns:
        df = X_raw.merge(ydf, on="subject_id", how="inner")
    else:
        df = X_raw.copy()
        df["label"] = ydf["label"]

    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    y = df["label"].map(label_map).values

    # Drop id + label from features
    X = df.drop(columns=["subject_id", "label"], errors="ignore")

    # Force numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN
    all_nan_cols = X.columns[X.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print(f"[INFO] Dropping {len(all_nan_cols)} all-NaN columns in robustness eval")
        X = X.drop(columns=all_nan_cols)

    # Mean-impute remaining NaNs
    X = X.fillna(X.mean())

    # Drop any rows still containing NaNs
    row_mask = ~X.isna().any(axis=1)
    if not row_mask.all():
        dropped = (~row_mask).sum()
        print(f"[INFO] Dropping {dropped} rows with NaNs after imputation (robustness)")
    X = X.loc[row_mask]
    y = y[row_mask.values]

    # Drop zero-variance columns
    var = X.var(axis=0)
    nonzero_var = var > 0
    if not nonzero_var.all():
        dropped_zero = (~nonzero_var).sum()
        print(f"[INFO] Dropping {dropped_zero} zero-variance columns (robustness)")
    X = X.loc[:, nonzero_var]

    # Keep column names for perturbation
    base_cols = X.columns.tolist()
    key_cols = [c for c in base_cols if any(k.lower() in c.lower() for k in feature_keys)]
    if not key_cols:
        key_cols = base_cols  # fallback: perturb all features

    results = {}
    LABELS = y  # just a clearer name

    for lvl in atrophy_levels:
        print(f"[INFO] Evaluating robustness at atrophy level = {lvl:.2f}")

        # Copy clean base features
        X_pert = X.copy()

        # Simulate atrophy: scale key features
        if lvl > 0:
            X_pert[key_cols] = X_pert[key_cols] * (1.0 - lvl)

            # Add small Gaussian noise
            noise_scale = 0.01 * X_pert.std().mean()
            if np.isfinite(noise_scale) and noise_scale > 0:
                X_pert += np.random.normal(0, noise_scale, size=X_pert.shape)

        # Standardize
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_pert.values.astype(np.float32))

        # SVM with 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")

        y_pred = cross_val_predict(clf, Xs, LABELS, cv=skf)
        acc = accuracy_score(LABELS, y_pred)
        f1 = f1_score(LABELS, y_pred, average="macro")

        results[lvl] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
        }
        print(f"  -> acc={acc:.3f}, f1_macro={f1:.3f}")

    return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_json", default="reports/robustness_svm.json")
    args = ap.parse_args()

    res = eval_svm(args.features_csv, args.labels_csv)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
