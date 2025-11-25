import os, argparse, glob, json, numpy as np, nibabel as nib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd

def gaussian_3d(shape, center=None, sigma=10.0):
    z, y, x = np.indices(shape)
    if center is None:
        center = (shape[0]//2, shape[1]//2, shape[2]//2)
    cz, cy, cx = center
    return np.exp(-(((z-cz)**2 + (y-cy)**2 + (x-cx)**2) / (2*sigma**2)))

def apply_atrophy(volume, level=0.0, local=True, sigma=10.0):
    v = volume.copy()
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    if local and level > 0:
        mask = gaussian_3d(v.shape, center=(v.shape[0]//2, v.shape[1]//2, v.shape[2]//2), sigma=sigma)
        v = v * (1.0 - level * mask.astype(v.dtype))
    if level > 0:
        factor = 1.0 - 0.3*level
        from skimage.transform import resize
        ds = (max(2, int(v.shape[0]*factor)), max(2, int(v.shape[1]*factor)), max(2, int(v.shape[2]*factor)))
        v = resize(resize(v, ds, mode='reflect', anti_aliasing=True), v.shape, mode='reflect', anti_aliasing=True)
    return v

def load_rows(preproc_dir, labels_csv):
    df = pd.read_csv(labels_csv)
    rows = []
    for _, r in df.iterrows():
        sid = str(r['subject_id'])
        matches = glob.glob(os.path.join(preproc_dir, f"*{sid}*.nii*"))
        if not matches:
            continue
        rows.append((sid, matches[0], r['label']))
    return rows

def eval_cnn(model_path, rows, atrophy_levels=(0.0,0.05,0.10,0.15)):
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)
    label_map = {'CN':0,'MCI':1,'AD':2}
    y_true = np.array([label_map[r[2]] for r in rows])
    metrics = {}
    for lvl in atrophy_levels:
        y_pred = []
        y_proba = []
        for sid, path, label in rows:
            v = nib.load(path).get_fdata().astype('float32')
            v = apply_atrophy(v, level=lvl, local=True, sigma=10.0)
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            v = np.expand_dims(v, axis=(0,-1))
            proba = model.predict(v, verbose=0)[0]
            y_proba.append(proba)
            y_pred.append(np.argmax(proba))
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            auc = float('nan')
        f1 = f1_score(y_true, y_pred, average='macro')
        metrics[lvl] = {'accuracy': float(acc), 'auc_ovr': float(auc), 'f1_macro': float(f1)}
    return metrics

def eval_svm(features_csv, labels_csv, atrophy_levels=(0.0,0.05,0.10,0.15), feature_keys=('hippocampus','entorhinal','temporal','ventricle')):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score
    X = pd.read_csv(features_csv)
    ydf = pd.read_csv(labels_csv)
    label_map = {'CN':0,'MCI':1,'AD':2}
    ydf['y'] = ydf['label'].map(label_map)
    if 'subject_id' in X.columns:
        X = X.merge(ydf[['subject_id','y']], on='subject_id', how='inner')
        y = X.pop('y').values
        X = X.drop(columns=['subject_id'], errors='ignore')
    else:
        y = ydf['y'].values

    base_cols = X.columns.tolist()
    key_cols = [c for c in base_cols if any(k.lower() in c.lower() for k in feature_keys)]
    if not key_cols:
        key_cols = base_cols

    results = {}
    for lvl in atrophy_levels:
        X_pert = X.copy()
        X_pert[key_cols] = X_pert[key_cols] * (1.0 - lvl)
        X_pert += np.random.normal(0, 0.01*X_pert.std().mean(), size=X_pert.shape)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_pert)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        y_pred = cross_val_predict(clf, Xs, y, cv=skf)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        results[lvl] = {'accuracy': float(acc), 'f1_macro': float(f1)}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preproc_dir', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--cnn_model', required=True)
    ap.add_argument('--features_csv', required=True)
    ap.add_argument('--out_json', default='reports/robustness.json')
    args = ap.parse_args()

    rows = load_rows(args.preproc_dir, args.labels_csv)
    cnn_metrics = eval_cnn(args.cnn_model, rows)
    svm_metrics = eval_svm(args.features_csv, args.labels_csv)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump({'cnn': cnn_metrics, 'svm': svm_metrics}, f, indent=2)
    print(json.dumps({'cnn': cnn_metrics, 'svm': svm_metrics}, indent=2))

if __name__ == '__main__':
    main()
