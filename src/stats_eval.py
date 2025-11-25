import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from scipy import stats

def bootstrap_ci(diff, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(diff, size=len(diff), replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds_csv', required=True)
    ap.add_argument('--out_json', default='reports/stats_eval.json')
    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)  # subject_id,y_true,y_pred_a,y_pred_b
    a_correct = (df['y_true'] == df['y_pred_a']).astype(int).values
    b_correct = (df['y_true'] == df['y_pred_b']).astype(int).values

    acc_a = a_correct.mean()
    acc_b = b_correct.mean()
    diff = a_correct - b_correct

    tstat, pval = stats.ttest_rel(a_correct, b_correct)
    lo, hi = bootstrap_ci(diff)

    out = {
        'accuracy_model_a': float(acc_a),
        'accuracy_model_b': float(acc_b),
        'paired_tstat': float(tstat),
        'paired_pval': float(pval),
        'acc_diff_mean': float(diff.mean()),
        'acc_diff_ci95': [lo, hi]
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
