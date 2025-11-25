import argparse, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {'pearson_r': None, 'pearson_p': None, 'spearman_rho': None, 'spearman_p': None}
    pr, pp = pearsonr(x[mask], y[mask])
    sr, sp = spearmanr(x[mask], y[mask])
    return {'pearson_r': float(pr), 'pearson_p': float(pp), 'spearman_rho': float(sr), 'spearman_p': float(sp)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs_csv', required=True)
    ap.add_argument('--clinical_csv', required=True)
    ap.add_argument('--out_json', default='reports/clinical_corr.json')
    args = ap.parse_args()

    probs = pd.read_csv(args.probs_csv)
    clin = pd.read_csv(args.clinical_csv)
    df = probs.merge(clin, on='subject_id', how='inner')
    p_ad = df['p_AD'].values.astype(float)
    mmse = df['MMSE'].values.astype(float) if 'MMSE' in df else np.full_like(p_ad, np.nan, dtype=float)
    cdr = df['CDR'].values.astype(float) if 'CDR' in df else np.full_like(p_ad, np.nan, dtype=float)

    out = {
        'N': int(len(df)),
        'corr_pAD_MMSE': safe_corr(p_ad, mmse),
        'corr_pAD_CDR': safe_corr(p_ad, cdr)
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
