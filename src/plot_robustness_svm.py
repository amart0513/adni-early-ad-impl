import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_robustness_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    # data is expected like: { "0.0": {"accuracy": ..., "f1_macro": ...}, "0.05": {...}, ... }
    levels = []
    accs = []
    f1s = []
    for k, v in data.items():
        lvl = float(k)
        levels.append(lvl)
        accs.append(v.get("accuracy", float("nan")))
        f1s.append(v.get("f1_macro", float("nan")))
    # sort by atrophy level
    levels = np.array(levels)
    order = np.argsort(levels)
    return levels[order], np.array(accs)[order], np.array(f1s)[order]


def plot_robustness(levels, accs, f1s, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(levels * 100, accs, marker="o", label="Accuracy")
    plt.plot(levels * 100, f1s, marker="s", linestyle="--", label="F1 (macro)")
    plt.xlabel("Synthetic atrophy level (%)")
    plt.ylabel("Score")
    plt.title("SVM Robustness to Progressive Atrophy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved robustness plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_json",
        default="reports/robustness_svm.json",
        help="JSON file produced by robustness_svm.py",
    )
    ap.add_argument(
        "--out_png",
        default="figures/robustness_svm.png",
        help="Output path for the degradation curve",
    )
    args = ap.parse_args()

    levels, accs, f1s = load_robustness_json(args.in_json)
    plot_robustness(levels, accs, f1s, args.out_png)


if __name__ == "__main__":
    main()
