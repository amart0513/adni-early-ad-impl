import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# =====================
# Load SVM Metrics
# =====================
with open("reports/metrics_svm.json") as f:
    svm_metrics = json.load(f)

with open("reports/svm_robustness.json") as f:
    rob = json.load(f)


print("Loaded SVM metrics and robustness data.")


# =====================
# Confusion Matrix Plot
# =====================
if "y_true" in svm_metrics and "y_pred" in svm_metrics:
    y_true = np.array(svm_metrics["y_true"])
    y_pred = np.array(svm_metrics["y_pred"])

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['CN','MCI','AD'],
                yticklabels=['CN','MCI','AD'])
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("reports/svm_confusion_matrix.png")
    plt.show()
else:
    print("y_true and y_pred not found in metrics JSON.")


# =====================
# Robustness Curve
# =====================
atrophy_levels = sorted([float(k) for k in rob.keys()])
svm_acc = [rob[str(l)]["accuracy"] for l in atrophy_levels]
svm_f1  = [rob[str(l)]["f1_macro"] for l in atrophy_levels]

plt.figure(figsize=(7,5))
plt.plot(atrophy_levels, svm_acc, marker='o', label='Accuracy')
plt.plot(atrophy_levels, svm_f1, marker='o', label='Macro F1')
plt.title("SVM Robustness Curve")
plt.xlabel("Atrophy Level")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/svm_robustness_curve.png")
plt.show()


# =====================
# Optional: SVM ROC Curves
# Requires val_proba or probability estimates
# =====================
if "probs" in svm_metrics:
    y_proba = np.array(svm_metrics["probs"])
    y_true = np.array(svm_metrics["y_true"])
    y_bin = label_binarize(y_true, classes=[0,1,2])

    fpr = {}; tpr = {}; roc_auc = {}

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7,5))
    labels = ['CN','MCI','AD']
    colors = ['blue','green','red']

    for i in range(3):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.title("SVM ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/svm_roc_curves.png")
    plt.show()

else:
    print("No probability outputs found for SVM ROC.")


print("Visuals generated!")
