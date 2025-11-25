import os, re, json, random, numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def parse_subject_id(filename: str):
    base = os.path.basename(filename)
    m = re.search(r"(sub-\d+)", base)
    if m:
        return m.group(1)
    m = re.search(r"(\d+_S_\d+)", base)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.splitext(base)[0])[0]

def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
