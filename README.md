# 🧠 Early Detection of Alzheimer’s Disease from MRI  
### *Hybrid Pipeline: Classical ML + 3D Deep Learning + Robustness + Grad-CAM Interpretability*

This repository implements a full Alzheimer’s classification pipeline using:

- **Support Vector Machine (SVM)** trained on tabular MRI-derived features  
- **3D Convolutional Neural Network (3D-CNN)** trained on full volumetric MRI  
- **BrainWeb synthetic atrophy** for robustness testing  
- **Grad-CAM interpretability** for CN, MCI, and AD  
- **ROC, AUC, confusion matrices, and metrics**

Due to computational limits, the **3D CNN was trained in Google Colab with GPU acceleration**.  
All other components (SVM, preprocessing, robustness tests, Grad-CAM overlays) run locally.

---

# 📁 Project Structure

```
adni-early-ad-impl/
├── data/
│   ├── raw/                
│   ├── preprocessed/       
│   ├── features/           
│   ├── clinical/           
│
├── outputs/
│   ├── metrics/            
│   ├── confusion/          
│   ├── roc/                
│   ├── gradcam/            
│   ├── robustness/         
│
├── src/
│   ├── preprocess.py       
│   ├── features_svm.py     
│   ├── robustness_svm.py   
│   ├── robustness.py       
│   ├── cnn3d.py            
│   ├── gradcam.py          
│   ├── utils.py            
│   ├── plot_roc.py         
│   ├── plot_confusion.py   
│   ├── stats_eval.py       
│
├── models/
│   ├── svm.pkl             
│   ├── scaler.pkl          
│   ├── best_3dcnn_model.keras  
│
├── figures/
│   ├── svm_roc.png
│   ├── cnn_roc.png
│   ├── svm_confusion.png
│   ├── cnn_confusion.png
│   ├── gradcam_CN.png
│   ├── gradcam_MCI.png
│   ├── gradcam_AD.png
│   ├── robustness_curve.png
│
└── README.md
```

---

# 🚀 Installation

```bash
pip install -r requirements.txt
```

Use:
- CPU for SVM + preprocessing  
- **Google Colab GPU** for the 3D CNN

---

# 🧩 Step 1 — Preprocess MRI Volumes

```bash
python src/preprocess.py     --input_dir data/raw     --output_dir data/preprocessed     --target_size 128
```

---

# 🧩 Step 2 — Train SVM Baseline

```bash
python src/features_svm.py     --features_csv data/features/adni_features.csv     --labels_csv data/features/labels.csv
```

Outputs:
- Confusion matrix  
- ROC curve  
- SVM model in `models/svm.pkl`

---

# 🧩 Step 3 — Train 3D CNN (Google Colab Required)

```python
!python src/cnn3d.py     --preproc_dir data/preprocessed     --labels_csv data/features/labels.csv     --epochs 40     --batch_size 2
```

Produces:
- `best_3dcnn_model.keras`
- CNN ROC + confusion matrix

---

# 🧩 Step 4 — Grad-CAM Interpretability

```bash
python src/gradcam.py     --model_path models/best_3dcnn_model.keras     --volume_path data/preprocessed/sub-001.npy     --label CN
```

---

# 🧩 Step 5 — Synthetic Atrophy Robustness (BrainWeb)

SVM:
```bash
python src/robustness_svm.py
```

CNN:
```bash
python src/robustness.py     --model_path models/best_3dcnn_model.keras
```

---

# 📊 Metrics Summary

| Metric | SVM | 3D CNN |
|--------|------|--------|
| Accuracy | ~0.58–0.65 | ~0.75–0.80 |
| Macro F1 | ~0.40 | ~0.54 |
| CN AUC | 0.82 | 0.81 |
| MCI AUC | 0.63 | 0.68 |
| AD AUC | 0.80 | 0.71 |
| Interpretability | Feature weights | 3D Grad-CAM |
| Robustness | Degrades faster | Stable under atrophy |

---

# 🙌 Authorship  
**Angie Martinez & Saul Espinoza Nalvarte**  
FIU — Early Alzheimer’s Detection Project (Fall 2025)
