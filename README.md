# Low-Latency DGA Detection

Detect Domain Generation Algorithm (DGA) traffic using lexical features alone.
Binary classification (DGA vs. benign) and multi-class malware family attribution.


## Overview
This project investigates whether a low-latency, low-parameter DGA detection system can maintain strong discriminative performance while remaining robust to entirely novel malware families, using only lexical patterns in raw domain strings. Two sub-problems are addressed:

- Sub-problem A — Binary classification: DGA vs. benign
- Sub-problem B — Multi-class malware family identification


## Dataset
Source: DGA Dataset on Kaggle

- 160,000 labelled domain strings (80,000 benign / 80,000 DGA)
- 7 DGA sub-families: bamital, cryptolocker, gameoverdga, goz, necurs, newgoz, nivdort


Download instructions:
bash# kagglehub.dataset_download("gtkcyber/dga-dataset")


## Project structure
```
project-root/
|
├── problem_a/
│   ├── autoencoder/   
│   ├── CNN/
│   ├── Logistic_Regression/
│   ├── transformer/
│   └── xgboost/
|
├── problem_b/
│   └── svm/
|
├── eda.ipynb
├── requirements.txt
└── README.md
```

## Setup
pip install -r requirements.txt

Note: The Transformer model requires GPU infrastructure for inference within the
sub-millisecond latency budget. All other models can run on CPU.


## Results Summary
Sub-problem A (Binary)
ModelAUC-ROCRecallF1P99 LatencyParametersLR – WBCE0.99770.94810.96840.622 ms100,001XGBoost – WBCE0.96430.78110.87230.972 ms200 treesCNN0.99890.98680.98881.371 ms854,705Transformer0.99880.98850.987214.72 ms132,084,482Autoencoder0.87380.81790.82830.099 ms627,815
Recommended deployment: Logistic Regression with Weighted BCE as the primary
inline detector, optionally paired with a multi-class attribution layer.
Sub-problem B (Multi-class)

Linear SVM test accuracy: 87.79%, macro F1: 0.7625
Known failure: necurs (F1 = 0.0137) — near-complete confusion with cryptolocker
Universal blind spot: nivdort (LOFO detection rate: 0%)


## Known Limitations

- nivdort cannot be detected by any model trained on the remaining families (LOFO F1 ≈ 0).
- Its generation strategy produces domains lexically similar to benign traffic.
- TF-IDF vocabularies are fixed at training time; novel n-gram patterns from unseen DGA families produce zero-weight features.
- Bag-of-n-grams representations discard positional ordering.
- Latency benchmarks were not conducted on standardised hardware.
- The Transformer requires dedicated GPU infrastructure for real-time inline DNS inspection.

## Citation
If you use this work, please cite the dataset:

Givre C., "DGA Dataset," Kaggle, 2021.
https://www.kaggle.com/datasets/gtkcyber/dga-dataset
