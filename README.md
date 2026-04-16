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

Project Structure
.
├── problem_a/
│   ├── autoencoder   
│   ├── CNN
│   ├── Logistic_Regression
│   ├── transformer
│   └── xgboost
├── problem_b/
│   └── svm
├── evaluation/
│   └── [FILL IN: LOFO protocol, threshold calibration, metrics]
├── notebooks/
│   └── [FILL IN: any EDA or results notebooks]
├── requirements.txt
└── README.md

Setup
Prerequisites

Python [FILL IN: version, e.g. 3.10+]
[FILL IN: CUDA version if GPU used for Transformer]

Installation
bashgit clone [FILL IN: repository URL]
cd [FILL IN: repo folder name]

pip install -r requirements.txt
Key dependencies (fill in exact versions used):
PackageVersionPurposescikit-learn[FILL IN]Logistic Regression, SVM, TF-IDFxgboost[FILL IN]Gradient Boosted Treestorch[FILL IN]CNN, Autoencodertransformers[FILL IN]CANINE Transformer (HuggingFace)numpy[FILL IN]Numerical operationspandas[FILL IN]Data handling[FILL IN][FILL IN]Any other dependencies

Note: The Transformer model requires GPU infrastructure for inference within the
sub-millisecond latency budget. All other models can run on CPU.


Usage
Sub-problem A — Binary Classification
bash# [FILL IN: command to train each model]
# e.g.
python models/logistic_regression.py --loss wbce --train data/[FILL IN]
python models/xgboost_model.py --train data/[FILL IN]
python models/cnn_model.py --train data/[FILL IN]
python models/transformer_model.py --train data/[FILL IN]
python models/autoencoder_model.py --train data/[FILL IN]
Sub-problem B — Multi-class Classification
bash# [FILL IN: command to train SVM and LR multi-class models]
Threshold Calibration
Post-training threshold calibration selects the highest threshold satisfying FPR ≤ 1%:
bash# [FILL IN: command or script to run threshold sweep]
Leave-One-Family-Out (LOFO) Robustness Evaluation
bash# [FILL IN: command to run LOFO protocol]
# Note: each fold retrains from scratch with a fresh vectoriser/feature extractor

Models
ModelSub-problemLoss Function(s)Logistic Regression + TF-IDFABCE, Weighted BCE, Focal LossXGBoostABCE, Weighted BCE, Focal LossCNN (1D conv + global max-pooling)ABCETransformer (CANINE)ABCEAutoencoder (anomaly detection)AMean Cross-Entropy (reconstruction)Linear SVMBMulti-class hinge lossLogistic RegressionBCategorical cross-entropy
TF-IDF configuration (Sub-problem A baseline):

Character-level n-grams: n = 2–4
Max features: 100,000
Sublinear TF scaling: enabled

Weighted BCE class weights: {benign: 3, DGA: 1}
Focal Loss parameters: α = 0.25, γ = 2.0
Autoencoder architecture:

Character embedding: 32-dimensional
Input shape: 75 × 32
Latent dimension: 32 (75:1 compression ratio)
[FILL IN: intermediate layer sizes, activation functions, decoder architecture]

CNN architecture:

[FILL IN: number of filters, kernel sizes, FC layer sizes, dropout]

Transformer: Pre-trained CANINE (Google, via HuggingFace transformers)

Evaluation
Metrics

Primary: AUC-ROC, FPR (constrained to ≤ 1%), Precision, Recall, F1
Robustness: Mean LOFO generalisation gap (baseline F1 − LOFO F1)
Operational: Per-sample inference latency (mean and P99), parameter count

Hardware

Note: Latency figures in the paper were not benchmarked on standardised hardware
and should be treated as indicative of architectural differences only.
[FILL IN: CPU/GPU specs used for each model during benchmarking]


Results Summary
Sub-problem A (Binary)
ModelAUC-ROCRecallF1P99 LatencyParametersLR – WBCE0.99770.94810.96840.622 ms100,001XGBoost – WBCE0.96430.78110.87230.972 ms200 treesCNN0.99890.98680.98881.371 ms854,705Transformer0.99880.98850.987214.72 ms132,084,482Autoencoder0.87380.81790.82830.099 ms627,815
Recommended deployment: Logistic Regression with Weighted BCE as the primary
inline detector, optionally paired with a multi-class attribution layer.
Sub-problem B (Multi-class)

Linear SVM test accuracy: 87.79%, macro F1: 0.7625
Known failure: necurs (F1 = 0.0137) — near-complete confusion with cryptolocker
Universal blind spot: nivdort (LOFO detection rate: 0%)


# Known Limitations

- nivdort cannot be detected by any model trained on the remaining families (LOFO F1 ≈ 0).
- Its generation strategy produces domains lexically similar to benign traffic.
- TF-IDF vocabularies are fixed at training time; novel n-gram patterns from unseen DGA families produce zero-weight features.
- Bag-of-n-grams representations discard positional ordering.
- Latency benchmarks were not conducted on standardised hardware.
- The Transformer requires dedicated GPU infrastructure for real-time inline DNS inspection.

# Citation
If you use this work, please cite the dataset:

Givre C., "DGA Dataset," Kaggle, 2021.
https://www.kaggle.com/datasets/gtkcyber/dga-dataset
