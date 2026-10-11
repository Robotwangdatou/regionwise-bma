# Regionwise Bayesian Model Averaging with CEIWC-Aware Floor Gating

This repository contains the code and data for the paper *Regionwise Bayesian Model Averaging with CEIWC-Aware Floor Gating* (submitted to Neurocomputing).

## Overview

We identify **Covariance-Estimation-Induced Weight Collapse (CEIWC)**: in data-scarce subregions, finite-sample covariance estimation drives optimal expert weights toward zero, silencing the very models minority-class decisions depend on. We derive a symmetric floor gate from classification Bayes risk, with provable admissibility and risk-degradation bounds, and validate it on 31 imbalanced tabular datasets and CIFAR-10 ResNet-18 embeddings.

## Repository Structure

```
├── code/                    # Core experiment scripts
│   ├── regionwise_bma_experiments.py   # Core BMA framework and gate
│   ├── run_paper_method.py             # Main gated regionwise BMA (31 datasets)
│   └── run_deep_track_v2.py            # CIFAR-10 deep track (5 seeds)
└── data/                    # Imbalanced datasets (CSV)
```
```

## Datasets

The 31 imbalanced datasets are from the KEEL and UCI repositories. Smaller datasets are included in `data/`. Larger datasets must be downloaded separately:

- **creditcard.csv**: [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) (126 MB)
- **covtype_4vs_2.csv**: UCI Covertype
- **cifar-10-python.tar.gz**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (170 MB)

## Key Results

| Metric | Value |
|---|---|
| Gated RW vs. Global BMA (31 datasets) | +0.0590 AUC, p < 0.0001, 28/3 wins |
| Highly imbalanced (IR > 20, N=9) | +0.1528 AUC, 9/9 wins |
| Gate effect (gated − ungated RW) | +0.0609 AUC, p < 0.0001 |
| creditcard fraud | 0.4843 → 0.7771 (+0.2928) |
| CIFAR-10 cat-vs-dog (5 seeds) | 0.8533 → 0.8886 (+0.037, p=0.015) |

## Requirements

- Python 3.8+
- scikit-learn, numpy, scipy, pandas, matplotlib
- PyTorch (for CIFAR-10 deep track)

