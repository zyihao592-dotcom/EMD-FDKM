# FEMD‑DKM: A method for identifying rock facies in oil and gas reservoirs based on empirical mode decomposition and fusion weights

This repository provides the official Python implementation of the **FEMD‑DKM** lithofacies identification model proposed in the manuscript *A method for identifying rock facies in oil and gas reservoirs based on empirical mode decomposition and fusion weights*.

## Overview

FEMD‑DKM is a multi‑step intelligent lithofacies identification workflow for well‑logging data. It addresses the challenges of non‑stationary logging signals, multi‑scale feature extraction, sample class imbalance, and nonlinear high‑dimensional feature learning for complex reservoir lithofacies classification.

The workflow consists of three sequential core modules:

1. **FEMD**: Empirical Mode Decomposition (EMD) together with Kendall‑correlation & mutual‑information‑driven weight fusion. It decomposes raw well‑logging curves into multi‑scale intrinsic mode functions (IMFs), computes fusion weights via Bayesian optimization, and generates enhanced multi‑scale feature representations from the original logging data.
2. **R‑WDLS**: A weighted k‑nearest‑neighbor relative‑density and locally shadowed‑samples oversampling algorithm. It conducts minority‑class sample augmentation to mitigate severe class imbalance within lithofacies datasets and optimizes the overall sample distribution.
3. **model**: Deep Kernel Method (DKM) classification module. It adopts residual‑enhanced Kernel Principal Component Analysis (KPCA) for nonlinear feature mapping, applies the RACOS algorithm for hyper‑parameter global optimization, and finally implements lithofacies recognition using a Random Forest classifier.
