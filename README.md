# CCL-NMF Pipeline
This repository implements the **Coupled Cross-sectional and Longitudinal Non-Negative Matrix Factorization (CCL-NMF)** framework described in  
*Skampardoni et al., “Coupled Cross-sectional and Longitudinal Non-Negative Matrix Factorization Reveals Dominant Brain Aging Trajectories in 48,949 Individuals,” Nature Communications, 2025*

## Overview
CCL-NMF is a machine learning framework designed to disentangle **heterogeneous brain aging patterns** by jointly analyzing cross-sectional and longitudinal data.
- **Cross-sectional deviation map** (C-map) estimated with a normative modeling approach, and  
- **Longitudinal change map** (L-map) estimated with linear mixed-effects models.  

These maps are jointly decomposed via NMF into a shared dictionary of **interpretable brain aging components** and **subject-specific loading coefficients for each data type**:
- **Cross-sectional loadings** reflect how strongly the patterns are expressed at baseline.
- **Longitudinal loadings** capture how rapidly these patterns progress over time.

Importantly, CCL-NMF outputs continuous loadings (not hard assignments), so individuals
can express multiple patterns simultaneously, capturing mixed neuropathologic processes.


## Install
```bash
conda create --name test python=3.10
source activate test
pip install -r requirements.txt
```

## Run
```bash
# 1) Train the autoencoder (AA)
python ccl_nmf/aa.py train \
  --features_csv data/aa_input/ref_features.csv \
  --covariates_csv data/aa_input/ref_covariates.csv \
  --outdir data/aa_output \
  --age_bins 5 \
  --z_dim 10 --h_dim 110 100 \
  --epochs 1000 --patience 50 \
  --batch_size 200 --seed 42 \
  --valheldout_size 0.35 --heldout_frac_within_val 0.40 \
  --base_lr 1e-4 --max_lr 5e-3 --gamma 0.98 --step_size 0

# 2) Run inference for validation, heldout, and test sets
# Validation:
python ccl_nmf/aa.py infer \
  --models_dir data/aa_output \
  --features_csv data/aa_output/val_features.csv \
  --covariates_csv data/aa_output/val_participants.csv \
  --name val \
  --seed 42

# Heldout:
python ccl_nmf/aa.py infer \
  --models_dir data/aa_output \
  --features_csv data/aa_output/heldout_features.csv \
  --covariates_csv data/aa_output/heldout_participants.csv \
  --name heldout \
  --seed 42

# Test:
python ccl_nmf/aa.py infer \
  --models_dir data/aa_output \
  --features_csv data/aa_input/target_features.csv \
  --covariates_csv data/aa_input/target_covariates.csv \
  --name test \
  --seed 42

# 3) Build the C-map
python ccl_nmf/C_map_calc.py \
  --output_dir data/aa_output \
  --save_dir data/jointNMF_input/C_map

# 4) Build the L-map
python ccl_nmf/L_map_calc.py \
  --residuals_csv data/jointNMF_input/C_map/C_map.csv \
  --rest_csv data/lme_input/target_longitudinal.csv \
  --out_dir data/jointNMF_input/L_map \
  --min_scans 3

# 5) Run joint NMF
python ccl_nmf/run_jointNMF.py \
  --cross data/jointNMF_input/C_map/C_map.csv \
  --betas data/jointNMF_input/L_map/L_map.csv \
  --output-dir data/jointNMF_output \
  --num-components 3 \
  --seed 42
```

## Input Data

The pipeline expects CSV files with specific formats for features, covariates, and longitudinal follow-up. Example toy datasets are provided below.

### 1. Reference cohort (for training the autoencoder)
- **`ref_features.csv`**  
  Shape: 1000 × 21  
  - `participant_id`: subject identifier  
  - `ROI_01` … `ROI_20`: regional gray matter volumes at baseline  

- **`ref_covariates.csv`**  
  Shape: 1000 × 2  
  - `participant_id`  
  - `Age`: baseline age in years  

These files provide the data of the *reference group* used to train the normative autoencoder.

---

### 2. Target cohort (to apply the autoencoder and generate C-map)
- **`target_features.csv`**  
  Shape: 4000 × 21  
  - `participant_id`  
  - `ROI_01` … `ROI_20`: regional gray matter volumes at baseline  

- **`target_covariates.csv`**  
  Shape: 4000 × 2  
  - `participant_id`  
  - `Age`: baseline age in years  

These files provide the data of the *target cohort*, used for generating deviations (C-map inputs).

---

### 3. Longitudinal data of the target cohort (for L-map estimation)
- **`target_longitudinal.csv`**  
  Shape: ~10,000 × 26  
  - `participant_id`  
  - `visit`: visit number  
  - `Age`: age at scan  
  - `Sex`: sex (coded 0/1)  
  - `ICV`: intracranial volume  
  - `Diagnosis`: diagnosis label (e.g., CN, MCI, AD)  
  - `ROI_01` … `ROI_20`: longitudinal gray matter volumes across visits  

This file is used to estimate **individual rates of change** with linear mixed-effects models (L-map).

---

## Notes
- `participant_id` must be consistent across all files.  
- ROI columns must have identical naming between reference, target, and longitudinal datasets.  
- Longitudinal table should include at least 3 scans per subject (or adjust `--min_scans`).  
- Additional covariates (Sex, ICV, Diagnosis) can be retained for provenance but are not required in the current modeling.  
