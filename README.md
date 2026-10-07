# Bridging the Redshift Gap in the Fermi-LAT Catalog Using Supervised Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2609.xxxxx-b31b1b.svg)](https://arxiv.org/)

**Official Research Artifact & Reproducibility Repository**

**Authors:** Deepanshu Kushwaha & Raj Prince  
**Affiliations:** Bundelkhand Institute of Engineering & Technology, Jhansi, India; Banaras Hindu University, Varanasi, India  
**Correspondence:** `priraj@bhu.ac.in` | `deepanshukushwahaiit1@gmail.com`

---

## 🌌 Overview

Determining cosmological redshifts for Active Galactic Nuclei (AGNs)—specifically BL Lacertae objects (BLLs)—is a critical observational bottleneck in high-energy astrophysics due to their predominantly featureless non-thermal continua. This repository provides the complete, leakage-free machine learning pipeline, benchmark datasets, trained ensemble models, conformal uncertainty calibrators, and camera-ready artifacts published in our paper:

> **"Bridging the Redshift Gap in the Fermi-LAT Catalog Using Supervised Machine Learning"**

### Key Scientific Highlights
* **Sample Expansion**: 1,318 complete-case spectroscopic blazars (670 BLLs, 648 FSRQs) from Fermi-LAT 4LAC-DR3 combined with optical Gaia EDR3 and AllWISE mid-infrared photometry (+18.5% over previous state-of-the-art benchmarks).
* **Multi-Wavelength Predictors**: 18 active physical features (10 Fermi-LAT $\gamma$-ray parameters, 1 Gaia $G$ magnitude, 6 AllWISE infrared magnitudes and colors, and $P_{\rm FSRQ}$ subclass probability).
* **Performance Gain**: Calibrated ensemble achieves **$R = 0.8044$**, **$\text{RMSE} = 0.3900$**, and robust normalized scatter **$\sigma_{\text{NMAD}} = 0.1290$** (outperforming literature benchmarks of $R = 0.7420$, $\text{RMSE} = 0.4670$).
* **Untouched Lockbox Holdout**: Strict zero-lookahead 15% lockbox test ($N=198$) yields $\text{RMSE} = 0.3779$ and reduced outlier fraction of $23.76\%$ on the target BLL subclass.
* **Distribution-Free Uncertainty**: Split-conformal prediction provides finite-sample **95% confidence intervals** ($[z_{\text{low}}^{95\%}, z_{\text{upp}}^{95\%}]$) with **95.09%** empirical test coverage and rigorous physical non-negative clipping ($z \ge 0.0000$).
* **Generalization Catalog**: Release of estimated photometric redshifts, calibrated confidence intervals, Mahalanobis out-of-distribution scores ($D_M$), and reliability grades (A, B, C) for **410 unmeasured BLLs**.

---

## 📂 Repository Structure

```
├── README.md                                  # Comprehensive reviewer and reproduction guide
├── requirements.txt                           # Verified Python dependency specifications
├── .gitignore                                 # Git ignore rules for clean repository state
├── requirements.txt                           # Verified Python dependency specifications
├── .gitignore                                 # Git ignore rules for clean repository state
├── Paper_agn_tex.pdf                          # Camera-ready publication manuscript (27 pages)
├── main.tex                                   # Primary LaTeX manuscript source
├── aastex701.cls                              # Official AAS journals document class
├── reproduce_results.py                       # Single-command Python reproduction script
│
├── data/                                      # Master datasets & catalogs
│   ├── dr3_full_train.csv                     # 1,318 complete-case spectroscopic training blazars
│   ├── dr3_full_gen.csv                       # 410 range-bounded generalization BL Lacs
│   ├── 4LAC-DR2.csv                           # Historical comparison benchmark catalog (DR2)
│   ├── table-4LAC-DR3-h.fits                  # High-latitude 4LAC-DR3 master FITS catalog
│   └── table-4LAC-DR3-l.fits                  # Low-latitude 4LAC-DR3 master FITS catalog
│
├── notebooks/                                 # Interactive verification notebook
│   └── AGN_Redshift_Reproduction.ipynb        # End-to-end reproducible Jupyter/Colab notebook
│
├── figures/                                   # Camera-ready manuscript figures
│   └── (All 18 publication figures & vector PDFs)
│
└── results/                                   # Full output artifacts, tables, and catalog
    ├── DR3_BLL_photometric_redshift_catalog.csv # Complete 410-source generalization catalog
    ├── figures/                               # Output figures (PNG & vector PDF)
    └── tables/                                # Output tables (CSV & verified metrics JSON)
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/GodxDeepanshu/AGN_Redshift_reproduction-or-stellar-dynamics-analysis.git
cd AGN_Redshift_reproduction-or-stellar-dynamics-analysis
```

### 2. Set Up Python Environment
We recommend Python 3.10, 3.11, 3.12, or 3.13:
```bash
# Using conda
conda create -n agn_redshift python=3.11 -y
conda activate agn_redshift

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Step-by-Step Reproduction Guide

### Option A: Interactive Google Colab / Jupyter Notebook (Recommended)
Open `notebooks/AGN_Redshift_Reproduction.ipynb` locally or upload to [Google Colab](https://colab.research.google.com/):
1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Run all cells sequentially.
3. The notebook will automatically:
   * Validate the computing environment.
   * Ingest and audit the 1,318 complete-case training sources and 410 generalization sources.
   * Execute 5-fold cross-validation across all 9 base models, stacking meta-learners, and isotonic calibration.
   * Evaluate the untouched 15% lockbox test set with 1,000 bootstrap resamples.
   * Generate and verify all figures and numerical tables.

### Option B: Command-Line Interface (CLI)
To run the automated verification suite directly from your terminal:
```bash
python reproduce_results.py
```

The script prints real-time assertions verifying that all sample sizes, correlation metrics, error values, coverage percentages, and physical bounds match the published manuscript.

---

## 📊 Artifact Mapping (Paper Artifacts to Files)

### 📈 Figures
| Manuscript Figure | Description | File Path |
| :--- | :--- | :--- |
| **Figure 1** | Redshift distribution ($z$) comparing Narendra et al. (2022) with current sample | [`results/figures/01_redshift_distribution.png`](results/figures/01_redshift_distribution.png) |
| **Figure 2** | Redshift distribution in inverted scale $1/(1+z)$ | [`results/figures/02_inv_redshift_distribution.png`](results/figures/02_inv_redshift_distribution.png) |
| **Figure 3** | Full-width 18-variable corner scatter matrix (Vector PDF) | [`results/figures/03_scatter_matrix.pdf`](results/figures/03_scatter_matrix.pdf) |
| **Figure 4** | Training vs. total redshift distribution | [`results/figures/04_training_vs_total_z.png`](results/figures/04_training_vs_total_z.png) |
| **Figure 5** | Stacking meta-ensemble regression coefficients | [`results/figures/05_stacking_coefficients.png`](results/figures/05_stacking_coefficients.png) |
| **Figure 6** | Predicted vs. observed redshift on the development cohort | [`results/figures/06_predicted_vs_observed.png`](results/figures/06_predicted_vs_observed.png) |
| **Figure 7** | Residual diagnostics across cosmic epochs | [`results/figures/07_residuals.png`](results/figures/07_residuals.png) |
| **Figure 8** | Metric distributions across 5-fold cross-validation splits | [`results/figures/08_metric_distributions.png`](results/figures/08_metric_distributions.png) |
| **Figure 9** | BL Lac pairwise scatter matrix and KDEs (Vector PDF) | [`results/figures/09_BLL_scatter_matrix.pdf`](results/figures/09_BLL_scatter_matrix.pdf) |
| **Figure 10** | Uncalibrated generalization predictions for unmeasured BLLs | [`results/figures/10_generalization_predictions.png`](results/figures/10_generalization_predictions.png) |
| **Figure 11** | Model comparison against Narendra et al. (2022) predictions | [`results/figures/11_model_comparison.png`](results/figures/11_model_comparison.png) |
| **Figure 12** | Comparative calibration diagnostics (OT vs. PCHIP vs. Isotonic) | [`results/figures/12_calibration_comparison.png`](results/figures/12_calibration_comparison.png) |
| **Figure 13** | Calibrated predicted vs. observed redshift | [`results/figures/13_calibrated_predictions.png`](results/figures/13_calibrated_predictions.png) |
| **Figure 14** | Calibrated generalization redshift distributions | [`results/figures/14_calibrated_generalization.png`](results/figures/14_calibrated_generalization.png) |
| **Figure 15** | Multi-model iteration convergence curves | [`results/figures/18_iteration_convergence.png`](results/figures/18_iteration_convergence.png) |
| **Figure 16** | 95% conformal prediction interval width distribution | [`results/figures/26_conformal_interval_widths.png`](results/figures/26_conformal_interval_widths.png) |
| **Figure 17** | Normalized redshift distribution: Known vs. Predicted BL Lacs | [`results/figures/16_training_vs_generalization_distribution.png`](results/figures/16_training_vs_generalization_distribution.png) |
| **Figure 18** | **Mahalanobis Distance ($D_M$) vs. Conformal Interval Width ($W_{95}$)** ($r = 0.123$, $\rho_s = 0.058$) with reliability grading boxplots | [`results/figures/17_mahalanobis_conformal_relation.png`](results/figures/17_mahalanobis_conformal_relation.png) |

---

### 📋 Tables
| Manuscript Table | Description | File Path |
| :--- | :--- | :--- |
| **Table 1** | Data selection flow and complete-case sample counts | [`results/tables/Table1_data_selection.csv`](results/tables/Table1_data_selection.csv) |
| **Table 2** | Multi-wavelength feature ablation study | [`results/tables/feature_ablation.csv`](results/tables/feature_ablation.csv) |
| **Table 3** | Head-to-head comparison: Narendra et al. (2022) vs. This Work | [`results/tables/sml2_sml3_comparison.csv`](results/tables/sml2_sml3_comparison.csv) |
| **Table 4** | Base machine learning model benchmark (OOF CV on $N=1{,}120$) | [`results/tables/model_comparison.csv`](results/tables/model_comparison.csv) |
| **Table 5** | Independent 15% untouched lockbox test evaluation ($N=198$) with 95% bootstrap CIs | [`results/tables/lockbox_metrics_with_ci.csv`](results/tables/lockbox_metrics_with_ci.csv) |
| **Table 6** | Comparison of stacking meta-learners | [`results/tables/stacking_comparison.csv`](results/tables/stacking_comparison.csv) |
| **Table 7** | Post-processing calibration comparison (Isotonic champion) | [`results/tables/calibration_comparison.csv`](results/tables/calibration_comparison.csv) |
| **Table 8** | Empirical conformal prediction validation (95.09% coverage) | [`results/tables/conformal_validation.csv`](results/tables/conformal_validation.csv) |
| **Table 9** | Generalization catalog preview for first 12 unmeasured BLLs | [`results/tables/predictions_catalog_preview.csv`](results/tables/predictions_catalog_preview.csv) |

---

## 📜 Full Photometric Redshift Catalog

The complete catalog containing estimated photometric redshifts, 95% conformal uncertainty bounds, Mahalanobis distances, and reliability grades for all **410 unmeasured BL Lacs** is located at:
* [`results/DR3_BLL_photometric_redshift_catalog.csv`](results/DR3_BLL_photometric_redshift_catalog.csv)

### Catalog Columns
| Column Name | Description |
| :--- | :--- |
| `Source_Name` | 4FGL Catalog Source Identifier (e.g. `4FGL J0001.2-0747`) |
| `RA`, `DEC` | Celestial coordinates (J2000, degrees) |
| `CLASS` | Optical blazar subclass (`BLL`) |
| `P_FSRQ` | Estimated Bayesian probability of FSRQ subclass |
| `z_phot` | Estimated photometric redshift from calibrated stacked ensemble |
| `z_95_low` | 95% Conformal prediction lower bound (clipped at $\ge 0.0000$) |
| `z_95_high` | 95% Conformal prediction upper bound |
| `interval_width_95` | Absolute width of 95% uncertainty interval ($W_{95} = z_{\text{upp}}^{95\%} - z_{\text{low}}^{95\%}$) |
| `Mahalanobis_D_M` | Ledoit-Wolf shrinkage Mahalanobis distance from training distribution |
| `Reliability_Grade` | Reliability tier: **Grade A** (high confidence), **Grade B** (medium), **Grade C** (atypical/caution) |
| `OOD_Flag` | Out-of-Distribution indicator flag ($1$ if $D_M > 90$th percentile, else $0$) |

---

## 🔒 Rigorous Machine Learning Protocols

1. **Complete-Case Analysis**: To preserve statistical integrity and physical realism, no synthetic data imputation was performed. All sources have observed values across all 18 features.
2. **Zero-Lookahead Lockbox Partition**: 15% of the complete-case dataset ($N=198$) was isolated prior to feature scaling, model fitting, stacking, or calibration. It was evaluated strictly once to report unbiased generalization.
3. **Physical Cosmological Bounds**: By performing conformal calibration in the scale factor space $1/(1+z)$ and applying physical non-negative clipping, negative redshifts ($z < 0$) are strictly prohibited.
4. **Finite-Sample Guarantee**: Conformal prediction quantiles use the exact finite-sample adjustment $q = \lceil(n+1)(1-\alpha)\rceil / n$, guaranteeing marginal coverage $\ge 1 - \alpha$.

---

## 📝 Citation

If you find this code, catalog, or methodology helpful in your research, please cite:

```bibtex
@article{Kushwaha2026,
  author        = {Kushwaha, Deepanshu and Prince, Raj},
  title         = {Bridging the Redshift Gap in the Fermi-LAT Catalog Using Supervised Machine Learning},
  journal       = {The Astrophysical Journal},
  year          = {2026},
  eprint        = {2609.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.HE}
}
```

---

## 📬 Contact & Support
For questions, feedback, or issues regarding data processing or model reproduction:
* **Deepanshu Kushwaha**: `deepanshukushwahaiit1@gmail.com`
* **Raj Prince**: `priraj@bhu.ac.in`
