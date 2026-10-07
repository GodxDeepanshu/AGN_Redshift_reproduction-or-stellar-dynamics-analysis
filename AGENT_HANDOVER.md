# AGN Redshift Project — Agent Handover Document

This document provides a complete summary of the tasks completed, methodology, final dataset metrics, and manuscript updates for any future agent taking over this project.

---

## 1. Project Context & Objectives
We are estimating the photometric redshifts ($z$) of active galactic nuclei (AGNs)—specifically BL Lacertae objects (BLLs) and Flat-Spectrum Radio Quasars (FSRQs)—in the **Fermi-LAT 4LAC-DR3** catalog. BLLs lack prominent spectral lines, making photometric estimation via machine learning crucial for cosmological and gamma-ray study.

---

## 2. Data Pipeline & Selection
*   **Sample Size**: Clean complete-case dataset of **1,318 spectroscopic sources** (no missing values, no median or other imputations used).
*   **Active Features (18 features total)**:
    *   *10 Fermi-LAT Gamma-Ray*: `LogFlux`, `LogEnergy_Flux`, `LogSignificance`, `LogVariability_Index`, `Lognu_syn`, `LognuFnu_syn`, `PL_Index`, `LogPivot_Energy`, `LP_Index`, `LP_beta`
    *   *6 AllWISE Infrared*: `W1mag`, `W2mag`, `W3mag`, `W4mag`, `W1_W2`, `W2_W3`
    *   *1 Gaia Optical*: `Gaia_G_Magnitude` (clean cases only, no missing Gaia values)
    *   *1 Subclass Probability*: $P(\text{FSRQ} | X)$ continuous Bayesian probability of being FSRQ
*   **Target Variable**: Trained in inverse space $1/(z+1)$ to map $z \in [0, \infty)$ to the bounded domain $(0,1]$, then transformed back to linear $z$ scale.

---

## 3. Champion Model Architecture
*   **Base Learners**: 12 regressors, including tree-based ensembles (Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, HistGradientBoosting) and tabular deep learning models (TabNet, Saint, FT-Transformer, TabM, MLP, TabPFN).
*   **Stacking Meta-Learner**: Regularized **LASSO** meta-learner fitted over out-of-fold base learner predictions (which replaced Bayesian Ridge as the best-performing stacker).
*   **CV Configuration**: 100 iterations of stratified 10-fold cross-validation (using all 16 CPU cores).
*   **Calibration**: Class-conditional non-parametric **Isotonic Regression** applied to predicted inverse redshifts to correct regression-to-the-mean bias.
*   **Uncertainty Quantification**: Split, Cross-Conformal, and Jackknife+ conformal prediction providing 95% prediction intervals.
*   **OOD safety**: Mahalanobis distance profiling to filter out-of-distribution blazars in the generalization catalog.

---

## 4. Final Verification Metrics (1,318-Source Champion Stack)
All metrics strictly follow standard photo-$z$ literature definitions, normalized by cosmological scaling $(1 + z_{\text{spec}})$:
*   $\text{Outlier Fraction } (\eta) = \frac{|z_{\text{pred}} - z_{\text{spec}}|}{1 + z_{\text{spec}}} > 0.15$
*   $\sigma_{\text{NMAD}} = 1.4826 \times \text{median}\left(\frac{|z_{\text{pred}} - z_{\text{spec}}|}{1 + z_{\text{spec}}}\right)$

### Comparative Performance Table
| Metric | Narendra et al. (2022) baseline | Our Champion Stack (Uncalibrated) | Our Champion Stack (Isotonic Calibrated) |
| :--- | :---: | :---: | :---: |
| **Pearson $R_z$** | 0.7420 | 0.8044 | **0.8114** |
| **$\text{RMSE}_z$** | 0.4570 | 0.3986 | **0.3904** |
| **Outlier Fraction $\eta$** | 43.50% | 30.96% | **29.59%** |
| **Scatter $\sigma_{\text{NMAD}, z}$** | 0.1879 | 0.1303 | **0.1212** |
| **Mean Bias** | — | -0.0720 | **-0.0692** |

*Note: Relative improvement over Narendra et al. (2022) is **9.35%** in Pearson $R_z$ and **14.57%** reduction in RMSE, achieved on a dataset that is 18.5% larger.*

---

## 5. Work Completed Up to Here

### A. Pipeline & Metric Corrections
*   Updated metric routines (`compute_metrics` and `compute_nmad`) in [run_publication_pipeline.py](file:///home/hea/AGN_RedShift_Project/Final_result/methodology/run_publication_pipeline.py) to implement $(1 + z_{\text{spec}})$ scaling.
*   Disabled range-bounding cuts on the generalization set to retain all **414 BLL sources** remaining after quality, spatial ($|GLAT| > 10^\circ$), and multi-wavelength completeness cuts (consistent with Narendra et al. 2022).
*   Successfully ran the full 100-iteration multi-core stacking pipeline. All predictions converged.

### B. Spec-Compliant Figure Regeneration
*   Checked the official specimen folder [Final_result/files/files](file:///home/hea/AGN_RedShift_Project/Final_result/files/files).
*   Modified [plot_figures.py](file:///home/hea/AGN_RedShift_Project/Final_result/files/files/plot_figures.py#L371-L393) to implement the real data loader `load_results()`.
*   Fixed a panel overwrite bug to successfully generate all **16 spec-compliant figures** in `figures_out/` at 300 DPI using publication-quality fonts.
*   Synchronized and copied the plots to the main project **[figures/](file:///home/hea/AGN_RedShift_Project/figures)** folder.
*   Ensured color schemes strictly match paper standards and user requests for Figure 3:
    *   `BLL Training`: Pink (`#FF69B4`)
    *   `FSRQ Training`: Green (`#27AE60`)
    *   `BLL Generalization`: Blue (`#2980B9`)
    *   `FSRQ Generalization`: Black (`#000000`)
*   Optimized figure sizes using adaptive palette conversions (reducing total size from 14.5MB to 2.9MB) to avoid compiler API upload limits.

### C. Manuscript Updates & PDF Compilation
*   Updated the LaTeX manuscript **[paper.tex](file:///home/hea/AGN_RedShift_Project/paper.tex)**:
    *   Revised abstract, introduction, comparison, results, and conclusion sections with final calibrated metrics.
    *   Updated all tables, including Table 8 (Appendix - first 30 prediction catalog sources with new redshifts).
    *   Aligned figure paths to point to the newly generated combined panels.
*   Successfully compiled `paper.tex` into the final PDF **[5 july paper.pdf](file:///home/hea/AGN_RedShift_Project/5%20july%20paper.pdf)** via the YtoTech LaTeX API.

---

## 6. Next Steps for Takeover Agent
1.  **Generalization Catalog**: The final calibrated redshift predictions and 95% conformal bounds for the 414 generalization BLLs are stored in **[predicted_redshift_DR3.csv](file:///home/hea/AGN_RedShift_Project/Final_result/results/predictions/predicted_redshift_DR3.csv)**.
2.  **PDF Deliverable**: The compiled manuscript is available in **[5 july paper.pdf](file:///home/hea/AGN_RedShift_Project/5%20july%20paper.pdf)**.

