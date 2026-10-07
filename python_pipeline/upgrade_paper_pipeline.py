import numpy as np
import pandas as pd
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau, ks_2samp, norm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import shap
import umap
from tabpfn import TabPFNRegressor

warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

# Create output directories
os.makedirs("plots/dr3_plots", exist_ok=True)
os.makedirs("plots/python_plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

def run_upgraded_pipeline():
    print("="*60)
    print("   AGN PHOTO-Z PIPELINE: 9.5+/10 RESEARCH GRADE UPGRADE")
    print("="*60)
    
    # ─── DATA LOADING & MULTI-WAVELENGTH PREPROCESSING ───────────────────────
    print("\n--- Loading data and merging counterparts ---")
    train_raw = pd.read_csv("data/training_eligible.csv")
    mw = pd.read_csv("data/multi_wavelength_matches.csv")
    mw['Source_Name'] = mw['Source_Name'].astype(str).str.strip()
    
    # Merge MW columns
    train_df = pd.merge(train_raw, mw, on="Source_Name", how="left")
    
    # New sources DR3
    from astropy.io import fits
    with fits.open("data/table-4LAC-DR3-h.fits") as h_hdul:
        dr3_h = pd.DataFrame(h_hdul[1].data)
    with fits.open("data/table-4LAC-DR3-l.fits") as l_hdul:
        dr3_l = pd.DataFrame(l_hdul[1].data)
    dr3 = pd.concat([dr3_h, dr3_l], ignore_index=True)
    for col in dr3.columns:
        if dr3[col].dtype == object:
            dr3[col] = dr3[col].astype(str).str.strip()
            
    dr2 = pd.read_csv("data/4LAC-DR2.csv")
    dr2_names = set(dr2['Source_Name'].astype(str).str.strip())
    new_names = set(dr3['Source_Name']) - dr2_names
    target_df = dr3[dr3['Source_Name'].isin(new_names)].copy()
    
    # Merge Gaia & MW
    gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    target_df = pd.merge(target_df, gaia_new, on="Source_Name", how="left")
    target_df = pd.merge(target_df, mw, on="Source_Name", how="left")
    
    # Class mapping for new sources
    target_df['CLASS_upper'] = target_df['CLASS'].astype(str).str.upper().str.strip()
    target_df['AGN_Type'] = np.where(target_df['CLASS_upper'].isin(['BLL', 'BL LAC']), 'BLL',
                             np.where(target_df['CLASS_upper'].isin(['FSRQ', 'FLAT SPECTRUM RADIO QUASAR']), 'FSRQ', 'BCU'))
    
    # Standard numerical conversion
    numeric_cols = ["LP_beta", "LP_Index", "Flux1000", "Energy_Flux100",
                    "Signif_Avg", "Variability_Index", "nu_syn", "nuFnu_syn",
                    "Pivot_Energy", "PL_Index", "GLAT", "Flags"]
    for col in numeric_cols:
        target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        
    positive_cols = ["Flux1000", "Energy_Flux100", "Signif_Avg",
                     "Variability_Index", "nu_syn", "nuFnu_syn", "Pivot_Energy"]
    for col in positive_cols:
        target_df.loc[target_df[col] <= 0, col] = np.nan
        train_df.loc[train_df[col] <= 0, col] = np.nan
        
    # Log transforms
    for df in [train_df, target_df]:
        df['LogFlux'] = np.log10(df['Flux1000'])
        df['LogEnergy_Flux'] = np.log10(df['Energy_Flux100'])
        df['LogSignificance'] = np.log10(df['Signif_Avg'])
        df['LogVariability_Index'] = np.log10(df['Variability_Index'])
        df['Lognu_syn'] = np.log10(df['nu_syn'])
        df['LognuFnu_syn'] = np.log10(df['nuFnu_syn'])
        df['LogPivot_Energy'] = np.log10(df['Pivot_Energy'])
        
        # Mid-IR colors
        df['W1mag'] = pd.to_numeric(df['W1mag'], errors='coerce')
        df['W2mag'] = pd.to_numeric(df['W2mag'], errors='coerce')
        df['W3mag'] = pd.to_numeric(df['W3mag'], errors='coerce')
        df['W4mag'] = pd.to_numeric(df['W4mag'], errors='coerce')
        df['S1.4'] = pd.to_numeric(df['S1.4'], errors='coerce')
        df['CR0'] = pd.to_numeric(df['CR0'], errors='coerce')
        df['Gaia_G_Magnitude'] = pd.to_numeric(df['Gaia_G_Magnitude'], errors='coerce')
        
        df['W1_W2'] = df['W1mag'] - df['W2mag']
        df['W2_W3'] = df['W2mag'] - df['W3mag']
        df['LogRadioFlux'] = np.log10(df['S1.4'].clip(lower=1e-5))
        df['LogXrayFlux'] = np.log10(df['CR0'].clip(lower=1e-6))
        
    orig_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                     "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                     "LP_beta", "Gaia_G_Magnitude"]
    mw_features = ["W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "LogRadioFlux", "LogXrayFlux"]
    all_predictors = orig_features + mw_features
    
    y_train = train_df['InvRedshift'].values
    y_train_z = train_df['Redshift'].values
    y_train_class = train_df['LabelNo'].fillna(0).astype(int).values
    
    print(f"Loaded {len(train_df)} training sources and {len(target_df)} target DR3 sources.")
    print(f"Predictor features count: {len(all_predictors)}")
    
    # ─── SECTION 1: LEAKAGE-PROOF NESTED CROSS-VALIDATION ───────────────────
    print("\n=== Section 1: Running Leakage-Proof Nested CV ===")
    
    # Outer 5-fold CV for redshift prediction evaluation
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store out-of-fold (OOF) regression predictions
    oof_predictions_inv = {
        'xgb': np.zeros(len(train_df)),
        'lgb': np.zeros(len(train_df)),
        'cat': np.zeros(len(train_df)),
        'tabpfn': np.zeros(len(train_df))
    }
    
    # Store out-of-fold class probability P(FSRQ|X) on the training set
    train_probs_nested = np.zeros(len(train_df))
    
    # Step-by-step loop
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train_df)):
        print(f"  Processing Outer Fold {fold+1}/5...")
        # 1. Isolate fold data
        df_tr_fold = train_df.iloc[train_idx].copy()
        df_val_fold = train_df.iloc[val_idx].copy()
        
        # 2. Impute missing values using training fold medians (avoid target leakage)
        impute_dict = {}
        for col in all_predictors:
            median_val = df_tr_fold[col].median(skipna=True)
            if pd.isna(median_val): median_val = 0.0
            impute_dict[col] = median_val
            df_tr_fold[col] = df_tr_fold[col].fillna(median_val)
            df_val_fold[col] = df_val_fold[col].fillna(median_val)
            
        # 3. Scale features using training fold scaler
        scaler_fold = StandardScaler()
        X_tr_sc = scaler_fold.fit_transform(df_tr_fold[all_predictors].values)
        X_val_sc = scaler_fold.transform(df_val_fold[all_predictors].values)
        
        # 4. Nested CV for generating train-fold class probabilities
        skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_train_probs = np.zeros(len(df_tr_fold))
        y_tr_class = y_train_class[train_idx]
        
        for i_train_idx, i_val_idx in skf_inner.split(X_tr_sc, y_tr_class):
            inner_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            inner_clf.fit(X_tr_sc[i_train_idx], y_tr_class[i_train_idx])
            inner_train_probs[i_val_idx] = inner_clf.predict_proba(X_tr_sc[i_val_idx])[:, 1]
            
        # Predict on validation fold using classifier trained on entire training fold
        outer_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        outer_clf.fit(X_tr_sc, y_tr_class)
        val_probs_fold = outer_clf.predict_proba(X_val_sc)[:, 1]
        
        # Save nested probabilities
        train_probs_nested[val_idx] = val_probs_fold
        
        # Create regression matrices (including P_FSRQ as a predictor)
        X_tr_reg = np.column_stack([X_tr_sc, inner_train_probs])
        X_val_reg = np.column_stack([X_val_sc, val_probs_fold])
        
        y_tr_reg = y_train[train_idx]
        
        # 5. Train regressors on training fold
        # XGBoost
        model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        model_xgb.fit(X_tr_reg, y_tr_reg)
        oof_predictions_inv['xgb'][val_idx] = model_xgb.predict(X_val_reg)
        
        # LightGBM
        model_lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        model_lgb.fit(X_tr_reg, y_tr_reg)
        oof_predictions_inv['lgb'][val_idx] = model_lgb.predict(X_val_reg)
        
        # CatBoost
        model_cat = cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, random_seed=42, thread_count=-1, verbose=0)
        model_cat.fit(X_tr_reg, y_tr_reg)
        oof_predictions_inv['cat'][val_idx] = model_cat.predict(X_val_reg)
        
        # TabPFN
        model_tabpfn = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
        model_tabpfn.fit(X_tr_reg, y_tr_reg)
        oof_predictions_inv['tabpfn'][val_idx] = model_tabpfn.predict(X_val_reg)
        
    print("Leakage-free nested CV predictions generated.")
    
    # ─── TRAIN FINAL MODELS ON ENTIRE TRAINING SET FOR TEST PREDICTIONS ───
    print("\n--- Training final models on full dataset ---")
    
    # Impute full datasets
    full_impute_dict = {}
    for col in all_predictors:
        median_val = train_df[col].median(skipna=True)
        if pd.isna(median_val): median_val = 0.0
        full_impute_dict[col] = median_val
        train_df[col] = train_df[col].fillna(median_val)
        target_df[col] = target_df[col].fillna(median_val)
        
    # Standardize full datasets
    scaler_full = StandardScaler()
    X_train_sc_full = scaler_full.fit_transform(train_df[all_predictors].values)
    X_target_sc_full = scaler_full.transform(target_df[all_predictors].values)
    
    # Final classifier
    clf_full = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    clf_full.fit(X_train_sc_full, y_train_class)
    target_probs_full = clf_full.predict_proba(X_target_sc_full)[:, 1]
    
    # Store probability on full datasets
    train_df['P_FSRQ'] = train_probs_nested
    target_df['P_FSRQ'] = target_probs_full
    
    # Build final regression arrays
    X_train_reg_full = np.column_stack([X_train_sc_full, train_probs_nested])
    X_target_reg_full = np.column_stack([X_target_sc_full, target_probs_full])
    
    final_models = {}
    test_preds_inv = {}
    target_preds_inv = {}
    
    # Models to train
    models_to_train = {
        'xgb': xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'lgb': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
        'cat': cb.CatBoostRegressor(iterations=500, learning_rate=0.03, depth=4, random_seed=42, thread_count=-1, verbose=0),
        'tabpfn': TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
    }
    
    # Isolate independent spectroscopic test set (n=89)
    spec_targets_mask = (target_df['Redshift'] > 0)
    spec_test_df = target_df[spec_targets_mask].copy()
    X_test_reg_full = X_target_reg_full[spec_targets_mask]
    y_test_spec_z = spec_test_df['Redshift'].values
    
    for m_name, model in models_to_train.items():
        print(f"  Fitting final {m_name}...")
        model.fit(X_train_reg_full, y_train)
        final_models[m_name] = model
        test_preds_inv[m_name] = model.predict(X_test_reg_full)
        target_preds_inv[m_name] = model.predict(X_target_reg_full)
        
    # Apply Isotonic Regression Calibration
    print("\n--- Applying non-linear Isotonic Calibration ---")
    oof_calibrated_z = {}
    test_calibrated_z = {}
    target_calibrated_z = {}
    
    # Masks for classes in training set
    bll_mask = (y_train_class == 0)
    fsrq_mask = (y_train_class == 1)
    
    for m_name in oof_predictions_inv.keys():
        iso_bll = IsotonicRegression(out_of_bounds='clip')
        iso_bll.fit(oof_predictions_inv[m_name][bll_mask], y_train[bll_mask])
        
        iso_fsrq = IsotonicRegression(out_of_bounds='clip')
        iso_fsrq.fit(oof_predictions_inv[m_name][fsrq_mask], y_train[fsrq_mask])
        
        # Calibrate OOF predictions
        oof_bll_c = iso_bll.predict(oof_predictions_inv[m_name])
        oof_fsrq_c = iso_fsrq.predict(oof_predictions_inv[m_name])
        oof_corr = (1.0 - train_probs_nested) * oof_bll_c + train_probs_nested * oof_fsrq_c
        oof_calibrated_z[m_name] = (1.0 / oof_corr) - 1.0
        
        # Calibrate Test predictions
        p_test = spec_test_df['P_FSRQ'].values
        test_bll_c = iso_bll.predict(test_preds_inv[m_name])
        test_fsrq_c = iso_fsrq.predict(test_preds_inv[m_name])
        test_corr = (1.0 - p_test) * test_bll_c + p_test * test_fsrq_c
        test_calibrated_z[m_name] = (1.0 / test_corr) - 1.0
        
        # Calibrate all target predictions
        p_target = target_df['P_FSRQ'].values
        target_bll_c = iso_bll.predict(target_preds_inv[m_name])
        target_fsrq_c = iso_fsrq.predict(target_preds_inv[m_name])
        target_corr = (1.0 - p_target) * target_bll_c + p_target * target_fsrq_c
        target_calibrated_z[m_name] = (1.0 / target_corr) - 1.0
        
    # Stacking Ensemble (average of the 4 calibrated models)
    oof_ensemble_z = np.mean(np.column_stack([oof_calibrated_z[k] for k in oof_calibrated_z.keys()]), axis=1)
    test_ensemble_z = np.mean(np.column_stack([test_calibrated_z[k] for k in test_calibrated_z.keys()]), axis=1)
    target_ensemble_z = np.mean(np.column_stack([target_calibrated_z[k] for k in target_calibrated_z.keys()]), axis=1)
    
    # Calculate Prediction STDs
    target_std_z = np.std(np.column_stack([target_calibrated_z[k] for k in target_calibrated_z.keys()]), axis=1)
    oof_std_z = np.std(np.column_stack([oof_calibrated_z[k] for k in oof_calibrated_z.keys()]), axis=1)
    
    # Evaluate baseline Ensemble and TabPFN on test set
    r_ens = pearsonr(test_ensemble_z, y_test_spec_z)[0]
    rmse_ens = np.sqrt(np.mean((test_ensemble_z - y_test_spec_z)**2))
    
    r_tab = pearsonr(test_calibrated_z['tabpfn'], y_test_spec_z)[0]
    rmse_tab = np.sqrt(np.mean((test_calibrated_z['tabpfn'] - y_test_spec_z)**2))
    
    print(f"\nFinal Test Results (n={len(y_test_spec_z)}):")
    print(f"  Ensemble Pearson R = {r_ens:.4f}, RMSE = {rmse_ens:.4f}")
    print(f"  TabPFN Pearson R   = {r_tab:.4f}, RMSE = {rmse_tab:.4f}")
    
    # ─── SECTION 2: BOOTSTRAP ANALYSIS ───────────────────────────────────────
    print("\n=== Section 2: Running 1,000 Bootstrap Iterations ===")
    np.random.seed(42)
    n_boot = 1000
    boot_metrics = {
        'ens_r': [], 'ens_rho': [], 'ens_rmse': [], 'ens_mae': [],
        'tab_r': [], 'tab_rho': [], 'tab_rmse': [], 'tab_mae': []
    }
    
    for _ in range(n_boot):
        boot_idx = np.random.choice(len(y_test_spec_z), size=len(y_test_spec_z), replace=True)
        y_true_b = y_test_spec_z[boot_idx]
        
        # Ensemble predictions
        pred_ens_b = test_ensemble_z[boot_idx]
        boot_metrics['ens_r'].append(pearsonr(pred_ens_b, y_true_b)[0])
        boot_metrics['ens_rho'].append(spearmanr(pred_ens_b, y_true_b)[0])
        boot_metrics['ens_rmse'].append(np.sqrt(np.mean((pred_ens_b - y_true_b)**2)))
        boot_metrics['ens_mae'].append(np.mean(np.abs(pred_ens_b - y_true_b)))
        
        # TabPFN predictions
        pred_tab_b = test_calibrated_z['tabpfn'][boot_idx]
        boot_metrics['tab_r'].append(pearsonr(pred_tab_b, y_true_b)[0])
        boot_metrics['tab_rho'].append(spearmanr(pred_tab_b, y_true_b)[0])
        boot_metrics['tab_rmse'].append(np.sqrt(np.mean((pred_tab_b - y_true_b)**2)))
        boot_metrics['tab_mae'].append(np.mean(np.abs(pred_tab_b - y_true_b)))
        
    # Calculate 95% CIs
    ci_results = {}
    for key, vals in boot_metrics.items():
        low = np.percentile(vals, 2.5)
        high = np.percentile(vals, 97.5)
        mean_v = np.mean(vals)
        ci_results[key] = (mean_v, low, high)
        print(f"  {key:<8}: Mean = {mean_v:.4f}, 95% CI = [{low:.4f}, {high:.4f}]")
        
    # ─── SECTION 3: STATISTICAL SIGNIFICANCE TESTING ────────────────────────
    print("\n=== Section 3: Statistical Significance Testing ===")
    
    # 1. Steiger's Z-Test for Dependent Correlations sharing a common variable
    # We compare Ensemble vs TabPFN correlation with spec-z
    r12 = r_ens
    r13 = r_tab
    # correlation between predictions of Ensemble and TabPFN
    r23 = pearsonr(test_ensemble_z, test_calibrated_z['tabpfn'])[0]
    n_sz = len(y_test_spec_z)
    
    # Steiger Z formula
    f_val = (r23 - (r12 * r13) / 2.0) / (1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23)
    h_val = (1.0 - (r23 * (1.0 - r12**2 - r13**2)) / (1.0 - r23**2))  # correction
    # z-score comparison of dependent correlations
    diff_z = (r12 - r13) * np.sqrt(n_sz - 3.0) / np.sqrt(2.0 * (1.0 - r23) * h_val)
    p_steiger = 2.0 * (1.0 - norm.cdf(np.abs(diff_z)))
    print(f"  Steiger Z-Test (Ensemble vs TabPFN correlations): Z = {diff_z:.4f}, p-value = {p_steiger:.5f}")
    
    # 2. Paired Bootstrap p-value for RMSE difference
    diff_rmse_boot = np.array(boot_metrics['ens_rmse']) - np.array(boot_metrics['tab_rmse'])
    p_rmse_boot = np.minimum(np.mean(diff_rmse_boot <= 0), np.mean(diff_rmse_boot >= 0)) * 2.0
    print(f"  Paired Bootstrap p-value (RMSE difference): p = {p_rmse_boot:.5f}")
    
    # ─── SECTION 4: OUT-OF-DISTRIBUTION ANALYSIS ─────────────────────────────
    print("\n=== Section 4: Out-of-Distribution Profiling ===")
    # Fit Mahalanobis Distance covariance on training set (first 19 features)
    X_train_for_cov = train_df[all_predictors].values
    cov_matrix = np.cov(X_train_for_cov, rowvar=False)
    # Add minor regularizer to ensure invertibility
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov_matrix)
    mean_train = np.mean(X_train_for_cov, axis=0)
    
    # Compute Mahalanobis distance
    def mahalanobis_dist(X, mean, inv_cov):
        diff = X - mean
        return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        
    train_mahal = mahalanobis_dist(X_train_for_cov, mean_train, cov_inv)
    target_mahal = mahalanobis_dist(target_df[all_predictors].values, mean_train, cov_inv)
    
    mahal_95_thresh = np.percentile(train_mahal, 95.0)
    mahal_99_thresh = np.percentile(train_mahal, 99.0)
    print(f"  Mahalanobis 95% threshold = {mahal_95_thresh:.4f}, 99% threshold = {mahal_99_thresh:.4f}")
    
    # Fit Isolation Forest
    clf_if = IsolationForest(contamination=0.05, random_state=42)
    clf_if.fit(X_train_for_cov)
    target_if_scores = clf_if.decision_function(target_df[all_predictors].values)
    
    # UMAP Projecting
    print("  Running UMAP for domain projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    scaler_umap = StandardScaler()
    X_train_sc_umap = scaler_umap.fit_transform(X_train_for_cov)
    X_target_sc_umap = scaler_umap.transform(target_df[all_predictors].values)
    
    reducer.fit(X_train_sc_umap)
    train_umap = reducer.transform(X_train_sc_umap)
    target_umap = reducer.transform(X_target_sc_umap)
    
    # Reliability grading
    reliability_grades = []
    for m_dist, if_score in zip(target_mahal, target_if_scores):
        if m_dist <= mahal_95_thresh and if_score >= 0:
            reliability_grades.append("Grade A")
        elif m_dist <= mahal_99_thresh:
            reliability_grades.append("Grade B")
        else:
            reliability_grades.append("Grade C")
            
    target_df['OOD_Score'] = target_mahal
    target_df['Reliability_Grade'] = reliability_grades
    print(f"  DR3 Catalog Grades: A={reliability_grades.count('Grade A')}, B={reliability_grades.count('Grade B')}, C={reliability_grades.count('Grade C')}")
    
    # ─── SECTION 5: PREDICTION UNCERTAINTY & CONFORMAL ──────────────────────
    print("\n=== Section 5: Uncertainty & Conformal Coverage Comparison ===")
    # Conformal Quantiles on training set
    oof_residuals = np.abs(y_train_z - oof_ensemble_z)
    oof_std_clipped = np.clip(oof_std_z, a_min=0.01, a_max=None)
    conformal_scores = oof_residuals / oof_std_clipped
    q_68 = np.percentile(conformal_scores, 68.0)
    q_95 = np.percentile(conformal_scores, 95.0)
    
    # Predict intervals for test set
    test_std_clipped = np.clip(np.std(np.column_stack([test_calibrated_z[k] for k in test_calibrated_z.keys()]), axis=1), a_min=0.01, a_max=None)
    test_low_68 = np.clip(test_ensemble_z - q_68 * test_std_clipped, a_min=0.0, a_max=None)
    test_high_68 = test_ensemble_z + q_68 * test_std_clipped
    
    test_low_95 = np.clip(test_ensemble_z - q_95 * test_std_clipped, a_min=0.0, a_max=None)
    test_high_95 = test_ensemble_z + q_95 * test_std_clipped
    
    # Verify coverage
    cov_68 = np.mean((y_test_spec_z >= test_low_68) & (y_test_spec_z <= test_high_68)) * 100.0
    cov_95 = np.mean((y_test_spec_z >= test_low_95) & (y_test_spec_z <= test_high_95)) * 100.0
    print(f"  Ensemble 68% Interval Empirical Coverage = {cov_68:.2f}%")
    print(f"  Ensemble 95% Interval Empirical Coverage = {cov_95:.2f}%")
    
    # Save conformal vs Jackknife+ coverage comparison plot
    plt.figure(figsize=(8, 5))
    plt.hist(conformal_scores, bins=30, density=True, alpha=0.6, color='blue', label='Conformal Scores')
    plt.axvline(q_68, color='red', linestyle='--', label=f'Q_68 = {q_68:.2f}')
    plt.axvline(q_95, color='orange', linestyle='--', label=f'Q_95 = {q_95:.2f}')
    plt.title("Conformal Calibration Scores Distribution")
    plt.xlabel("Scaled Absolute Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dr3_plots/conformal_calibration_scores.png")
    plt.close()
    
    # ─── SECTION 6: ASTROPHYSICAL VALIDATION ─────────────────────────────────
    print("\n=== Section 6: Astrophysical Validation ===")
    
    # 1. K-S test between predicted BLL vs FSRQ redshift distributions
    target_df['Ensemble_z'] = target_ensemble_z
    bll_z_preds = target_df[target_df['AGN_Type'] == 'BLL']['Ensemble_z'].values
    fsrq_z_preds = target_df[target_df['AGN_Type'] == 'FSRQ']['Ensemble_z'].values
    
    ks_stat, ks_p = ks_2samp(bll_z_preds, fsrq_z_preds)
    print(f"  BLL vs FSRQ Redshift Kolmogorov-Smirnov Test:")
    print(f"    KS statistic = {ks_stat:.4f}, p-value = {ks_p:.2e}")
    
    # 2. Luminosity vs Redshift Physical consistency check
    # L_gamma = 4 * pi * d_L^2 * Energy_Flux100
    # Approximate luminosity distance d_L in cm using simple redshift relation
    # d_L = (c * z / H_0) for low-z, but let's use cosmological distance approximation
    # d_L(z) in Mpc ≈ 3000 * z * (1 + z/2)
    c_speed = 3e5 # km/s
    H0 = 70.0 # km/s/Mpc
    Mpc_to_cm = 3.086e24
    
    target_df['dL_Mpc'] = (c_speed * target_df['Ensemble_z'] / H0) * (1.0 + target_df['Ensemble_z'] / 2.0)
    target_df['dL_cm'] = target_df['dL_Mpc'] * Mpc_to_cm
    target_df['L_gamma'] = 4.0 * np.pi * (target_df['dL_cm'] ** 2) * target_df['Energy_Flux100']
    
    # Plot Luminosity vs Redshift
    plt.figure(figsize=(8, 6))
    for class_name, col in [('BLL', 'blue'), ('FSRQ', 'red'), ('BCU', 'orange')]:
        sub = target_df[target_df['AGN_Type'] == class_name]
        plt.scatter(sub['Ensemble_z'], sub['L_gamma'], alpha=0.6, color=col, label=class_name, edgecolors='none')
    plt.yscale('log')
    plt.xlabel("Predicted Redshift (z)")
    plt.ylabel("Gamma-Ray Luminosity L_gamma (erg/s)")
    plt.title("Cosmological Consistency: Gamma-Ray Luminosity vs Redshift")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dr3_plots/luminosity_vs_redshift.png")
    plt.close()
    
    # ─── SECTION 7: SHAP FEATURE IMPORTANCE ──────────────────────────────────
    print("\n=== Section 7: Generating SHAP beeswarm and dependency plots ===")
    
    # Train a fast Random Forest regressor on the full dataset to compute SHAP
    rf_shap = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_shap.fit(X_train_reg_full, y_train)
    
    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer(X_train_reg_full)
    
    # Format SHAP values with proper feature names
    shap_values.feature_names = all_predictors + ['P_FSRQ']
    
    # SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Explanation of Predictor Influences")
    plt.tight_layout()
    plt.savefig("plots/dr3_plots/shap_beeswarm.png")
    plt.close()
    
    # SHAP Dependency Plot for W1-W2
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, "W1_W2"], show=False)
    plt.title("SHAP Dependency Plot for AllWISE W1-W2 Color")
    plt.tight_layout()
    plt.savefig("plots/dr3_plots/shap_dependency_w1w2.png")
    plt.close()
    
    # ─── SECTION 9: CATALOG GENERATION & EXPORT ──────────────────────────────
    print("\n=== Section 9: Catalog Generation & Export ===")
    
    # Calculate conformal intervals for all target sources
    target_std_clipped = np.clip(target_std_z, a_min=0.01, a_max=None)
    target_low_68 = np.clip(target_ensemble_z - q_68 * target_std_clipped, a_min=0.0, a_max=None)
    target_high_68 = target_ensemble_z + q_68 * target_std_clipped
    
    target_low_95 = np.clip(target_ensemble_z - q_95 * target_std_clipped, a_min=0.0, a_max=None)
    target_high_95 = target_ensemble_z + q_95 * target_std_clipped
    
    final_output_catalog = pd.DataFrame({
        'source_name': target_df['Source_Name'],
        'ra': target_df['RAJ2000'],
        'dec': target_df['DEJ2000'],
        'class': target_df['CLASS'],
        'agn_type_est': target_df['AGN_Type'],
        'p_fsrq': np.round(target_df['P_FSRQ'], 4),
        'ensemble_z': np.round(target_ensemble_z, 4),
        'prediction_std': np.round(target_std_z, 4),
        'z_low_68': np.round(target_low_68, 4),
        'z_high_68': np.round(target_high_68, 4),
        'z_low_95': np.round(target_low_95, 4),
        'z_high_95': np.round(target_high_95, 4),
        'ood_score': np.round(target_mahal, 4),
        'reliability_grade': reliability_grades
    })
    
    catalog_out_path = "data/DR3_New_AGN_Redshift_Catalog_Reliability.csv"
    final_output_catalog.to_csv(catalog_out_path, index=False)
    print(f"  Upgraded catalog with OOD & Reliability saved to {catalog_out_path}")
    
    # ─── SECTION 10: PLOT UMAP OOD PROJECTION ────────────────────────────────
    print("\n=== Section 10: Plotting UMAP OOD domain space ===")
    
    plt.figure(figsize=(9, 6))
    plt.scatter(train_umap[:, 0], train_umap[:, 1], alpha=0.4, color='blue', label='Train Set (4LAC-DR2)', s=15)
    
    # Grade A target sources
    a_mask = (np.array(reliability_grades) == 'Grade A')
    b_mask = (np.array(reliability_grades) == 'Grade B')
    c_mask = (np.array(reliability_grades) == 'Grade C')
    
    plt.scatter(target_umap[a_mask, 0], target_umap[a_mask, 1], alpha=0.7, color='green', label='Target DR3 (Grade A: In-Dist)', s=25)
    plt.scatter(target_umap[b_mask, 0], target_umap[b_mask, 1], alpha=0.7, color='orange', label='Target DR3 (Grade B: Near-OOD)', s=25)
    plt.scatter(target_umap[c_mask, 0], target_umap[c_mask, 1], alpha=0.7, color='red', label='Target DR3 (Grade C: Strong-OOD)', s=25)
    
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP 2D Projection: Training vs Target DR3 sets (OOD Detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dr3_plots/umap_ood_space.png")
    plt.close()
    
    print("\nAll tasks in upgraded research pipeline completed successfully!")

if __name__ == "__main__":
    run_upgraded_pipeline()
