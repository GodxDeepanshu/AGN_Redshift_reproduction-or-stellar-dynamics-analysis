import numpy as np
import pandas as pd
import os
import sys
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wilcoxon, ttest_rel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

# Insert path for helper modules
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics
from tabpfn import TabPFNRegressor

# Create output directories
os.makedirs("plots/dr3_plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

def run_pipeline():
    print("======================================================================")
    print("  RUNNING ROBUST REFINED PHOTOMETRIC REDSHIFT PIPELINE")
    print("======================================================================")
    
    # ─── 1. DATA LOADING & COORDINATE DE-DUPLICATION ──────────────────────────
    print("\n--- Phase 1: Data Loading & Spatial De-Duplication ---")
    train_raw = pd.read_csv("data/training_eligible.csv")
    mw = pd.read_csv("data/multi_wavelength_matches.csv")
    mw['Source_Name'] = mw['Source_Name'].astype(str).str.strip()
    
    # Merge MW columns to train
    train_df = pd.merge(train_raw, mw, on="Source_Name", how="left")
    
    # Load target DR3 catalog
    with fits.open("data/table-4LAC-DR3-h.fits") as h_hdul:
        dr3_h = pd.DataFrame(h_hdul[1].data)
    with fits.open("data/table-4LAC-DR3-l.fits") as l_hdul:
        dr3_l = pd.DataFrame(l_hdul[1].data)
    dr3 = pd.concat([dr3_h, dr3_l], ignore_index=True)
    for col in dr3.columns:
        if dr3[col].dtype == object:
            dr3[col] = dr3[col].astype(str).str.strip()
    dr3['Source_Name'] = dr3['Source_Name'].astype(str).str.strip()
            
    # Set difference to isolate new sources
    dr2 = pd.read_csv("data/4LAC-DR2.csv")
    dr2_names = set(dr2['Source_Name'].astype(str).str.strip())
    new_names = set(dr3['Source_Name']) - dr2_names
    target_raw = dr3[dr3['Source_Name'].isin(new_names)].copy()
    
    # Merge Gaia & MW to target
    gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    target_df = pd.merge(target_raw, gaia_new, on="Source_Name", how="left")
    target_df = pd.merge(target_df, mw, on="Source_Name", how="left")
    
    print(f"Initial training catalog size: {len(train_df)}")
    print(f"Initial target DR3 new sources count: {len(target_df)}")
    
    # Coordinate de-duplication
    train_coords = SkyCoord(ra=train_df['RAJ2000'].values*u.deg, dec=train_df['DEJ2000'].values*u.deg)
    target_coords = SkyCoord(ra=target_df['RAJ2000'].values*u.deg, dec=target_df['DEJ2000'].values*u.deg)
    
    idx, d2d, _ = target_coords.match_to_catalog_sky(train_coords)
    is_duplicate = d2d < (10.0 / 3600.0) * u.deg
    duplicate_count = is_duplicate.sum()
    print(f"Found {duplicate_count} target sources within 10 arcsec of training sources.")
    if duplicate_count > 0:
        target_df = target_df[~is_duplicate].copy()
        print(f"Target catalog size after spatial de-duplication: {len(target_df)}")
    else:
        print("No spatial duplicates found; set-difference names are coordinate-unique.")
        
    # ─── 2. PREPROCESSING & MISSINGNESS HANDLING ────────────────────────────────
    print("\n--- Phase 2: Missingness & Physical Upper Limits Ingestion ---")
    
    name_map = {
        'Flux1000': 'LogFlux',
        'Energy_Flux100': 'LogEnergy_Flux',
        'Signif_Avg': 'LogSignificance',
        'Variability_Index': 'LogVariability_Index',
        'nu_syn': 'Lognu_syn',
        'nuFnu_syn': 'LognuFnu_syn',
        'Pivot_Energy': 'LogPivot_Energy'
    }
    
    # Process numeric columns & apply log transforms
    for df in [train_df, target_df]:
        df['PL_Index'] = pd.to_numeric(df['PL_Index'], errors='coerce')
        df['LP_Index'] = pd.to_numeric(df['LP_Index'], errors='coerce')
        df['LP_beta'] = pd.to_numeric(df['LP_beta'], errors='coerce')
        df['Gaia_G_Magnitude'] = pd.to_numeric(df['Gaia_G_Magnitude'], errors='coerce')
        
        for raw_col, log_col in name_map.items():
            val = pd.to_numeric(df[raw_col], errors='coerce')
            val = np.where(val > 0, val, np.nan)
            df[log_col] = np.log10(val)
        
        df['Redshift'] = pd.to_numeric(df['Redshift'], errors='coerce')
        df['InvRedshift'] = 1.0 / (1.0 + df['Redshift'])
        
        df['W1mag'] = pd.to_numeric(df['W1mag'], errors='coerce')
        df['W2mag'] = pd.to_numeric(df['W2mag'], errors='coerce')
        df['W3mag'] = pd.to_numeric(df['W3mag'], errors='coerce')
        df['W4mag'] = pd.to_numeric(df['W4mag'], errors='coerce')
        df['S1.4'] = pd.to_numeric(df['S1.4'], errors='coerce')
        df['CR0'] = pd.to_numeric(df['CR0'], errors='coerce')
        
        # Hardcode Swift-XRT upper limit first & add indicator
        # Missing X-ray flux is a physical non-detection (flux limit)
        df['Is_Upper_Limit_Xray'] = df['CR0'].isna().astype(float)
        df['LogXrayFlux'] = np.where(df['CR0'].isna(), -4.0, np.log10(df['CR0'].clip(lower=1e-6)))
        
        # Radio flux
        df['LogRadioFlux'] = np.log10(df['S1.4'].clip(lower=1e-5))
        
    # Standard numerical predictors list
    orig_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                     "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                     "LP_beta", "Gaia_G_Magnitude"]
    
    # Features to impute (WISE bands + NVSS radio flux)
    impute_features = orig_features + ["W1mag", "W2mag", "W3mag", "W4mag", "LogRadioFlux"]
    
    # Iterative Imputer (MICE)
    print("Fitting MICE (IterativeImputer) on training set predictors...")
    imputer = IterativeImputer(max_iter=15, random_state=42)
    
    # Fit and transform
    train_df[impute_features] = imputer.fit_transform(train_df[impute_features])
    target_df[impute_features] = imputer.transform(target_df[impute_features])
    
    # Calculate engineered features POST-IMPUTATION to keep mathematical relationships exact
    for df in [train_df, target_df]:
        df['W1_W2'] = df['W1mag'] - df['W2mag']
        df['W2_W3'] = df['W2mag'] - df['W3mag']
        df['W3_W4'] = df['W3mag'] - df['W4mag']
        df['RadioLoudness'] = df['LogRadioFlux'] - df['LogFlux']
        df['XrayToOptical'] = df['LogXrayFlux'] + 0.4 * df['Gaia_G_Magnitude']
        df['IRToRadio'] = -0.4 * df['W1mag'] - df['LogRadioFlux']
        df['IndexDiff'] = df['PL_Index'] - df['LP_Index']
        
    print("Imputation and feature engineering completed.")
    
    # Define full predictor list
    predictors = orig_features + ["W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "W3_W4",
                                  "LogRadioFlux", "LogXrayFlux", "RadioLoudness", "XrayToOptical",
                                  "IRToRadio", "IndexDiff", "Is_Upper_Limit_Xray"]
    
    # ─── 3. CLASSIFICATION & DOMAIN ADAPTATION (DRIW) ──────────────────────────
    print("\n--- Phase 3: Classifier Probability & Density Ratio Importance Weighting ---")
    
    # Class probability model P(FSRQ|X)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_probs = np.zeros(len(train_df))
    X_train_class = train_df[predictors]
    y_train_class = train_df['LabelNo'].fillna(0).astype(int).values
    X_target_class = target_df[predictors]
    
    for train_idx, val_idx in skf.split(X_train_class, y_train_class):
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train_class.iloc[train_idx], y_train_class[train_idx])
        train_probs[val_idx] = clf.predict_proba(X_train_class.iloc[val_idx])[:, 1]
        
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_class, y_train_class)
    target_df['P_FSRQ'] = final_clf.predict_proba(X_target_class)[:, 1]
    train_df['P_FSRQ'] = train_probs
    
    # Include P_FSRQ in predictors list
    predictors = predictors + ['P_FSRQ']
    
    # DRIW: Domain classifier to estimate sample weights
    print("Training domain classifier to estimate density ratio weights...")
    X_all = np.vstack([train_df[predictors].values, target_df[predictors].values])
    y_domain = np.hstack([np.zeros(len(train_df)), np.ones(len(target_df))])
    
    domain_clf = LogisticRegression(C=0.1, random_state=42)
    domain_clf.fit(X_all, y_domain)
    
    p_target_train = domain_clf.predict_proba(train_df[predictors].values)[:, 1]
    weights = p_target_train / (1.0 - p_target_train + 1e-6)
    weights = np.clip(weights, 0.05, 5.0)  # Clip weights to prevent variance explosion
    
    # ─── 4. TARGET VARIANCE STABILIZATION (QuantileTransformer) ─────────────────
    print("\n--- Phase 4: Target Variance Stabilization ---")
    y_train = train_df['InvRedshift'].values
    
    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
    y_train_trans = qt.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # ─── 5. REGRESSOR BASE LEARNERS & NNLS STACKING ──────────────────────────────
    print("\n--- Phase 5: Base Learners & Stacking Level 1 Stacking ---")
    
    X = train_df[predictors].values
    X_target = target_df[predictors].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_target_sc = scaler.transform(X_target)
    
    # Base learners (fitted with sample weights where supported)
    xgb_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    lgb_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    cat_reg = cb.CatBoostRegressor(iterations=300, learning_rate=0.04, depth=4, random_seed=42, thread_count=-1, verbose=0)
    tabpfn_reg = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
    
    # Stacking level 0 CV (fitted to transformed target, with weight replication)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_trans = {
        'xgb': np.zeros(len(train_df)),
        'lgb': np.zeros(len(train_df)),
        'cat': np.zeros(len(train_df)),
        'tabpfn': np.zeros(len(train_df))
    }
    
    print("Running 5-fold CV to generate meta-features...")
    for train_idx, val_idx in cv.split(X):
        # Fit base models on fold training set (using DRIW weights for tree models)
        w_fold = weights[train_idx]
        
        xgb_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        lgb_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        cat_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        
        # TabPFN is a prior-fitted network; no weights supported
        X_tr_sc = scaler.fit_transform(X[train_idx])
        X_va_sc = scaler.transform(X[val_idx])
        tabpfn_reg.fit(X_tr_sc, y_train_trans[train_idx])
        
        # Predict on fold validation set
        oof_preds_trans['xgb'][val_idx] = xgb_reg.predict(X[val_idx])
        oof_preds_trans['lgb'][val_idx] = lgb_reg.predict(X[val_idx])
        oof_preds_trans['cat'][val_idx] = cat_reg.predict(X[val_idx])
        oof_preds_trans['tabpfn'][val_idx] = tabpfn_reg.predict(X_va_sc)
        
    # Inverse transform OOF predictions back to inverse redshift scale
    oof_preds_inv = {}
    for key in oof_preds_trans.keys():
        oof_preds_inv[key] = qt.inverse_transform(oof_preds_trans[key].reshape(-1, 1)).flatten()
        
    # NNLS Stacking Meta-Learner (Non-Negative Least Squares)
    X_level1 = np.column_stack([oof_preds_inv['xgb'], oof_preds_inv['lgb'], oof_preds_inv['cat'], oof_preds_inv['tabpfn']])
    meta_model = LinearRegression(positive=True)
    meta_model.fit(X_level1, y_train)
    
    print("\nConstrained NNLS Meta-Learner Weights:")
    print(f"  XGBoost:      {meta_model.coef_[0]:.4f}")
    print(f"  LightGBM:     {meta_model.coef_[1]:.4f}")
    print(f"  CatBoost:     {meta_model.coef_[2]:.4f}")
    print(f"  TabPFN:       {meta_model.coef_[3]:.4f}")
    print(f"  Intercept:     {meta_model.intercept_:.4f}")
    
    # Train final models on entire training set
    xgb_reg.fit(X, y_train_trans, sample_weight=weights)
    lgb_reg.fit(X, y_train_trans, sample_weight=weights)
    cat_reg.fit(X, y_train_trans, sample_weight=weights)
    tabpfn_reg.fit(X_sc, y_train_trans)
    
    # Predict on target set (and inverse transform to inverse redshift)
    pred_trans = {
        'xgb': xgb_reg.predict(X_target),
        'lgb': lgb_reg.predict(X_target),
        'cat': cat_reg.predict(X_target),
        'tabpfn': tabpfn_reg.predict(X_target_sc)
    }
    
    pred_inv = {}
    for key in pred_trans.keys():
        pred_inv[key] = qt.inverse_transform(pred_trans[key].reshape(-1, 1)).flatten()
        
    # Meta prediction for target
    X_level1_target = np.column_stack([pred_inv['xgb'], pred_inv['lgb'], pred_inv['cat'], pred_inv['tabpfn']])
    target_df['pred_stack_inv'] = meta_model.predict(X_level1_target)
    
    # Also save individual model predictions
    for key in pred_inv.keys():
        target_df[f'pred_{key}_inv'] = pred_inv[key]
        
    # ─── 6. BIAS CALIBRATION (ISOTONIC) & CONFORMAL PREDICTION ─────────────────
    print("\n--- Phase 6: Calibration & Conformal Prediction ---")
    
    # Fit class-specific Isotonic regression
    oof_stack_inv = meta_model.predict(X_level1)
    
    bll_mask = (train_df['LabelNo'].values == 0)
    fsrq_mask = (train_df['LabelNo'].values == 1)
    
    iso_bll = IsotonicRegression(out_of_bounds='clip').fit(oof_stack_inv[bll_mask], y_train[bll_mask])
    iso_fsrq = IsotonicRegression(out_of_bounds='clip').fit(oof_stack_inv[fsrq_mask], y_train[fsrq_mask])
    
    # Calibrate out-of-fold predictions
    oof_bll_c = iso_bll.predict(oof_stack_inv)
    oof_fsrq_c = iso_fsrq.predict(oof_stack_inv)
    p_train = train_df['P_FSRQ'].values
    oof_corr_inv = (1.0 - p_train) * oof_bll_c + p_train * oof_fsrq_c
    oof_ensemble_z = (1.0 / oof_corr_inv) - 1.0
    
    # Calibrate target predictions
    p_target = target_df['P_FSRQ'].values
    new_bll_c = iso_bll.predict(target_df['pred_stack_inv'].values)
    new_fsrq_c = iso_fsrq.predict(target_df['pred_stack_inv'].values)
    new_corr_inv = (1.0 - p_target) * new_bll_c + p_target * new_fsrq_c
    target_df['ensemble_z'] = (1.0 / new_corr_inv) - 1.0
    
    # Locally Weighted Conformal Prediction
    # Epistemic uncertainty = standard deviation among base models
    pred_inv_matrix = np.column_stack([pred_inv['xgb'], pred_inv['lgb'], pred_inv['cat'], pred_inv['tabpfn']])
    pred_z_matrix = 1.0 / pred_inv_matrix - 1.0
    std_z = np.std(pred_z_matrix, axis=1)
    target_df['ensemble_var_std'] = std_z
    
    oof_inv_matrix = np.column_stack([oof_preds_inv['xgb'], oof_preds_inv['lgb'], oof_preds_inv['cat'], oof_preds_inv['tabpfn']])
    oof_z_matrix = 1.0 / oof_inv_matrix - 1.0
    oof_std_z = np.std(oof_z_matrix, axis=1)
    
    # Compute scaled residuals
    residuals_z = np.abs(train_df['Redshift'].values - oof_ensemble_z)
    oof_std_z_clipped = np.clip(oof_std_z, a_min=0.01, a_max=None)
    conformal_scores = residuals_z / oof_std_z_clipped
    
    q_68 = np.percentile(conformal_scores, 68.0)
    q_95 = np.percentile(conformal_scores, 95.0)
    print(f"Conformal Quantiles: Q_68 = {q_68:.4f}, Q_95 = {q_95:.4f}")
    
    # ─── 7. INDEPENDENT TEST SET EVALUATION ─────────────────────────────────────
    print("\n--- Phase 7: Spectroscopic Test Set Performance (n=89) ---")
    spec_sources = target_df[target_df['Redshift'] > 0].copy()
    y_test_spec_z = spec_sources['Redshift'].values
    
    # Stacking ensemble predictions
    pred_stack_z = spec_sources['ensemble_z'].values
    r_val, _ = pearsonr(pred_stack_z, y_test_spec_z)
    rmse_val = np.sqrt(np.mean((pred_stack_z - y_test_spec_z)**2))
    print(f"Robust Stacking Ensemble | Pearson R = {r_val:.4f} | RMSE = {rmse_val:.4f}")
    
    # Standalone TabPFN
    tabpfn_spec_z = 1.0 / spec_sources['pred_tabpfn_inv'].values - 1.0
    r_tabpfn, _ = pearsonr(tabpfn_spec_z, y_test_spec_z)
    rmse_tabpfn = np.sqrt(np.mean((tabpfn_spec_z - y_test_spec_z)**2))
    print(f"Robust TabPFN            | Pearson R = {r_tabpfn:.4f} | RMSE = {rmse_tabpfn:.4f}")
    
    # Compare significance against baseline (pred_xgb_inv from older run)
    # Pairwise significance tests
    pred_xgb_z = 1.0 / spec_sources['pred_xgb_inv'].values - 1.0
    res_stack = np.abs(pred_stack_z - y_test_spec_z)
    res_base = np.abs(pred_xgb_z - y_test_spec_z)
    
    stat_wilc, p_wilc = wilcoxon(res_stack, res_base)
    stat_t, p_t = ttest_rel(res_stack, res_base)
    print(f"Wilcoxon signed-rank p-value: {p_wilc:.4e}")
    print(f"Paired t-test p-value:         {p_t:.4e}")
    
    # ─── 8. SAVING OUTPUT FILES ─────────────────────────────────────────────────
    print("\n--- Phase 8: Saving Refined Catalogs ---")
    
    # Assign reliability grades A, B, C, D
    std_z = target_df['ensemble_var_std'].values
    
    # We calculate the combined OOD score using Mahalanobis on predictors
    # Fit Mahalanobis on training predictors
    scaler_ood = StandardScaler()
    X_train_sc_ood = scaler_ood.fit_transform(train_df[predictors].values)
    X_target_sc_ood = scaler_ood.transform(target_df[predictors].values)
    
    mean_tr = np.mean(X_train_sc_ood, axis=0)
    cov_tr = np.cov(X_train_sc_ood, rowvar=False)
    inv_cov_tr = np.linalg.pinv(cov_tr)
    
    def mahalanobis_distance(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        
    mah_target = np.array([mahalanobis_distance(x, mean_tr, inv_cov_tr) for x in X_target_sc_ood])
    target_df['Mahalanobis_Dist'] = mah_target
    
    # Normalize score
    mah_min, mah_max = mah_target.min(), mah_target.max()
    target_df['OOD_Score'] = (mah_target - mah_min) / (mah_max - mah_min + 1e-6)
    
    conf_width = 2.0 * q_68 * std_z
    grades = []
    for idx in range(len(target_df)):
        o_s = target_df['OOD_Score'].values[idx]
        s_z = std_z[idx]
        w_c = conf_width[idx]
        
        if o_s < 0.3 and s_z < 0.1 and w_c < 0.3:
            grades.append('Grade A')
        elif o_s < 0.5 and s_z < 0.2 and w_c < 0.6:
            grades.append('Grade B')
        elif o_s < 0.7 and s_z < 0.4 and w_c < 1.0:
            grades.append('Grade C')
        else:
            grades.append('Grade D')
            
    target_df['Reliability_Grade'] = grades
    
    # Estimate AGN Type
    est_types = []
    for idx, row in target_df.iterrows():
        cls = str(row['CLASS']).upper().strip()
        if cls == 'BCU':
            est_types.append('FSRQ' if row['P_FSRQ'] >= 0.5 else 'BLL')
        else:
            est_types.append(cls)
    target_df['AGN_Type_Est'] = est_types
    
    # Conformal boundaries
    z_low = np.clip(target_df['ensemble_z'].values - q_68 * std_z, a_min=0, a_max=None)
    z_high = target_df['ensemble_z'].values + q_68 * std_z
    
    reliability_catalog = pd.DataFrame({
        'source_name': target_df['Source_Name'],
        'ra': target_df['RAJ2000'],
        'dec': target_df['DEJ2000'],
        'agn_class': target_df['CLASS'],
        'agn_type_est': target_df['AGN_Type_Est'],
        'p_fsrq': np.round(target_df['P_FSRQ'], 4),
        'ensemble_z': np.round(target_df['ensemble_z'], 4),
        'conformal_z_low': np.round(z_low, 4),
        'conformal_z_high': np.round(z_high, 4),
        'prediction_std': np.round(std_z, 4),
        'conformal_68_width': np.round(conf_width, 4),
        'ood_score': np.round(target_df['OOD_Score'], 4),
        'reliability_grade': target_df['Reliability_Grade']
    }).sort_values(by='ensemble_z', ascending=False)
    
    reliability_catalog.to_csv("data/DR3_New_AGN_Redshift_Catalog_Reliability.csv", index=False)
    print("Saved refined reliability catalog to data/DR3_New_AGN_Redshift_Catalog_Reliability.csv")
    print("\nRobust improvements execution finished successfully!")

if __name__ == '__main__':
    run_pipeline()
