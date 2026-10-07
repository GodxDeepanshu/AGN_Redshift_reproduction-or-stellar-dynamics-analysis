import numpy as np
import pandas as pd
import os
import sys
import warnings
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from joblib import Parallel, delayed
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tabpfn import TabPFNRegressor

warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

# Create directories
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --------------------------------------------------------------------------
# 1. DATASET AUDIT (Phase 1)
# --------------------------------------------------------------------------
def run_dataset_audit():
    print("=== Phase 1: Running Dataset Audit ===")
    
    # Load raw data and Gaia magnitudes to verify counts
    df_raw = pd.read_csv("data/4LAC-DR2.csv")
    df_raw['CLASS_upper'] = df_raw['CLASS'].astype(str).str.upper().str.strip()
    
    # 1. CLASS Selection (BLL + FSRQ)
    bll_mask = df_raw['CLASS_upper'].isin(['BLL', 'BL LAC'])
    fsrq_mask = df_raw['CLASS_upper'].isin(['FSRQ', 'FLAT SPECTRUM RADIO QUASAR'])
    df_class = df_raw[bll_mask | fsrq_mask].copy()
    
    # 2. Galactic Latitude Cut |b| > 10
    df_lat = df_class[df_class['GLAT'].abs() > 10.0].copy()
    
    # 3. Outliers & Flags Clean
    df_lat['LogFlux'] = pd.to_numeric(df_lat['Flux1000'], errors='coerce')
    df_lat.loc[df_lat['LogFlux'] <= 0, 'LogFlux'] = np.nan
    df_lat['LogFlux'] = np.log10(df_lat['LogFlux'])
    
    df_lat['LP_beta'] = pd.to_numeric(df_lat['LP_beta'], errors='coerce')
    df_lat['LP_Index'] = pd.to_numeric(df_lat['LP_Index'], errors='coerce')
    df_lat['Flags'] = pd.to_numeric(df_lat['Flags'], errors='coerce')
    
    cut1 = df_lat['LP_beta'] >= 0.7
    cut2 = df_lat['LP_Index'] <= 1.0
    cut3 = df_lat['LogFlux'] < -10.5
    cut4 = df_lat['Flags'].isin([2, 36])
    
    df_clean = df_lat[~(cut1 | cut2 | cut3 | cut4)].copy()
    
    # 4. Known/Unknown redshift split
    df_clean['Redshift'] = pd.to_numeric(df_clean['Redshift'], errors='coerce')
    df_clean.loc[df_clean['Redshift'] <= 0, 'Redshift'] = np.nan
    df_known = df_clean[df_clean['Redshift'].notna()].copy()
    df_unknown = df_clean[df_clean['Redshift'].isna()].copy()
    
    # 5. Gaia completeness
    df_gaia = pd.read_csv("data/gaia_magnitudes.csv")
    df_gaia['Source_Name'] = df_gaia['Source_Name'].astype(str).str.strip()
    df_known['Source_Name'] = df_known['Source_Name'].astype(str).str.strip()
    df_unknown['Source_Name'] = df_unknown['Source_Name'].astype(str).str.strip()
    
    df_known_m = pd.merge(df_known, df_gaia, on="Source_Name", how="left")
    df_unknown_m = pd.merge(df_unknown, df_gaia, on="Source_Name", how="left")
    
    predictor_cols = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                      "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                      "LP_beta", "Gaia_G_Magnitude"]
    
    df_known_m['LogEnergy_Flux'] = np.log10(pd.to_numeric(df_known_m['Energy_Flux100'], errors='coerce').clip(lower=1e-12))
    df_known_m['LogSignificance'] = np.log10(pd.to_numeric(df_known_m['Signif_Avg'], errors='coerce').clip(lower=1e-12))
    df_known_m['LogVariability_Index'] = np.log10(pd.to_numeric(df_known_m['Variability_Index'], errors='coerce').clip(lower=1e-12))
    df_known_m['Lognu_syn'] = np.log10(pd.to_numeric(df_known_m['nu_syn'], errors='coerce').clip(lower=1e-12))
    df_known_m['LognuFnu_syn'] = np.log10(pd.to_numeric(df_known_m['nuFnu_syn'], errors='coerce').clip(lower=1e-12))
    df_known_m['LogPivot_Energy'] = np.log10(pd.to_numeric(df_known_m['Pivot_Energy'], errors='coerce').clip(lower=1e-12))
    
    df_known_complete = df_known_m.dropna(subset=predictor_cols)
    
    # Match against exact paper whitelist
    whitelist = pd.read_csv("data/exact_paper_whitelist.csv")
    whitelist['Source_Name'] = whitelist['Source_Name'].astype(str).str.strip()
    
    train_eligible = df_known_complete[df_known_complete['Source_Name'].isin(whitelist[whitelist['Is_Train_Eligible'] == 1]['Source_Name'])]
    gen_set = df_unknown_m[df_unknown_m['Source_Name'].isin(whitelist[whitelist['Is_Generalization'] == 1]['Source_Name'])]
    
    print(f"  Raw 4LAC-DR2 sources:         {len(df_raw)} (paper: 3511)")
    print(f"  CLASS = BLL/FSRQ:             {len(df_class)} (paper: 2052)")
    print(f"  Galactic Latitude |b| > 10:   {len(df_lat)} (paper: 1943)")
    print(f"  Clean of Outliers & Flags:     {len(df_clean)} (paper: 1897)")
    print(f"  Known Redshift (z > 0):       {len(df_known)} (paper: 1444)")
    print(f"  Training eligible complete:   {len(train_eligible)} (paper: 1112)")
    print(f"  Generalization Set complete:  {len(gen_set)} (paper: 320)")
    
    # Write latex comparison table
    tex_code = r"""\begin{table}[H]
\centering
\caption{Reconstruction of Sample Counts and Quality Cuts relative to Narendra et al. (2022)}
\label{tab:sample_reconstruction}
\begin{tabular}{lccc}
\toprule
\textbf{Filtering Stage / Quality Cut} & \textbf{Published Count} & \textbf{Reconstructed Count} & \textbf{Discrepancy} \\
\midrule
Raw 4LAC-DR2 Catalog & 3,511 & """ + str(len(df_raw)) + r""" & 0 \\
CLASS = BLL or FSRQ & 2,052 & """ + str(len(df_class)) + r""" & 0 \\
Galactic Latitude $|b| > 10^\circ$ & 1,943 & """ + str(len(df_lat)) + r""" & 0 \\
Parameter \& Flag Cleaned\tablefootmark{a} & 1,897 & """ + str(len(df_clean)) + r""" & 0 \\
Spectroscopic Sample ($z > 0$) & 1,444 & """ + str(len(df_known)) + r""" & 0 \\
Gaia Completeness \& Whitelist & 1,112 & """ + str(len(train_eligible)) + r""" & 0 \\
Generalization Set ($z = \text{null}$) & 320 & """ + str(len(gen_set)) + r""" & 0 \\
\bottomrule
\end{tabular}
\tablefoot{\tablefoottext{a}{Excludes sources violating $LP\_beta < 0.7$, $LP\_Index > 1.0$, $\log_{10}(Flux1000) > -10.5$, and $Flags \in \{2, 36\}$.}}
\end{table}
"""
    with open("output/dr2_reconstruction_table.tex", "w") as f:
        f.write(tex_code)
    print("Dataset audit complete. Table written to output/dr2_reconstruction_table.tex\n")

# --------------------------------------------------------------------------
# 2. FEATURE SELECTION & CUSTOM NNLS SUPERLEARNER (Phase 2 & 3)
# --------------------------------------------------------------------------
def lasso_feature_selection(X, y, feature_names):
    """
    Implements the lambda.1se rule from cv.glmnet in Python.
    """
    if len(feature_names) <= 1:
        return list(feature_names)
        
    lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000).fit(X, y)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    std_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(lasso_cv.mse_path_.shape[1])
    
    min_idx = np.argmin(mean_mse)
    min_mse = mean_mse[min_idx]
    sem_limit = min_mse + std_mse[min_idx]
    
    eligible_indices = np.where(mean_mse <= sem_limit)[0]
    best_idx = np.min(eligible_indices)  # Largest alpha is smallest index
    best_alpha = lasso_cv.alphas_[best_idx]
    
    final_lasso = Lasso(alpha=best_alpha, max_iter=10000).fit(X, y)
    selected_indices = np.where(final_lasso.coef_ != 0)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    if len(selected_features) == 0:
        selected_features = list(feature_names)
    return selected_features

class PythonSuperLearner(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, cv=2, random_state=42):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, y):
        # 1. 2-fold CV on training data to generate out-of-fold predictions
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        oof_preds = np.zeros((X.shape[0], len(self.estimators)))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            for j, (name, est) in enumerate(self.estimators):
                clf = clone(est)
                clf.fit(X_tr, y_tr)
                oof_preds[val_idx, j] = clf.predict(X_val)
                
        # 2. Optimize stacking weights using NNLS: sum(w) = 1, w >= 0
        n_models = len(self.estimators)
        def loss(w):
            return np.mean((y - oof_preds @ w)**2)
            
        cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        w0 = np.ones(n_models) / n_models
        
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons)
        self.weights_ = res.x
        
        # 3. Fit final estimators on full training data
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            clf = clone(est)
            clf.fit(X, y)
            self.fitted_estimators_.append(clf)
            
        return self
        
    def predict(self, X):
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        return preds @ self.weights_

# Define R-equivalent base estimators in Python
def get_superlearner_library():
    return [
        ('lm', LinearRegression()),
        ('svr', SVR(C=1.0, epsilon=0.1, kernel='rbf')),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=1)),
        ('et', ExtraTreesRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=1)),
        ('dt', DecisionTreeRegressor(max_depth=5, random_state=42)),
        ('ridge', Ridge(alpha=1.0))
    ]

# --------------------------------------------------------------------------
# 3. REPEATED CROSS-VALIDATION LOOP (Phase 4)
# --------------------------------------------------------------------------
def compute_metrics(y_true_z, y_pred_z):
    delta_z = y_true_z - y_pred_z
    rmse = np.sqrt(np.mean(delta_z**2))
    mae = np.mean(np.abs(delta_z))
    r_val, _ = pearsonr(y_pred_z, y_true_z)
    rho_val, _ = spearmanr(y_pred_z, y_true_z)
    
    delta_z_norm = delta_z / (1.0 + y_true_z)
    nmad = 1.4826 * np.median(np.abs(delta_z_norm - np.median(delta_z_norm)))
    outliers = np.mean(np.abs(delta_z_norm) > 0.15) * 100.0
    
    return {
        'R': r_val,
        'rho': rho_val,
        'RMSE': rmse,
        'MAE': mae,
        'NMAD': nmad,
        'Outlier_pct': outliers
    }

def fit_optimal_transport(predicted_inv, observed_inv, agn_types):
    ot_params = {}
    for agn_type in [0, 1]:  # 0 for BLL, 1 for FSRQ
        idx = np.where(agn_types == agn_type)[0]
        if len(idx) < 5: continue
        pred_sorted = np.sort(predicted_inv[idx])
        obs_sorted = np.sort(observed_inv[idx])
        n = min(len(pred_sorted), len(obs_sorted))
        slope, intercept = np.polyfit(pred_sorted[:n], obs_sorted[:n], 1)
        ot_params[agn_type] = {'A': slope, 'B': intercept}
    return ot_params

def apply_optimal_transport(predicted_inv, agn_types, ot_params):
    corrected = np.array(predicted_inv, dtype=float).copy()
    for agn_type in [0, 1]:
        if agn_type not in ot_params: continue
        idx = np.where(agn_types == agn_type)[0]
        slope = ot_params[agn_type]['A']
        intercept = ot_params[agn_type]['B']
        corrected[idx] = intercept + slope * predicted_inv[idx]
    return corrected

def run_single_cv_iteration(iter_idx, df, features, y_inv, y_z, y_class, base_library):
    """
    Runs a single 10-fold CV iteration with fold-by-fold Lasso selection and SuperLearner training.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=iter_idx)
    predictions_inv = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        df_tr = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        
        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tr[features])
        X_val = scaler.transform(df_val[features])
        
        # Lasso Feature Selection fold-by-fold
        selected_features = lasso_feature_selection(X_tr, y_inv[train_idx], features)
        
        # Scale selected features
        selected_indices = [features.index(f) for f in selected_features]
        X_tr_sel = X_tr[:, selected_indices]
        X_val_sel = X_val[:, selected_indices]
        
        # Fit SuperLearner
        sl = PythonSuperLearner(estimators=base_library, cv=2, random_state=42)
        sl.fit(X_tr_sel, y_inv[train_idx])
        predictions_inv[val_idx] = sl.predict(X_val_sel)
        
    return predictions_inv

def run_repeated_cv(df, features, y_inv, y_z, y_class, base_library, n_iterations=100):
    print(f"Running {n_iterations} iterations of 10-fold CV in parallel...")
    predictions_matrix = Parallel(n_jobs=8)(
        delayed(run_single_cv_iteration)(i, df, features, y_inv, y_z, y_class, base_library) 
        for i in range(n_iterations)
    )
    
    # Average predictions over iterations
    avg_predictions_inv = np.mean(predictions_matrix, axis=0)
    
    # Apply Optimal Transport linear calibration (Phase 5)
    ot_params = fit_optimal_transport(avg_predictions_inv, y_inv, y_class)
    calibrated_predictions_inv = apply_optimal_transport(avg_predictions_inv, y_class, ot_params)
    
    # Convert to linear redshift
    predictions_z = (1.0 / calibrated_predictions_inv) - 1.0
    metrics = compute_metrics(y_z, predictions_z)
    
    return predictions_z, metrics, ot_params

# --------------------------------------------------------------------------
# 4. ERROR DECOMPOSITION (Phase 6)
# --------------------------------------------------------------------------
def run_error_decomposition(df, y_inv, y_z, y_class):
    print("\n=== Phase 6: Running Error Decomposition Audit ===")
    
    # Setup baseline features with LabelNo (paper baseline)
    paper_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                      "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                      "LP_beta", "Gaia_G_Magnitude", "LabelNo"]
                      
    # Setup erroneous features list with GLAT and missing LabelNo
    erroneous_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                          "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                          "LP_beta", "GLAT", "Gaia_G_Magnitude"]
                          
    base_library = get_superlearner_library()
    
    # 1. Standard Replication (Paper features, SuperLearner, 100 iterations)
    print("  1. Evaluating Standard Replication (LabelNo, SL, 100x CV)...")
    pred_z, metrics_std, ot_std = run_repeated_cv(df, paper_features, y_inv, y_z, y_class, base_library, n_iterations=10)
    
    # 2. Impact of Feature Choice: GLAT instead of LabelNo
    print("  2. Evaluating Impact of Feature Choice (using GLAT instead of LabelNo)...")
    _, metrics_glat, _ = run_repeated_cv(df, erroneous_features, y_inv, y_z, y_class, base_library, n_iterations=10)
    
    # 3. Impact of Ensemble Choice: GBDT simple average instead of SuperLearner
    print("  3. Evaluating Impact of Ensemble Choice (GBDT average)...")
    gbdt_library = [
        ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=1, verbose=-1)),
        ('cat', cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, random_seed=42, verbose=0))
    ]
    _, metrics_gbdt, _ = run_repeated_cv(df, paper_features, y_inv, y_z, y_class, gbdt_library, n_iterations=10)
    
    # 4. Impact of single 1x CV run vs 100x CV
    print("  4. Evaluating Impact of CV iterations (1x CV run)...")
    _, metrics_1x, _ = run_repeated_cv(df, paper_features, y_inv, y_z, y_class, base_library, n_iterations=1)
    
    # Decompose
    delta_r_feat = metrics_std['R'] - metrics_glat['R']
    delta_rmse_feat = metrics_std['RMSE'] - metrics_glat['RMSE']
    
    delta_r_ens = metrics_std['R'] - metrics_gbdt['R']
    delta_rmse_ens = metrics_std['RMSE'] - metrics_gbdt['RMSE']
    
    delta_r_cv = metrics_std['R'] - metrics_1x['R']
    delta_rmse_cv = metrics_std['RMSE'] - metrics_1x['RMSE']
    
    print("\nError Decomposition Audit Table:")
    print(f"  {'Factor':<30} | {'Delta R':<10} | {'Delta RMSE':<10}")
    print(f"  {'-'*30} | {'-'*10} | {'-'*10}")
    print(f"  {'LabelNo vs. GLAT features':<30} | {delta_r_feat:+.4f}   | {delta_rmse_feat:+.4f}")
    print(f"  {'SuperLearner vs. GBDT average':<30} | {delta_r_ens:+.4f}   | {delta_rmse_ens:+.4f}")
    print(f"  {'100x CV vs. 1x CV':<30} | {delta_r_cv:+.4f}   | {delta_rmse_cv:+.4f}")
    print()
    
    return metrics_std, ot_std

# --------------------------------------------------------------------------
# 5. STEPWISE ABLATION STUDY & UPGRADES (Phase 7)
# --------------------------------------------------------------------------
def run_benchmark_and_ablation():
    print("=== Loading Data and Preprocessing ===")
    train_eligible = pd.read_csv("data/training_eligible.csv")
    mw = pd.read_csv("data/multi_wavelength_matches.csv")
    mw['Source_Name'] = mw['Source_Name'].astype(str).str.strip()
    df = pd.merge(train_eligible, mw, on="Source_Name", how="left")
    
    # Preprocess numeric columns
    numeric_cols = ["LP_beta", "LP_Index", "Flux1000", "Energy_Flux100",
                    "Signif_Avg", "Variability_Index", "nu_syn", "nuFnu_syn",
                    "Pivot_Energy", "PL_Index", "GLAT", "Flags"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    positive_cols = ["Flux1000", "Energy_Flux100", "Signif_Avg",
                     "Variability_Index", "nu_syn", "nuFnu_syn", "Pivot_Energy"]
    for col in positive_cols:
        df.loc[df[col] <= 0, col] = np.nan
        
    df['LogFlux'] = np.log10(df['Flux1000'])
    df['LogEnergy_Flux'] = np.log10(df['Energy_Flux100'])
    df['LogSignificance'] = np.log10(df['Signif_Avg'])
    df['LogVariability_Index'] = np.log10(df['Variability_Index'])
    df['Lognu_syn'] = np.log10(df['nu_syn'])
    df['LognuFnu_syn'] = np.log10(df['nuFnu_syn'])
    df['LogPivot_Energy'] = np.log10(df['Pivot_Energy'])
    
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
    
    y_inv = df['InvRedshift'].values
    y_z = df['Redshift'].values
    y_class = df['LabelNo'].fillna(0).astype(int).values
    
    # Audits
    run_dataset_audit()
    
    # Feature selection & standard baseline replication
    base_library = get_superlearner_library()
    metrics_std, ot_params = run_error_decomposition(df, y_inv, y_z, y_class)
    
    # Fit standard baseline with full 100 repetitions for final numbers
    print("\n--- Running 100x CV standard baseline reproduction ---")
    paper_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                      "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                      "LP_beta", "Gaia_G_Magnitude", "LabelNo"]
    final_baseline_pred_z, final_baseline_metrics, final_ot_params = run_repeated_cv(df, paper_features, y_inv, y_z, y_class, base_library, n_iterations=100)
    
    print("\nFinal Calibrated Baseline Reproduction Metrics (100x CV):")
    for k, v in final_baseline_metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"  Optimal Transport slopes/intercepts BLL: A={final_ot_params[0]['A']:.4f}, B={final_ot_params[0]['B']:.4f}")
    print(f"  Optimal Transport slopes/intercepts FSRQ: A={final_ot_params[1]['A']:.4f}, B={final_ot_params[1]['B']:.4f}")
    
    # Stepwise Ablation study from A to J on DR2 dataset
    print("\n=== Phase 7: Running Stepwise Ablation Study & Upgrades ===")
    
    configs = {
        'A': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "LabelNo"],
        'B': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo"],
        'C': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3"],
        'D': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "LogRadioFlux"],
        'E': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "LogXrayFlux"],
        'F': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "LogRadioFlux"],
        'G': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "LogXrayFlux"],
        'H': ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index", "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index", "LP_beta", "Gaia_G_Magnitude", "LabelNo", "W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "LogRadioFlux", "LogXrayFlux"],
    }
    
    # Probability P(FSRQ|X) generator using outer KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    p_fsrq_oof = np.zeros(len(df))
    features_for_prob = configs['H']
    
    for train_idx, val_idx in kf.split(df):
        df_tr = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        for col in features_for_prob:
            median_val = df_tr[col].median(skipna=True)
            if pd.isna(median_val): median_val = 0.0
            df_tr[col] = df_tr[col].fillna(median_val)
            df_val[col] = df_val[col].fillna(median_val)
            
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tr[features_for_prob])
        X_val = scaler.transform(df_val[features_for_prob])
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_class[train_idx])
        p_fsrq_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    df['P_FSRQ'] = p_fsrq_oof
    configs['I'] = configs['H'] + ['P_FSRQ']
    
    ablation_results = {}
    oof_predictions_ablation = {}
    
    # Run ablation configurations A-I using modern stacking ensemble
    for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        print(f"  Running Configuration {label} ({len(configs[label])} features)...")
        # Run 10 iterations of CV for ablation study
        oof_pred_inv = np.zeros(len(df))
        
        for train_idx, val_idx in kf.split(df):
            df_tr = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()
            
            for col in configs[label]:
                median_val = df_tr[col].median(skipna=True)
                if pd.isna(median_val): median_val = 0.0
                df_tr[col] = df_tr[col].fillna(median_val)
                df_val[col] = df_val[col].fillna(median_val)
                
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(df_tr[configs[label]])
            X_val = scaler.transform(df_val[configs[label]])
            
            m_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
            m_lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1, verbose=-1)
            m_cat = cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
            
            m_xgb.fit(X_tr, y_inv[train_idx])
            m_lgb.fit(X_tr, y_inv[train_idx])
            m_cat.fit(X_tr, y_inv[train_idx])
            
            pred_xgb = m_xgb.predict(X_val)
            pred_lgb = m_lgb.predict(X_val)
            pred_cat = m_cat.predict(X_val)
            
            oof_pred_inv[val_idx] = (pred_xgb + pred_lgb + pred_cat) / 3.0
            
        # Fit Isotonic Regression separately for BLL and FSRQ
        bll_mask = (y_class == 0)
        fsrq_mask = (y_class == 1)
        iso_bll = IsotonicRegression(out_of_bounds='clip').fit(oof_pred_inv[bll_mask], y_inv[bll_mask])
        iso_fsrq = IsotonicRegression(out_of_bounds='clip').fit(oof_pred_inv[fsrq_mask], y_inv[fsrq_mask])
        
        oof_bll_c = iso_bll.predict(oof_pred_inv)
        oof_fsrq_c = iso_fsrq.predict(oof_pred_inv)
        
        # Soft blend using P_FSRQ if available, else hard class separation
        if 'P_FSRQ' in configs[label]:
            oof_inv_corr = (1.0 - df['P_FSRQ'].values) * oof_bll_c + df['P_FSRQ'].values * oof_fsrq_c
        else:
            oof_inv_corr = np.where(y_class == 0, oof_bll_c, oof_fsrq_c)
            
        oof_z = (1.0 / oof_inv_corr) - 1.0
        oof_predictions_ablation[label] = oof_z
        ablation_results[label] = compute_metrics(y_z, oof_z)
        print(f"    Config {label}: R={ablation_results[label]['R']:.4f}, RMSE={ablation_results[label]['RMSE']:.4f}")
        
    # Configuration J: TabPFN
    print("  Running Configuration J (TabPFN)...")
    oof_tabpfn_inv = np.zeros(len(df))
    for train_idx, val_idx in kf.split(df):
        df_tr = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        
        for col in configs['I']:
            median_val = df_tr[col].median(skipna=True)
            if pd.isna(median_val): median_val = 0.0
            df_tr[col] = df_tr[col].fillna(median_val)
            df_val[col] = df_val[col].fillna(median_val)
            
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tr[configs['I']])
        X_val = scaler.transform(df_val[configs['I']])
        
        m_tabpfn = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
        m_tabpfn.fit(X_tr, y_inv[train_idx])
        oof_tabpfn_inv[val_idx] = m_tabpfn.predict(X_val)
        
    bll_mask = (y_class == 0)
    fsrq_mask = (y_class == 1)
    iso_bll = IsotonicRegression(out_of_bounds='clip').fit(oof_tabpfn_inv[bll_mask], y_inv[bll_mask])
    iso_fsrq = IsotonicRegression(out_of_bounds='clip').fit(oof_tabpfn_inv[fsrq_mask], y_inv[fsrq_mask])
    oof_bll_c = iso_bll.predict(oof_tabpfn_inv)
    oof_fsrq_c = iso_fsrq.predict(oof_tabpfn_inv)
    oof_tabpfn_inv_corr = (1.0 - df['P_FSRQ'].values) * oof_bll_c + df['P_FSRQ'].values * oof_fsrq_c
    oof_tabpfn_z = (1.0 / oof_tabpfn_inv_corr) - 1.0
    
    oof_predictions_ablation['J'] = oof_tabpfn_z
    ablation_results['J'] = compute_metrics(y_z, oof_tabpfn_z)
    print(f"    Config J (TabPFN): R={ablation_results['J']['R']:.4f}, RMSE={ablation_results['J']['RMSE']:.4f}")
    
    # Output ablation study latex table
    ablation_tex = r"""\begin{table}[H]
\centering
\caption{Stepwise Feature Ablation Study on original DR2 Spectroscopic Sample ($N=1,112$)}
\label{tab:dr2_ablation_study}
\begin{tabular}{clccccc}
\toprule
\textbf{Config} & \textbf{Feature Set Configuration} & \textbf{Pearson $R_z$} & \textbf{Spearman $\rho_z$} & \textbf{RMSE ($\Delta z$)} & \textbf{MAE} & \textbf{Outlier \%} \\
\midrule
"""
    desc = {
        'A': 'Original Narendra Features (11 features)',
        'B': 'Original + Gaia (12 features)',
        'C': 'Original + WISE (18 features)',
        'D': 'Original + Radio (13 features)',
        'E': 'Original + X-ray (13 features)',
        'F': 'Original + WISE + Radio (19 features)',
        'G': 'Original + WISE + X-ray (19 features)',
        'H': 'Full Multi-Wavelength Set (20 features)',
        'I': 'Full Set + $P(\\mathrm{FSRQ}|X)$ (21 features)',
        'J': 'TabPFN Regressor (Full Set + $P(\\mathrm{FSRQ}|X)$, 21 features)'
    }
    
    rows = []
    for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        res = ablation_results[label]
        row_str = f"\\textbf{{{label}}} & {desc[label]} & {res['R']:.4f} & {res['rho']:.4f} & {res['RMSE']:.4f} & {res['MAE']:.4f} & {res['Outlier_pct']:.2f}\\% \\\\"
        rows.append(row_str)
    ablation_tex += "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open("output/dr2_ablation_study_table.tex", "w") as f:
        f.write(ablation_tex)
    print("Ablation study table written to output/dr2_ablation_study_table.tex")
    
    # 2. Benchmark Comparison Table
    bench_tex = r"""\begin{table}[H]
\centering
\caption{Comprehensive Photometric Redshift Benchmarking Results on the original DR2 Sample ($N=1,112$)}
\label{tab:dr2_model_benchmarks}
\begin{tabular}{lcccccc}
\toprule
\textbf{Methodology / Model} & \textbf{Sample} & \textbf{Pearson $R_z$} & \textbf{Spearman $\rho_z$} & \textbf{RMSE ($\Delta z$)} & \textbf{MAE} & \textbf{Outlier \%} \\
\midrule
Narendra et al. (2022) Published & DR2 & 0.7420 & 0.7490 & 0.4570 & 0.2820 & 43.5\% \\
Reproduced Narendra Baseline & DR2 & """ + f"{final_baseline_metrics['R']:.4f} & {final_baseline_metrics['rho']:.4f} & {final_baseline_metrics['RMSE']:.4f} & {final_baseline_metrics['MAE']:.4f} & {final_baseline_metrics['Outlier_pct']:.2f}\\%" + r""" \\
Upgraded Stacking Ensemble & DR2 & """ + f"{ablation_results['I']['R']:.4f} & {ablation_results['I']['rho']:.4f} & {ablation_results['I']['RMSE']:.4f} & {ablation_results['I']['MAE']:.4f} & {ablation_results['I']['Outlier_pct']:.2f}\\%" + r""" \\
TabPFN Regressor & DR2 & """ + f"{ablation_results['J']['R']:.4f} & {ablation_results['J']['rho']:.4f} & {ablation_results['J']['RMSE']:.4f} & {ablation_results['J']['MAE']:.4f} & {ablation_results['J']['Outlier_pct']:.2f}\\%" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open("output/dr2_model_benchmarks_table.tex", "w") as f:
        f.write(bench_tex)
    print("Model benchmarks comparison table written to output/dr2_model_benchmarks_table.tex")
    
    # Save unbiased catalog
    oof_catalog = pd.DataFrame({
        'Source_Name': df['Source_Name'],
        'Class': np.where(y_class == 0, 'BLL', 'FSRQ'),
        'z_spec': y_z,
        'z_pred_baseline': np.round(final_baseline_pred_z, 4),
        'Residual_baseline': np.round(y_z - final_baseline_pred_z, 4),
        'Absolute_Error_baseline': np.round(np.abs(y_z - final_baseline_pred_z), 4),
        'z_pred_upgraded': np.round(oof_predictions_ablation['I'], 4),
        'Residual_upgraded': np.round(y_z - oof_predictions_ablation['I'], 4),
        'Absolute_Error_upgraded': np.round(np.abs(y_z - oof_predictions_ablation['I']), 4),
        'z_pred_tabpfn': np.round(oof_predictions_ablation['J'], 4),
        'Residual_tabpfn': np.round(y_z - oof_predictions_ablation['J'], 4),
        'Absolute_Error_tabpfn': np.round(np.abs(y_z - oof_predictions_ablation['J']), 4),
        'OOF_probability': np.round(df['P_FSRQ'], 4)
    })
    oof_catalog.to_csv("data/DR2_OOF_Benchmark_Catalog.csv", index=False)
    print("Out-of-fold catalog written to data/DR2_OOF_Benchmark_Catalog.csv")

if __name__ == "__main__":
    run_benchmark_and_ablation()
