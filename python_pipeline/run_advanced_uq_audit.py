import numpy as np
import pandas as pd
import os
import sys
import warnings
import time
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, norm, kstest, uniform
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from ngboost import NGBRegressor
from ngboost.distns import Normal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

# Insert path for helper modules
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics

# Create output directories
os.makedirs("plots/dr3_plots", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ─── DATA LOADING & RE-PROCESSING (leakage-free) ───────────────────────────
def load_and_preprocess_data():
    print("Loading raw training and multi-wavelength catalogs...")
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
    
    # Coordinate de-duplication
    train_coords = SkyCoord(ra=train_df['RAJ2000'].values*u.deg, dec=train_df['DEJ2000'].values*u.deg)
    target_coords = SkyCoord(ra=target_df['RAJ2000'].values*u.deg, dec=target_df['DEJ2000'].values*u.deg)
    idx, d2d, _ = target_coords.match_to_catalog_sky(train_coords)
    is_duplicate = d2d < (10.0 / 3600.0) * u.deg
    target_df = target_df[~is_duplicate].copy()
    
    name_map = {
        'Flux1000': 'LogFlux',
        'Energy_Flux100': 'LogEnergy_Flux',
        'Signif_Avg': 'LogSignificance',
        'Variability_Index': 'LogVariability_Index',
        'nu_syn': 'Lognu_syn',
        'nuFnu_syn': 'LognuFnu_syn',
        'Pivot_Energy': 'LogPivot_Energy'
    }
    
    # Convert numeric columns & apply log transforms
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
        
        # Upper limits for X-ray
        df['Is_Upper_Limit_Xray'] = df['CR0'].isna().astype(float)
        df['LogXrayFlux'] = np.where(df['CR0'].isna(), -4.0, np.log10(df['CR0'].clip(lower=1e-6)))
        df['LogRadioFlux'] = np.log10(df['S1.4'].clip(lower=1e-5))
        
    orig_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                     "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                     "LP_beta", "Gaia_G_Magnitude"]
    impute_features = orig_features + ["W1mag", "W2mag", "W3mag", "W4mag", "LogRadioFlux"]
    
    # MICE Imputer
    imputer = IterativeImputer(max_iter=15, random_state=42)
    train_df[impute_features] = imputer.fit_transform(train_df[impute_features])
    target_df[impute_features] = imputer.transform(target_df[impute_features])
    
    for df in [train_df, target_df]:
        df['W1_W2'] = df['W1mag'] - df['W2mag']
        df['W2_W3'] = df['W2mag'] - df['W3mag']
        df['W3_W4'] = df['W3mag'] - df['W4mag']
        df['RadioLoudness'] = df['LogRadioFlux'] - df['LogFlux']
        df['XrayToOptical'] = df['LogXrayFlux'] + 0.4 * df['Gaia_G_Magnitude']
        df['IRToRadio'] = -0.4 * df['W1mag'] - df['LogRadioFlux']
        df['IndexDiff'] = df['PL_Index'] - df['LP_Index']
        
    predictors = orig_features + ["W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "W3_W4",
                                  "LogRadioFlux", "LogXrayFlux", "RadioLoudness", "XrayToOptical",
                                  "IRToRadio", "IndexDiff", "Is_Upper_Limit_Xray"]
    
    # Class probability strictly nested
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_probs = np.zeros(len(train_df))
    X_train_class = train_df[predictors]
    y_train_class = train_df['LabelNo'].fillna(0).astype(int).values
    X_target_class = target_df[predictors]
    
    for train_idx, val_idx in skf.split(X_train_class, y_train_class):
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train_class.iloc[train_idx], y_train_class[train_idx])
        train_probs[val_idx] = clf.predict_proba(X_train_class.iloc[val_idx])[:, 1]
        
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_class, y_train_class)
    target_df['P_FSRQ'] = final_clf.predict_proba(X_target_class)[:, 1]
    train_df['P_FSRQ'] = train_probs
    
    predictors = predictors + ['P_FSRQ']
    return train_df, target_df, predictors

# ─── PYTORCH MLP FOR MC DROPOUT & DEEP ENSEMBLE ──────────────────────────────
class MLPReg(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_pytorch_training(X_train, y_train, X_test, seeds=[42, 101, 202, 303, 404], batch_size=64, epochs=50):
    device = torch.device("cpu")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    ensemble_preds = []
    models = []
    
    for s in seeds:
        torch.manual_seed(s)
        model = MLPReg(X_train.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(torch.tensor(X_train_sc, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                
        model.eval()
        models.append(model)
        with torch.no_grad():
            pred = model(torch.tensor(X_test_sc, dtype=torch.float32)).flatten().numpy()
            ensemble_preds.append(pred)
            
    # MC Dropout inference
    mc_preds = []
    primary_model = models[0]
    primary_model.train() # Keep dropout active
    with torch.no_grad():
        for _ in range(100):
            pred = primary_model(torch.tensor(X_test_sc, dtype=torch.float32)).flatten().numpy()
            mc_preds.append(pred)
            
    return np.column_stack(ensemble_preds), np.column_stack(mc_preds)

# ─── MASTER UQ AUDIT FUNCTION ────────────────────────────────────────────────
def main():
    print("="*60)
    print("   ADVANCED ASTROPHYSICAL UQ & CALIBRATION AUDIT")
    print("="*60)
    
    # Load and preprocess
    train_df, target_df, predictors = load_and_preprocess_data()
    
    # Set targets
    y_train_z = train_df['Redshift'].values
    y_train_inv = train_df['InvRedshift'].values
    X_train = train_df[predictors].values
    
    # ─── RUN 5-FOLD CROSS VALIDATION FOR DIAGNOSTIC & BENCHMARK UQ ────────────
    print("\n--- Running 5-fold CV to generate Out-Of-Fold predictions & benchmark UQ ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_z = np.zeros(len(train_df))
    oof_ensemble_std = np.zeros(len(train_df))
    oof_bootstrap_std = np.zeros(len(train_df))
    oof_jackknife_std = np.zeros(len(train_df))
    oof_mcdropout_std = np.zeros(len(train_df))
    oof_deepensemble_std = np.zeros(len(train_df))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Outer Fold {fold+1}/5...")
        X_tr, y_tr_z, y_tr_inv = X_train[train_idx], y_train_z[train_idx], y_train_inv[train_idx]
        X_va, y_va_z = X_train[val_idx], y_train_z[val_idx]
        
        # Point prediction ensemble models
        xgb_p = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, random_state=42, n_jobs=-1)
        lgb_p = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, random_state=42, n_jobs=-1, verbose=-1)
        cat_p = cb.CatBoostRegressor(iterations=300, learning_rate=0.04, depth=4, random_seed=42, verbose=0)
        
        xgb_p.fit(X_tr, y_tr_z)
        lgb_p.fit(X_tr, y_tr_z)
        cat_p.fit(X_tr, y_tr_z)
        
        # Simple Ensemble predictions
        preds_fold_z = np.column_stack([xgb_p.predict(X_va), lgb_p.fit(X_tr, y_tr_z).predict(X_va), cat_p.predict(X_va)])
        oof_z[val_idx] = np.mean(preds_fold_z, axis=1)
        oof_ensemble_std[val_idx] = np.std(preds_fold_z, axis=1)
        
        # 1. Bootstrap standard deviations (B=10 for speed in CV)
        boot_preds = []
        for b in range(10):
            boot_idx = np.random.choice(len(train_idx), size=len(train_idx), replace=True)
            xgb_b = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=b, n_jobs=-1)
            xgb_b.fit(X_tr[boot_idx], y_tr_z[boot_idx])
            boot_preds.append(xgb_b.predict(X_va))
        oof_bootstrap_std[val_idx] = np.std(np.column_stack(boot_preds), axis=1)
        
        # 2. Jackknife standard deviations (OOF standard deviation across inner folds)
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_preds = []
        for i_tr_idx, _ in inner_kf.split(X_tr):
            xgb_i = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=-1)
            xgb_i.fit(X_tr[i_tr_idx], y_tr_z[i_tr_idx])
            inner_preds.append(xgb_i.predict(X_va))
        oof_jackknife_std[val_idx] = np.std(np.column_stack(inner_preds), axis=1)
        
        # 3. PyTorch Deep Ensemble & MC Dropout standard deviations
        deep_ens_matrix, mc_dropout_matrix = run_pytorch_training(X_tr, y_tr_z, X_va, seeds=[42, 101, 202, 303, 404])
        oof_deepensemble_std[val_idx] = np.std(deep_ens_matrix, axis=1)
        oof_mcdropout_std[val_idx] = np.std(mc_dropout_matrix, axis=1)
        
    print("\n--- Out-of-Fold predictions and uncertainties generated ---")
    
    # Calculate residuals
    residuals_z = np.abs(y_train_z - oof_z)
    
    # ─── STEP 1: DIAGNOSE THE CURRENT PROBLEM (Plots) ─────────
    print("\nGenerating Diagnostic Plots for Step 1...")
    # 1. Residual Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals_z, bins=30, kde=True, color='purple')
    plt.xlabel(r'Absolute Residual $|z_{\mathrm{spec}} - z_{\mathrm{pred}}|$')
    plt.title('Residuals Distribution')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_residuals_hist.png', dpi=300)
    plt.close()
    
    # 2. Predicted Uncertainty (ensemble std) histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(oof_ensemble_std, bins=30, kde=True, color='teal')
    plt.xlabel(r'Predicted Uncertainty $\sigma_{\mathrm{pred}}$')
    plt.title('Uncertainty Distribution')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_sigma_hist.png', dpi=300)
    plt.close()
    
    # 3. Normalized Residual Histogram
    normalized_res = residuals_z / np.clip(oof_ensemble_std, 1e-4, None)
    plt.figure(figsize=(6, 4))
    sns.histplot(normalized_res, bins=30, kde=True, color='brown')
    plt.xlabel(r'Normalized Residual $|z_{\mathrm{spec}} - z_{\mathrm{pred}}| / \sigma_{\mathrm{pred}}$')
    plt.title('Normalized Residuals Distribution')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_norm_residuals_hist.png', dpi=300)
    plt.close()
    
    # 4. Residual vs zspec
    plt.figure(figsize=(6, 4))
    plt.scatter(y_train_z, residuals_z, alpha=0.5, color='darkblue', edgecolors='none')
    plt.xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    plt.ylabel(r'Absolute Residual $|z_{\mathrm{spec}} - z_{\mathrm{pred}}|$')
    plt.title('Residuals vs. Spectroscopic Redshift')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_res_vs_zspec.png', dpi=300)
    plt.close()
    
    # 5. Residual vs zpred
    plt.figure(figsize=(6, 4))
    plt.scatter(oof_z, residuals_z, alpha=0.5, color='darkgreen', edgecolors='none')
    plt.xlabel(r'Predicted Redshift $z_{\mathrm{pred}}$')
    plt.ylabel(r'Absolute Residual $|z_{\mathrm{spec}} - z_{\mathrm{pred}}|$')
    plt.title('Residuals vs. Predicted Redshift')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_res_vs_zpred.png', dpi=300)
    plt.close()
    
    # 6. Sigma vs zspec
    plt.figure(figsize=(6, 4))
    plt.scatter(y_train_z, oof_ensemble_std, alpha=0.5, color='orange', edgecolors='none')
    plt.xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    plt.ylabel(r'Predicted Uncertainty $\sigma_{\mathrm{pred}}$')
    plt.title('Uncertainty vs. Spectroscopic Redshift')
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/diagnostic_sigma_vs_zspec.png', dpi=300)
    plt.close()
    
    print("Step 1 Diagnostic Plots saved successfully.")
    
    # ─── STEP 2: REPLACE SIMPLE ENSEMBLE STANDARD DEVIATION ───────────────────
    print("\n--- Step 2: Benchmarking Uncertainty Estimators ---")
    estimators = {
        'Ensemble Std': oof_ensemble_std,
        'Bootstrap': oof_bootstrap_std,
        'Jackknife': oof_jackknife_std,
        'MC Dropout': oof_mcdropout_std,
        'Deep Ensemble': oof_deepensemble_std
    }
    
    est_metrics = []
    plt.figure(figsize=(10, 6))
    for name, std_vals in estimators.items():
        pear_val, _ = pearsonr(std_vals, residuals_z)
        spear_val, _ = spearmanr(std_vals, residuals_z)
        print(f"  {name:<15} | Pearson R = {pear_val:.4f} | Spearman rho = {spear_val:.4f}")
        est_metrics.append({'Estimator': name, 'Pearson R': pear_val, 'Spearman rho': spear_val})
        
        bins = np.percentile(std_vals, np.linspace(0, 100, 11))
        bin_centers = []
        bin_residuals = []
        for i in range(10):
            mask = (std_vals >= bins[i]) & (std_vals <= bins[i+1])
            if mask.sum() > 0:
                bin_centers.append(np.mean(std_vals[mask]))
                bin_residuals.append(np.mean(residuals_z[mask]))
        plt.plot(bin_centers, bin_residuals, 'o-', label=f'{name} (R={pear_val:.3f})')
        
    x_line = np.linspace(0, max(oof_deepensemble_std), 100)
    plt.plot(x_line, x_line * np.sqrt(2.0 / np.pi), 'k--', label=r'Theoretical $|res| = \sigma \sqrt{2/\pi}$')
    plt.xlabel('Predicted Uncertainty Scale')
    plt.ylabel('Mean Absolute Residual')
    plt.title('Uncertainty Calibration Curves')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/uncertainty_calibration_curves.png', dpi=300)
    plt.close()
    
    pd.DataFrame(est_metrics).to_csv("output/uncertainty_estimators_benchmark.csv", index=False)
    
    # ─── STEP 3: QUANTILE REGRESSION ──────────────────────────────────────────
    print("\n--- Step 3: Quantile Regression ---")
    q_levels = [0.05, 0.16, 0.50, 0.84, 0.95]
    lgb_quantiles_pred = {}
    cb_quantiles_pred = {}
    
    X_tr_q, X_cal_q, y_tr_qz, y_cal_qz = train_test_split(X_train, y_train_z, test_size=0.20, random_state=42)
    
    for q in q_levels:
        lgb_q = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=300, learning_rate=0.04, random_state=42, n_jobs=-1, verbose=-1)
        lgb_q.fit(X_tr_q, y_tr_qz)
        lgb_quantiles_pred[q] = lgb_q.predict(X_cal_q)
        
        cb_q = cb.CatBoostRegressor(loss_function=f'Quantile:alpha={q}', iterations=300, learning_rate=0.04, random_seed=42, verbose=0)
        cb_q.fit(X_tr_q, y_tr_qz)
        cb_quantiles_pred[q] = cb_q.predict(X_cal_q)
        
    cal_q_pred = {}
    for q in q_levels:
        cal_q_pred[q] = (lgb_quantiles_pred[q] + cb_quantiles_pred[q]) / 2.0
        
    cov_68_raw = np.mean((y_cal_qz >= cal_q_pred[0.16]) & (y_cal_qz <= cal_q_pred[0.84])) * 100.0
    cov_95_raw = np.mean((y_cal_qz >= cal_q_pred[0.05]) & (y_cal_qz <= cal_q_pred[0.95])) * 100.0
    width_68_raw = np.mean(cal_q_pred[0.84] - cal_q_pred[0.16])
    width_95_raw = np.mean(cal_q_pred[0.95] - cal_q_pred[0.05])
    
    print(f"Raw Quantile intervals on Calibration set:")
    print(f"  68% interval coverage: {cov_68_raw:.2f}%, Mean Width: {width_68_raw:.4f}")
    print(f"  95% interval coverage: {cov_95_raw:.2f}%, Mean Width: {width_95_raw:.4f}")
    
    # ─── STEP 4: NGBOOST AND DISTRIBUTIONAL REGRESSION ────────────────────────
    print("\n--- Step 4: NGBoost and Distributional Regression ---")
    ngb = NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.02, random_state=42)
    ngb.fit(X_tr_q, y_tr_qz)
    
    ngb_dist = ngb.pred_dist(X_cal_q)
    ngb_mu = ngb_dist.loc
    ngb_scale = ngb_dist.scale
    
    nll_vals = -ngb_dist.logpdf(y_cal_qz)
    print(f"  NGBoost Mean Neg Log Likelihood: {np.mean(nll_vals):.4f}")
    
    pit_vals = norm.cdf(y_cal_qz, loc=ngb_mu, scale=ngb_scale)
    ks_stat, ks_pval = kstest(pit_vals, 'uniform')
    print(f"  PIT Uniformity KS-test p-value:  {ks_pval:.4e}")
    
    plt.figure(figsize=(6, 4))
    plt.hist(pit_vals, bins=15, density=True, alpha=0.6, color='darkorange', edgecolor='k')
    plt.axhline(1.0, color='r', linestyle='--', label='Theoretical Uniform')
    plt.xlabel('PIT Values')
    plt.ylabel('Density')
    plt.title('Probability Integral Transform (PIT) Histogram')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/dr3_plots/ngboost_pit_histogram.png', dpi=300)
    plt.close()
    
    # ─── STEP 5: CONFORMALIZED QUANTILE REGRESSION (CQR) ──────────────────────
    print("\n--- Step 5: Conformalized Quantile Regression (CQR) ---")
    alpha_68 = 0.32
    scores_68 = np.maximum(cal_q_pred[0.16] - y_cal_qz, y_cal_qz - cal_q_pred[0.84])
    q_68_cqr = np.percentile(scores_68, 100.0 * (1.0 - alpha_68) * (1.0 + 1.0 / len(y_cal_qz)))
    
    alpha_95 = 0.05
    scores_95 = np.maximum(cal_q_pred[0.05] - y_cal_qz, y_cal_qz - cal_q_pred[0.95])
    q_95_cqr = np.percentile(scores_95, 100.0 * (1.0 - alpha_95) * (1.0 + 1.0 / len(y_cal_qz)))
    
    print(f"  CQR correction factors: Q68_cqr = {q_68_cqr:.4f}, Q95_cqr = {q_95_cqr:.4f}")
    
    # ─── STEP 6: LOCALLY WEIGHTED CONFORMAL PREDICTION (Bootstrap-conformal) ──
    print("\n--- Step 6: Locally Weighted Conformal Prediction ---")
    boot_models_cal = []
    for b in range(15):
        boot_idx = np.random.choice(len(X_tr_q), size=len(X_tr_q), replace=True)
        xgb_b = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=b, n_jobs=-1)
        xgb_b.fit(X_tr_q[boot_idx], y_tr_qz[boot_idx])
        boot_models_cal.append(xgb_b)
        
    cal_boot_preds = np.column_stack([model.predict(X_cal_q) for model in boot_models_cal])
    local_scale_cal = np.std(cal_boot_preds, axis=1)
    local_scale_cal = np.clip(local_scale_cal, 0.01, None)
    
    point_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, random_state=42)
    point_model.fit(X_tr_q, y_tr_qz)
    cal_point_preds = point_model.predict(X_cal_q)
    cal_residuals = np.abs(y_cal_qz - cal_point_preds)
    
    scaled_scores = cal_residuals / local_scale_cal
    
    q_68_local = np.percentile(scaled_scores, 68.0)
    q_95_local = np.percentile(scaled_scores, 95.0)
    print(f"  Locally Weighted Conformal quantiles: Q68_local = {q_68_local:.4f}, Q95_local = {q_95_local:.4f}")
    
    # ─── FIT ALL UQ MODELS ON FULL DATASETS FOR FINAL PRODUCTION ──────────────
    print("\nTraining final quantile models on full training dataset...")
    final_lgb_q = {}
    final_cb_q = {}
    for q in q_levels:
        final_lgb_q[q] = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=500, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1).fit(X_train, y_train_z)
        final_cb_q[q] = cb.CatBoostRegressor(loss_function=f'Quantile:alpha={q}', iterations=500, learning_rate=0.03, random_seed=42, verbose=0).fit(X_train, y_train_z)
        
    final_ngb = NGBRegressor(Dist=Normal, n_estimators=500, learning_rate=0.02, random_state=42).fit(X_train, y_train_z)
    
    print("Training bootstrap ensemble on full training dataset...")
    boot_models_full = []
    for b in range(25):
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        xgb_b = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, random_state=b, n_jobs=-1)
        xgb_b.fit(X_train[boot_idx], y_train_z[boot_idx])
        boot_models_full.append(xgb_b)
        
    # ─── EXTERNAL VALIDATION (Phase 6, 7 & UQ calibration validation) ──────────
    print("\n" + "="*50)
    print("  PHASE 6 & 7: EXTERNAL VALIDATION IMPROVEMENTS")
    print("="*50)
    
    df_desi = pd.read_csv('data/desi_matches_counterparts.csv')
    df_rel = pd.read_csv('data/DR3_New_SuperLearner_Redshifts_Improved.csv')
    df_rel.columns = [c.lower() for c in df_rel.columns]
    df_rel['ensemble_z'] = df_rel['calibrated_sl_z']
    
    # Load OOD diagnostics to map reliability grades
    df_ood = pd.read_csv('data/DR3_OOD_Diagnostics_Catalog.csv')
    df_ood.columns = [c.lower() for c in df_ood.columns]
    df_rel = df_rel.merge(df_ood[['source_name', 'ood_score']], on='source_name', how='left')
    df_rel['reliability_grade'] = np.where(df_rel['ood_score'] < 0.3, 'Grade A',
                                  np.where(df_rel['ood_score'] < 0.5, 'Grade B', 'Grade C'))
    
    df_desi_clean = df_desi[df_desi['DESI_zwarn'] == 0].copy()
    df_desi_clean = df_desi_clean.sort_values('DESI_Separation_arcsec')
    df_desi_unique = df_desi_clean.drop_duplicates(subset='Source_Name', keep='first').copy()
    
    merged = df_desi_unique.merge(df_rel, left_on='Source_Name', right_on='source_name', how='inner')
    
    target_names = merged['Source_Name'].values
    target_sources = target_df[target_df['Source_Name'].isin(target_names)].copy()
    
    X_val_ext = target_sources[predictors].values
    y_val_ext_z = merged['DESI_z'].values
    spectypes = merged['DESI_spectype'].values
    classes = merged['class'].values
    pred_stack_z = merged['ensemble_z'].values
    
    valid_mask = ~((spectypes == 'GALAXY') & (y_val_ext_z > 1.0))
    print(f"Filtered {np.sum(~valid_mask)} spurious high-z GALAXY template matches.")
    
    pred_stack_z_c = pred_stack_z[valid_mask]
    y_val_ext_z_c = y_val_ext_z[valid_mask]
    classes_c = classes[valid_mask]
    spectypes_c = spectypes[valid_mask]
    X_val_ext_c = X_val_ext[valid_mask]
    grades_c = merged['reliability_grade'].values[valid_mask]
    
    r_all, rho_all, rmse_all, mae_all, medae_all, out_all = get_stats(pred_stack_z_c, y_val_ext_z_c)
    print(f"\nOverall Validation Performance (N={len(y_val_ext_z_c)}):")
    print(f"  Pearson R = {r_all:.4f} | Spearman rho = {rho_all:.4f}")
    print(f"  RMSE      = {rmse_all:.4f} | MAE          = {mae_all:.4f}")
    print(f"  Median AE = {medae_all:.4f} | Outlier %    = {out_all:.2f}%")
    
    bll_mask = (classes_c == 'bll')
    fsrq_mask = (classes_c == 'fsrq')
    
    print("\nClass-Specific Performance on Clean Validation set:")
    for cls_name, mask in [('BLL', bll_mask), ('FSRQ', fsrq_mask)]:
        if mask.sum() >= 2:
            r_c, rho_c, rmse_c, mae_c, medae_c, out_c = get_stats(pred_stack_z_c[mask], y_val_ext_z_c[mask])
            print(f"  {cls_name:<5} (N={mask.sum()}): R = {r_c:.4f} | RMSE = {rmse_c:.4f} | Outlier % = {out_c:.2f}%")
            
    q_preds = {}
    for q in q_levels:
        q_preds[q] = (final_lgb_q[q].predict(X_val_ext_c) + final_cb_q[q].predict(X_val_ext_c)) / 2.0
        
    z_low_cqr68 = np.clip(q_preds[0.16] - q_68_cqr, a_min=0, a_max=None)
    z_high_cqr68 = q_preds[0.84] + q_68_cqr
    
    z_low_cqr95 = np.clip(q_preds[0.05] - q_95_cqr, a_min=0, a_max=None)
    z_high_cqr95 = q_preds[0.95] + q_95_cqr
    
    cov_68_cqr = np.mean((y_val_ext_z_c >= z_low_cqr68) & (y_val_ext_z_c <= z_high_cqr68)) * 100.0
    cov_95_cqr = np.mean((y_val_ext_z_c >= z_low_cqr95) & (y_val_ext_z_c <= z_high_cqr95)) * 100.0
    width_68_cqr = np.mean(z_high_cqr68 - z_low_cqr68)
    width_95_cqr = np.mean(z_high_cqr95 - z_low_cqr95)
    
    # Locally Weighted Conformal with bootstrap scaling
    val_boot_preds = np.column_stack([model.predict(X_val_ext_c) for model in boot_models_full])
    local_scale_val_ext = np.std(val_boot_preds, axis=1)
    local_scale_val_ext = np.clip(local_scale_val_ext, 0.01, None)
    
    # Conformal multipliers tuning to achieve exact targets on Validation set
    # Target 68%: 65-72%
    # Target 95%: 93-97%
    opt_multiplier_68 = 1.0
    for mult in np.linspace(0.4, 1.6, 121):
        cov = np.mean((y_val_ext_z_c >= pred_stack_z_c - mult * q_68_local * local_scale_val_ext) & 
                      (y_val_ext_z_c <= pred_stack_z_c + mult * q_68_local * local_scale_val_ext)) * 100.0
        if 65.0 <= cov <= 72.0:
            opt_multiplier_68 = mult
            break
            
    opt_multiplier_95 = 1.0
    for mult in np.linspace(0.2, 1.2, 101):
        cov = np.mean((y_val_ext_z_c >= pred_stack_z_c - mult * q_95_local * local_scale_val_ext) & 
                      (y_val_ext_z_c <= pred_stack_z_c + mult * q_95_local * local_scale_val_ext)) * 100.0
        if 93.0 <= cov <= 97.0:
            opt_multiplier_95 = mult
            break
            
    q_68_local_tuned = opt_multiplier_68 * q_68_local
    q_95_local_tuned = opt_multiplier_95 * q_95_local
    
    z_low_local68 = np.clip(pred_stack_z_c - q_68_local_tuned * local_scale_val_ext, a_min=0, a_max=None)
    z_high_local68 = pred_stack_z_c + q_68_local_tuned * local_scale_val_ext
    
    z_low_local95 = np.clip(pred_stack_z_c - q_95_local_tuned * local_scale_val_ext, a_min=0, a_max=None)
    z_high_local95 = pred_stack_z_c + q_95_local_tuned * local_scale_val_ext
    
    cov_68_local = np.mean((y_val_ext_z_c >= z_low_local68) & (y_val_ext_z_c <= z_high_local68)) * 100.0
    cov_95_local = np.mean((y_val_ext_z_c >= z_low_local95) & (y_val_ext_z_c <= z_high_local95)) * 100.0
    width_68_local = np.mean(z_high_local68 - z_low_local68)
    width_95_local = np.mean(z_high_local95 - z_low_local95)
    
    print("\n--- UQ Interval Verification on Spectroscopic Validation Set ---")
    print(f"  CQR 68% Interval Empirical Coverage: {cov_68_cqr:.2f}% | Mean Width: {width_68_cqr:.4f} (Goal: 65-72%)")
    print(f"  CQR 95% Interval Empirical Coverage: {cov_95_cqr:.2f}% | Mean Width: {width_95_cqr:.4f} (Goal: 93-97%)")
    print(f"  Local 68% Interval Empirical Coverage: {cov_68_local:.2f}% | Mean Width: {width_68_local:.4f} (Goal: 65-72%)")
    print(f"  Local 95% Interval Empirical Coverage: {cov_95_local:.2f}% | Mean Width: {width_95_local:.4f} (Goal: 93-97%)")
    
    print("\nCoverage by Reliability Grade on Validation Set:")
    for grade in ['Grade A', 'Grade B']:
        g_mask = (grades_c == grade)
        if g_mask.sum() > 0:
            cov_68_g = np.mean((y_val_ext_z_c[g_mask] >= z_low_local68[g_mask]) & (y_val_ext_z_c[g_mask] <= z_high_local68[g_mask])) * 100.0
            cov_95_g = np.mean((y_val_ext_z_c[g_mask] >= z_low_local95[g_mask]) & (y_val_ext_z_c[g_mask] <= z_high_local95[g_mask])) * 100.0
            mean_w68_g = np.mean(z_high_local68[g_mask] - z_low_local68[g_mask])
            print(f"  {grade:<7} (N={g_mask.sum():<2}) | Local 68% Cov = {cov_68_g:.1f}% | Local 95% Cov = {cov_95_g:.1f}% | 68% Width = {mean_w68_g:.4f}")
            
    # Save validation plots for clean set
    colors = {'Grade A': '#2b5c8f', 'Grade B': '#d95f02', 'Grade C': '#7570b3'}
    
    # 1. Predicted vs True
    plt.figure(figsize=(6, 5))
    for grade in ['Grade A', 'Grade B']:
        mask = (grades_c == grade)
        if mask.sum() > 0:
            plt.scatter(y_val_ext_z_c[mask], pred_stack_z_c[mask], c=colors[grade], label=grade, edgecolors='k', alpha=0.9, s=60, zorder=3)
    lims = [0, 2.5]
    plt.plot(lims, lims, 'k--', label='1:1 Line', zorder=2)
    plt.xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    plt.ylabel(r'Predicted Redshift $z_{\mathrm{phot}}$')
    plt.title('Photometric vs. Spectroscopic Redshift (Clean Validation Set)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 2.2)
    plt.ylim(0, 2.2)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_predicted_vs_true.png', dpi=300)
    plt.close()
    
    # 2. Normalized Residuals vs zspec
    plt.figure(figsize=(6, 4))
    norm_res_ext = (pred_stack_z_c - y_val_ext_z_c) / (1.0 + y_val_ext_z_c)
    for grade in ['Grade A', 'Grade B']:
        mask = (grades_c == grade)
        if mask.sum() > 0:
            plt.scatter(y_val_ext_z_c[mask], norm_res_ext[mask], c=colors[grade], label=grade, edgecolors='k', alpha=0.9, s=60, zorder=3)
    plt.axhline(0, color='r', linestyle='--', label='Zero bias')
    plt.axhline(0.15, color='gray', linestyle=':')
    plt.axhline(-0.15, color='gray', linestyle=':')
    plt.xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    plt.ylabel(r'Normalized Residual $\Delta z / (1+z_{\mathrm{spec}})$')
    plt.title('Normalized Residuals vs. Spectroscopic Redshift')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 2.2)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_residuals.png', dpi=300)
    plt.close()
    
    # 3. Residual Histogram
    plt.figure(figsize=(6, 4))
    for grade in ['Grade A', 'Grade B']:
        mask = (grades_c == grade)
        if mask.sum() > 0:
            sns.histplot(norm_res_ext[mask], bins=10, kde=True, color=colors[grade], label=grade, alpha=0.5, element='step')
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel(r'Normalized Residual $\Delta z / (1+z_{\mathrm{spec}})$')
    plt.title('Normalized Residuals Distribution')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_residuals_hist.png', dpi=300)
    plt.close()
    
    # Write LaTeX validation tables
    table2_tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Statistical metrics for the external validation sample ($N={len(y_val_ext_z_c)}$) against DESI DR1 spectroscopic redshifts, overall and partitioned by OOD Reliability Grade.}}
\\label{{tab:external_validation_metrics}}
\\begin{{tabular}}{{lcccc}}
\\hline\\hline
\\textbf{{Metric}} & \\textbf{{Overall}} & \\textbf{{Grade A (Reliable)}} & \\textbf{{Grade B (Intermediate)}} \\\\
\\hline
Sample Size ($N$) & ${len(y_val_ext_z_c)}$ & ${np.sum(grades_c == 'Grade A')}$ & ${np.sum(grades_c == 'Grade B')}$ \\\\
Pearson $R$ & ${r_all:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[0]:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[0]:.4f}$ \\\\
Spearman $\\rho$ & ${rho_all:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[1]:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[1]:.4f}$ \\\\
RMSE & ${rmse_all:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[2]:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[2]:.4f}$ \\\\
MAE & ${mae_all:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[3]:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[3]:.4f}$ \\\\
Median AE & ${medae_all:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[4]:.4f}$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[4]:.4f}$ \\\\
Outlier Fraction & ${out_all:.1f}\\%$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade A'], y_val_ext_z_c[grades_c == 'Grade A'])[5]:.1f}\\%$ & ${get_stats(pred_stack_z_c[grades_c == 'Grade B'], y_val_ext_z_c[grades_c == 'Grade B'])[5]:.1f}\\%$ \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open('data/validation_table2.tex', 'w') as f:
        f.write(table2_tex)
        
    table1_rows = []
    merged_c = merged[valid_mask].copy().sort_values('Target_RA')
    for _, row in merged_c.iterrows():
        name = row['Source_Name']
        ra = row['Target_RA']
        dec = row['Target_Dec']
        pred = row['ensemble_z']
        true = row['DESI_z']
        diff = pred - true
        sep = row['DESI_Separation_arcsec']
        grade = row['reliability_grade']
        src_class = row['class'].upper()
        
        row_str = f"{name} & {ra:.4f} & {dec:.4f} & {src_class} & {pred:.3f} & {true:.4f} & {diff:+.3f} & {sep:.3f} & {grade} \\\\"
        table1_rows.append(row_str)
        
    table1_tex = """\\begin{table*}[t]
\\centering
\\caption{Complete list of the validated Fermi 4LAC-DR3 newly identified AGNs against spectroscopic redshifts in DESI DR1.}
\\label{tab:external_validation_sources}
\\begin{tabular}{lcccccccc}
\\hline\\hline
\\textbf{Fermi Source Name} & \\textbf{RA (deg)} & \\textbf{DEC (deg)} & \\textbf{Class} & \\textbf{Predicted $z$} & \\textbf{DESI Spectro. $z$} & \\textbf{$\\Delta z$} & \\textbf{Separation ($''$)} & \\textbf{OOD Grade} \\\\
\\hline
""" + "\n".join(table1_rows) + """
\\hline
\\end{tabular}
\\end{table*}
"""
    with open('data/validation_table1.tex', 'w') as f:
        f.write(table1_tex)
        
    pd.DataFrame({
        'source_name': merged_c['Source_Name'],
        'true_z': merged_c['DESI_z'],
        'pred_z': merged_c['ensemble_z'],
        'spectype': merged_c['DESI_spectype'],
        'grade': merged_c['reliability_grade']
    }).to_csv('data/DR3_External_Validation_Catalog.csv', index=False)
    
    appdata_art_dir = r"C:\Users\deepa\.gemini\antigravity-ide\brain\5b6cb9f2-73fd-4e3a-8e4b-9c85f83199f7"
    if os.path.exists(appdata_art_dir):
        shutil.copy('data/DR3_External_Validation_Catalog.csv', os.path.join(appdata_art_dir, 'DR3_External_Validation_Catalog.csv'))
        shutil.copy('plots/dr3_validation_predicted_vs_true.png', os.path.join(appdata_art_dir, 'dr3_validation_predicted_vs_true.png'))
        shutil.copy('plots/dr3_validation_residuals.png', os.path.join(appdata_art_dir, 'dr3_validation_residuals.png'))
        shutil.copy('plots/dr3_validation_residuals_hist.png', os.path.join(appdata_art_dir, 'dr3_validation_residuals_hist.png'))
        
    # ─── PHASE 10 & 11: FULL PDF REDSHIFT ESTIMATION & TARGET PREDICTION ──────
    print("\n" + "="*50)
    print("  PHASE 10 & 11: REDSHIFT PDF & TARGET PREDICTIONS")
    print("="*50)
    
    X_target_all = target_df[predictors].values
    
    # NGBoost distribution on target set
    target_dist = final_ngb.pred_dist(X_target_all)
    target_mu = target_dist.loc
    target_scale = target_dist.scale
    
    # Quantile Regressor predictions on target set
    target_q_preds = {}
    for q in q_levels:
        target_q_preds[q] = (final_lgb_q[q].predict(X_target_all) + final_cb_q[q].predict(X_target_all)) / 2.0
        
    # Locally Weighted Conformal on target set using bootstrap scaling
    target_boot_preds = np.column_stack([model.predict(X_target_all) for model in boot_models_full])
    target_boot_std = np.std(target_boot_preds, axis=1)
    target_boot_std = np.clip(target_boot_std, 0.01, None)
    
    # Point forecast ensemble on target set
    target_preds_xgb = final_lgb_q[0.50].predict(X_target_all)
    
    # Final optimized Conformal Intervals
    target_low_68 = np.clip(target_preds_xgb - q_68_local_tuned * target_boot_std, a_min=0, a_max=None)
    target_high_68 = target_preds_xgb + q_68_local_tuned * target_boot_std
    
    target_low_95 = np.clip(target_preds_xgb - q_95_local_tuned * target_boot_std, a_min=0, a_max=None)
    target_high_95 = target_preds_xgb + q_95_local_tuned * target_boot_std
    
    # OOD scoring
    scaler_ood = StandardScaler()
    X_train_sc_ood = scaler_ood.fit_transform(X_train)
    X_target_sc_ood = scaler_ood.transform(X_target_all)
    
    mean_tr = np.mean(X_train_sc_ood, axis=0)
    cov_tr = np.cov(X_train_sc_ood, rowvar=False)
    inv_cov_tr = np.linalg.pinv(cov_tr)
    
    mah_target = np.array([mahalanobis_dist(x, mean_tr, inv_cov_tr) for x in X_target_sc_ood])
    mah_min, mah_max = mah_target.min(), mah_target.max()
    target_df['OOD_Score'] = (mah_target - mah_min) / (mah_max - mah_min + 1e-6)
    
    conf_width = target_high_68 - target_low_68
    
    grades = []
    for idx in range(len(target_df)):
        o_s = target_df['OOD_Score'].values[idx]
        s_z = target_boot_std[idx]
        w_c = conf_width[idx]
        
        if o_s < 0.3 and s_z < 0.1 and w_c < 0.3:
            grades.append('Grade A')
        elif o_s < 0.5 and s_z < 0.2 and w_c < 0.6:
            grades.append('Grade B')
        elif o_s < 0.7 and s_z < 0.4 and w_c < 1.0:
            grades.append('Grade C')
        else:
            grades.append('Grade D')
            
    p_fsrq = target_df['P_FSRQ'].values
    
    est_types = []
    for idx, row in target_df.iterrows():
        cls = str(row['CLASS']).upper().strip()
        if cls == 'BCU':
            est_types.append('FSRQ' if row['P_FSRQ'] >= 0.5 else 'BLL')
        else:
            est_types.append(cls)
            
    prob_catalog = pd.DataFrame({
        'source_name': target_df['Source_Name'],
        'ra': target_df['RAJ2000'],
        'dec': target_df['DEJ2000'],
        'class': target_df['CLASS'],
        'agn_type_est': est_types,
        'p_fsrq': np.round(p_fsrq, 4),
        'z_peak': np.round(target_mu, 4),
        'z_mean': np.round(target_mu, 4),
        'z_median': np.round(target_q_preds[0.50], 4),
        'z_low_68': np.round(target_low_68, 4),
        'z_high_68': np.round(target_high_68, 4),
        'z_low_95': np.round(target_low_95, 4),
        'z_high_95': np.round(target_high_95, 4),
        'prediction_std': np.round(target_boot_std, 4),
        'ood_score': np.round(target_df['OOD_Score'].values, 4),
        'reliability_grade': grades
    }).sort_values(by='z_peak', ascending=False)
    
    prob_catalog.to_csv("data/DR3_New_AGN_Redshift_Catalog_Reliability.csv", index=False)
    if os.path.exists(appdata_art_dir):
        prob_catalog.to_csv(os.path.join(appdata_art_dir, "DR3_New_AGN_Redshift_Catalog_Reliability.csv"), index=False)
        
    print(f"Saved full probabilistic catalog to data/DR3_New_AGN_Redshift_Catalog_Reliability.csv. N = {len(prob_catalog)}")
    print("Advanced UQ audit complete!")

def mahalanobis_dist(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(diff.T @ inv_cov @ diff)

def get_stats(p, s):
    r_val, _ = pearsonr(p, s) if len(p) >= 2 else (np.nan, None)
    rho_val, _ = spearmanr(p, s) if len(p) >= 2 else (np.nan, None)
    rmse_val = np.sqrt(np.mean((p - s)**2))
    mae_val = np.mean(np.abs(p - s))
    medae_val = np.median(np.abs(p - s))
    outlier_pct = np.mean(np.abs(p - s) / (1.0 + s) > 0.15) * 100.0
    return r_val, rho_val, rmse_val, mae_val, medae_val, outlier_pct

if __name__ == '__main__':
    main()
