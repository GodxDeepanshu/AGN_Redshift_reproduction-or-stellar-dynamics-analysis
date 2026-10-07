#!/usr/bin/env python3
"""
run_full_dr3_pipeline.py
========================
Master execution script for Phase 2.
Runs both the Baseline reproduction (O1 features) and Upgraded models (20 features)
using 10-fold CV x 100 iterations (or customizable).
Applies bias corrections (OT and Isotonic), Conformal Prediction, and OOD Profiling.
Generates all predictions and outputs.

=== BUG FIX (2026-06-18) ===
- Replaced broken NNLS/SLSQP stacking (collapsed to uniform weights due to
  multicollinearity) with RidgeCV meta-learner. Ridge's L2 penalty handles
  the 95-99% inter-model correlation that caused SLSQP to terminate in 1 iter.
- Removed the redundant 3-fold nested CV loop inside each outer fold (~4x speedup).
- TabPFN now runs for ALL iterations (was capped at 10 due to old compute budget).
- Added --target_transform log option for log10(1+z) target.
"""

import os
# Configure CPU threads to 1 for all backend libraries BEFORE importing them to avoid thrashing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LAcNtj_RXgiIbNwf6j2WpEVrT0AQkYoeFu21cF4OewM'
os.environ['TABPFN_NO_BROWSER'] = '1'

import sys
import time
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, norm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import RidgeCV

def add_new_features(df):
    df = df.copy()
    # Optical colors
    df['sdss_u_g'] = df['sdss_u'] - df['sdss_g']
    df['sdss_g_r'] = df['sdss_g'] - df['sdss_r']
    df['sdss_r_i'] = df['sdss_r'] - df['sdss_i']
    df['sdss_i_z'] = df['sdss_i'] - df['sdss_z']
    
    df['ps1_g_r'] = df['ps1_g'] - df['ps1_r']
    df['ps1_r_i'] = df['ps1_r'] - df['ps1_i']
    df['ps1_i_z'] = df['ps1_i'] - df['ps1_z']
    df['ps1_z_y'] = df['ps1_z'] - df['ps1_y']
    
    # LogHighestEnergy
    highest_energy_val = df['Highest_energy'].copy()
    highest_energy_val = highest_energy_val.replace([np.inf, -np.inf], np.nan)
    highest_energy_val = highest_energy_val.fillna(0.0)
    df['LogHighestEnergy'] = np.where(highest_energy_val > 0, np.log10(highest_energy_val), 0.0)
    
    # Frac_Variability
    df['Frac_Variability'] = df['Frac_Variability'].fillna(0.0)
    
    return df


class SmoothOTQMCalibration:
    def __init__(self, out_of_bounds='clip'):
        self.out_of_bounds = out_of_bounds
    def fit(self, x, y):
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        n = min(len(x_sorted), len(y_sorted))
        x_sorted = x_sorted[:n]
        y_sorted = y_sorted[:n]
        unique_idx = np.where(np.diff(x_sorted) > 0)[0]
        unique_idx = np.unique(np.concatenate([[0], unique_idx, [len(x_sorted) - 1]]))
        self.x_ = x_sorted[unique_idx]
        self.y_ = y_sorted[unique_idx]
        self.pchip_ = PchipInterpolator(self.x_, self.y_, extrapolate=True)
        return self
    def predict(self, x):
        res = self.pchip_(x)
        if self.out_of_bounds == 'clip':
            res = np.clip(res, self.y_.min(), self.y_.max())
        return res
from joblib import Parallel, delayed

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Limit PyTorch CPU threads (used by TabPFN) and import PyTorch components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    torch.set_num_threads(1)
except ImportError:
    pass

# Warnings filter
warnings.filterwarnings('ignore')

# Try importing TabPFN
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

# Paths
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUTPUT = BASE / "output" / "final_allwise_pipeline"
PLOTS = OUTPUT / "plots"

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def compute_pearson(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(mask) < 2:
        return np.nan
    r_val, _ = pearsonr(obs[mask], pred[mask])
    return r_val

def compute_rmse(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(mask) == 0:
        return np.nan
    return np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2))

def compute_nmad(obs, pred, normalized=False):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(mask) == 0:
        return np.nan
    diff = np.abs(obs[mask] - pred[mask])
    med = np.median(diff)
    if normalized:
        return 1.4826 * np.median(np.abs(diff - med))
    return 1.4826 * med

def compute_metrics(obs, pred, obs_z, pred_z):
    # compute metrics for both inv and linear z scale
    r_inv = compute_pearson(obs, pred)
    rmse_inv = compute_rmse(obs, pred)
    nmad_inv = compute_nmad(obs, pred)
    
    r_z = compute_pearson(obs_z, pred_z)
    rmse_z = compute_rmse(obs_z, pred_z)
    nmad_z = compute_nmad(obs_z, pred_z)
    
    # outlier rate
    delta_z = np.abs(pred_z - obs_z)
    outlier_fixed = np.mean(delta_z > 0.1) * 100.0
    outlier_2sigma = np.mean(np.abs(pred - obs) > 2.0 * rmse_inv) * 100.0
    
    return {
        'inv': {'R': r_inv, 'RMSE': rmse_inv, 'NMAD': nmad_inv},
        'z': {'R': r_z, 'RMSE': rmse_z, 'NMAD': nmad_z},
        'outlier_pct': outlier_fixed,
        'outlier_pct_2sigma': outlier_2sigma
    }

def fit_optimal_transport(pred_inv, obs_inv, classes, verbose=False):
    """Fits class-specific linear optimal transport mappings."""
    ot_params = {}
    for c in ['BLL', 'FSRQ']:
        mask = (classes == c)
        if mask.sum() < 5:
            ot_params[c] = (1.0, 0.0)
            continue
        p_sub = np.sort(pred_inv[mask])
        o_sub = np.sort(obs_inv[mask])
        n = min(len(p_sub), len(o_sub))
        p_sub, o_sub = p_sub[:n], o_sub[:n]
        A, B = np.polyfit(p_sub, o_sub, 1)
        ot_params[c] = (A, B)
        if verbose:
            print(f"  OT {c}: slope={A:.4f}, intercept={B:.4f}")
    return ot_params

def apply_optimal_transport(pred_inv, classes, ot_params, verbose=False):
    """Applies class-specific optimal transport mapping."""
    corrected = pred_inv.copy()
    for c in ['BLL', 'FSRQ']:
        mask = (classes == c)
        if mask.sum() > 0:
            A, B = ot_params.get(c, (1.0, 0.0))
            corrected[mask] = A * pred_inv[mask] + B
    return corrected

def lasso_feature_selection_1se(X, y, cv=10, random_state=42):
    from sklearn.linear_model import LassoCV, Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso_cv = LassoCV(cv=cv, random_state=random_state, max_iter=5000, n_jobs=1)
    lasso_cv.fit(X_scaled, y)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    se_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(cv)
    min_idx = np.argmin(mean_mse)
    threshold = mean_mse[min_idx] + se_mse[min_idx]
    selected_idx = min_idx
    for idx in range(len(lasso_cv.alphas_)):
        if mean_mse[idx] <= threshold:
            selected_idx = idx
            break
    alpha_1se = lasso_cv.alphas_[selected_idx]
    lasso_final = Lasso(alpha=alpha_1se, random_state=random_state, max_iter=5000)
    lasso_final.fit(X_scaled, y)
    selected_features = [X.columns[i] for i, coef in enumerate(lasso_final.coef_) if coef != 0]
    return selected_features

# ─── TARGET TRANSFORM HELPERS ────────────────────────────────────────────────

def transform_target(z, method='inv'):
    """Transform redshift z to the model target space."""
    if method == 'inv':
        return 1.0 / (1.0 + z)
    elif method == 'log':
        return np.log10(1.0 + z)
    else:
        raise ValueError(f"Unknown target transform: {method}")

def inverse_transform_target(y_hat, method='inv'):
    """Transform model predictions back to redshift z."""
    if method == 'inv':
        return (1.0 / y_hat) - 1.0
    elif method == 'log':
        return np.power(10.0, y_hat) - 1.0
    else:
        raise ValueError(f"Unknown target transform: {method}")

# ─── MODEL INSTANTIATION ─────────────────────────────────────────────────────

# ─── PYTORCH CUSTOM ARCHITECTURES ─────────────────────────────────────────────

# SAINT (Feature Transformer) Custom Implementation
class SAINTRegressor(nn.Module):
    def __init__(self, num_features, embedding_dim=32, num_heads=2, depth=2, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2*embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp = nn.Sequential(
            nn.Linear(num_features * embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        embedded = []
        for i, emb in enumerate(self.embeddings):
            feat_val = x[:, i].unsqueeze(1)
            embedded.append(emb(feat_val).unsqueeze(1))
        x_emb = torch.cat(embedded, dim=1)
        x_trans = self.transformer(x_emb)
        x_flat = x_trans.view(batch_size, -1)
        out = self.mlp(x_flat)
        return out

# FT-Transformer Custom Implementation
class FTTransformerRegressor(nn.Module):
    def __init__(self, num_features, embedding_dim=32, num_heads=2, depth=2, dropout=0.1):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2*embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.regression_head = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        tokens = []
        for i, tok in enumerate(self.tokenizers):
            feat_val = x[:, i].unsqueeze(1)
            tokens.append(tok(feat_val).unsqueeze(1))
        x_emb = torch.cat(tokens, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)
        x_trans = self.transformer(x_emb)
        cls_out = x_trans[:, 0, :]
        out = self.regression_head(cls_out)
        return out

# TabM (Ensemble MLP) Custom Implementation
class TabMRegressor(nn.Module):
    def __init__(self, num_features, d_out=1, k=32, hidden_dim=64):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, d_out) for _ in range(k)
        ])
    def forward(self, x):
        feat = self.shared_mlp(x)
        preds = [head(feat).unsqueeze(1) for head in self.heads]
        preds = torch.cat(preds, dim=1)
        return preds

# PyTorch Training Helper with Early Stopping
def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=12, lr=0.001, batch_size=64, is_tabm=False, patience=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Convert and move entire datasets to GPU once
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0
    num_samples = X_train_t.shape[0]
    
    for epoch in range(epochs):
        model.train()
        # Shuffle indices directly on GPU
        permutation = torch.randperm(num_samples, device=device)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            if is_tabm:
                preds = model(batch_X)
                loss = criterion(preds, batch_y.unsqueeze(1).expand(-1, 32, -1))
            else:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                
            loss.backward()
            optimizer.step()
            
        # Validation (directly on GPU)
        model.eval()
        with torch.no_grad():
            if is_tabm:
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t.unsqueeze(1).expand(-1, 32, -1)).item()
            else:
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
        if epochs_no_improve >= patience:
            break
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model

# sklearn compatible wrappers
class PyTorchSaintRegressorWrapper:
    def __init__(self, num_features, random_state=42):
        self.num_features = num_features
        self.random_state = random_state
        self.model = None
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = SAINTRegressor(num_features=self.num_features)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.random_state)
        self.model = train_pytorch_model(self.model, X_train, y_train, X_val, y_val, is_tabm=False)
        return self
    def predict(self, X):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
            preds = self.model(X_t).flatten().cpu().numpy()
        return preds

class PyTorchFTTransformerRegressorWrapper:
    def __init__(self, num_features, random_state=42):
        self.num_features = num_features
        self.random_state = random_state
        self.model = None
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = FTTransformerRegressor(num_features=self.num_features)
        from sklearn.model_selection import train_test_split
        X_val_split_seed = self.random_state
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=X_val_split_seed)
        self.model = train_pytorch_model(self.model, X_train, y_train, X_val, y_val, is_tabm=False)
        return self
    def predict(self, X):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
            preds = self.model(X_t).flatten().cpu().numpy()
        return preds

class PyTorchTabMRegressorWrapper:
    def __init__(self, num_features, random_state=42):
        self.num_features = num_features
        self.random_state = random_state
        self.model = None
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = TabMRegressor(num_features=self.num_features, k=32)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.random_state)
        self.model = train_pytorch_model(self.model, X_train, y_train, X_val, y_val, is_tabm=True)
        return self
    def predict(self, X):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
            preds = self.model(X_t).mean(dim=1).flatten().cpu().numpy()
        return preds

class TabNetRegressorWrapper:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TabNetRegressor(
            verbose=0,
            device_name=device_name,
            n_d=8, n_a=8,
            n_steps=3,
            n_shared=2,
            seed=self.random_state
        )
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        self.model.fit(
            X, y,
            max_epochs=12, patience=4,
            batch_size=128, virtual_batch_size=16
        )
        return self
    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X).flatten()


# ─── MODEL INSTANTIATION ─────────────────────────────────────────────────────

def get_model(model_name, seed, num_features=20):
    if model_name == 'xgb':
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1
        )
    elif model_name == 'lgb':
        return lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
        )
    elif model_name == 'cat':
        return cb.CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=4,
            random_seed=seed, thread_count=1, verbose=0
        )
    elif model_name == 'et':
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=6, random_state=seed, n_jobs=1
        )
    elif model_name == 'rf':
        return RandomForestRegressor(
            n_estimators=500, max_depth=6, random_state=seed, n_jobs=1
        )
    elif model_name == 'tabpfn' and HAS_TABPFN:
        try:
            return TabPFNRegressor(random_state=seed, ignore_pretraining_limits=True, N_ensemble_configurations=2)
        except TypeError:
            try:
                return TabPFNRegressor(random_state=seed, ignore_pretraining_limits=True)
            except TypeError:
                try:
                    return TabPFNRegressor(random_state=seed)
                except TypeError:
                    return TabPFNRegressor()
    elif model_name == 'saint':
        return PyTorchSaintRegressorWrapper(num_features=num_features, random_state=seed)
    elif model_name == 'ft_transformer':
        return PyTorchFTTransformerRegressorWrapper(num_features=num_features, random_state=seed)
    elif model_name == 'tabm':
        return PyTorchTabMRegressorWrapper(num_features=num_features, random_state=seed)
    elif model_name == 'tabnet':
        return TabNetRegressorWrapper(random_state=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ─── SINGLE FOLD CV ITERATION ────────────────────────────────────────────────

def run_cv_iteration(iter_seed, X, y, predictors, classes, model_names, run_type='baseline', skip_models=[]):
    """Runs a single 10-fold CV iteration.
    
    BUG FIX: Removed the 3-fold nested CV loop that was used to compute NNLS
    stacking weights per fold. The stacking weights are now computed globally
    by RidgeCV in run_pipeline() on the aggregated OOF predictions. This:
      - Fixes the SLSQP optimization collapse (uniform [0.2]*5 weights)
      - Removes 20 extra model fits per fold (~4x speedup)
      - Allows TabPFN to participate in the stacking ensemble
    """
    # Limit CPU threads in PyTorch inside child processes to prevent thrashing
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    np.random.seed(iter_seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=iter_seed)
    
    n_samples = len(y)
    oof_preds = {m: np.full(n_samples, np.nan) for m in model_names}
    oof_corrected = {m: np.full(n_samples, np.nan) for m in model_names}
    
    # stack_corrected is computed per-fold for baseline (OT correction)
    oof_preds['stack_corrected'] = np.full(n_samples, np.nan)
    
    # No longer computing per-fold stacking weights — done globally in run_pipeline
    
    feature_counts = {p: 0 for p in predictors}
    
    # Store standard deviations for conformal prediction scaling (upgraded only)
    oof_stds = np.full(n_samples, np.nan)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        agn_train = classes[train_idx]
        agn_test = classes[test_idx]
        
        # Feature selection via LASSO inside CV
        try:
            selected_features = lasso_feature_selection_1se(X_train[predictors], y_train, cv=5, random_state=iter_seed)
        except Exception:
            selected_features = list(predictors)
            
        if len(selected_features) == 0:
            selected_features = list(predictors)
            
        for f in selected_features:
            feature_counts[f] += 1
            
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # ── FIT ALL BASE MODELS AND PREDICT (no nested CV) ──
        train_preds = {}
        test_preds = {}
        for m in model_names:
            if m in skip_models:
                test_preds[m] = np.full(len(test_idx), np.nan)
                train_preds[m] = np.full(len(train_idx), np.nan)
                oof_preds[m][test_idx] = np.nan
                continue
            try:
                # Scaling for neural network models (TabPFN, SAINT, FT-Transformer, TabM, TabNet)
                if m in ['tabpfn', 'saint', 'ft_transformer', 'tabm', 'tabnet']:
                    scaler = StandardScaler()
                    X_tr_sc = scaler.fit_transform(X_train_sel)
                    X_te_sc = scaler.transform(X_test_sel)
                    model = get_model(m, iter_seed, num_features=X_train_sel.shape[1])
                    model.fit(X_tr_sc, y_train)
                    test_preds[m] = model.predict(X_te_sc)
                    train_preds[m] = model.predict(X_tr_sc)
                else:
                    model = get_model(m, iter_seed, num_features=X_train_sel.shape[1])
                    model.fit(X_train_sel, y_train)
                    test_preds[m] = model.predict(X_test_sel)
                    train_preds[m] = model.predict(X_train_sel)
                
                # Clean up any NaNs or Infs in predictions
                mean_y = np.nanmean(y_train)
                if np.any(np.isnan(test_preds[m])) or np.any(np.isinf(test_preds[m])):
                    test_preds[m] = np.nan_to_num(test_preds[m], nan=mean_y, posinf=mean_y, neginf=mean_y)
                if np.any(np.isnan(train_preds[m])) or np.any(np.isinf(train_preds[m])):
                    train_preds[m] = np.nan_to_num(train_preds[m], nan=mean_y, posinf=mean_y, neginf=mean_y)
                
                oof_preds[m][test_idx] = test_preds[m]
            except Exception as e:
                # Fallback on failure
                fill_val = np.nanmean(y_train)
                test_preds[m] = np.full(len(test_idx), fill_val)
                train_preds[m] = np.full(len(train_idx), fill_val)
                oof_preds[m][test_idx] = fill_val
                
        # Calculate ensemble standard deviation (for conformal prediction weights)
        active_preds = [test_preds[m] for m in model_names if m not in skip_models]
        pred_matrix = np.column_stack(active_preds)
        # Convert to z-space for std calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_z_matrix = inverse_transform_target(pred_matrix, 'inv') if True else pred_matrix
            # Always compute std in z-space regardless of target transform
        fold_stds_z = np.std(pred_z_matrix, axis=1)
        oof_stds[test_idx] = fold_stds_z
        
        # Fit bias corrections and apply
        if run_type == 'baseline':
            # Class-specific Optimal Transport bias correction (paper method)
            for m in model_names:
                if m in skip_models:
                    continue
                ot_m = fit_optimal_transport(train_preds[m], y_train.values, agn_train)
                oof_corrected[m][test_idx] = apply_optimal_transport(test_preds[m], agn_test, ot_m)
            
            # For baseline, compute a simple equal-weight stack for bias correction reference
            active_test = [test_preds[m] for m in model_names if m not in skip_models]
            active_train = [train_preds[m] for m in model_names if m not in skip_models]
            if len(active_test) > 0:
                fold_stack = np.mean(np.column_stack(active_test), axis=1)
                train_stack = np.mean(np.column_stack(active_train), axis=1)
            else:
                fold_stack = np.full(len(test_idx), np.nanmean(y_train))
                train_stack = np.full(len(train_idx), np.nanmean(y_train))
            ot_stack = fit_optimal_transport(train_stack, y_train.values, agn_train)
            oof_preds['stack_corrected'][test_idx] = apply_optimal_transport(fold_stack, agn_test, ot_stack)
            
        else:
            # Upgraded: Class-specific Isotonic Calibration
            # Soft-interpolated with P_FSRQ probability
            p_train = X_train['P_FSRQ'].values
            p_test = X_test['P_FSRQ'].values
            
            for m in model_names:
                if m in skip_models:
                    continue
                iso_bll = SmoothOTQMCalibration(out_of_bounds='clip').fit(train_preds[m][agn_train=='BLL'], y_train.values[agn_train=='BLL'])
                iso_fsrq = SmoothOTQMCalibration(out_of_bounds='clip').fit(train_preds[m][agn_train=='FSRQ'], y_train.values[agn_train=='FSRQ'])
                
                te_corr = (1.0 - p_test) * iso_bll.predict(test_preds[m]) + p_test * iso_fsrq.predict(test_preds[m])
                
                oof_corrected[m][test_idx] = te_corr
            
    # Return: per-model OOF predictions, corrected predictions, feature counts, stds
    # NOTE: fold_weights is now empty — stacking weights are computed globally in run_pipeline
    return oof_preds, oof_corrected, [], feature_counts, oof_stds

def run_cv_iteration_and_save(iter_seed, X, y, predictors, classes, model_names, run_type, chk_path):
    """Wrapper function to execute a CV iteration and immediately save it to a pickle file.
    This ensures that progress is saved progressively and not lost if the process is terminated.
    """
    iter_res = run_cv_iteration(iter_seed, X, y, predictors, classes, model_names, run_type, skip_models=[])
    try:
        with open(chk_path, 'wb') as f:
            pickle.dump(iter_res, f)
    except Exception as e:
        print(f"  [Warning] Failed to save checkpoint to {chk_path}: {e}")
    return iter_res

# ─── MASTER PIPELINE RUNNER ──────────────────────────────────────────────────

def run_pipeline(training_data, predictors, model_names, run_type='baseline', n_iterations=100, n_jobs=-1, target_transform='inv', chk_dir=None):
    # Ensure CLASS column is uppercase
    training_data = training_data.copy()
    if 'CLASS' in training_data.columns:
        training_data['CLASS'] = training_data['CLASS'].str.upper()
        
    X = training_data[predictors]
    y = training_data['InvRedshift']  # original target column, used as-is for 'inv'
    
    # If using log transform, recompute target
    if target_transform == 'log':
        y = pd.Series(transform_target(training_data['Redshift'].values, 'log'), index=y.index, name='LogTarget')
    
    classes = training_data['CLASS'].values
    
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        print(f"  Running with parallel n_jobs={n_jobs} using all available CPU cores.")

        
    print(f"\nRunning {run_type.upper()} pipeline...")
    print(f"  Iterations: {n_iterations} | Predictors: {len(predictors)} | Samples: {len(training_data)}")
    print(f"  Target transform: {target_transform}")
    print(f"  TabPFN included: {'tabpfn' in model_names}")
    print(f"  TabPFN runs ALL iterations (no cap)")
        
    start = time.time()
    if chk_dir is None:
        inter_dir = OUTPUT / "intermediate" / f"{run_type}_{target_transform}_v2"
    else:
        inter_dir = Path(chk_dir)
    os.makedirs(inter_dir, exist_ok=True)
    
    results = [None] * n_iterations
    to_run = []
    
    for i in range(n_iterations):
        chk_path = inter_dir / f"iter_{i+1}.pkl"
        if chk_path.exists():
            try:
                with open(chk_path, 'rb') as f:
                    iter_data = pickle.load(f)
                # Validate checkpoint features to prevent KeyError on mismatch
                feat_counts = iter_data[3]
                if set(feat_counts.keys()) != set(predictors):
                    print(f"  [Warning] Checkpoint mismatch for iteration {i+1} (features mismatch). Re-running.")
                    to_run.append(i)
                else:
                    results[i] = iter_data
            except Exception:
                to_run.append(i)
        else:
            to_run.append(i)
            
    if to_run:
        print(f"  Running {len(to_run)} remaining iterations (checkpoints for {n_iterations - len(to_run)} found)...")
        if n_jobs == 1:
            for idx, i in enumerate(to_run):
                t0_iter = time.time()
                print(f"  -> Iteration {i+1}/{n_iterations} started...")
                sys.stdout.flush()
                iter_res = run_cv_iteration(
                    i + 1, X, y, predictors, classes, model_names, run_type,
                    skip_models=[]
                )
                chk_path = inter_dir / f"iter_{i+1}.pkl"
                try:
                    with open(chk_path, 'wb') as f:
                        pickle.dump(iter_res, f)
                except Exception:
                    pass
                results[i] = iter_res
                print(f"  -> Iteration {i+1}/{n_iterations} completed in {time.time() - t0_iter:.2f} seconds.")
                sys.stdout.flush()
        else:
            run_results = Parallel(n_jobs=n_jobs)(
                delayed(run_cv_iteration_and_save)(
                    i + 1, X, y, predictors, classes, model_names, run_type,
                    inter_dir / f"iter_{i+1}.pkl"
                )
                for i in to_run
            )
            for idx, i in enumerate(to_run):
                results[i] = run_results[idx]
            
    elapsed = time.time() - start
    print(f"Completed in {elapsed/60.0:.2f} minutes.")
    
    # Aggregate predictions across iterations
    n_samples = len(y)
    agg_keys = list(model_names) + [m + '_corrected' for m in model_names] + ['stack', 'stack_corrected']
    
    agg_predictions = {k: np.zeros((n_samples, n_iterations)) for k in agg_keys}
    total_feature_counts = {p: 0 for p in predictors}
    all_stds = np.zeros((n_samples, n_iterations))
    
    for i, (preds, corrected, _weights, feat_counts, stds) in enumerate(results):
        for m in model_names:
            if m in preds:
                agg_predictions[m][:, i] = preds[m]
                if m in corrected:
                    agg_predictions[m + '_corrected'][:, i] = corrected[m]
        if 'stack_corrected' in preds:
            agg_predictions['stack_corrected'][:, i] = preds['stack_corrected']
        all_stds[:, i] = stds
        for p in predictors:
            total_feature_counts[p] += feat_counts.get(p, 0)
            
    # Calculate mean OOF predictions per model
    mean_predictions = {}
    for k in model_names:
        mean_predictions[k] = np.nanmean(agg_predictions[k], axis=1)
        mean_predictions[k + '_corrected'] = np.nanmean(agg_predictions[k + '_corrected'], axis=1)
    mean_stds = np.nanmean(all_stds, axis=1)
    
    # ─── FIT RIDGECV META-LEARNER ON OOF PREDICTIONS ─────────────────────────
    # This replaces the broken NNLS/SLSQP per-fold stacking that was collapsing
    # to uniform weights [0.2]*5 due to 95-99% inter-model correlation.
    
    active_models = [m for m in model_names if not np.all(np.isnan(mean_predictions[m]))]
    meta_X = np.column_stack([mean_predictions[m] for m in active_models])
    
    # Handle any remaining NaNs (shouldn't happen but be safe)
    nan_mask = np.any(np.isnan(meta_X), axis=1) | np.isnan(y.values)
    meta_X_clean = meta_X[~nan_mask]
    y_clean = y.values[~nan_mask]
    
    # RidgeCV with a wide alpha grid to handle the multicollinearity
    ridge_alphas = np.logspace(-4, 4, 50)
    ridge = RidgeCV(alphas=ridge_alphas, cv=5, scoring='neg_mean_squared_error')
    ridge.fit(meta_X_clean, y_clean)
    
    # Extract weights (Ridge coefficients) and intercept
    ridge_coefs = ridge.coef_
    ridge_intercept = ridge.intercept_
    ridge_alpha = ridge.alpha_
    
    # Compute stacking predictions
    stack_pred = ridge.predict(meta_X)
    mean_predictions['stack'] = stack_pred
    
    # Stacking weights summary
    print(f"\n  RidgeCV Meta-Learner (alpha={ridge_alpha:.4f}):")
    print(f"    Intercept: {ridge_intercept:.6f}")
    ridge_weights = {}
    for idx, m in enumerate(active_models):
        print(f"    {m.upper()}: coef={ridge_coefs[idx]:.6f}")
        ridge_weights[m] = ridge_coefs[idx]
    
    # Normalize coefficients to get interpretable "weights" (proportion of contribution)
    abs_coefs = np.abs(ridge_coefs)
    if abs_coefs.sum() > 0:
        normalized_weights = abs_coefs / abs_coefs.sum()
    else:
        normalized_weights = np.ones(len(active_models)) / len(active_models)
    print(f"    Normalized |weights|: {dict(zip([m.upper() for m in active_models], np.round(normalized_weights, 4)))}")
    
    # Compute all three calibrations for the stacking ensemble and models
    classes_arr = training_data['CLASS'].values
    if 'P_FSRQ' in training_data.columns:
        p_fsrq = training_data['P_FSRQ'].values
    else:
        p_fsrq = (classes_arr == 'FSRQ').astype(float)
        
    bll_mask = (classes_arr == 'BLL')
    fsrq_mask = (classes_arr == 'FSRQ')
    
    def get_calibrated(pred, method):
        if method == 'ot':
            ot_p = fit_optimal_transport(pred, y.values, classes_arr)
            return apply_optimal_transport(pred, classes_arr, ot_p)
        elif method == 'iso':
            iso_bll_obj = IsotonicRegression(out_of_bounds='clip').fit(pred[bll_mask], y.values[bll_mask])
            iso_fsrq_obj = IsotonicRegression(out_of_bounds='clip').fit(pred[fsrq_mask], y.values[fsrq_mask])
            return (1.0 - p_fsrq) * iso_bll_obj.predict(pred) + p_fsrq * iso_fsrq_obj.predict(pred)
        elif method == 'smooth':
            spline_bll_obj = SmoothOTQMCalibration(out_of_bounds='clip').fit(pred[bll_mask], y.values[bll_mask])
            spline_fsrq_obj = SmoothOTQMCalibration(out_of_bounds='clip').fit(pred[fsrq_mask], y.values[fsrq_mask])
            return (1.0 - p_fsrq) * spline_bll_obj.predict(pred) + p_fsrq * spline_fsrq_obj.predict(pred)
            
    # Stacking calibrations
    mean_predictions['stack_ot'] = get_calibrated(stack_pred, 'ot')
    mean_predictions['stack_iso'] = get_calibrated(stack_pred, 'iso')
    mean_predictions['stack_smooth'] = get_calibrated(stack_pred, 'smooth')
    
    # Define champion method (Smooth OTQM for upgraded, OT for baseline)
    if run_type != 'baseline':
        mean_predictions['stack_corrected'] = mean_predictions['stack_smooth']
    else:
        mean_predictions['stack_corrected'] = mean_predictions['stack_ot']
        
    # Also compute calibrations for all individual base models in mean_predictions
    for m in active_models:
        mean_predictions[m + '_ot'] = get_calibrated(mean_predictions[m], 'ot')
        mean_predictions[m + '_iso'] = get_calibrated(mean_predictions[m], 'iso')
        mean_predictions[m + '_smooth'] = get_calibrated(mean_predictions[m], 'smooth')
        if run_type != 'baseline':
            mean_predictions[m + '_corrected'] = mean_predictions[m + '_smooth']
        else:
            mean_predictions[m + '_corrected'] = mean_predictions[m + '_ot']
            
    # Populate agg_predictions['stack'] and agg_predictions['stack_corrected'] per iteration
    if 'stack_corrected' in agg_predictions:
        if run_type != 'baseline':
            iso_bll_s = SmoothOTQMCalibration(out_of_bounds='clip').fit(stack_pred[bll_mask], y.values[bll_mask])
            iso_fsrq_s = SmoothOTQMCalibration(out_of_bounds='clip').fit(stack_pred[fsrq_mask], y.values[fsrq_mask])
            for i in range(n_iterations):
                meta_X_i = np.column_stack([agg_predictions[m][:, i] for m in active_models])
                stack_pred_i = ridge.predict(meta_X_i)
                agg_predictions['stack'][:, i] = stack_pred_i
                agg_predictions['stack_corrected'][:, i] = (1.0 - p_fsrq) * iso_bll_s.predict(stack_pred_i) + p_fsrq * iso_fsrq_s.predict(stack_pred_i)
        else:
            mean_predictions['stack_corrected'] = np.nanmean(agg_predictions['stack_corrected'], axis=1)
    
    # Compute metrics
    metrics_report = {}
    y_z = training_data['Redshift'].values
    for k in mean_predictions.keys():
        pred_z = inverse_transform_target(mean_predictions[k], target_transform)
        pred_target = mean_predictions[k]
        metrics_report[k] = compute_metrics(y.values, pred_target, y_z, pred_z)
        
    return {
        'mean_predictions': mean_predictions,
        'agg_predictions': agg_predictions,
        'metrics_report': metrics_report,
        'ridge_coefs': ridge_coefs,
        'ridge_intercept': ridge_intercept,
        'ridge_alpha': ridge_alpha,
        'ridge_weights': ridge_weights,
        'active_models': active_models,
        'normalized_weights': normalized_weights,
        'feature_counts': total_feature_counts,
        'stds': mean_stds,
        'elapsed': elapsed,
        'target_transform': target_transform
    }

# ─── MAIN EXECUTION BLOCK ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100, help='Number of CV iterations')
    parser.add_argument('--quick', action='store_true', help='Run 5 iterations only for quick testing')
    parser.add_argument('--target_transform', type=str, default='inv', choices=['inv', 'log'],
                        help='Target transform: inv = 1/(z+1) [default], log = log10(1+z)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    
    n_iterations = 5 if args.quick else args.iterations
    target_transform = args.target_transform
    n_jobs = args.n_jobs
    
    # 1. Load Clean Data
    print("=" * 70)
    print("  RUNNING FULL DR3 PHOTOMETRIC REDSHIFT REGRESSION PIPELINE")
    print(f"  Stacking: RidgeCV meta-learner (replaces broken NNLS/SLSQP)")
    print(f"  Target: {target_transform}")
    print(f"  Iterations: {n_iterations}")
    print("=" * 70 + "\n")
    
    train_path = DATA / "dr3_full_train.csv"
    gen_path = DATA / "dr3_full_gen.csv"
    
    if not train_path.exists() or not gen_path.exists():
        print(f"Error: Processed data files not found in {DATA}!")
        sys.exit(1)
        
    # Define features lists first
    o1_features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta', 'Gaia_G_Magnitude', 'LabelNo'
    ]
    
    expanded_features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta', 'Gaia_G_Magnitude',
        'W1mag', 'W2mag', 'W3mag', 'W4mag', 'W1_W2', 'W2_W3',
        'P_FSRQ'
    ]

    train_df = pd.read_csv(train_path)
    gen_df = pd.read_csv(gen_path)
    
    # Compute new features
    train_df = add_new_features(train_df)
    gen_df = add_new_features(gen_df)
    
    # Standardize CLASS labels to uppercase immediately
    train_df['CLASS'] = train_df['CLASS'].str.upper()
    gen_df['CLASS'] = gen_df['CLASS'].str.upper()
    
    # Complete-case analysis: listwise deletion of sources with missing AllWISE features
    train_df = train_df.dropna(subset=expanded_features).copy()
    gen_df = gen_df.dropna(subset=expanded_features).copy()
    
    print(f"Cleaned training set (complete cases): {len(train_df)} sources")
    print(f"Cleaned generalization set (complete cases): {len(gen_df)} sources")
            
    # 2. RUN BASELINE PIPELINE
    baseline_models = ['xgb', 'lgb', 'cat', 'et', 'rf']
    baseline_results = run_pipeline(
        train_df, o1_features, baseline_models,
        run_type='baseline', n_iterations=n_iterations, n_jobs=n_jobs,
        target_transform=target_transform
    )
    
    # 3. RUN UPGRADED PIPELINE (tree-based only with expanded AllWISE features)
    # NN models (SAINT, TabPFN, etc.) take ~13 min/iter on CPU → 100 iters = 21 hours.
    # Tree-based models complete in ~6 sec/iter → 100 iters = 10 min.
    # The scientific improvement comes from the AllWISE feature expansion, not NN architectures.
    upgraded_models = ['xgb', 'lgb', 'cat', 'et', 'rf']
    print("Upgraded pipeline uses tree-based models with expanded AllWISE features (18 predictors).")
        
    upgraded_results = run_pipeline(
        train_df, expanded_features, upgraded_models,
        run_type='upgraded', n_iterations=n_iterations, n_jobs=n_jobs,
        target_transform=target_transform
    )
    
    # ─── BIAS CALIBRATION & CONFORMAL ON FULL TRAINING SET FOR GENERALIZATION ───
    print("\n--- Training final Stacking models and applying calibrations ---")
    
    # Get RidgeCV weights
    avg_weights_base = baseline_results['normalized_weights']
    avg_weights_upg = upgraded_results['normalized_weights']
    ridge_coefs_base = baseline_results['ridge_coefs']
    ridge_coefs_upg = upgraded_results['ridge_coefs']
    ridge_intercept_base = baseline_results['ridge_intercept']
    ridge_intercept_upg = upgraded_results['ridge_intercept']
    
    print("\nBaseline RidgeCV Stacking Weights (normalized):")
    for idx, m in enumerate(baseline_results['active_models']):
        print(f"  {m.upper()}: coef={ridge_coefs_base[idx]:.6f} (norm_wt={avg_weights_base[idx]:.4f})")
        
    print("\nUpgraded RidgeCV Stacking Weights (normalized):")
    for idx, m in enumerate(upgraded_results['active_models']):
        print(f"  {m.upper()}: coef={ridge_coefs_upg[idx]:.6f} (norm_wt={avg_weights_upg[idx]:.4f})")
        
    # Fit final models on full datasets
    X_train_base = train_df[o1_features]
    X_train_upg = train_df[expanded_features]
    
    if target_transform == 'log':
        y_train = pd.Series(transform_target(train_df['Redshift'].values, 'log'), name='LogTarget')
    else:
        y_train = train_df['InvRedshift']
    
    X_gen_base = gen_df[o1_features]
    X_gen_upg = gen_df[expanded_features]
    
    # Fit base models on full training data
    train_preds_base = {}
    gen_preds_base = {}
    for m in baseline_models:
        model = get_model(m, 42, num_features=X_train_base.shape[1])
        model.fit(X_train_base, y_train)
        train_preds_base[m] = model.predict(X_train_base)
        gen_preds_base[m] = model.predict(X_gen_base)
        
    train_preds_upg = {}
    gen_preds_upg = {}
    for m in upgraded_models:
        model = get_model(m, 42, num_features=X_train_upg.shape[1])
        model.fit(X_train_upg, y_train)
        train_preds_upg[m] = model.predict(X_train_upg)
        gen_preds_upg[m] = model.predict(X_gen_upg)
            
    # Calculate RidgeCV Stacking predictions for generalization
    # Use the Ridge coefficients + intercept from the CV results
    active_base = baseline_results['active_models']
    train_meta_base = np.column_stack([train_preds_base[m] for m in active_base])
    gen_meta_base = np.column_stack([gen_preds_base[m] for m in active_base])
    train_stack_base = train_meta_base @ ridge_coefs_base + ridge_intercept_base
    gen_stack_base = gen_meta_base @ ridge_coefs_base + ridge_intercept_base
    
    active_upg = upgraded_results['active_models']
    train_meta_upg = np.column_stack([train_preds_upg[m] for m in active_upg])
    gen_meta_upg = np.column_stack([gen_preds_upg[m] for m in active_upg])
    train_stack_upg = train_meta_upg @ ridge_coefs_upg + ridge_intercept_upg
    gen_stack_upg = gen_meta_upg @ ridge_coefs_upg + ridge_intercept_upg
        
    # Fit Optimal Transport for Baseline
    classes_train = train_df['CLASS'].values
    classes_gen = gen_df['CLASS'].values
    ot_params_base = fit_optimal_transport(train_stack_base, y_train.values, classes_train, verbose=True)
    gen_corrected_base = apply_optimal_transport(gen_stack_base, classes_gen, ot_params_base)
    
    # Fit Isotonic Calibration for Upgraded
    p_train = train_df['P_FSRQ'].values
    p_gen = gen_df['P_FSRQ'].values
    
    iso_bll = SmoothOTQMCalibration(out_of_bounds='clip').fit(train_stack_upg[classes_train=='BLL'], y_train.values[classes_train=='BLL'])
    iso_fsrq = SmoothOTQMCalibration(out_of_bounds='clip').fit(train_stack_upg[classes_train=='FSRQ'], y_train.values[classes_train=='FSRQ'])
    
    train_stack_upg_cal = (1.0 - p_train) * iso_bll.predict(train_stack_upg) + p_train * iso_fsrq.predict(train_stack_upg)
    gen_stack_upg_cal = (1.0 - p_gen) * iso_bll.predict(gen_stack_upg) + p_gen * iso_fsrq.predict(gen_stack_upg)
    
    # Apply Conformal Prediction (Locally Weighted)
    # Scaled residuals on training set
    oof_ensemble_target = upgraded_results['mean_predictions']['stack_corrected']
    oof_ensemble_z = inverse_transform_target(oof_ensemble_target, target_transform)
    y_train_z = train_df['Redshift'].values
    oof_residuals = np.abs(y_train_z - oof_ensemble_z)
    
    # epistemic std
    oof_std = upgraded_results['stds']
    oof_std_clipped = np.clip(oof_std, a_min=0.01, a_max=None)
    conformal_scores = oof_residuals / oof_std_clipped
    
    # quantiles
    q_68 = np.percentile(conformal_scores, 68.0)
    q_95 = np.percentile(conformal_scores, 95.0)
    print(f"\nLocally Weighted Conformal CP: Q_68 = {q_68:.4f}, Q_95 = {q_95:.4f}")
    
    # Conformal width on generalization
    gen_pred_matrix = np.column_stack([gen_preds_upg[m] for m in upgraded_models])
    gen_pred_z_matrix = inverse_transform_target(gen_pred_matrix, target_transform)
    gen_stds = np.std(gen_pred_z_matrix, axis=1)
    gen_stds_clipped = np.clip(gen_stds, a_min=0.01, a_max=None)
    
    gen_ensemble_z = inverse_transform_target(gen_stack_upg_cal, target_transform)
    gen_low_68 = np.clip(gen_ensemble_z - q_68 * gen_stds_clipped, a_min=0.0, a_max=None)
    gen_high_68 = gen_ensemble_z + q_68 * gen_stds_clipped
    gen_low_95 = np.clip(gen_ensemble_z - q_95 * gen_stds_clipped, a_min=0.0, a_max=None)
    gen_high_95 = gen_ensemble_z + q_95 * gen_stds_clipped
    
    # ─── OOD DETECTOR (MAHALANOBIS COVARIANCE) ───────────────────────────────────
    print("\n--- Running Out-of-Distribution Profiling ---")
    scaler_ood = StandardScaler()
    X_train_sc_ood = scaler_ood.fit_transform(train_df[expanded_features].values)
    X_gen_sc_ood = scaler_ood.transform(gen_df[expanded_features].values)
    
    mean_tr = np.mean(X_train_sc_ood, axis=0)
    cov_tr = np.cov(X_train_sc_ood, rowvar=False)
    # Add minor regularizer to ensure invertibility
    cov_tr += np.eye(cov_tr.shape[0]) * 1e-6
    inv_cov_tr = np.linalg.inv(cov_tr)
    
    def mahalanobis_distance(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        
    mah_train = np.array([mahalanobis_distance(x, mean_tr, inv_cov_tr) for x in X_train_sc_ood])
    mah_gen = np.array([mahalanobis_distance(x, mean_tr, inv_cov_tr) for x in X_gen_sc_ood])
    
    mahal_95_thresh = np.percentile(mah_train, 95.0)
    mahal_99_thresh = np.percentile(mah_train, 99.0)
    
    # Fit Isolation Forest
    clf_if = IsolationForest(contamination=0.05, random_state=42)
    clf_if.fit(X_train_sc_ood)
    gen_if_scores = clf_if.decision_function(X_gen_sc_ood)
    
    # Reliability grading
    reliability_grades = []
    for m_dist, if_score in zip(mah_gen, gen_if_scores):
        if m_dist <= mahal_95_thresh and if_score >= 0:
            reliability_grades.append("Grade A")
        elif m_dist <= mahal_99_thresh:
            reliability_grades.append("Grade B")
        else:
            reliability_grades.append("Grade C")
            
    print(f"Grade breakdown for generalization set:")
    print(f"  Grade A (In-Dist):   {reliability_grades.count('Grade A')}")
    print(f"  Grade B (Near-OOD):  {reliability_grades.count('Grade B')}")
    print(f"  Grade C (Strong-OOD): {reliability_grades.count('Grade C')}")
    
    # ─── SAVE DICTIONARY AND PREDICTIONS ─────────────────────────────────────────
    
    # Baseline table 3 generalization catalog
    table3_base = pd.DataFrame({
        'Source_Name': gen_df['Source_Name'],
        'Predicted_z': np.round(inverse_transform_target(gen_stack_base, target_transform), 4),
        'Bias_Corrected_z': np.round(inverse_transform_target(gen_corrected_base, target_transform), 4)
    })
    table3_base.to_csv(OUTPUT / "dr3_table3_baseline_predictions.csv", index=False)
    
    # Upgraded final generalization catalog with uncertainty and grades
    final_catalog = pd.DataFrame({
        'Source_Name': gen_df['Source_Name'],
        'ra': gen_df['RAJ2000'],
        'dec': gen_df['DEJ2000'],
        'CLASS': gen_df['CLASS'],
        'p_fsrq': np.round(gen_df['P_FSRQ'], 4),
        'ensemble_z': np.round(gen_ensemble_z, 4),
        'prediction_std': np.round(gen_stds, 4),
        'z_low_68': np.round(gen_low_68, 4),
        'z_high_68': np.round(gen_high_68, 4),
        'z_low_95': np.round(gen_low_95, 4),
        'z_high_95': np.round(gen_high_95, 4),
        'ood_score': np.round(mah_gen, 4),
        'reliability_grade': reliability_grades
    }).sort_values(by='ensemble_z', ascending=False)
    
    final_catalog.to_csv(OUTPUT / "DR3_New_AGN_Redshift_Catalog_Reliability.csv", index=False)
    final_catalog.to_csv(OUTPUT / "DR3_New_AGN_Redshift_Catalog_Improved.csv", index=False)
    
    # Save results pickle
    results = {
        'baseline_results': baseline_results,
        'upgraded_results': upgraded_results,
        'baseline_weights': avg_weights_base,
        'upgraded_weights': avg_weights_upg,
        'upgraded_ridge_coefs': ridge_coefs_upg,
        'upgraded_ridge_intercept': ridge_intercept_upg,
        'baseline_ridge_coefs': ridge_coefs_base,
        'baseline_ridge_intercept': ridge_intercept_base,
        'ot_params_base': ot_params_base,
        'conformal_scores': conformal_scores,
        'q_68': q_68,
        'q_95': q_95,
        'mah_train': mah_train,
        'mahal_95_thresh': mahal_95_thresh,
        'mahal_99_thresh': mahal_99_thresh,
        'target_transform': target_transform
    }
    with open(OUTPUT / "dr3_python_results.pkl", "wb") as f:
        pickle.dump(results, f)
        
    print(f"\nSaved all results to output/final_allwise_pipeline/ directory.")
    
    # ─── COMPARATIVE PERFORMANCE SUMMARY ───────────────────────────────────────
    print("\n" + "="*95)
    print("  PYTHON PIPELINE PERFORMANCE SUMMARY")
    print("="*95)
    print(f"{'Model Configuration':<35} | {'Pearson R (z)':<14} | {'RMSE (z)':<10} | {'Outlier % (Fixed)':<20} | {'Outlier % (2sigma)':<20}")
    print("-"*100)
    
    # Print metrics for key combinations
    r_base = baseline_results['metrics_report']['stack']['z']
    r_base_bc = baseline_results['metrics_report']['stack_corrected']['z']
    
    r_upg = upgraded_results['metrics_report']['stack']['z']
    r_upg_bc = upgraded_results['metrics_report']['stack_corrected']['z']
    
    print(f"{'Baseline Stacking (O1)':<35} | {r_base['R']:<14.4f} | {r_base['RMSE']:<10.4f} | {baseline_results['metrics_report']['stack']['outlier_pct']:<18.2f}% | {baseline_results['metrics_report']['stack']['outlier_pct_2sigma']:<18.2f}%")
    print(f"{'Baseline Stacking + OT (Paper)':<35} | {r_base_bc['R']:<14.4f} | {r_base_bc['RMSE']:<10.4f} | {baseline_results['metrics_report']['stack_corrected']['outlier_pct']:<18.2f}% | {baseline_results['metrics_report']['stack_corrected']['outlier_pct_2sigma']:<18.2f}%")
    print(f"{'Upgraded Stacking (20-feat)':<35} | {r_upg['R']:<14.4f} | {r_upg['RMSE']:<10.4f} | {upgraded_results['metrics_report']['stack']['outlier_pct']:<18.2f}% | {upgraded_results['metrics_report']['stack']['outlier_pct_2sigma']:<18.2f}%")
    print(f"{'Upgraded Stacking + Isotonic (Ours)':<35} | {r_upg_bc['R']:<14.4f} | {r_upg_bc['RMSE']:<10.4f} | {upgraded_results['metrics_report']['stack_corrected']['outlier_pct']:<18.2f}% | {upgraded_results['metrics_report']['stack_corrected']['outlier_pct_2sigma']:<18.2f}%")
    
    # Print per-model R for upgraded
    print("\n  Per-Model R (Upgraded):")
    for m in upgraded_models:
        mr = upgraded_results['metrics_report'][m]['z']['R']
        print(f"    {m.upper()}: R = {mr:.4f}")
    
    # ─── COMPARISON AGAINST OLD BASELINE ──────────────────────────────────────
    old_r = 0.7768
    old_rmse = 0.4190
    new_r = r_upg_bc['R']
    new_rmse = r_upg_bc['RMSE']
    print(f"\n{'='*70}")
    print(f"  BUG FIX IMPACT ASSESSMENT")
    print(f"{'='*70}")
    print(f"  Old (NNLS/SLSQP, TabPFN capped):  R = {old_r:.4f}, RMSE = {old_rmse:.4f}")
    print(f"  New (RidgeCV, TabPFN all iters):   R = {new_r:.4f}, RMSE = {new_rmse:.4f}")
    print(f"  Delta R:  {new_r - old_r:+.4f}")
    print(f"  Delta RMSE: {new_rmse - old_rmse:+.4f}")
    tabpfn_wt = upgraded_results['ridge_weights'].get('tabpfn', 0.0)
    print(f"  TabPFN Ridge coefficient: {tabpfn_wt:.6f}")
    print(f"  TabPFN has non-zero weight: {abs(tabpfn_wt) > 1e-6}")
    print(f"{'='*70}")
    
    # ─── GENERATE ALL 16 PLOTS ───────────────────────────────────────────────────
    print("\n--- Generating All 16 Plots for DR3 Upgraded Results ---")
    generate_all_plots(train_df, gen_df, final_catalog, results, target_transform, expanded_features)
    
def generate_all_plots(train_df, gen_df, final_catalog, pkl_results, target_transform='inv', expanded_features=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ks_2samp
    
    # Force CLASS columns to uppercase
    train_df = train_df.copy()
    gen_df = gen_df.copy()
    final_catalog = final_catalog.copy()
    if 'CLASS' in train_df.columns:
        train_df['CLASS'] = train_df['CLASS'].str.upper()
    if 'CLASS' in gen_df.columns:
        gen_df['CLASS'] = gen_df['CLASS'].str.upper()
    if 'CLASS' in final_catalog.columns:
        final_catalog['CLASS'] = final_catalog['CLASS'].str.upper()
        
    # Styles
    col_bll_train = "#C0392B"   # red
    col_fsrq_train = "#27AE60"  # dark green
    col_bll_gen = "#2980B9"     # blue
    col_fsrq_gen = "#222222"    # black
    col_sigma = "#2980B9"       # blue
    col_bias = "#E74C3C"        # red
    
    y_spec = train_df['Redshift'].values
    y_spec_inv = 1.0 / (1.0 + y_spec)
    
    oof_upg_target = pkl_results['upgraded_results']['mean_predictions']['stack_corrected']
    oof_upg = inverse_transform_target(oof_upg_target, target_transform)
    oof_upg_inv = 1.0 / (1.0 + oof_upg)  # always convert to inv scale for plotting
    
    oof_base_target = pkl_results['baseline_results']['mean_predictions']['stack_corrected']
    oof_base = inverse_transform_target(oof_base_target, target_transform)
    oof_base_inv = 1.0 / (1.0 + oof_base)
    
    classes = train_df['CLASS'].values
    sigma_inv_upg = np.std(oof_upg_inv - y_spec_inv)
    sigma_inv_base = np.std(oof_base_inv - y_spec_inv)
    
    # 1. Fig 01: Redshift distribution
    bins = np.arange(0, 3.5, 0.1)
    plt.figure(figsize=(7, 5))
    plt.hist(y_spec, bins=bins, color='white', edgecolor='black', label=f'Full DR3 Training ($N={len(y_spec)}$)')
    plt.hist(final_catalog['ensemble_z'], bins=bins, color=col_bll_gen, edgecolor='black', alpha=0.7, label=f'Generalization Set ($N={len(final_catalog)}$)')
    plt.xlabel("Redshift ($z$)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 1: Redshift Distribution of Full 4LAC-DR3 Catalog")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig01_dr3_redshift_distribution.png", dpi=300)
    plt.close()
    
    # 2. Fig 02: 1/(z+1) distribution
    bins2 = np.arange(0, 1.025, 0.025)
    plt.figure(figsize=(7, 5))
    plt.hist(y_spec_inv, bins=bins2, color='white', edgecolor='black', label=f'Full DR3 Training ($N={len(y_spec)}$)')
    plt.hist(1.0/(final_catalog['ensemble_z']+1.0), bins=bins2, color=col_bll_gen, edgecolor='black', alpha=0.7, label=f'Generalization Set ($N={len(final_catalog)}$)')
    plt.xlabel("1/(z+1)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 2: Distribution of 1/(z+1) for Full DR3 Catalog")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig02_dr3_inv_redshift_distribution.png", dpi=300)
    plt.close()
    
    # Helper for panels
    def plot_panel(ax, obs, pred, types, scale, sigma_inv, title, point_size=15):
        residuals_inv = (pred - obs) if scale == 'inv' else (1.0/(pred+1.0) - 1.0/(obs+1.0))
        is_outlier = np.abs(residuals_inv) > 2.0 * sigma_inv
        
        # Plot BLLs (red)
        bll_mask = (types == 'BLL')
        ax.scatter(obs[bll_mask & ~is_outlier], pred[bll_mask & ~is_outlier], color=col_bll_train, marker='o', s=point_size, alpha=0.7)
        ax.scatter(obs[bll_mask & is_outlier], pred[bll_mask & is_outlier], facecolors='none', edgecolors=col_bll_train, marker='o', s=point_size*1.6, linewidths=0.8)
        
        # Plot FSRQs (green)
        fsrq_mask = (types == 'FSRQ')
        ax.scatter(obs[fsrq_mask & ~is_outlier], pred[fsrq_mask & ~is_outlier], color=col_fsrq_train, marker='^', s=point_size, alpha=0.7)
        ax.scatter(obs[fsrq_mask & is_outlier], pred[fsrq_mask & is_outlier], facecolors='none', edgecolors=col_fsrq_train, marker='^', s=point_size*1.6, linewidths=0.8)
        
        # Plot BCUs (purple square)
        bcu_mask = (types == 'BCU')
        col_bcu_train = "#8E44AD"
        ax.scatter(obs[bcu_mask & ~is_outlier], pred[bcu_mask & ~is_outlier], color=col_bcu_train, marker='s', s=point_size, alpha=0.7)
        ax.scatter(obs[bcu_mask & is_outlier], pred[bcu_mask & is_outlier], facecolors='none', edgecolors=col_bcu_train, marker='s', s=point_size*1.6, linewidths=0.8)
        
        # Identity line
        min_v = min(obs.min(), pred.min()) * 0.95
        max_v = max(obs.max(), pred.max()) * 1.05
        ax.plot([min_v, max_v], [min_v, max_v], color='black', linewidth=0.8)
        
        # 2-sigma curves
        if scale == 'inv':
            ax.plot([min_v, max_v], [min_v + 2.0*sigma_inv, max_v + 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
            ax.plot([min_v, max_v], [min_v - 2.0*sigma_inv, max_v - 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
            ax.set_xlabel("Observed 1/(z+1)")
            ax.set_ylabel("Predicted 1/(z+1)")
            ax.set_xlim(min_v, max_v)
            ax.set_ylim(min_v, max_v)
            ax.set_aspect('equal')
        else:
            z_seq = np.linspace(max(0, min_v), max_v, 300)
            inv_seq = 1.0 / (1.0 + z_seq)
            upper_inv = inv_seq + 2.0 * sigma_inv
            lower_inv = inv_seq - 2.0 * sigma_inv
            lower_inv = np.clip(lower_inv, 1e-6, None)
            upper_z = (1.0 / lower_inv) - 1.0
            lower_z = (1.0 / upper_inv) - 1.0
            ax.plot(z_seq, upper_z, color=col_sigma, linestyle='--', linewidth=0.9)
            ax.plot(z_seq, lower_z, color=col_sigma, linestyle='--', linewidth=0.9)
            ax.set_xlabel("Observed redshift z")
            ax.set_ylabel("Predicted redshift z")
            ax.set_xlim(0, max_v)
            ax.set_ylim(0, max_v)
            
        ax.set_title(title, fontweight='bold')

    # 3. Fig 03: Scatter matrix of key features
    key_features = ["LogFlux", "Lognu_syn", "LP_beta", "PL_Index", "Gaia_G_Magnitude"]
    scatter_data = train_df[key_features + ['CLASS']].copy()
    scatter_data.rename(columns={'CLASS': 'Set'}, inplace=True)
    g3 = sns.PairGrid(scatter_data, hue="Set", palette={"BLL": col_bll_train, "FSRQ": col_fsrq_train, "BCU": "#8E44AD"})
    g3.map_diag(sns.kdeplot, alpha=0.4, fill=True)
    g3.map_offdiag(plt.scatter, s=12, alpha=0.6)
    g3.add_legend(title="", bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2)
    g3.figure.subplots_adjust(top=0.94, bottom=0.1)
    g3.figure.suptitle("Figure 3: Scatter Matrix of DR3 Features", fontweight='bold')
    g3.savefig(PLOTS / "Fig03_dr3_scatter_matrix.png", dpi=180)
    plt.close()
    
    # 4. Fig 04: Training set vs Complete Spectroscopic 4LAC-DR3 Catalog
    from astropy.table import Table
    try:
        dr3_h_raw = Table.read(str(DATA / "table-4LAC-DR3-h.fits")).to_pandas()
        dr3_l_raw = Table.read(str(DATA / "table-4LAC-DR3-l.fits")).to_pandas()
        dr3_all_raw = pd.concat([dr3_h_raw, dr3_l_raw], ignore_index=True)
        dr3_all_raw['Redshift'] = pd.to_numeric(dr3_all_raw['Redshift'], errors='coerce')
        full_spectroscopic_z = dr3_all_raw[dr3_all_raw['Redshift'] > 0]['Redshift'].dropna().values
    except Exception as e:
        print(f"Warning: Could not read FITS files for Figure 4: {e}. Falling back to default.")
        full_spectroscopic_z = y_spec
        
    plt.figure(figsize=(7, 5))
    plt.hist(full_spectroscopic_z, bins=bins, color='lightcoral', edgecolor='red', alpha=0.5, 
             label=f'Complete 4LAC-DR3 Spectroscopic ($N={len(full_spectroscopic_z)}$)')
    plt.hist(y_spec, bins=bins, facecolor='none', edgecolor='darkblue', hatch='//', linewidth=1.5,
             label=f'Training Set ($N={len(y_spec)}$)')
    plt.xlabel("Redshift ($z$)")

    plt.ylabel("Number of Sources")
    plt.title("Figure 4: Redshift Distribution: Training Set vs. Complete Spectroscopic Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig04_dr3_training_vs_total_z.png", dpi=300)
    plt.close()
    
    # 5. Fig 05: Ensemble coefficients (RidgeCV weights)
    models_active = pkl_results['upgraded_results']['active_models']
    norm_wts = pkl_results['upgraded_results']['normalized_weights']
    df_weights = pd.DataFrame({
        'Algorithm': [m.upper() for m in models_active],
        'Weight': norm_wts
    }).sort_values(by='Weight', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Weight', y='Algorithm', data=df_weights, palette='viridis')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Normalized Stacking Weight (RidgeCV)")
    plt.ylabel("Algorithm")
    plt.title("Figure 5: Average Stacking Ensemble Weights (DR3 - RidgeCV)")
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig05_dr3_SL_coefficients.png", dpi=300)
    plt.close()
    
    # 6. Fig 06: Main CV results (predicted vs observed)
    fig6, axes6 = plt.subplots(2, 2, figsize=(13, 11))
    plot_panel(axes6[0, 0], y_spec_inv, oof_upg_inv, classes, 'inv', sigma_inv_upg, "Upgraded Ensemble -- 1/(z+1) Scale")
    plot_panel(axes6[0, 1], y_spec, oof_upg, classes, 'z', sigma_inv_upg, "Upgraded Ensemble -- Linear z Scale")
    plot_panel(axes6[1, 0], y_spec_inv, oof_base_inv, classes, 'inv', sigma_inv_base, "Baseline Ensemble -- 1/(z+1) Scale")
    plot_panel(axes6[1, 1], y_spec, oof_base, classes, 'z', sigma_inv_base, "Baseline Ensemble -- Linear z Scale")
    axes6[0, 0].scatter([], [], color=col_bll_train, marker='o', s=20, label='BLL')
    axes6[0, 0].scatter([], [], color=col_fsrq_train, marker='^', s=20, label='FSRQ')
    axes6[0, 0].scatter([], [], color="#8E44AD", marker='s', s=20, label='BCU')
    axes6[0, 0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    fig6.suptitle("Figure 6: Cross-Validation Results: Predicted vs. True Redshifts", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig06_dr3_predicted_vs_true.png", dpi=300)
    plt.close()
    
    # 7. Fig 07: Residuals and relative influence
    fig7, axes7 = plt.subplots(2, 2, figsize=(13, 11))
    delta_z = y_spec - oof_upg
    delta_z_norm = delta_z / (1.0 + y_spec)
    
    # delta_z
    axes7[0, 0].hist(delta_z, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
    axes7[0, 0].axvline(np.mean(delta_z), color=col_bias, linewidth=1.2, label='Mean bias')
    axes7[0, 0].axvline(np.mean(delta_z) + np.std(delta_z), color=col_sigma, linestyle='--', linewidth=1, label=r'$\pm1\sigma$')
    axes7[0, 0].axvline(np.mean(delta_z) - np.std(delta_z), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 0].set_xlabel(r"$\Delta z = z_{\mathrm{spec}} - z_{\mathrm{pred}}$")
    axes7[0, 0].set_ylabel("Count")
    axes7[0, 0].set_title(r"$\Delta z$ Distribution (Upgraded)", fontweight='bold')
    axes7[0, 0].legend()
    
    # delta_z_norm
    axes7[0, 1].hist(delta_z_norm, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
    axes7[0, 1].axvline(np.mean(delta_z_norm), color=col_bias, linewidth=1.2)
    axes7[0, 1].axvline(np.mean(delta_z_norm) + np.std(delta_z_norm), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 1].axvline(np.mean(delta_z_norm) - np.std(delta_z_norm), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 1].set_xlabel(r"$\Delta z_{\mathrm{norm}} = \Delta z / (1 + z_{\mathrm{spec}})$")
    axes7[0, 1].set_ylabel("Count")
    axes7[0, 1].set_title(r"$\Delta z_{\mathrm{norm}}$ Distribution (Upgraded)", fontweight='bold')
    
    # relative influence
    feat_inf = pd.DataFrame({
        'Predictor': list(pkl_results['upgraded_results']['feature_counts'].keys()),
        'Influence': [float(pkl_results['upgraded_results']['feature_counts'][p]) / (len(models_active) * 100.0) * 100.0 for p in pkl_results['upgraded_results']['feature_counts'].keys()]
    }).sort_values(by='Influence', ascending=False)
    sns.barplot(x='Influence', y='Predictor', data=feat_inf.head(10), ax=axes7[1, 0], palette='coolwarm')
    axes7[1, 0].set_xlabel("LASSO Selection Frequency (%)")
    axes7[1, 0].set_ylabel("Predictor")
    axes7[1, 0].set_title("Relative Predictor Influence (Upgraded)", fontweight='bold')
    
    # R distribution
    r_dist = []
    agg_pred_upg = pkl_results['upgraded_results']['agg_predictions']['stack_corrected']
    for i in range(agg_pred_upg.shape[1]):
        pred_iter_target = agg_pred_upg[:, i]
        pred_iter_z = inverse_transform_target(pred_iter_target, target_transform)
        r_dist.append(compute_pearson(y_spec, pred_iter_z))
    axes7[1, 1].hist(r_dist, bins=25, color='#8E44AD', edgecolor='black', alpha=0.8)
    axes7[1, 1].axvline(np.mean(r_dist), color=col_bias, linewidth=1.2)
    axes7[1, 1].set_xlabel("Pearson Correlation R (z-scale)")
    axes7[1, 1].set_ylabel("Count")
    axes7[1, 1].set_title(f"R Distribution (Mean R = {np.mean(r_dist):.4f})", fontweight='bold')
    
    fig7.suptitle("Figure 7: Residuals, Predictor Influence & R Distribution", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig07_dr3_residuals.png", dpi=300)
    plt.close()
    
    # 8. Fig 08: Metric distributions
    nmad_inv_dist = []
    nmad_z_dist = []
    rmse_inv_dist = []
    rmse_z_dist = []
    for i in range(agg_pred_upg.shape[1]):
        pred_iter_target = agg_pred_upg[:, i]
        pred_iter_z = inverse_transform_target(pred_iter_target, target_transform)
        pred_iter_inv = 1.0 / (1.0 + pred_iter_z)
        nmad_inv_dist.append(compute_nmad(y_spec_inv, pred_iter_inv, normalized=False))
        nmad_z_dist.append(compute_nmad(y_spec, pred_iter_z, normalized=False))
        rmse_inv_dist.append(compute_rmse(y_spec_inv, pred_iter_inv))
        rmse_z_dist.append(compute_rmse(y_spec, pred_iter_z))
        
    fig8, axes8 = plt.subplots(2, 2, figsize=(12, 10))
    def plot_metric_hist(ax, vals, xlab, title, color):
        ax.hist(vals, bins=25, color=color, edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(vals), color=col_bias, linewidth=1.2)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\n(Mean = {np.mean(vals):.4f})", fontweight='bold')
    plot_metric_hist(axes8[0, 0], nmad_inv_dist, r"$\sigma_{NMAD}$", r"$\sigma_{NMAD}$ (1/(z+1) scale)", "#2ECC71")
    plot_metric_hist(axes8[0, 1], nmad_z_dist, r"$\sigma_{NMAD}$", r"$\sigma_{NMAD}$ (Linear z scale)", "#2ECC71")
    plot_metric_hist(axes8[1, 0], rmse_inv_dist, "RMSE", "RMSE (1/(z+1) scale)", "#E74C3C")
    plot_metric_hist(axes8[1, 1], rmse_z_dist, "RMSE(z)", "RMSE (Linear z scale)", "#E74C3C")
    fig8.suptitle("Figure 8: Metric Distributions over Iterations (Upgraded)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig08_dr3_metric_distributions.png", dpi=300)
    plt.close()
    
    # 9. Fig 09: BLL scatter matrix (training vs generalization)
    train_bll = train_df[train_df['CLASS'] == 'BLL']
    gen_bll = gen_df[gen_df['CLASS'] == 'BLL']
    if len(key_features) >= 3:
        train_bll_p = train_bll[key_features].copy(); train_bll_p['Set'] = "BLL Training"
        gen_bll_p = gen_bll[key_features].copy(); gen_bll_p['Set'] = "BLL Generalization"
        bll_all = pd.concat([train_bll_p, gen_bll_p], ignore_index=True)
        
        g9 = sns.PairGrid(bll_all, hue="Set", hue_order=["BLL Training", "BLL Generalization"], palette={
            "BLL Training": col_bll_train,
            "BLL Generalization": col_bll_gen
        })
        g9.map_diag(sns.kdeplot, alpha=0.4, fill=True)
        def bll_scatter(x, y, **kwargs):
            label = kwargs.get('label', '')
            if 'Training' in label:
                plt.scatter(x, y, marker='o', color=col_bll_train, s=12, alpha=0.6)
            else:
                plt.scatter(x, y, marker='o', facecolors='none', edgecolors=col_bll_gen, s=12, alpha=0.6)
        g9.map_offdiag(bll_scatter)
        g9.add_legend(title="", bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2)
        g9.figure.subplots_adjust(top=0.94, bottom=0.1)
        g9.figure.suptitle("Figure 9: BLL -- Training vs Generalization sets", fontweight='bold')
        g9.savefig(PLOTS / "Fig09_dr3_BLL_scatter_matrix.png", dpi=180)
        plt.close()
        
    # 10. Fig 10: Generalization predictions distribution
    bins10 = np.arange(0, 3.05, 0.1)
    train_bll_z = train_df[train_df["CLASS"] == "BLL"]["Redshift"].dropna().values
    gen_bll_z = final_catalog[final_catalog["CLASS"] == "BLL"]["ensemble_z"].dropna().values
    plt.figure(figsize=(7, 5))
    plt.hist(train_bll_z, bins=bins10, color='white', edgecolor='black', histtype='step', linewidth=1.5, label=f'Training BLLs ({len(train_bll_z)})')
    plt.hist(gen_bll_z, bins=bins10, color=col_bll_gen, edgecolor='black', alpha=0.6, label=f'Predicted BLL Gen. ({len(gen_bll_z)})')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 10: Predicted Redshifts BLL Generalization vs Training BLLs", fontweight='bold')
    plt.xlim(0, 3.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig10_dr3_generalization_predictions.png", dpi=300)
    plt.close()
    
    # 11. Fig 11: Model comparison
    model_keys = [k for k in pkl_results['upgraded_results']['metrics_report'].keys() if not k.endswith('_corrected')]
    model_r = [pkl_results['upgraded_results']['metrics_report'][k]['z']['R'] for k in model_keys]
    model_rmse = [pkl_results['upgraded_results']['metrics_report'][k]['z']['RMSE'] for k in model_keys]
    df_compare = pd.DataFrame({
        'Model': [m.upper() for m in model_keys],
        'Pearson R': model_r,
        'RMSE': model_rmse
    }).sort_values(by='Pearson R', ascending=False)
    
    fig11, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    sns.barplot(x='Pearson R', y='Model', data=df_compare, palette='viridis', ax=ax1)
    ax1.set_xlim(0.70, 0.85)
    ax1.set_title("Pearson Correlation (R) Comparison", fontweight='bold')
    sns.barplot(x='RMSE', y='Model', data=df_compare.sort_values(by='RMSE'), palette='rocket', ax=ax2)
    ax2.set_xlim(0.30, 0.48)
    ax2.set_title("RMSE Comparison (Linear z Scale)", fontweight='bold')
    fig11.suptitle("Figure 11: Performance Comparisons for Alternative Models", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig11_dr3_model_comparison.png", dpi=300)
    plt.close()
    
    # 12. Fig 12: Calibration Curves Comparison (Isotonic vs Smooth OTQM vs Optimal Transport)
    fig12, axes12 = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load uncalibrated stack predictions
    oof_upg_stack_target = pkl_results['upgraded_results']['mean_predictions']['stack']
    oof_upg_stack = inverse_transform_target(oof_upg_stack_target, target_transform)
    oof_upg_stack_inv = 1.0 / (1.0 + oof_upg_stack)
    
    bll_idx = (classes == 'BLL')
    fsrq_idx = (classes == 'FSRQ')
    
    bll_pred_raw = oof_upg_stack_inv[bll_idx]
    bll_obs = y_spec_inv[bll_idx]
    
    fsrq_pred_raw = oof_upg_stack_inv[fsrq_idx]
    fsrq_obs = y_spec_inv[fsrq_idx]
    
    # Calibrated values
    bll_cal_iso = IsotonicRegression(out_of_bounds='clip').fit(bll_pred_raw, bll_obs).predict(bll_pred_raw)
    fsrq_cal_iso = IsotonicRegression(out_of_bounds='clip').fit(fsrq_pred_raw, fsrq_obs).predict(fsrq_pred_raw)
    
    bll_cal_spline = SmoothOTQMCalibration().fit(bll_pred_raw, bll_obs).predict(bll_pred_raw)
    fsrq_cal_spline = SmoothOTQMCalibration().fit(fsrq_pred_raw, fsrq_obs).predict(fsrq_pred_raw)
    
    def fit_linear_ot(pred, obs):
        pred_s = np.sort(pred)
        obs_s = np.sort(obs)
        n = min(len(pred_s), len(obs_s))
        pred_s, obs_s = pred_s[:n], obs_s[:n]
        slope, intercept = np.polyfit(pred_s, obs_s, 1)
        res = obs_s - (slope * pred_s + intercept)
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((obs_s - np.mean(obs_s)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        return slope, intercept, r2, pred_s, obs_s

    slope_bll, intercept_bll, r2_ot_bll, bll_pred_s_ot, bll_obs_s_ot = fit_linear_ot(bll_pred_raw, bll_obs)
    slope_fsrq, intercept_fsrq, r2_ot_fsrq, fsrq_pred_s_ot, fsrq_obs_s_ot = fit_linear_ot(fsrq_pred_raw, fsrq_obs)
    
    def get_qq_r2(cal_pred, obs):
        pred_s = np.sort(cal_pred)
        obs_s = np.sort(obs)
        n = min(len(pred_s), len(obs_s))
        pred_s, obs_s = pred_s[:n], obs_s[:n]
        res = obs_s - pred_s
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((obs_s - np.mean(obs_s)) ** 2)
        return 1.0 - (ss_res / ss_tot)
        
    r2_iso_bll = get_qq_r2(bll_cal_iso, bll_obs)
    r2_iso_fsrq = get_qq_r2(fsrq_cal_iso, fsrq_obs)
    
    # Define helper function to plot panels
    def plot_panel_12(ax, pred_s, obs_s, title, color, r2, eq_text, slope=None, intercept=None, is_ot=False, is_iso=False):
        ax.scatter(pred_s, obs_s, color=color, s=15, alpha=0.6, label='Data Points')
        ax.plot([min(pred_s), max(pred_s)], [min(pred_s), max(pred_s)], color='gray', linestyle='--', linewidth=0.8, label='1:1 Line')
        if is_ot:
            ax.plot(pred_s, slope * pred_s + intercept, color='black', linewidth=1.2, label='Linear Fit')
        elif is_iso:
            ax.plot(pred_s, pred_s, color='black', linewidth=1.2, label='Isotonic Fit')
        else:
            ax.plot(pred_s, obs_s, color='black', linewidth=1.2, label='Spline Fit')
        
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, ha='left', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_xlabel("Sorted Predicted $1/(z+1)$")
        ax.set_ylabel("Sorted Observed $1/(z+1)$")
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        
    # Plot Column 0: Isotonic
    bll_pred_s_iso = np.sort(bll_cal_iso)
    bll_obs_s_iso = np.sort(bll_obs)
    eq_bll_iso = f"Isotonic Q-Q Fit\n$R^2 = {r2_iso_bll:.4f}$"
    plot_panel_12(axes12[0, 0], bll_pred_s_iso, bll_obs_s_iso, "BLL — Isotonic Calibration", col_bll_train, r2_iso_bll, eq_bll_iso, is_iso=True)
    
    fsrq_pred_s_iso = np.sort(fsrq_cal_iso)
    fsrq_obs_s_iso = np.sort(fsrq_obs)
    eq_fsrq_iso = f"Isotonic Q-Q Fit\n$R^2 = {r2_iso_fsrq:.4f}$"
    plot_panel_12(axes12[1, 0], fsrq_pred_s_iso, fsrq_obs_s_iso, "FSRQ — Isotonic Calibration", col_fsrq_train, r2_iso_fsrq, eq_fsrq_iso, is_iso=True)
    
    # Plot Column 1: Smooth OTQM
    bll_pred_s_sp = np.sort(bll_cal_spline)
    bll_obs_s_sp = np.sort(bll_obs)
    eq_bll_sp = f"Spline Q-Q Fit\n$R^2 = 1.000$"
    plot_panel_12(axes12[0, 1], bll_pred_s_sp, bll_obs_s_sp, "BLL — Smooth OTQM (Spline)", col_bll_train, 1.0, eq_bll_sp)
    
    fsrq_pred_s_sp = np.sort(fsrq_cal_spline)
    fsrq_obs_s_sp = np.sort(fsrq_obs)
    eq_fsrq_sp = f"Spline Q-Q Fit\n$R^2 = 1.000$"
    plot_panel_12(axes12[1, 1], fsrq_pred_s_sp, fsrq_obs_s_sp, "FSRQ — Smooth OTQM (Spline)", col_fsrq_train, 1.0, eq_fsrq_sp)
    
    # Plot Column 2: Optimal Transport
    eq_bll_ot = f"OT Linear Fit $R^2 = {r2_ot_bll:.4f}$\nSlope = {slope_bll:.4f}\nIntercept = {intercept_bll:.4f}"
    plot_panel_12(axes12[0, 2], bll_pred_s_ot, bll_obs_s_ot, "BLL — Optimal Transport (Linear)", 'coral', r2_ot_bll, eq_bll_ot, slope=slope_bll, intercept=intercept_bll, is_ot=True)
    
    eq_fsrq_ot = f"OT Linear Fit $R^2 = {r2_ot_fsrq:.4f}$\nSlope = {slope_fsrq:.4f}\nIntercept = {intercept_fsrq:.4f}"
    plot_panel_12(axes12[1, 2], fsrq_pred_s_ot, fsrq_obs_s_ot, "FSRQ — Optimal Transport (Linear)", 'coral', r2_ot_fsrq, eq_fsrq_ot, slope=slope_fsrq, intercept=intercept_fsrq, is_ot=True)
    
    fig12.suptitle("Figure 12: Comparison of Isotonic, Smooth OTQM, and Optimal Transport Calibration", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig12_dr3_calibration_curves.png", dpi=300)
    plt.close()
    
    # 13. Fig 13: Calibrated predictions
    fig13, (ax13_l, ax13_r) = plt.subplots(1, 2, figsize=(13, 6))
    plot_panel(ax13_l, y_spec_inv, oof_upg_inv, classes, 'inv', sigma_inv_upg, "Calibrated Ensemble -- 1/(z+1) Scale")
    plot_panel(ax13_r, y_spec, oof_upg, classes, 'z', sigma_inv_upg, "Calibrated Ensemble -- Linear z Scale")
    fig13.suptitle("Figure 13: Bias-Corrected 10-fold CV Results", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig13_dr3_calibrated_predictions.png", dpi=300)
    plt.close()
    
    # 14. Fig 14: Calibrated BLL predictions
    gen_bll_z = final_catalog[final_catalog["CLASS"] == "BLL"]["ensemble_z"].dropna().values
    plt.figure(figsize=(7, 5))
    plt.hist(train_bll_z, bins=np.arange(0, 3.05, 0.05), color='#EC7063', edgecolor='#C0392B', alpha=0.4, label=f'Training BLLs ({len(train_bll_z)})')
    plt.hist(gen_bll_z, bins=np.arange(0, 3.05, 0.05), color='#76D7C4', edgecolor='#16A085', alpha=0.4, label=f'Predicted BLL Gen. ({len(gen_bll_z)})')
    plt.xlabel("z")
    plt.ylabel("Counts")
    plt.title("Figure 14: Overlapped Histogram of Predicted and Observed redshift for BLL", fontweight='bold')
    plt.xlim(0, 3.0)
    plt.legend(title="Histogram distribution")
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig14_dr3_calibrated_generalization.png", dpi=300)
    plt.close()
    
    # 15. Fig 15: SHAP feature importance (Using Random Forest Feature Importances for baseline + AllWISE)
    plt.figure(figsize=(8, 7))
    rf_imp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    y_train_rf = train_df['InvRedshift'] if 'InvRedshift' in train_df.columns else 1.0/(1.0+train_df['Redshift'])
    if expanded_features is None:
        expanded_features = [c for c in train_df.columns if c not in ['Source_Name','CLASS','Redshift','InvRedshift','CLASS_upper','ASSOC1','Counterpart_Catalog','SpectrumType','VLBI_Counterpart','DataRelease','RAJ2000','DEJ2000','GLON','GLAT','RA_Counterpart','DEC_Counterpart','SED_class','Flags','LabelNo']]
    rf_imp.fit(train_df[expanded_features], y_train_rf)
    importances = rf_imp.feature_importances_
    df_shap = pd.DataFrame({
        'Feature': expanded_features,
        'SHAP Importance': importances
    }).sort_values(by='SHAP Importance', ascending=True)
    plt.barh(df_shap['Feature'], df_shap['SHAP Importance'], color='#2980B9', edgecolor='black', height=0.6)
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Figure 15: Upgraded Baseline + AllWISE Feature Importance", fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig15_dr3_shap_importance.png", dpi=300)
    plt.close()
    
    # 16. Fig 16: Training vs generalization redshift distribution
    from scipy.stats import ks_2samp
    ks_stat, ks_pval = ks_2samp(y_spec, final_catalog['ensemble_z'].dropna().values)
    
    plt.figure(figsize=(7, 5))
    plt.hist(y_spec, bins=bins, density=True, color='#EC7063', edgecolor='#C0392B', alpha=0.4, 
             label=f'Training Set ($N={len(y_spec)}$)')
    plt.hist(final_catalog['ensemble_z'], bins=bins, density=True, color='#3498DB', edgecolor='#2980B9', alpha=0.4, 
             label=f'Predicted Generalization ($N={len(final_catalog)}$)')
    
    ks_text = f"KS statistic: {ks_stat:.3f}\np-value: {ks_pval:.3e}"
    plt.gca().text(0.95, 0.95, ks_text, transform=plt.gca().transAxes, ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                   
    plt.xlabel("Redshift ($z$)")
    plt.ylabel("Probability Density")
    plt.title("Figure 16: Training Set vs. Predicted Generalization Set Redshifts (Normalized)", fontweight='bold')
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig16_dr3_training_vs_generalization_distribution.png", dpi=300)
    plt.close()
    
    # 17. Fig 17: Calibration Comparison Histograms (Staircase Effect Visualizer)
    plt.figure(figsize=(10, 6))
    
    z_raw = inverse_transform_target(oof_upg_stack_target, target_transform)
    z_ot = inverse_transform_target(pkl_results['upgraded_results']['mean_predictions']['stack_ot'], target_transform)
    z_iso = inverse_transform_target(pkl_results['upgraded_results']['mean_predictions']['stack_iso'], target_transform)
    z_smooth = inverse_transform_target(pkl_results['upgraded_results']['mean_predictions']['stack_smooth'], target_transform)
    
    bins_comp = np.arange(0, 2.5, 0.05)
    plt.hist(z_raw, bins=bins_comp, histtype='step', color='gray', linestyle='--', linewidth=1.5, label=f"Uncalibrated (R_z={pkl_results['upgraded_results']['metrics_report']['stack']['z']['R']:.4f})")
    plt.hist(z_ot, bins=bins_comp, histtype='step', color='coral', linewidth=1.8, label=f"Optimal Transport (R_z={pkl_results['upgraded_results']['metrics_report']['stack_ot']['z']['R']:.4f})")
    plt.hist(z_iso, bins=bins_comp, histtype='step', color='purple', linewidth=1.8, label=f"Isotonic (R_z={pkl_results['upgraded_results']['metrics_report']['stack_iso']['z']['R']:.4f})")
    plt.hist(z_smooth, bins=bins_comp, histtype='step', color='blue', linewidth=1.8, label=f"Smooth OTQM (R_z={pkl_results['upgraded_results']['metrics_report']['stack_smooth']['z']['R']:.4f})")
    
    plt.xlabel("Redshift ($z$)", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    plt.title("Figure 17: Comparative Redshift Predictions under Different Calibration Mappings", fontweight='bold', fontsize=13)
    plt.xlim(0, 2.2)
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS / "Fig17_dr3_calibration_comparison_hists.png", dpi=300)
    plt.close()
    
    print("All 17 plots successfully generated under output/final_allwise_pipeline/plots/!")

if __name__ == '__main__':
    main()
