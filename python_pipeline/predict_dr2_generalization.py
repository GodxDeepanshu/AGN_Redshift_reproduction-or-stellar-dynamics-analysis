import numpy as np
import pandas as pd
import os
import sys
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
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
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

# Insert path for helper modules
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics
from tabpfn import TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig
import tabm

# Create output directories
os.makedirs("plots/dr3_plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ─── SAINT Custom Implementation ─────────────────────────────────────────────
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

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=80, lr=0.001, batch_size=64, is_tabm=False):
    device = torch.device("cpu")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if is_tabm:
                preds = model(batch_X)
                loss = criterion(preds, batch_y.unsqueeze(1))
            else:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_y_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            
            if is_tabm:
                val_preds = model(val_X_t)
                val_loss = criterion(val_preds, val_y_t.unsqueeze(1)).item()
            else:
                val_preds = model(val_X_t)
                val_loss = criterion(val_preds, val_y_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model

def run_predictions():
    print("======================================================================")
    print("  RUNNING UNIFIED 19-FEATURE PHOTOMETRIC REDSHIFT PIPELINE")
    print("======================================================================")
    
    # ─── 1. DATA LOADING ──────────────────────────────────────────────────────
    print("\n--- Phase 1: Data Loading ---")
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
            
    dr2 = pd.read_csv("data/4LAC-DR2.csv")
    dr2_names = set(dr2['Source_Name'].astype(str).str.strip())
    new_names = set(dr3['Source_Name']) - dr2_names
    target_df = dr3[dr3['Source_Name'].isin(new_names)].copy()
    
    # Merge Gaia & MW to target
    gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    target_df = pd.merge(target_df, gaia_new, on="Source_Name", how="left")
    target_df = pd.merge(target_df, mw, on="Source_Name", how="left")
    
    # Load generalization DR2 catalog
    gen_df = pd.read_csv("data/generalization_set.csv")
    gen_df['Source_Name'] = gen_df['Source_Name'].astype(str).str.strip()
    gen_df = pd.merge(gen_df, mw, on="Source_Name", how="left")
    
    print(f"Loaded training eligible sources: {len(train_df)}")
    print(f"Loaded target DR3 new sources:    {len(target_df)}")
    print(f"Loaded generalization DR2 set:    {len(gen_df)}")
    
    # ─── 2. PREPROCESSING & MISSINGNESS HANDLING ────────────────────────────────
    print("\n--- Phase 2: Preprocessing & Physical Upper Limits ---")
    
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
    for df in [train_df, target_df, gen_df]:
        df['CLASS_upper'] = df['CLASS'].astype(str).str.upper()
        df['AGN_Type'] = np.where(df['CLASS_upper'].isin(['BLL', 'BL LAC']), 'BLL',
                         np.where(df['CLASS_upper'].isin(['FSRQ', 'FLAT SPECTRUM RADIO QUASAR']), 'FSRQ', 'BCU'))
        
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
        df['Is_Upper_Limit_Xray'] = df['CR0'].isna().astype(float)
        df['LogXrayFlux'] = np.where(df['CR0'].isna(), -4.0, np.log10(df['CR0'].clip(lower=1e-6)))
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
    gen_df[impute_features] = imputer.transform(gen_df[impute_features])
    
    # Calculate engineered features POST-IMPUTATION to keep mathematical relationships exact
    for df in [train_df, target_df, gen_df]:
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
    
    for train_idx, val_idx in skf.split(X_train_class, y_train_class):
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train_class.iloc[train_idx], y_train_class[train_idx])
        train_probs[val_idx] = clf.predict_proba(X_train_class.iloc[val_idx])[:, 1]
        
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_class, y_train_class)
    
    target_df['P_FSRQ'] = final_clf.predict_proba(target_df[predictors])[:, 1]
    gen_df['P_FSRQ'] = final_clf.predict_proba(gen_df[predictors])[:, 1]
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
    X_gen = gen_df[predictors].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_target_sc = scaler.transform(X_target)
    X_gen_sc = scaler.transform(X_gen)
    
    # Base learners
    xgb_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    lgb_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    cat_reg = cb.CatBoostRegressor(iterations=300, learning_rate=0.04, depth=4, random_seed=42, thread_count=-1, verbose=0)
    tabpfn_reg = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
    
    # Stacking level 0 CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_trans = {
        'xgb': np.zeros(len(train_df)),
        'lgb': np.zeros(len(train_df)),
        'cat': np.zeros(len(train_df)),
        'tabpfn': np.zeros(len(train_df))
    }
    
    print("Running 5-fold CV to generate meta-features...")
    for train_idx, val_idx in cv.split(X):
        w_fold = weights[train_idx]
        
        xgb_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        lgb_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        cat_reg.fit(X[train_idx], y_train_trans[train_idx], sample_weight=w_fold)
        
        X_tr_sc = scaler.fit_transform(X[train_idx])
        X_va_sc = scaler.transform(X[val_idx])
        tabpfn_reg.fit(X_tr_sc, y_train_trans[train_idx])
        
        oof_preds_trans['xgb'][val_idx] = xgb_reg.predict(X[val_idx])
        oof_preds_trans['lgb'][val_idx] = lgb_reg.predict(X[val_idx])
        oof_preds_trans['cat'][val_idx] = cat_reg.predict(X[val_idx])
        oof_preds_trans['tabpfn'][val_idx] = tabpfn_reg.predict(X_va_sc)
        
    oof_preds_inv = {}
    for key in oof_preds_trans.keys():
        oof_preds_inv[key] = qt.inverse_transform(oof_preds_trans[key].reshape(-1, 1)).flatten()
        
    # NNLS Stacking Meta-Learner
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
    print("Training final base regressors...")
    xgb_reg.fit(X, y_train_trans, sample_weight=weights)
    lgb_reg.fit(X, y_train_trans, sample_weight=weights)
    cat_reg.fit(X, y_train_trans, sample_weight=weights)
    tabpfn_reg.fit(X_sc, y_train_trans)
    
    # Predict on target & generalization sets (and inverse transform to inverse redshift)
    pred_trans_target = {
        'xgb': xgb_reg.predict(X_target),
        'lgb': lgb_reg.predict(X_target),
        'cat': cat_reg.predict(X_target),
        'tabpfn': tabpfn_reg.predict(X_target_sc)
    }
    
    pred_trans_gen = {
        'xgb': xgb_reg.predict(X_gen),
        'lgb': lgb_reg.predict(X_gen),
        'cat': cat_reg.predict(X_gen),
        'tabpfn': tabpfn_reg.predict(X_gen_sc)
    }
    
    pred_inv_target = {}
    pred_inv_gen = {}
    for key in pred_trans_target.keys():
        pred_inv_target[key] = qt.inverse_transform(pred_trans_target[key].reshape(-1, 1)).flatten()
        pred_inv_gen[key] = qt.inverse_transform(pred_trans_gen[key].reshape(-1, 1)).flatten()
        
    # Meta predictions
    X_level1_target = np.column_stack([pred_inv_target['xgb'], pred_inv_target['lgb'], pred_inv_target['cat'], pred_inv_target['tabpfn']])
    X_level1_gen = np.column_stack([pred_inv_gen['xgb'], pred_inv_gen['lgb'], pred_inv_gen['cat'], pred_inv_gen['tabpfn']])
    
    target_df['pred_stack_inv'] = meta_model.predict(X_level1_target)
    gen_df['pred_stack_inv'] = meta_model.predict(X_level1_gen)
    
    # ─── 6. TRAINING DEEP MODELS ON CPU ──────────────────────────────────────────
    print("\n--- Phase 6: Training Tabular Deep Learning Models (TabNet, TabM, SAINT, FT-Transformer) ---")
    
    # Out of fold predictions for deep learning models on train set
    # (Since training them in CV is slow, we use a single validation split to predict on target sets)
    # We split training set to train the neural models
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X))
    val_set_size = int(len(X) * 0.15)
    val_idx_nn = shuffled_indices[:val_set_size]
    train_idx_nn = shuffled_indices[val_set_size:]
    
    X_tr, X_val = X_sc[train_idx_nn], X_sc[val_idx_nn]
    y_tr, y_val = y_train[train_idx_nn], y_train[val_idx_nn]
    
    # TabNet
    print("Training TabNet...")
    model_tabnet = TabNetRegressor(verbose=0, device_name='cpu', n_d=8, n_a=8, n_steps=3, n_shared=2, seed=42)
    model_tabnet.fit(X_sc[train_idx_nn], y_train[train_idx_nn].reshape(-1, 1),
                       eval_set=[(X_sc[val_idx_nn], y_train[val_idx_nn].reshape(-1, 1))],
                       max_epochs=80, patience=15, batch_size=128, virtual_batch_size=16)
    
    pred_inv_target['tabnet'] = model_tabnet.predict(X_target_sc).flatten()
    pred_inv_gen['tabnet'] = model_tabnet.predict(X_gen_sc).flatten()
    
    # TabM
    print("Training TabM...")
    model_tabm = tabm.TabM.make(n_num_features=X_sc.shape[1], cat_cardinalities=None, d_out=1, k=32)
    model_tabm = train_pytorch_model(model_tabm, X_tr, y_tr, X_val, y_val, epochs=80, lr=0.001, batch_size=64, is_tabm=True)
    
    model_tabm.eval()
    with torch.no_grad():
        pred_inv_target['tabm'] = model_tabm(torch.tensor(X_target_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()
        pred_inv_gen['tabm'] = model_tabm(torch.tensor(X_gen_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()
        
    # SAINT
    print("Training SAINT...")
    model_saint = SAINTRegressor(num_features=X_sc.shape[1], embedding_dim=32, num_heads=2, depth=2, dropout=0.1)
    model_saint = train_pytorch_model(model_saint, X_tr, y_tr, X_val, y_val, epochs=80, lr=0.001, batch_size=64, is_tabm=False)
    
    model_saint.eval()
    with torch.no_grad():
        pred_inv_target['saint'] = model_saint(torch.tensor(X_target_sc, dtype=torch.float32)).flatten().numpy()
        pred_inv_gen['saint'] = model_saint(torch.tensor(X_gen_sc, dtype=torch.float32)).flatten().numpy()
        
    # FT-Transformer
    print("Training FT-Transformer...")
    train_df_inner = pd.DataFrame(X_sc[train_idx_nn], columns=predictors)
    train_df_inner['target'] = y_train[train_idx_nn]
    
    val_df_inner = pd.DataFrame(X_sc[val_idx_nn], columns=predictors)
    val_df_inner['target'] = y_train[val_idx_nn]
    
    data_config = DataConfig(target=['target'], continuous_cols=predictors, categorical_cols=[])
    trainer_config = TrainerConfig(max_epochs=20, batch_size=128, checkpoints=None, progress_bar="none", accelerator="cpu", devices=1, seed=42)
    model_config = FTTransformerConfig(task="regression", num_attn_blocks=2, num_heads=2, input_embed_dim=16, seed=42)
    
    model_ft = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config, optimizer_config=OptimizerConfig())
    model_ft.logger = False
    model_ft.fit(train_df_inner, validation=val_df_inner)
    
    new_df_inner = pd.DataFrame(X_target_sc, columns=predictors)
    pred_inv_target['ft'] = model_ft.predict(new_df_inner)['target_prediction'].values
    
    gen_df_inner_sc = pd.DataFrame(X_gen_sc, columns=predictors)
    pred_inv_gen['ft'] = model_ft.predict(gen_df_inner_sc)['target_prediction'].values

    # ─── 7. BIAS CALIBRATION (ISOTONIC) & CONFORMAL PREDICTION ─────────────────
    print("\n--- Phase 7: Calibration & Conformal Prediction ---")
    
    # Fit class-specific Isotonic regression
    oof_stack_inv = meta_model.predict(X_level1)
    
    bll_mask = (train_df['LabelNo'].values == 0)
    fsrq_mask = (train_df['LabelNo'].values == 1)
    
    iso_bll = IsotonicRegression(out_of_bounds='clip').fit(oof_stack_inv[bll_mask], y_train[bll_mask])
    iso_fsrq = IsotonicRegression(out_of_bounds='clip').fit(oof_stack_inv[fsrq_mask], y_train[fsrq_mask])
    
    # Calibrate training out-of-fold predictions
    p_train = train_df['P_FSRQ'].values
    oof_bll_c = iso_bll.predict(oof_stack_inv)
    oof_fsrq_c = iso_fsrq.predict(oof_stack_inv)
    oof_corr_inv = (1.0 - p_train) * oof_bll_c + p_train * oof_fsrq_c
    oof_ensemble_z = (1.0 / oof_corr_inv) - 1.0
    
    # Calibrate target set (DR3) predictions
    p_target = target_df['P_FSRQ'].values
    new_bll_c = iso_bll.predict(target_df['pred_stack_inv'].values)
    new_fsrq_c = iso_fsrq.predict(target_df['pred_stack_inv'].values)
    new_corr_inv = (1.0 - p_target) * new_bll_c + p_target * new_fsrq_c
    ensemble_z_target = (1.0 / new_corr_inv) - 1.0
    
    # Calibrate generalization set (DR2) predictions
    p_gen = gen_df['P_FSRQ'].values
    gen_bll_c = iso_bll.predict(gen_df['pred_stack_inv'].values)
    gen_fsrq_c = iso_fsrq.predict(gen_df['pred_stack_inv'].values)
    gen_corr_inv = (1.0 - p_gen) * gen_bll_c + p_gen * gen_fsrq_c
    ensemble_z_gen = (1.0 / gen_corr_inv) - 1.0
    
    # Locally Weighted Conformal Prediction
    # Epistemic uncertainty = standard deviation among base models
    pred_inv_matrix_target = np.column_stack([pred_inv_target['xgb'], pred_inv_target['lgb'], pred_inv_target['cat'], pred_inv_target['tabpfn']])
    pred_z_matrix_target = 1.0 / pred_inv_matrix_target - 1.0
    std_z_target = np.std(pred_z_matrix_target, axis=1)
    
    pred_inv_matrix_gen = np.column_stack([pred_inv_gen['xgb'], pred_inv_gen['lgb'], pred_inv_gen['cat'], pred_inv_gen['tabpfn']])
    pred_z_matrix_gen = 1.0 / pred_inv_matrix_gen - 1.0
    std_z_gen = np.std(pred_z_matrix_gen, axis=1)
    
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
    
    # Target set conformal intervals
    z_low_target = np.clip(ensemble_z_target - q_68 * std_z_target, a_min=0, a_max=None)
    z_high_target = ensemble_z_target + q_68 * std_z_target
    z_low_95_target = np.clip(ensemble_z_target - q_95 * std_z_target, a_min=0, a_max=None)
    z_high_95_target = ensemble_z_target + q_95 * std_z_target
    confidence_score_target = np.exp(-std_z_target / (1 + ensemble_z_target))
    
    # Generalization set conformal intervals
    z_low_gen = np.clip(ensemble_z_gen - q_68 * std_z_gen, a_min=0, a_max=None)
    z_high_gen = ensemble_z_gen + q_68 * std_z_gen
    z_low_95_gen = np.clip(ensemble_z_gen - q_95 * std_z_gen, a_min=0, a_max=None)
    z_high_95_gen = ensemble_z_gen + q_95 * std_z_gen
    confidence_score_gen = np.exp(-std_z_gen / (1 + ensemble_z_gen))
    
    # ─── 8. SAVING OUTPUT FILES ─────────────────────────────────────────────────
    print("\n--- Phase 8: Saving Refined Catalogs ---")
    
    # Out of distribution (OOD) score using Mahalanobis distance
    scaler_ood = StandardScaler()
    X_train_sc_ood = scaler_ood.fit_transform(train_df[predictors].values)
    X_target_sc_ood = scaler_ood.transform(target_df[predictors].values)
    X_gen_sc_ood = scaler_ood.transform(gen_df[predictors].values)
    
    mean_tr = np.mean(X_train_sc_ood, axis=0)
    cov_tr = np.cov(X_train_sc_ood, rowvar=False)
    inv_cov_tr = np.linalg.pinv(cov_tr)
    
    def mahalanobis_distance(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        
    mah_target = np.array([mahalanobis_distance(x, mean_tr, inv_cov_tr) for x in X_target_sc_ood])
    mah_gen = np.array([mahalanobis_distance(x, mean_tr, inv_cov_tr) for x in X_gen_sc_ood])
    
    # Normalize score
    mah_min, mah_max = min(mah_target.min(), mah_gen.min()), max(mah_target.max(), mah_gen.max())
    target_df['OOD_Score'] = (mah_target - mah_min) / (mah_max - mah_min + 1e-6)
    gen_df['OOD_Score'] = (mah_gen - mah_min) / (mah_max - mah_min + 1e-6)
    
    # Function to assign reliability grades
    def assign_grades(df, std_z, q_68):
        conf_width = 2.0 * q_68 * std_z
        grades = []
        for idx in range(len(df)):
            o_s = df['OOD_Score'].values[idx]
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
        return grades
        
    target_df['Reliability_Grade'] = assign_grades(target_df, std_z_target, q_68)
    gen_df['Reliability_Grade'] = assign_grades(gen_df, std_z_gen, q_68)
    
    # Save DR3 target set catalog
    final_catalog_target = pd.DataFrame({
        'source_name': target_df['Source_Name'],
        '4fgl_name': target_df['Source_Name'],
        'ra': target_df['RAJ2000'],
        'dec': target_df['DEJ2000'],
        'agn_class': target_df['CLASS'],
        'agn_type_est': np.where(target_df['AGN_Type'] == 'BCU', np.where(target_df['P_FSRQ'] >= 0.5, 'FSRQ', 'BLL'), target_df['AGN_Type']),
        'p_fsrq': np.round(target_df['P_FSRQ'], 4),
        'xgb_z': np.round(1.0 / pred_inv_target['xgb'] - 1.0, 4),
        'lgbm_z': np.round(1.0 / pred_inv_target['lgb'] - 1.0, 4),
        'catboost_z': np.round(1.0 / pred_inv_target['cat'] - 1.0, 4),
        'ft_z': np.round(1.0 / pred_inv_target['ft'] - 1.0, 4),
        'tabpfn_z': np.round(1.0 / pred_inv_target['tabpfn'] - 1.0, 4),
        'tabnet_z': np.round(1.0 / pred_inv_target['tabnet'] - 1.0, 4),
        'tabm_z': np.round(1.0 / pred_inv_target['tabm'] - 1.0, 4),
        'saint_z': np.round(1.0 / pred_inv_target['saint'] - 1.0, 4),
        'ensemble_z': np.round(ensemble_z_target, 4),
        'prediction_std': np.round(std_z_target, 4),
        'confidence_score': np.round(confidence_score_target, 4),
        'z_low_68': np.round(z_low_target, 4),
        'z_high_68': np.round(z_high_target, 4),
        'z_low_95': np.round(z_low_95_target, 4),
        'z_high_95': np.round(z_high_95_target, 4),
        'ood_score': np.round(target_df['OOD_Score'], 4),
        'reliability_grade': target_df['Reliability_Grade']
    })
    
    final_catalog_target.to_csv("data/DR3_New_AGN_Redshift_Catalog_Improved.csv", index=False)
    print("Saved DR3 catalog to data/DR3_New_AGN_Redshift_Catalog_Improved.csv")
    
    # Save DR2 generalization set catalog
    final_catalog_gen = pd.DataFrame({
        'source_name': gen_df['Source_Name'],
        '4fgl_name': gen_df['Source_Name'],
        'ra': gen_df['RAJ2000'],
        'dec': gen_df['DEJ2000'],
        'agn_class': gen_df['CLASS'],
        'agn_type_est': np.where(gen_df['AGN_Type'] == 'BCU', np.where(gen_df['P_FSRQ'] >= 0.5, 'FSRQ', 'BLL'), gen_df['AGN_Type']),
        'p_fsrq': np.round(gen_df['P_FSRQ'], 4),
        'xgb_z': np.round(1.0 / pred_inv_gen['xgb'] - 1.0, 4),
        'lgbm_z': np.round(1.0 / pred_inv_gen['lgb'] - 1.0, 4),
        'catboost_z': np.round(1.0 / pred_inv_gen['cat'] - 1.0, 4),
        'ft_z': np.round(1.0 / pred_inv_gen['ft'] - 1.0, 4),
        'tabpfn_z': np.round(1.0 / pred_inv_gen['tabpfn'] - 1.0, 4),
        'tabnet_z': np.round(1.0 / pred_inv_gen['tabnet'] - 1.0, 4),
        'tabm_z': np.round(1.0 / pred_inv_gen['tabm'] - 1.0, 4),
        'saint_z': np.round(1.0 / pred_inv_gen['saint'] - 1.0, 4),
        'ensemble_z': np.round(ensemble_z_gen, 4),
        'prediction_std': np.round(std_z_gen, 4),
        'confidence_score': np.round(confidence_score_gen, 4),
        'z_low_68': np.round(z_low_gen, 4),
        'z_high_68': np.round(z_high_gen, 4),
        'z_low_95': np.round(z_low_95_gen, 4),
        'z_high_95': np.round(z_high_95_gen, 4),
        'ood_score': np.round(gen_df['OOD_Score'], 4),
        'reliability_grade': gen_df['Reliability_Grade']
    })
    
    final_catalog_gen.to_csv("data/DR2_Generalization_AGN_Redshift_Catalog_Improved.csv", index=False)
    print("Saved DR2 catalog to data/DR2_Generalization_AGN_Redshift_Catalog_Improved.csv")
    
    # Save output/Table3_generalization_predictions.csv (the simple summary predictions in old format)
    table3_out = pd.DataFrame({
        'Name': gen_df['Source_Name'],
        'Predicted_z': np.round(1.0 / gen_df['pred_stack_inv'] - 1.0, 2),
        'Bias_Corrected_z': np.round(ensemble_z_gen, 2)
    })
    table3_out.to_csv("output/Table3_generalization_predictions.csv", index=False)
    print("Saved Table3 predictions to output/Table3_generalization_predictions.csv")
    
    # Evaluate performance on independent spectroscopic test set (DR3 n=89)
    spec_sources = target_df[target_df['Redshift'] > 0].copy()
    y_test_spec_z = spec_sources['Redshift'].values
    
    if len(y_test_spec_z) > 0:
        spec_names = set(spec_sources['Source_Name'])
        final_spec = final_catalog_target[final_catalog_target['source_name'].isin(spec_names)].copy()
        final_spec['order'] = final_spec['source_name'].map(lambda x: list(spec_sources['Source_Name']).index(x))
        final_spec = final_spec.sort_values('order')
        
        print("\n=== EVALUATION ON SPECTROSCOPIC SUBSET (n=89) ===")
        print(f"{'Model':<20} | {'Pearson R':<10} | {'RMSE':<8}")
        print("-" * 45)
        
        for m_key, m_name in [('xgb_z', 'XGBoost'), ('lgbm_z', 'LightGBM'), ('catboost_z', 'CatBoost'), 
                              ('ft_z', 'FT-Transformer'), ('tabpfn_z', 'TabPFN'), ('tabnet_z', 'TabNet'),
                              ('tabm_z', 'TabM'), ('saint_z', 'SAINT'), ('ensemble_z', 'Ensemble')]:
            preds = final_spec[m_key].values
            r_val, _ = pearsonr(preds, y_test_spec_z)
            rmse_val = np.sqrt(np.mean((preds - y_test_spec_z) ** 2))
            print(f"{m_name:<20} | {r_val:<10.4f} | {rmse_val:<8.4f}")
            
    print("\nRobust pipeline execution completed successfully!")

if __name__ == '__main__':
    run_predictions()
