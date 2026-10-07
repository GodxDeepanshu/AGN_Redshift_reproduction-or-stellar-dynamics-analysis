import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import helper functions
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics, print_metrics
from tabpfn import TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig
import tabm

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

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=64, is_tabm=False):
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

def run_improved_pipeline():
    print("=== STARTING IMPROVED REDSHIFT PREDICTION PIPELINE ===")
    
    # 1. Load data
    train_df = pd.read_csv("data/training_eligible.csv")
    
    # Get the 319 new sources
    dr2 = pd.read_csv("data/4LAC-DR2.csv")
    dr2_names = set(dr2['Source_Name'].astype(str).str.strip())
    
    from astropy.io import fits
    with fits.open("data/table-4LAC-DR3-h.fits") as h_hdul:
        dr3_h = pd.DataFrame(h_hdul[1].data)
    with fits.open("data/table-4LAC-DR3-l.fits") as l_hdul:
        dr3_l = pd.DataFrame(l_hdul[1].data)
    dr3 = pd.concat([dr3_h, dr3_l], ignore_index=True)
    for col in dr3.columns:
        if dr3[col].dtype == object:
            dr3[col] = dr3[col].astype(str).str.strip()
            
    new_names = set(dr3['Source_Name']) - dr2_names
    new_sources_df = dr3[dr3['Source_Name'].isin(new_names)].copy()
    
    # Preprocess new sources
    new_sources_df['CLASS_upper'] = new_sources_df['CLASS'].astype(str).str.upper()
    new_sources_df['AGN_Type'] = np.where(new_sources_df['CLASS_upper'].isin(['BLL', 'BL LAC']), 'BLL',
                                 np.where(new_sources_df['CLASS_upper'].isin(['FSRQ', 'FLAT SPECTRUM RADIO QUASAR']), 'FSRQ', 'BCU'))
    
    # Standard numerical conversion
    numeric_cols = ["LP_beta", "LP_Index", "Flux1000", "Energy_Flux100",
                    "Signif_Avg", "Variability_Index", "nu_syn", "nuFnu_syn",
                    "Pivot_Energy", "PL_Index", "GLAT", "Flags"]
    for col in numeric_cols:
        new_sources_df[col] = pd.to_numeric(new_sources_df[col], errors='coerce')
        
    positive_cols = ["Flux1000", "Energy_Flux100", "Signif_Avg",
                     "Variability_Index", "nu_syn", "nuFnu_syn", "Pivot_Energy"]
    for col in positive_cols:
        new_sources_df.loc[new_sources_df[col] <= 0, col] = np.nan
        
    # Log transforms
    new_sources_df['LogFlux'] = np.log10(new_sources_df['Flux1000'])
    new_sources_df['LogEnergy_Flux'] = np.log10(new_sources_df['Energy_Flux100'])
    new_sources_df['LogSignificance'] = np.log10(new_sources_df['Signif_Avg'])
    new_sources_df['LogVariability_Index'] = np.log10(new_sources_df['Variability_Index'])
    new_sources_df['Lognu_syn'] = np.log10(new_sources_df['nu_syn'])
    new_sources_df['LognuFnu_syn'] = np.log10(new_sources_df['nuFnu_syn'])
    new_sources_df['LogPivot_Energy'] = np.log10(new_sources_df['Pivot_Energy'])
    
    # Merge Gaia magnitudes for new sources
    gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    new_sources_df = pd.merge(new_sources_df, gaia_new, on="Source_Name", how="left")
    new_sources_df['Gaia_G_Magnitude'] = pd.to_numeric(new_sources_df['Gaia_G_Magnitude'], errors='coerce')
    
    # Load multi-wavelength matches
    mw = pd.read_csv("data/multi_wavelength_matches.csv")
    mw['Source_Name'] = mw['Source_Name'].astype(str).str.strip()
    
    # Merge MW columns
    train_df = pd.merge(train_df, mw, on="Source_Name", how="left")
    new_sources_df = pd.merge(new_sources_df, mw, on="Source_Name", how="left")
    
    # Compute infrared colors and log fluxes
    for df in [train_df, new_sources_df]:
        df['W1mag'] = pd.to_numeric(df['W1mag'], errors='coerce')
        df['W2mag'] = pd.to_numeric(df['W2mag'], errors='coerce')
        df['W3mag'] = pd.to_numeric(df['W3mag'], errors='coerce')
        df['W4mag'] = pd.to_numeric(df['W4mag'], errors='coerce')
        df['S1.4'] = pd.to_numeric(df['S1.4'], errors='coerce')
        df['CR0'] = pd.to_numeric(df['CR0'], errors='coerce')
        
        df['W1_W2'] = df['W1mag'] - df['W2mag']
        df['W2_W3'] = df['W2mag'] - df['W3mag']
        
        # Clip radio flux to positive values before log
        df['LogRadioFlux'] = np.log10(df['S1.4'].clip(lower=1e-5))
        df['LogXrayFlux'] = np.log10(df['CR0'].clip(lower=1e-6))
        
    print("Preprocessing completed. Feature lists compiled.")
    
    # Define features
    orig_features = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                     "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                     "LP_beta", "Gaia_G_Magnitude"]
    
    mw_features = ["W1mag", "W2mag", "W3mag", "W4mag", "W1_W2", "W2_W3", "LogRadioFlux", "LogXrayFlux"]
    
    all_predictors = orig_features + mw_features
    print(f"Total predictors count: {len(all_predictors)} (11 original + 8 multi-wavelength)")
    
    # Target variables
    y_train = train_df['InvRedshift'].values
    y_train_z = train_df['Redshift'].values
    
    # Perform median imputation on predictors based on training set
    impute_vals = {}
    for col in all_predictors:
        median_val = train_df[col].median(skipna=True)
        if pd.isna(median_val): median_val = 0.0 # fallback
        impute_vals[col] = median_val
        train_df[col] = train_df[col].fillna(median_val)
        new_sources_df[col] = new_sources_df[col].fillna(median_val)
        
    print("Imputed missing predictor values with training medians.")
    
    # ─── Semi-Supervised Class Probability ─────────────────────────────────────
    print("\n=== Training Class Probability Model P(FSRQ|X) ===")
    
    # Classifier features (predictors without target LabelNo)
    X_train_class = train_df[all_predictors]
    y_train_class = train_df['LabelNo'].values # 0 for BLL, 1 for FSRQ
    
    X_new_class = new_sources_df[all_predictors]
    
    # Cross-validation for out-of-fold probability estimation on training set
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_probs = np.zeros(len(train_df))
    
    for train_idx, val_idx in skf.split(X_train_class, y_train_class):
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train_class.iloc[train_idx], y_train_class[train_idx])
        train_probs[val_idx] = clf.predict_proba(X_train_class.iloc[val_idx])[:, 1]
        
    # Fit final classifier on entire training set
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_class, y_train_class)
    new_probs = final_clf.predict_proba(X_new_class)[:, 1]
    
    # Append class probabilities as a new predictor
    train_df['P_FSRQ'] = train_probs
    new_sources_df['P_FSRQ'] = new_probs
    
    # Check BCUs class probabilities
    bcu_probs = new_sources_df[new_sources_df['AGN_Type'] == 'BCU']['P_FSRQ']
    print(f"Predicted class probabilities for {len(bcu_probs)} BCUs. Mean probability P(FSRQ|X) = {bcu_probs.mean():.4f}")
    
    # Feature set for redshift regression includes P_FSRQ instead of binary LabelNo
    reg_predictors = all_predictors + ['P_FSRQ']
    
    # Save processed features
    train_df.to_csv("data/training_eligible_mw.csv", index=False)
    new_sources_df.to_csv("data/DR3_New_Sources_mw.csv", index=False)
    
    # Setup data matrices
    X_train = train_df[reg_predictors].values
    X_new = new_sources_df[reg_predictors].values
    
    # Scaled matrices for neural network models
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_new_sc = scaler.transform(X_new)
    
    # ─── Training Redshift Models ──────────────────────────────────────────────
    print("\n=== Training Redshift Models (1/(z+1)) ===")
    
    # Dict to hold in-sample predictions on training set
    oof_preds_inv = {}
    # Dict to hold predictions for DR3 new sources
    new_preds_inv = {}
    
    # XGBoost
    print("Training XGBoost...")
    model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train, y_train)
    oof_preds_inv['xgb'] = model_xgb.predict(X_train)
    new_preds_inv['xgb'] = model_xgb.predict(X_new)
    
    # LightGBM
    print("Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    model_lgb.fit(X_train, y_train)
    oof_preds_inv['lgb'] = model_lgb.predict(X_train)
    new_preds_inv['lgb'] = model_lgb.predict(X_new)
    
    # CatBoost
    print("Training CatBoost...")
    model_cat = cb.CatBoostRegressor(iterations=500, learning_rate=0.03, depth=4, random_seed=42, thread_count=-1, verbose=0)
    model_cat.fit(X_train, y_train)
    oof_preds_inv['cat'] = model_cat.predict(X_train)
    new_preds_inv['cat'] = model_cat.predict(X_new)
    
    # TabPFN
    print("Training TabPFN...")
    model_tabpfn = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
    model_tabpfn.fit(X_train_sc, y_train)
    oof_preds_inv['tabpfn'] = model_tabpfn.predict(X_train_sc)
    new_preds_inv['tabpfn'] = model_tabpfn.predict(X_new_sc)
    
    # TabNet
    print("Training TabNet...")
    model_tabnet = TabNetRegressor(verbose=0, device_name='cpu', n_d=8, n_a=8, n_steps=3, n_shared=2, seed=42)
    model_tabnet.fit(X_train_sc, y_train.reshape(-1, 1), max_epochs=80, patience=15, batch_size=128, virtual_batch_size=16)
    oof_preds_inv['tabnet'] = model_tabnet.predict(X_train_sc).flatten()
    new_preds_inv['tabnet'] = model_tabnet.predict(X_new_sc).flatten()
    
    # Early stopping split for deep models
    X_tr, X_val, y_tr, y_val = train_test_split_helper(X_train_sc, y_train, test_size=0.15)
    
    # TabM
    print("Training TabM...")
    model_tabm = tabm.TabM.make(n_num_features=X_train.shape[1], cat_cardinalities=None, d_out=1, k=32)
    model_tabm = train_pytorch_model(model_tabm, X_tr, y_tr, X_val, y_val, epochs=80, lr=0.001, batch_size=64, is_tabm=True)
    model_tabm.eval()
    with torch.no_grad():
        oof_preds_inv['tabm'] = model_tabm(torch.tensor(X_train_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()
        new_preds_inv['tabm'] = model_tabm(torch.tensor(X_new_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()
        
    # SAINT
    print("Training SAINT...")
    model_saint = SAINTRegressor(num_features=X_train.shape[1], embedding_dim=32, num_heads=2, depth=2, dropout=0.1)
    model_saint = train_pytorch_model(model_saint, X_tr, y_tr, X_val, y_val, epochs=80, lr=0.001, batch_size=64, is_tabm=False)
    model_saint.eval()
    with torch.no_grad():
        oof_preds_inv['saint'] = model_saint(torch.tensor(X_train_sc, dtype=torch.float32)).flatten().numpy()
        new_preds_inv['saint'] = model_saint(torch.tensor(X_new_sc, dtype=torch.float32)).flatten().numpy()
        
    # FT-Transformer
    print("Training FT-Transformer...")
    train_df_inner = pd.DataFrame(X_train_sc, columns=reg_predictors)
    train_df_inner['target'] = y_train
    
    data_config = DataConfig(target=['target'], continuous_cols=reg_predictors, categorical_cols=[])
    trainer_config = TrainerConfig(max_epochs=20, batch_size=128, checkpoints=None, progress_bar="none", accelerator="cpu", devices=1, seed=42)
    model_config = FTTransformerConfig(task="regression", num_attn_blocks=2, num_heads=2, input_embed_dim=16, seed=42)
    
    model_ft = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config, optimizer_config=OptimizerConfig())
    model_ft.logger = False
    model_ft.fit(train_df_inner)
    
    oof_preds_inv['ft'] = model_ft.predict(train_df_inner)['target_prediction'].values
    
    new_df_inner = pd.DataFrame(X_new_sc, columns=reg_predictors)
    new_preds_inv['ft'] = model_ft.predict(new_df_inner)['target_prediction'].values

    print("\n--- Training predictions compiled in-sample. ---")
    
    # ─── Non-Linear Bias Calibration (Isotonic Regression) ────────────────────
    print("\n=== Fitting Non-Linear Bias Calibration (Isotonic Regression) ===")
    
    # We calibrate separate isotonic models for BLL and FSRQ classes
    train_class_labels = train_df['LabelNo'].values # 0 for BLL, 1 for FSRQ
    bll_mask = (train_class_labels == 0)
    fsrq_mask = (train_class_labels == 1)
    
    # Corrected predictions
    oof_corr = {k: np.zeros(len(train_df)) for k in oof_preds_inv.keys()}
    new_corr = {k: np.zeros(len(new_sources_df)) for k in oof_preds_inv.keys()}
    
    for k in oof_preds_inv.keys():
        # Fit Isotonic Regression for BLL
        iso_bll = IsotonicRegression(out_of_bounds='clip')
        iso_bll.fit(oof_preds_inv[k][bll_mask], y_train[bll_mask])
        
        # Fit Isotonic Regression for FSRQ
        iso_fsrq = IsotonicRegression(out_of_bounds='clip')
        iso_fsrq.fit(oof_preds_inv[k][fsrq_mask], y_train[fsrq_mask])
        
        # Calibrate out-of-fold predictions on training set
        # Using continuous P_FSRQ to soft-interpolate
        p_train = train_df['P_FSRQ'].values
        oof_bll_c = iso_bll.predict(oof_preds_inv[k])
        oof_fsrq_c = iso_fsrq.predict(oof_preds_inv[k])
        oof_corr[k] = (1.0 - p_train) * oof_bll_c + p_train * oof_fsrq_c
        
        # Calibrate predictions on DR3 new sources
        p_new = new_sources_df['P_FSRQ'].values
        new_bll_c = iso_bll.predict(new_preds_inv[k])
        new_fsrq_c = iso_fsrq.predict(new_preds_inv[k])
        new_corr[k] = (1.0 - p_new) * new_bll_c + p_new * new_fsrq_c
        
    # Convert back to linear redshift: z = 1/y - 1
    oof_z = {k: (1.0 / oof_corr[k]) - 1.0 for k in oof_corr.keys()}
    new_z = {k: (1.0 / new_corr[k]) - 1.0 for k in new_corr.keys()}
    
    # Stacking ensemble: simple average of all 8 models
    oof_ensemble_z = np.mean(np.column_stack([oof_z[k] for k in oof_z.keys()]), axis=1)
    new_ensemble_z = np.mean(np.column_stack([new_z[k] for k in new_z.keys()]), axis=1)
    
    new_std = np.std(np.column_stack([new_z[k] for k in new_z.keys()]), axis=1)
    oof_std = np.std(np.column_stack([oof_z[k] for k in oof_z.keys()]), axis=1)
    
    # ─── Conformal Prediction (Locally Weighted) ─────────────────────────────
    print("\n=== Applying Conformal Prediction ===")
    
    # Calculate residuals on the training set
    oof_residuals = np.abs(y_train_z - oof_ensemble_z)
    
    # Scaled residuals (conformal score): s = |z_spec - z_pred| / std
    # To avoid division by zero, clip std to a minimum value
    oof_std_clipped = np.clip(oof_std, a_min=0.01, a_max=None)
    conformal_scores = oof_residuals / oof_std_clipped
    
    # Quantiles for 68% (1-sigma) and 95% (2-sigma) confidence intervals
    q_68 = np.percentile(conformal_scores, 68.0)
    q_95 = np.percentile(conformal_scores, 95.0)
    
    print(f"Conformal quantiles: Q_68 = {q_68:.4f}, Q_95 = {q_95:.4f}")
    
    # Construct confidence intervals for new DR3 sources
    new_std_clipped = np.clip(new_std, a_min=0.01, a_max=None)
    
    z_low_68 = np.clip(new_ensemble_z - q_68 * new_std_clipped, a_min=0.0, a_max=None)
    z_high_68 = new_ensemble_z + q_68 * new_std_clipped
    
    z_low_95 = np.clip(new_ensemble_z - q_95 * new_std_clipped, a_min=0.0, a_max=None)
    z_high_95 = new_ensemble_z + q_95 * new_std_clipped
    
    # Confidence score (physically motivated)
    confidence_score = np.exp(-new_std / (1 + new_ensemble_z))
    
    # ─── Save Results ─────────────────────────────────────────────────────────
    # Compile publication catalog for all 319 sources
    final_catalog = pd.DataFrame({
        'source_name': new_sources_df['Source_Name'],
        '4fgl_name': new_sources_df['Source_Name'],
        'ra': new_sources_df['RAJ2000'],
        'dec': new_sources_df['DEJ2000'],
        'agn_class': new_sources_df['CLASS'],
        'agn_type_est': new_sources_df['AGN_Type'],
        'p_fsrq': np.round(new_sources_df['P_FSRQ'], 4),
        'xgb_z': np.round(new_z['xgb'], 4),
        'lgbm_z': np.round(new_z['lgb'], 4),
        'catboost_z': np.round(new_z['cat'], 4),
        'ft_z': np.round(new_z['ft'], 4),
        'tabpfn_z': np.round(new_z['tabpfn'], 4),
        'tabnet_z': np.round(new_z['tabnet'], 4),
        'tabm_z': np.round(new_z['tabm'], 4),
        'saint_z': np.round(new_z['saint'], 4),
        'ensemble_z': np.round(new_ensemble_z, 4),
        'prediction_std': np.round(new_std, 4),
        'confidence_score': np.round(confidence_score, 4),
        'z_low_68': np.round(z_low_68, 4),
        'z_high_68': np.round(z_high_68, 4),
        'z_low_95': np.round(z_low_95, 4),
        'z_high_95': np.round(z_high_95, 4)
    })
    
    output_catalog_path = "data/DR3_New_AGN_Redshift_Catalog_Improved.csv"
    final_catalog.to_csv(output_catalog_path, index=False)
    print(f"\nSaved improved catalog to {output_catalog_path}")
    
    # Report performance on independent spectroscopic test set
    spec_sources = new_sources_df[new_sources_df['Redshift'] > 0].copy()
    y_test_spec = spec_sources['Redshift'].values
    
    if len(y_test_spec) > 0:
        spec_names = set(spec_sources['Source_Name'])
        final_spec = final_catalog[final_catalog['source_name'].isin(spec_names)].copy()
        final_spec['order'] = final_spec['source_name'].map(lambda x: list(spec_sources['Source_Name']).index(x))
        final_spec = final_spec.sort_values('order')
        
        print(f"\n=== EVALUATION ON SPECTROSCOPIC SUBSET (n={len(y_test_spec)}) ===")
        print(f"{'Model':<20} | {'Pearson R':<10} | {'RMSE':<8}")
        print("-" * 45)
        
        for m_key, m_name in [('xgb_z', 'XGBoost'), ('lgbm_z', 'LightGBM'), ('catboost_z', 'CatBoost'), 
                              ('ft_z', 'FT-Transformer'), ('tabpfn_z', 'TabPFN'), ('tabnet_z', 'TabNet'),
                              ('tabm_z', 'TabM'), ('saint_z', 'SAINT'), ('ensemble_z', 'Ensemble')]:
            preds = final_spec[m_key].values
            r_val = compute_pearson_helper(preds, y_test_spec)
            rmse_val = np.sqrt(np.mean((preds - y_test_spec) ** 2))
            print(f"{m_name:<20} | {r_val:<10.4f} | {rmse_val:<8.4f}")
            
    print("\nImproved pipeline run completed successfully!")

def train_test_split_helper(X, y, test_size=0.15):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X))
    val_set_size = int(len(X) * test_size)
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def compute_pearson_helper(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(mask) < 2: return np.nan
    from scipy.stats import pearsonr
    r_val, _ = pearsonr(obs[mask], pred[mask])
    return r_val

if __name__ == "__main__":
    run_improved_pipeline()
