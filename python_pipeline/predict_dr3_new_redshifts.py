import pandas as pd
from astropy.io import fits
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style for publication-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18
})

def angular_distance(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    dlon = ra2 - ra1
    dlat = dec2 - dec1
    a = np.sin(dlat/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c) * 3600.0

# ─── SAINT (Feature Transformer) Custom Implementation ──────────────────────
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# ─── PyTorch Training Helper ────────────────────────────────────────────────
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
                preds = model(batch_X) # (B, k, 1)
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

# -------------------------------------------------------------------------
# TASK A: CATALOG COMPARISON
# -------------------------------------------------------------------------
print("=== TASK A: Catalog Comparison ===")
dr2_raw = pd.read_csv("data/4LAC-DR2.csv")
dr2_raw['Source_Name'] = dr2_raw['Source_Name'].astype(str).str.strip()

with fits.open("data/table-4LAC-DR3-h.fits") as h_hdul:
    dr3_h = pd.DataFrame(h_hdul[1].data)
with fits.open("data/table-4LAC-DR3-l.fits") as l_hdul:
    dr3_l = pd.DataFrame(l_hdul[1].data)

dr3_raw = pd.concat([dr3_h, dr3_l], ignore_index=True)
for col in dr3_raw.columns:
    if dr3_raw[col].dtype == object:
        dr3_raw[col] = dr3_raw[col].astype(str).str.strip()

dr2_names = set(dr2_raw['Source_Name'])
dr3_names = set(dr3_raw['Source_Name'])

new_names = dr3_names - dr2_names
common_names = dr3_names.intersection(dr2_names)
removed_names = dr2_names - dr3_names

# coordinate distance to verify
dr3_ra = dr3_raw['RAJ2000'].values
dr3_dec = dr3_raw['DEJ2000'].values
dr2_ra = dr2_raw['RAJ2000'].values
dr2_dec = dr2_raw['DEJ2000'].values

# Identify the new sources df
new_sources_df = dr3_raw[dr3_raw['Source_Name'].isin(new_names)].copy()

# Find nearest DR2 source for each new source
nearest_dists = []
for idx, row in new_sources_df.iterrows():
    dists = angular_distance(row['RAJ2000'], row['DEJ2000'], dr2_ra, dr2_dec)
    nearest_dists.append(np.min(dists))
new_sources_df['nearest_dr2_dist_arcsec'] = nearest_dists

# Compile DR3_New_Sources.csv columns
# source_name, 4fgl_name, ra, dec, agn_class, associated_source, available_features
new_sources_out = pd.DataFrame({
    'source_name': new_sources_df['Source_Name'],
    '4fgl_name': new_sources_df['Source_Name'],
    'ra': new_sources_df['RAJ2000'],
    'dec': new_sources_df['DEJ2000'],
    'agn_class': new_sources_df['CLASS'],
    'associated_source': new_sources_df['ASSOC1']
})

# Compile list of non-null features for each source
predictors_11 = ["Flux1000", "Energy_Flux100", "Signif_Avg", "Variability_Index",
                 "nu_syn", "nuFnu_syn", "PL_Index", "Pivot_Energy", "LP_Index",
                 "LP_beta", "Gaia_G_Magnitude"]

# Merge Gaia magnitudes downloaded
gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
new_sources_df = pd.merge(new_sources_df, gaia_new, on="Source_Name", how="left")

available_feats_list = []
for idx, row in new_sources_df.iterrows():
    avail = []
    for p in predictors_11:
        val = row.get(p, np.nan)
        if pd.notna(val) and not (isinstance(val, (int, float)) and np.isnan(val)):
            avail.append(p)
    available_feats_list.append(", ".join(avail))

new_sources_out['available_features'] = available_feats_list
new_sources_out.to_csv("data/DR3_New_Sources.csv", index=False)
print(f"Saved {len(new_sources_out)} sources to data/DR3_New_Sources.csv")

# Print Summary Statistics
total_dr2 = len(dr2_raw)
total_dr3 = len(dr3_raw)
common_count = len(common_names)
new_count = len(new_names)
removed_count = len(removed_names)

print("\n--- Summary Statistics ---")
print(f"Total DR2 AGNs:                 {total_dr2}")
print(f"Total DR3 AGNs:                 {total_dr3}")
print(f"Number of common AGNs:          {common_count}")
print(f"Number of newly added DR3 AGNs: {new_count}")
print(f"Number of removed sources:      {removed_count}")

# -------------------------------------------------------------------------
# TASK B: VALIDATION OF NEW SOURCES
# -------------------------------------------------------------------------
print("\n=== TASK B: Validation of New Sources ===")

# Keep only BLL and FSRQ
new_sources_df['CLASS_upper'] = new_sources_df['CLASS'].str.upper()
bll_mask = new_sources_df['CLASS_upper'].isin(['BLL', 'BL LAC'])
fsrq_mask = new_sources_df['CLASS_upper'].isin(['FSRQ', 'FLAT SPECTRUM RADIO QUASAR'])
new_sources_bll_fsrq = new_sources_df[bll_mask | fsrq_mask].copy()

# Add standardised type and LabelNo
new_sources_bll_fsrq['AGN_Type'] = np.where(new_sources_bll_fsrq['CLASS_upper'].isin(['BLL', 'BL LAC']), 'BLL', 'FSRQ')
new_sources_bll_fsrq['LabelNo'] = np.where(new_sources_bll_fsrq['AGN_Type'] == 'BLL', 0, 1)

# Apply numerical conversions and set non-positive values to NA for logs
numeric_cols = ["LP_beta", "LP_Index", "Flux1000", "Energy_Flux100",
                "Signif_Avg", "Variability_Index", "nu_syn", "nuFnu_syn",
                "Pivot_Energy", "PL_Index", "Gaia_G_Magnitude", "GLAT", "Flags"]

for col in numeric_cols:
    new_sources_bll_fsrq[col] = pd.to_numeric(new_sources_bll_fsrq[col], errors='coerce')

# Apply Galactic latitude cut: |b| > 10
new_sources_bll_fsrq = new_sources_bll_fsrq[new_sources_bll_fsrq['GLAT'].abs() > 10.0].copy()

positive_cols = ["Flux1000", "Energy_Flux100", "Signif_Avg",
                 "Variability_Index", "nu_syn", "nuFnu_syn", "Pivot_Energy"]
for col in positive_cols:
    new_sources_bll_fsrq.loc[new_sources_bll_fsrq[col] <= 0, col] = np.nan

# Compute log features
new_sources_bll_fsrq['LogFlux'] = np.log10(new_sources_bll_fsrq['Flux1000'])
new_sources_bll_fsrq['LogEnergy_Flux'] = np.log10(new_sources_bll_fsrq['Energy_Flux100'])
new_sources_bll_fsrq['LogSignificance'] = np.log10(new_sources_bll_fsrq['Signif_Avg'])
new_sources_bll_fsrq['LogVariability_Index'] = np.log10(new_sources_bll_fsrq['Variability_Index'])
new_sources_bll_fsrq['Lognu_syn'] = np.log10(new_sources_bll_fsrq['nu_syn'])
new_sources_bll_fsrq['LognuFnu_syn'] = np.log10(new_sources_bll_fsrq['nuFnu_syn'])
new_sources_bll_fsrq['LogPivot_Energy'] = np.log10(new_sources_bll_fsrq['Pivot_Energy'])

# Apply outlier cuts: LP_beta < 0.7, LP_Index > 1.0, LogFlux >= -10.5, Flags not in {2, 36}
outlier_mask = (
    (new_sources_bll_fsrq['LP_beta'] >= 0.7) |
    (new_sources_bll_fsrq['LP_Index'] <= 1.0) |
    (new_sources_bll_fsrq['LogFlux'] < -10.5) |
    new_sources_bll_fsrq['Flags'].isin([2, 36])
)
new_sources_bll_fsrq = new_sources_bll_fsrq[~outlier_mask].copy()

# Validate complete features
predictors_12 = ["LogFlux", "LogEnergy_Flux", "LogSignificance", "LogVariability_Index",
                 "Lognu_syn", "LognuFnu_syn", "PL_Index", "LogPivot_Energy", "LP_Index",
                 "LP_beta", "Gaia_G_Magnitude", "LabelNo"]

# Drop missing values across predictors (this is our validation cut)
new_sources_valid = new_sources_bll_fsrq.dropna(subset=predictors_12).copy()
print(f"BLL/FSRQ new sources after missing values removal: {len(new_sources_valid)}")
print(f"  BLL:  {sum(new_sources_valid['AGN_Type'] == 'BLL')}")
print(f"  FSRQ: {sum(new_sources_valid['AGN_Type'] == 'FSRQ')}")

new_sources_valid.to_csv("data/DR3_New_Valid_Sources.csv", index=False)
print("Saved valid sources to data/DR3_New_Valid_Sources.csv")

# -------------------------------------------------------------------------
# TASK C: REDSHIFT PREDICTION FOR NEW SOURCES
# -------------------------------------------------------------------------
print("\n=== TASK C: Redshift Prediction for New Sources ===")

# Load training data
training_data = pd.read_csv("data/training_eligible.csv")
X_train_full = training_data[predictors_12]
y_train_full = training_data['InvRedshift']
agn_train = training_data['AGN_Type'].values

X_new = new_sources_valid[predictors_12]
agn_new = new_sources_valid['AGN_Type'].values

# Fit trees
print("Training XGBoost...")
model_xgb = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model_xgb.fit(X_train_full, y_train_full)
xgb_train_preds = model_xgb.predict(X_train_full)
xgb_new_preds_inv = model_xgb.predict(X_new)

print("Training LightGBM...")
model_lgb = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
)
model_lgb.fit(X_train_full, y_train_full)
lgb_train_preds = model_lgb.predict(X_train_full)
lgb_new_preds_inv = model_lgb.predict(X_new)

print("Training CatBoost...")
model_cat = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=4,
    random_seed=42, thread_count=-1, verbose=0
)
model_cat.fit(X_train_full, y_train_full)
cat_train_preds = model_cat.predict(X_train_full)
cat_new_preds_inv = model_cat.predict(X_new)

# Scale features for neural networks and TabPFN
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_full)
X_new_sc = scaler.transform(X_new)

print("Training FT-Transformer...")
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig

train_df = pd.DataFrame(X_train_sc, columns=predictors_12)
train_df['target'] = y_train_full.values

data_config = DataConfig(
    target=['target'],
    continuous_cols=predictors_12,
    categorical_cols=[]
)
trainer_config = TrainerConfig(
    max_epochs=20,
    batch_size=128,
    checkpoints=None,
    progress_bar="none",
    accelerator="cpu",
    devices=1,
    seed=42
)
model_config = FTTransformerConfig(
    task="regression",
    num_attn_blocks=2,
    num_heads=2,
    input_embed_dim=16,
    seed=42
)
optimizer_config = OptimizerConfig()

model_ft = TabularModel(
    data_config=data_config, 
    model_config=model_config, 
    trainer_config=trainer_config,
    optimizer_config=optimizer_config
)
model_ft.logger = False
model_ft.fit(train_df)

ft_train_preds = model_ft.predict(train_df)['target_prediction'].values
new_df = pd.DataFrame(X_new_sc, columns=predictors_12)
ft_new_preds_inv = model_ft.predict(new_df)['target_prediction'].values

# --- Training new neural models ---

# TabPFN
print("Training TabPFN...")
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
from tabpfn import TabPFNRegressor
model_tabpfn = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
model_tabpfn.fit(X_train_sc, y_train_full.values)
tabpfn_train_preds = model_tabpfn.predict(X_train_sc)
tabpfn_new_preds_inv = model_tabpfn.predict(X_new_sc)

# TabNet
print("Training TabNet...")
from pytorch_tabnet.tab_model import TabNetRegressor
model_tabnet = TabNetRegressor(
    verbose=0,
    device_name='cpu',
    n_d=8, n_a=8,
    n_steps=3,
    n_shared=2,
    seed=42
)
model_tabnet.fit(
    X_train_sc, y_train_full.values.reshape(-1, 1),
    max_epochs=100, patience=15,
    batch_size=128, virtual_batch_size=16
)
tabnet_train_preds = model_tabnet.predict(X_train_sc).flatten()
tabnet_new_preds_inv = model_tabnet.predict(X_new_sc).flatten()

# Split 15% for early stopping for SAINT and TabM to prevent overfitting
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_sc, y_train_full.values, test_size=0.15, random_state=42)

# TabM
print("Training TabM...")
import tabm
model_tabm = tabm.TabM.make(
    n_num_features=X_train_sc.shape[1],
    cat_cardinalities=None,
    d_out=1,
    k=32
)
model_tabm = train_pytorch_model(
    model_tabm, X_tr, y_tr, X_val, y_val,
    epochs=100, lr=0.001, batch_size=64, is_tabm=True
)
model_tabm.eval()
with torch.no_grad():
    tabm_train_preds = model_tabm(torch.tensor(X_train_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()
    tabm_new_preds_inv = model_tabm(torch.tensor(X_new_sc, dtype=torch.float32)).mean(dim=1).flatten().numpy()

# SAINT
print("Training SAINT...")
model_saint = SAINTRegressor(
    num_features=X_train_sc.shape[1],
    embedding_dim=32,
    num_heads=2,
    depth=2,
    dropout=0.1
)
model_saint = train_pytorch_model(
    model_saint, X_tr, y_tr, X_val, y_val,
    epochs=100, lr=0.001, batch_size=64, is_tabm=False
)
model_saint.eval()
with torch.no_grad():
    saint_train_preds = model_saint(torch.tensor(X_train_sc, dtype=torch.float32)).flatten().numpy()
    saint_new_preds_inv = model_saint(torch.tensor(X_new_sc, dtype=torch.float32)).flatten().numpy()

# Fit Optimal Transport (OT) bias correction parameters for each model
from py_bias_correction import fit_optimal_transport, apply_optimal_transport

print("\nFitting bias correction parameters...")
xgb_ot = fit_optimal_transport(xgb_train_preds, y_train_full.values, agn_train, verbose=False)
lgb_ot = fit_optimal_transport(lgb_train_preds, y_train_full.values, agn_train, verbose=False)
cat_ot = fit_optimal_transport(cat_train_preds, y_train_full.values, agn_train, verbose=False)
ft_ot = fit_optimal_transport(ft_train_preds, y_train_full.values, agn_train, verbose=False)
tabpfn_ot = fit_optimal_transport(tabpfn_train_preds, y_train_full.values, agn_train, verbose=False)
tabnet_ot = fit_optimal_transport(tabnet_train_preds, y_train_full.values, agn_train, verbose=False)
tabm_ot = fit_optimal_transport(tabm_train_preds, y_train_full.values, agn_train, verbose=False)
saint_ot = fit_optimal_transport(saint_train_preds, y_train_full.values, agn_train, verbose=False)

# Apply OT to new predictions
print("Applying bias correction...")
xgb_new_corr = apply_optimal_transport(xgb_new_preds_inv, agn_new, xgb_ot, verbose=False)
lgb_new_corr = apply_optimal_transport(lgb_new_preds_inv, agn_new, lgb_ot, verbose=False)
cat_new_corr = apply_optimal_transport(cat_new_preds_inv, agn_new, cat_ot, verbose=False)
ft_new_corr = apply_optimal_transport(ft_new_preds_inv, agn_new, ft_ot, verbose=False)
tabpfn_new_corr = apply_optimal_transport(tabpfn_new_preds_inv, agn_new, tabpfn_ot, verbose=False)
tabnet_new_corr = apply_optimal_transport(tabnet_new_preds_inv, agn_new, tabnet_ot, verbose=False)
tabm_new_corr = apply_optimal_transport(tabm_new_preds_inv, agn_new, tabm_ot, verbose=False)
saint_new_corr = apply_optimal_transport(saint_new_preds_inv, agn_new, saint_ot, verbose=False)

# Convert to linear redshift: z = 1/y - 1
xgb_z = (1.0 / xgb_new_corr) - 1.0
lgb_z = (1.0 / lgb_new_corr) - 1.0
cat_z = (1.0 / cat_new_corr) - 1.0
ft_z = (1.0 / ft_new_corr) - 1.0
tabpfn_z = (1.0 / tabpfn_new_corr) - 1.0
tabnet_z = (1.0 / tabnet_new_corr) - 1.0
tabm_z = (1.0 / tabm_new_corr) - 1.0
saint_z = (1.0 / saint_new_corr) - 1.0

# Calculate ensembles and uncertainties (over all 8 models)
preds_matrix = np.column_stack([xgb_z, lgb_z, cat_z, ft_z, tabpfn_z, tabnet_z, tabm_z, saint_z])
ensemble_z = np.mean(preds_matrix, axis=1)
median_z = np.median(preds_matrix, axis=1)
prediction_std = np.std(preds_matrix, axis=1)
confidence_score = np.exp(-prediction_std / (1 + ensemble_z))

# -------------------------------------------------------------------------
# TASK D: FINAL NEW-AGN CATALOG
# -------------------------------------------------------------------------
print("\n=== TASK D: Final New-AGN Catalog ===")

# Check Out of Distribution (OOD) flag
# Flag as 1 if any predictor is outside the training range
train_min = X_train_full.min()
train_max = X_train_full.max()

ood_flags = []
for idx, row in X_new.iterrows():
    is_ood = 0
    for p in predictors_12:
        if p == 'LabelNo': continue
        val = row[p]
        if val < train_min[p] or val > train_max[p]:
            is_ood = 1
            break
    ood_flags.append(is_ood)

# Compile DR3_New_AGN_Redshift_Catalog.csv
# Columns: source_name, 4fgl_name, ra, dec, agn_class, xgb_z, lgbm_z, catboost_z, ft_z, tabpfn_z, tabnet_z, tabm_z, saint_z, ensemble_z, prediction_std, confidence_score, ood_flag
final_catalog = pd.DataFrame({
    'source_name': new_sources_valid['Source_Name'],
    '4fgl_name': new_sources_valid['Source_Name'],
    'ra': new_sources_valid['RAJ2000'],
    'dec': new_sources_valid['DEJ2000'],
    'agn_class': new_sources_valid['CLASS'],
    'xgb_z': np.round(xgb_z, 4),
    'lgbm_z': np.round(lgb_z, 4),
    'catboost_z': np.round(cat_z, 4),
    'ft_z': np.round(ft_z, 4),
    'tabpfn_z': np.round(tabpfn_z, 4),
    'tabnet_z': np.round(tabnet_z, 4),
    'tabm_z': np.round(tabm_z, 4),
    'saint_z': np.round(saint_z, 4),
    'ensemble_z': np.round(ensemble_z, 4),
    'prediction_std': np.round(prediction_std, 4),
    'confidence_score': np.round(confidence_score, 4),
    'ood_flag': ood_flags
})

output_catalog_path = "data/DR3_New_AGN_Redshift_Catalog.csv"
final_catalog.to_csv(output_catalog_path, index=False)
print(f"Saved publication-ready catalog to {output_catalog_path}")

# -------------------------------------------------------------------------
# TASK E: SCIENTIFIC ANALYSIS & PLOT GENERATION
# -------------------------------------------------------------------------
print("\n=== TASK E: Scientific Analysis & Plots ===")
os.makedirs("plots/dr3_plots", exist_ok=True)

# 1. Histogram of predicted redshifts of new DR3 AGNs
plt.figure(figsize=(10, 6))
sns.histplot(ensemble_z, bins=15, kde=True, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Redshift ($z_{\\mathrm{phot}}$)")
plt.ylabel("Number of Sources")
plt.title("Redshift Distribution of Newly Identified Fermi 4LAC DR3 AGNs (8-Model Ensemble)")
plt.savefig("plots/dr3_plots/dr3_new_redshifts_hist.png", dpi=300, bbox_inches='tight')
plt.close()

# 2 & 3. BL Lac & FSRQ predicted redshift distributions
plt.figure(figsize=(10, 6))
sns.histplot(data=final_catalog, x='ensemble_z', hue='agn_class', kde=True,
             bins=15, multiple='layer', palette={'bll': '#C0392B', 'fsrq': '#27AE60'},
             edgecolor='black', alpha=0.6)
plt.xlabel("Predicted Redshift ($z_{\\mathrm{phot}}$)")
plt.ylabel("Number of Sources")
plt.title("Redshift Distribution by AGN Class (BLL vs FSRQ)")
plt.legend(title="AGN Class", labels=["FSRQ", "BL Lac"])
plt.savefig("plots/dr3_plots/dr3_redshift_distribution_by_class.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Ensemble vs individual model comparison (Pairplot / Scatter Matrix - 4x2 layout)
fig, axes = plt.subplots(4, 2, figsize=(12, 18))
models_z_dict = {
    'XGBoost': xgb_z, 'LightGBM': lgb_z, 'CatBoost': cat_z, 'FT-Transformer': ft_z,
    'TabPFN': tabpfn_z, 'TabNet': tabnet_z, 'TabM': tabm_z, 'SAINT': saint_z
}
model_names = list(models_z_dict.keys())

for idx, m_name in enumerate(model_names):
    ax = axes[idx//2, idx%2]
    sns.scatterplot(x=models_z_dict[m_name], y=ensemble_z, ax=ax, alpha=0.8, color='#2980B9')
    # add 1-1 line
    lims = [0, max(ensemble_z.max(), models_z_dict[m_name].max()) + 0.2]
    ax.plot(lims, lims, '--r', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{m_name} $z$")
    ax.set_ylabel("Ensemble $z$")
    ax.set_title(f"{m_name} vs Ensemble ($R$ = {np.corrcoef(models_z_dict[m_name], ensemble_z)[0,1]:.3f})")

plt.tight_layout()
plt.savefig("plots/dr3_plots/model_agreement_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Outlier source analysis
# Sources with prediction_std > 0.3 (high model disagreement)
outliers = final_catalog[final_catalog['prediction_std'] > 0.3].copy()
print(f"Number of high-disagreement outlier sources (std > 0.3): {len(outliers)} / {len(final_catalog)}")

# 6. Most distant predicted AGNs
most_distant = final_catalog.sort_values(by='ensemble_z', ascending=False).head(10)
print("\n--- Top 10 Most Distant Predicted AGNs ---")
print(most_distant[['source_name', 'agn_class', 'ensemble_z', 'prediction_std', 'confidence_score']])

# 7. Top 100 highest-redshift candidates
top_100_z = final_catalog.sort_values(by='ensemble_z', ascending=False).head(100)
top_100_z.to_csv("data/DR3_Top100_Highest_Redshift.csv", index=False)
print("Saved Top 100 Highest-Redshift candidates to data/DR3_Top100_Highest_Redshift.csv")

# 8. Top 100 most reliable predictions (highest confidence_score, lowest std)
top_100_conf = final_catalog.sort_values(by='confidence_score', ascending=False).head(100)
top_100_conf.to_csv("data/DR3_Top100_Most_Reliable.csv", index=False)
print("Saved Top 100 Most Reliable predictions to data/DR3_Top100_Most_Reliable.csv")

# 9. Perform independent validation evaluation on the spectroscopic subset (n=67)
print("\n=== TASK F: Spectroscopic Test Set Performance ===")
test_spec_df = new_sources_valid[new_sources_valid['Redshift'] > 0].copy()
y_test_spec = test_spec_df['Redshift'].values

# Corresponding prediction columns from final_catalog for the same sources
spec_names = set(test_spec_df['Source_Name'])
final_catalog_spec = final_catalog[final_catalog['source_name'].isin(spec_names)].copy()
# Order them to match test_spec_df exactly
final_catalog_spec['order'] = final_catalog_spec['source_name'].map(lambda x: list(test_spec_df['Source_Name']).index(x))
final_catalog_spec = final_catalog_spec.sort_values('order')

from scipy.stats import pearsonr

print(f"Independent test set size with valid spectroscopic redshifts: {len(y_test_spec)}")
print(f"{'Model':<20} | {'Pearson R':<10} | {'RMSE':<8}")
print("-" * 45)

for m_key, m_name in [('xgb_z', 'XGBoost'), ('lgbm_z', 'LightGBM'), ('catboost_z', 'CatBoost'), 
                      ('ft_z', 'FT-Transformer'), ('tabpfn_z', 'TabPFN'), ('tabnet_z', 'TabNet'),
                      ('tabm_z', 'TabM'), ('saint_z', 'SAINT'), ('ensemble_z', 'Ensemble')]:
    preds = final_catalog_spec[m_key].values
    r_val, _ = pearsonr(preds, y_test_spec)
    rmse_val = np.sqrt(np.mean((preds - y_test_spec) ** 2))
    print(f"{m_name:<20} | {r_val:<10.4f} | {rmse_val:<8.4f}")

# Compute overall properties for LaTeX table
print("\nAll tasks completed successfully!")

