import numpy as np
import pandas as pd
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics, print_metrics

# ─── SAINT (Feature Transformer) Custom Implementation ──────────────────────

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

# ─── Main Neural Evaluation Function ────────────────────────────────────────

def main():
    print("=" * 70)
    print("  NEURAL MODELS SINGLE-TRAIN EVALUATION (Phase 6)")
    print("=" * 70)
    
    # 1. Load Data
    data_dir = "data"
    training_data = pd.read_csv(os.path.join(data_dir, "training_eligible.csv"))
    
    with open(os.path.join(data_dir, "predictor_columns.txt"), "r") as f:
        predictors = [line.strip() for line in f if line.strip()]
        
    X = training_data[predictors].values
    y = training_data['InvRedshift'].values
    
    # 2. Train-Validation Split (85% / 15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    neural_results = {}
    
    # ─── 1. TabNet ──────────────────────────────────────────────────────────
    print("\nTraining TabNet...")
    t0 = time.time()
    try:
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
            X_train_sc, y_train.reshape(-1, 1),
            max_epochs=100, patience=15,
            batch_size=128, virtual_batch_size=16
        )
        preds_tabnet_inv = model_tabnet.predict(X_val_sc).flatten()
        
        # Calculate metrics in linear z
        metrics_tabnet = compute_all_metrics(preds_tabnet_inv, y_val)
        neural_results['tabnet'] = metrics_tabnet
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(metrics_tabnet, "TabNet")
    except Exception as e:
        print(f"  [ERROR] TabNet failed: {e}")
        
    # ─── 2. FT-Transformer ──────────────────────────────────────────────────
    print("\nTraining FT-Transformer...")
    t0 = time.time()
    try:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig, OptimizerConfig
        from pytorch_tabular.models import FTTransformerConfig
        
        train_df = pd.DataFrame(X_train_sc, columns=predictors)
        train_df['target'] = y_train
        
        val_df = pd.DataFrame(X_val_sc, columns=predictors)
        
        data_config = DataConfig(
            target=['target'],
            continuous_cols=predictors,
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
        
        preds_ft_inv = model_ft.predict(val_df)['target_prediction'].values
        metrics_ft = compute_all_metrics(preds_ft_inv, y_val)
        neural_results['ft_transformer'] = metrics_ft
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(metrics_ft, "FT-Transformer")
    except Exception as e:
        print(f"  [ERROR] FT-Transformer failed: {e}")
        
    # ─── 3. TabM ────────────────────────────────────────────────────────────
    print("\nTraining TabM...")
    t0 = time.time()
    try:
        import tabm
        model_tabm = tabm.TabM.make(
            n_num_features=len(predictors),
            cat_cardinalities=None,
            d_out=1,
            k=32
        )
        
        model_tabm = train_pytorch_model(
            model_tabm, X_train_sc, y_train, X_val_sc, y_val,
            epochs=100, lr=0.001, batch_size=64, is_tabm=True
        )
        
        model_tabm.eval()
        with torch.no_grad():
            val_X_t = torch.tensor(X_val_sc, dtype=torch.float32)
            # Output of TabM is (B, k, 1) -> average predictions across ensemble
            preds_tabm_inv = model_tabm(val_X_t).mean(dim=1).flatten().numpy()
            
        metrics_tabm = compute_all_metrics(preds_tabm_inv, y_val)
        neural_results['tabm'] = metrics_tabm
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(metrics_tabm, "TabM")
    except Exception as e:
        print(f"  [ERROR] TabM failed: {e}")
        
    # ─── 4. SAINT (Custom Feature Transformer) ──────────────────────────────
    print("\nTraining SAINT...")
    t0 = time.time()
    try:
        model_saint = SAINTRegressor(
            num_features=len(predictors),
            embedding_dim=32,
            num_heads=2,
            depth=2,
            dropout=0.1
        )
        
        model_saint = train_pytorch_model(
            model_saint, X_train_sc, y_train, X_val_sc, y_val,
            epochs=100, lr=0.001, batch_size=64, is_tabm=False
        )
        
        model_saint.eval()
        with torch.no_grad():
            val_X_t = torch.tensor(X_val_sc, dtype=torch.float32)
            preds_saint_inv = model_saint(val_X_t).flatten().numpy()
            
        metrics_saint = compute_all_metrics(preds_saint_inv, y_val)
        neural_results['saint'] = metrics_saint
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(metrics_saint, "SAINT")
    except Exception as e:
        print(f"  [ERROR] SAINT failed: {e}")
        
    # 5. Save results to JSON
    os.makedirs("output/python_results", exist_ok=True)
    json_path = "output/python_results/neural_metrics.json"
    
    # Load existing to avoid race conditions (e.g. TabPFN writing before/after)
    cleaned_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                cleaned_results = json.load(f)
        except Exception:
            pass
            
    for model_name, model_metrics in neural_results.items():
        cleaned_results[model_name] = {
            'R_z': float(model_metrics['z']['R']),
            'RMSE_z': float(model_metrics['z']['RMSE']),
            'outlier_pct_fixed': float(model_metrics['outlier_pct']),
            'outlier_pct_2sigma': float(model_metrics['outlier_pct_2sigma'])
        }
        
    with open(json_path, "w") as f:
        json.dump(cleaned_results, f, indent=4)
        
    print(f"\nSaved neural evaluation results to {json_path}")

if __name__ == "__main__":
    main()
