#!/usr/bin/env python3
"""
classify_bcu_probabilities.py
=============================
Trains a Random Forest classifier on BLL vs FSRQ labels to estimate P(FSRQ|X)
using out-of-fold predictions on the training set and full-model predictions
on the generalization set. Saves P_FSRQ back to the datasets as a predictor.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE = Path(r"c:\Users\deepa\OneDrive\文档\AGN_RedShift_Project")
DATA = BASE / "data"

TRAIN_PATH = DATA / "dr3_full_train.csv"
GEN_PATH   = DATA / "dr3_full_gen.csv"

def main():
    print("=" * 70)
    print("  ESTIMATING CLASS PROBABILITIES P(FSRQ|X)")
    print("=" * 70 + "\n")
    
    # Load data
    if not TRAIN_PATH.exists() or not GEN_PATH.exists():
        print(f"Error: Datasets not found at {TRAIN_PATH} or {GEN_PATH}")
        sys.exit(1)
        
    train = pd.read_csv(TRAIN_PATH)
    gen = pd.read_csv(GEN_PATH)
    
    print(f"Loaded training set: {len(train)} rows")
    print(f"Loaded generalization set: {len(gen)} rows")
    
    # Define features to use in classifier (11 original numeric features + 8 MW features)
    features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta', 'Gaia_G_Magnitude',
        'W1mag', 'W2mag', 'W3mag', 'W4mag', 'W1_W2', 'W2_W3',
        'LogRadioFlux', 'LogXrayFlux'
    ]
    
    # Target label: 0 for BLL, 1 for FSRQ
    y_train = train['LabelNo'].values
    
    # Prepare feature matrices
    X_train = train[features].copy()
    X_gen = gen[features].copy()
    
    # Perform median imputation for safety inside this step
    for col in features:
        train_med = X_train[col].median(skipna=True)
        if pd.isna(train_med):
            train_med = 0.0
        X_train[col] = X_train[col].fillna(train_med)
        X_gen[col] = X_gen[col].fillna(train_med)
        
    print(f"Features used for classification ({len(features)}):")
    for f in features:
        print(f"  - {f}")
        
    # Perform 10-fold Stratified Cross-Validation on training set
    print("\nEstimating out-of-fold class probabilities on training set...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_probs = np.zeros(len(train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train.iloc[train_idx], y_train[train_idx])
        # Probability of class 1 (FSRQ)
        train_probs[val_idx] = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
        print(f"  Fold {fold}/10 completed.")
        
    # Train final classifier on the full training set
    print("\nTraining final classifier on full training set...")
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final_clf.fit(X_train, y_train)
    
    # Predict probabilities for the generalization set
    print("Predicting class probabilities for generalization set...")
    gen_probs = final_clf.predict_proba(X_gen)[:, 1]
    
    # Append P_FSRQ to datasets
    train['P_FSRQ'] = train_probs
    gen['P_FSRQ'] = gen_probs
    
    # Save back to CSV files
    train.to_csv(TRAIN_PATH, index=False)
    gen.to_csv(GEN_PATH, index=False)
    
    print("\nSaved updated datasets with 'P_FSRQ' column:")
    print(f"  - Training: {TRAIN_PATH}")
    print(f"  - Generalization: {GEN_PATH}")
    
    # Print summary statistics
    print(f"\nMean P(FSRQ|X) on training set:")
    print(f"  Overall: {train['P_FSRQ'].mean():.4f}")
    print(f"  For FSRQs (true): {train[train['LabelNo']==1]['P_FSRQ'].mean():.4f}")
    print(f"  For BLLs (true):  {train[train['LabelNo']==0]['P_FSRQ'].mean():.4f}")
    print(f"Mean P(FSRQ|X) on generalization set (all BLLs): {gen['P_FSRQ'].mean():.4f}")
    
    # Save feature list with P_FSRQ added
    expanded_features_path = DATA / "dr3_expanded_features.txt"
    if expanded_features_path.exists():
        with open(expanded_features_path, 'r') as f:
            expanded_features = f.read().splitlines()
        if 'P_FSRQ' not in expanded_features:
            expanded_features.append('P_FSRQ')
            with open(expanded_features_path, 'w') as f:
                f.write('\n'.join(expanded_features))
            print(f"\nUpdated expanded features list in {expanded_features_path}")

if __name__ == '__main__':
    main()
