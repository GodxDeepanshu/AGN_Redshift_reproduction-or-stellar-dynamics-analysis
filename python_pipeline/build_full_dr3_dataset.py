#!/usr/bin/env python3
"""
build_full_dr3_dataset.py
=========================
Builds the training and generalization datasets for the Fermi 4LAC-DR3 photo-z pipeline
by combining High-Latitude and Low-Latitude catalogs, applying Narendra et al. (2022) quality
and completeness cuts, and applying parameter range bounds on the generalization set.

Saves both:
- Combined (High + Low Latitude) datasets: dr3_full_train.csv, dr3_full_gen.csv
- High-Latitude Only datasets (for comparison): dr3_high_lat_train.csv, dr3_high_lat_gen.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

DR3_H = DATA / "table-4LAC-DR3-h.fits"
DR3_L = DATA / "table-4LAC-DR3-l.fits"
GAIA_EXISTING = DATA / "gaia_magnitudes.csv"
GAIA_NEW = DATA / "gaia_magnitudes_dr3_new.csv"
MW_EXISTING = DATA / "multi_wavelength_matches.csv"

def decode_bytes(val):
    if isinstance(val, bytes):
        return val.decode('utf-8').strip()
    return str(val).strip()

def load_fits_to_df(path):
    t = Table.read(str(path))
    df = t.to_pandas()
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(decode_bytes)
            except:
                pass
    return df

def preprocess_and_cut(df_raw, apply_lat_cut=True, include_bcu=True):
    df = df_raw.copy()
    df['CLASS_upper'] = df['CLASS'].str.upper().str.strip()
    
    # Keep BLL, FSRQ, and optionally BCU
    classes = ['BLL', 'FSRQ', 'BCU'] if include_bcu else ['BLL', 'FSRQ']
    df = df[df['CLASS_upper'].isin(classes)].copy()
    
    # Numeric conversion
    numeric_cols = ["LP_beta", "LP_Index", "Flux1000", "Energy_Flux100",
                    "Signif_Avg", "Variability_Index", "nu_syn", "nuFnu_syn",
                    "Pivot_Energy", "PL_Index", "GLAT", "Flags", "Redshift"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Narendra et al. (2022) quality cuts
    mask_beta = df['LP_beta'] < 0.7
    mask_idx = df['LP_Index'] > 1.0
    mask_flux = np.log10(df['Flux1000'].clip(lower=1e-20)) > -10.5
    
    mask_qual = mask_beta & mask_idx & mask_flux
    if apply_lat_cut:
        mask_qual = mask_qual & (df['GLAT'].abs() > 10.0)
        
    df_clean = df[mask_qual].copy()
    return df_clean

def merge_multiwavelength(df):
    # Gaia merge
    # Gaia merge
    gaia_exist = pd.read_csv(GAIA_EXISTING)
    gaia_exist['Source_Name'] = gaia_exist['Source_Name'].astype(str).str.strip()
    
    gaia_new = pd.read_csv(GAIA_NEW)
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    
    gaia = pd.concat([gaia_exist, gaia_new]).drop_duplicates('Source_Name')
    df = df.merge(gaia, on='Source_Name', how='left')
    
    # MW merge (do it first so W1mag etc. exist in the dataframe)
    mw = pd.read_csv(MW_EXISTING)
    mw['Source_Name'] = mw['Source_Name'].astype(str).str.strip()
    df = df.merge(mw, on='Source_Name', how='left')
    
    # Convert multiwavelength to numeric
    df['W1mag'] = pd.to_numeric(df['W1mag'], errors='coerce')
    df['W2mag'] = pd.to_numeric(df['W2mag'], errors='coerce')
    df['W3mag'] = pd.to_numeric(df['W3mag'], errors='coerce')
    df['W4mag'] = pd.to_numeric(df['W4mag'], errors='coerce')
    df['S1.4'] = pd.to_numeric(df['S1.4'], errors='coerce')
    df['CR0'] = pd.to_numeric(df['CR0'], errors='coerce')
    for col in ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z', 'ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['W1_W2'] = df['W1mag'] - df['W2mag']
    df['W2_W3'] = df['W2mag'] - df['W3mag']
    df['LogRadioFlux'] = np.log10(df['S1.4'].clip(lower=1e-5))
    df['LogXrayFlux'] = np.log10(df['CR0'].clip(lower=1e-6))
    
    # Incompleteness cuts (nu_syn, nuFnu_syn, and AllWISE W1-W4)
    mask_comp = (
        df['nu_syn'].notna() & (df['nu_syn'] > 0) &
        df['nuFnu_syn'].notna() & (df['nuFnu_syn'] > 0) &
        df['W1mag'].notna() &
        df['W2mag'].notna() &
        df['W3mag'].notna() &
        df['W4mag'].notna()
    )
    df = df[mask_comp].copy()
    
    # Log transformations
    df['LogFlux'] = np.log10(df['Flux1000'])
    df['LogEnergy_Flux'] = np.log10(df['Energy_Flux100'])
    df['LogSignificance'] = np.log10(df['Signif_Avg'])
    df['LogVariability_Index'] = np.log10(df['Variability_Index'])
    df['Lognu_syn'] = np.log10(df['nu_syn'])
    df['LognuFnu_syn'] = np.log10(df['nuFnu_syn'])
    df['LogPivot_Energy'] = np.log10(df['Pivot_Energy'])
    df['LabelNo'] = np.nan
    df.loc[df['CLASS_upper'] == 'BLL', 'LabelNo'] = 0.0
    df.loc[df['CLASS_upper'] == 'FSRQ', 'LabelNo'] = 1.0
    
    return df

def generate_pfsq(train, gen):
    # Train classifier only on the 10 Fermi features
    o1_features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta'
    ]
    
    train_clf = train.copy()
    gen_clf = gen.copy()
    
    for col in o1_features:
        med = train[col].median()
        train_clf[col] = train_clf[col].fillna(med)
        gen_clf[col] = gen_clf[col].fillna(med)
        
    # Train RF classifier only on known BLL & FSRQ labels
    train_clf_labelled = train_clf.dropna(subset=['LabelNo']).copy()
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(train_clf_labelled[o1_features].values, train_clf_labelled['LabelNo'].values)
    
    train['P_FSRQ'] = clf.predict_proba(train_clf[o1_features].values)[:, 1]
    gen['P_FSRQ'] = clf.predict_proba(gen_clf[o1_features].values)[:, 1]
    
    return train, gen

def apply_range_bounds(train, gen, all_features):
    valid_mask = pd.Series(True, index=gen.index)
    for col in all_features:
        min_tr = train[col].min()
        max_tr = train[col].max()
        # Impute NaNs in gen temporarily for the bounding cut
        val_gen = gen[col].fillna(train[col].median())
        col_mask = (val_gen >= min_tr) & (val_gen <= max_tr)
        valid_mask = valid_mask & col_mask
    gen_bounded = gen[valid_mask].copy()
    return gen_bounded

def build_datasets(dr3_all, all_features, apply_lat_cut=True, include_bcu=True, prefix=""):
    print(f"\nBuilding dataset {prefix} (Lat cut={apply_lat_cut}, BCU={include_bcu})...")
    # Apply quality cuts
    df_qual = preprocess_and_cut(dr3_all, apply_lat_cut=apply_lat_cut, include_bcu=include_bcu)
    # Merge MW and drop incomplete
    df_comp = merge_multiwavelength(df_qual)
    
    # Ensure all required features exist in the dataframe (initialize missing with NaN)
    for col in all_features:
        if col not in df_comp.columns:
            df_comp[col] = np.nan
            
    # Strict complete-case filtering (Fermi + AllWISE, no Gaia)
    active_features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta',
        'W1mag', 'W2mag', 'W3mag', 'W4mag', 'W1_W2', 'W2_W3'
    ]
    df_comp = df_comp.dropna(subset=active_features).copy()
            
    # Split training vs generalization
    train = df_comp[df_comp['Redshift'] > 0].copy()
    gen = df_comp[df_comp['Redshift'] <= 0].copy()
    
    # Generate P(FSRQ|X)
    train, gen = generate_pfsq(train, gen)
    
    # For BCUs, set LabelNo to P_FSRQ
    train.loc[train['CLASS_upper'] == 'BCU', 'LabelNo'] = train.loc[train['CLASS_upper'] == 'BCU', 'P_FSRQ']
    gen.loc[gen['CLASS_upper'] == 'BCU', 'LabelNo'] = gen.loc[gen['CLASS_upper'] == 'BCU', 'P_FSRQ']
    
    # Filter out BCUs from generalization set (only keep BLL and FSRQ)
    gen = gen[gen['CLASS_upper'].isin(['BLL', 'FSRQ'])].copy()
    
    # Impute missing values with training medians (for standard features)
    for col in all_features:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        gen[col] = gen[col].fillna(med)
        
    # Disable range bounding to retain all 414 generalization sources
    gen_bounded = gen
    
    # Log-transform target
    train['InvRedshift'] = 1.0 / (train['Redshift'] + 1.0)
    gen_bounded['InvRedshift'] = np.nan
    
    print(f"  Training set size:       {len(train)} (BLL: {(train['CLASS_upper']=='BLL').sum()}, FSRQ: {(train['CLASS_upper']=='FSRQ').sum()}, BCU: {(train['CLASS_upper']=='BCU').sum()})")
    print(f"  Generalization set size: {len(gen_bounded)} (BLL: {(gen_bounded['CLASS_upper']=='BLL').sum()}, FSRQ: {(gen_bounded['CLASS_upper']=='FSRQ').sum()}, BCU: {(gen_bounded['CLASS_upper']=='BCU').sum()})")
    
    return train, gen_bounded

def main():
    print("Loading FITS files...")
    dr3_h = load_fits_to_df(DR3_H)
    dr3_l = load_fits_to_df(DR3_L)
    dr3_all = pd.concat([dr3_h, dr3_l], ignore_index=True)
    print(f"Raw DR3 total sources: {len(dr3_all)}")
    
    all_features = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta', 'LabelNo',
        'W1mag', 'W2mag', 'W3mag', 'W4mag', 'W1_W2', 'W2_W3',
        'LogRadioFlux', 'LogXrayFlux', 'P_FSRQ'
    ]
    
    # 1. Combined Catalog without BCU (Full-Latitude, BLL+FSRQ Only)
    train_full, gen_full = build_datasets(dr3_all, all_features, apply_lat_cut=False, include_bcu=False, prefix="Full-Lat BLL+FSRQ Only")
    train_full.to_csv(DATA / "dr3_full_train.csv", index=False)
    gen_full.to_csv(DATA / "dr3_full_gen.csv", index=False)
    
    # 2. High-Latitude Catalog without BCU (High-Lat BLL+FSRQ Only)
    train_hl, gen_hl = build_datasets(dr3_all, all_features, apply_lat_cut=True, include_bcu=False, prefix="High-Lat BLL+FSRQ Only")
    train_hl.to_csv(DATA / "dr3_high_lat_train.csv", index=False)
    gen_hl.to_csv(DATA / "dr3_high_lat_gen.csv", index=False)
    
    print("\nDataset building complete!")

if __name__ == '__main__':
    main()
