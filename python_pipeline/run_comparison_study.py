#!/usr/bin/env python3
"""
run_comparison_study.py
=======================
Executes the four main comparative study configurations:
  Config 1: Earlier Cuts (High-Lat Only), 12 Features, 5 Trees, No Calibration
  Config 2: Earlier Cuts (High-Lat Only), 12 Features, 5 Trees, OT Calibration (Narendra et al. 2022)
  Config 3: Earlier Cuts (High-Lat Only), 20 Features, 10 Models, Isotonic Calibration
  Config 4: New Cuts (Combined High+Low), 20 Features, 10 Models, Isotonic Calibration (Our Upgrade)

Saves comparison results to a CSV and generates a LaTeX table.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add python_pipeline to system path to import modules
sys.path.insert(0, str(Path(__file__).parent))
from run_full_dr3_pipeline import run_pipeline

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


# Paths
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUTPUT = BASE / "output" / "dr3_results"
os.makedirs(OUTPUT, exist_ok=True)

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
    'LogRadioFlux', 'LogXrayFlux', 'P_FSRQ',
    'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z',
    'ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y',
    'sdss_u_g', 'sdss_g_r', 'sdss_r_i', 'sdss_i_z',
    'ps1_g_r', 'ps1_r_i', 'ps1_i_z', 'ps1_z_y',
    'Frac_Variability', 'LogHighestEnergy'
]

def format_row(name, metrics):
    r_z = metrics['z']['R']
    rmse_z = metrics['z']['RMSE']
    outlier_fixed = metrics['outlier_pct']
    outlier_2sigma = metrics['outlier_pct_2sigma']
    return {
        "Configuration": name,
        "Pearson R_z": r_z,
        "RMSE (Dz)": rmse_z,
        "Outlier % (Fixed)": outlier_fixed,
        "Outlier % (2sigma)": outlier_2sigma
    }

def main():
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    n_iterations = args.iterations
    n_jobs = args.n_jobs
    
    print("=" * 80)
    print(f"  RUNNING COMPARATIVE STUDY ({n_iterations} Iterations of 10-Fold CV)")
    print("=" * 80 + "\n")
    
    # Load High-Lat datasets
    hl_train_path = DATA / "dr3_high_lat_train.csv"
    if not hl_train_path.exists():
        print("Error: High-lat training data not found! Run build_full_dr3_dataset.py first.")
        sys.exit(1)
    hl_train = pd.read_csv(hl_train_path)
    hl_train = add_new_features(hl_train)
    
    # Load Combined datasets
    comb_train_path = DATA / "dr3_full_train.csv"
    if not comb_train_path.exists():
        print("Error: Combined training data not found! Run build_full_dr3_dataset.py first.")
        sys.exit(1)
    comb_train = pd.read_csv(comb_train_path)
    comb_train = add_new_features(comb_train)
    
    # Dynamic upgraded models selection based on TabPFN availability
    try:
        from tabpfn import TabPFNRegressor
        HAS_TABPFN = True
    except ImportError:
        HAS_TABPFN = False
        
    upgraded_models = ['xgb', 'lgb', 'cat', 'et', 'rf', 'saint', 'ft_transformer', 'tabm', 'tabnet']
    if HAS_TABPFN:
        upgraded_models.append('tabpfn')
        print("TabPFN is available and included in upgraded models.")
    else:
        print("TabPFN is NOT available, running tree models and remaining NNs.")
        
    rows = []
    
    # --- CONFIG 1 & 2: Replication Baseline (Full-Latitude), 12 Features, 5 Trees ---
    print("\nRunning Config 1 & 2 (Full-Lat BLL+FSRQ, 12 features, 5 trees, inv transform)...")
    baseline_models = ['xgb', 'lgb', 'cat', 'et', 'rf']
    
    baseline_results = run_pipeline(
        comb_train, o1_features, baseline_models,
        run_type='baseline', n_iterations=n_iterations, n_jobs=n_jobs,
        target_transform='inv', chk_dir=OUTPUT / "intermediate" / "baseline_inv_v2"
    )
    
    # Config 1: Stacking without calibration (uncalibrated stack)
    metrics_c1 = baseline_results['metrics_report']['stack']
    rows.append(format_row("Config 1: Full-Lat BLL+FSRQ, 12 feat, 5 trees, No Calib (inv)", metrics_c1))
    
    # Config 2: Stacking with Optimal Transport calibration (Narendra baseline)
    metrics_c2 = baseline_results['metrics_report']['stack_corrected']
    rows.append(format_row("Config 2: Full-Lat BLL+FSRQ, 12 feat, 5 trees, OT Calib (inv)", metrics_c2))
    
    # --- CONFIG 3: Upgraded Stacking Model (High-Latitude), 40 Features, 10 Models, Calibration Comparison ---
    print("\nRunning Config 3 (High-Lat BLL+FSRQ, 40 features, 10 models, inv transform)...")
    
    upgraded_hl_inv_results = run_pipeline(
        hl_train, expanded_features, upgraded_models,
        run_type='upgraded', n_iterations=n_iterations, n_jobs=n_jobs,
        target_transform='inv', chk_dir=OUTPUT / "intermediate" / "upgraded_hl_inv"
    )
    metrics_c3_ot = upgraded_hl_inv_results['metrics_report']['stack_ot']
    rows.append(format_row("Config 3: High-Lat BLL+FSRQ, 40 feat, 10 models, OT Calib (inv)", metrics_c3_ot))
    
    metrics_c3_iso = upgraded_hl_inv_results['metrics_report']['stack_iso']
    rows.append(format_row("Config 3: High-Lat BLL+FSRQ, 40 feat, 10 models, Isotonic (inv)", metrics_c3_iso))
    
    metrics_c3_smooth = upgraded_hl_inv_results['metrics_report']['stack_smooth']
    rows.append(format_row("Config 3: High-Lat BLL+FSRQ, 40 feat, 10 models, Smooth OTQM (inv)", metrics_c3_smooth))
    
    # --- CONFIG 4: Upgraded Stacking Model (Full-Latitude), 40 Features, 10 Models, Calibration Comparison ---
    print("\nRunning Config 4 (Full-Lat BLL+FSRQ, 40 features, 10 models, inv transform)...")
    
    upgraded_full_inv_results = run_pipeline(
        comb_train, expanded_features, upgraded_models,
        run_type='upgraded', n_iterations=n_iterations, n_jobs=n_jobs,
        target_transform='inv', chk_dir=OUTPUT / "intermediate" / "upgraded_inv_v2"
    )
    metrics_c4_ot = upgraded_full_inv_results['metrics_report']['stack_ot']
    rows.append(format_row("Config 4: Full-Lat BLL+FSRQ, 40 feat, 10 models, OT Calib (inv)", metrics_c4_ot))
    
    metrics_c4_iso = upgraded_full_inv_results['metrics_report']['stack_iso']
    rows.append(format_row("Config 4: Full-Lat BLL+FSRQ, 40 feat, 10 models, Isotonic (inv)", metrics_c4_iso))
    
    metrics_c4_smooth = upgraded_full_inv_results['metrics_report']['stack_smooth']
    rows.append(format_row("Config 4: Full-Lat BLL+FSRQ, 40 feat, 10 models, Smooth OTQM (inv)", metrics_c4_smooth))
    
    # Compile comparison DataFrame
    df_compare = pd.DataFrame(rows)
    csv_path = OUTPUT / "selection_cuts_comparison.csv"
    df_compare.to_csv(csv_path, index=False)
    print("\nComparison results saved to selection_cuts_comparison.csv")
    print("\n" + df_compare.to_string(index=False).encode('ascii', errors='replace').decode('ascii'))
    
    # Generate LaTeX table snippet
    latex_table = r"""\begin{table*}[t]
\centering
\caption{Comparative Study of Photometric Redshift Performance across Data Selection Cuts, Feature Sets, and Calibration Methods.}
\label{tab:selection_cuts_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Pearson $R_z$} & \textbf{RMSE ($\Delta z$)} & \textbf{Outlier \% (Fixed)} & \textbf{Outlier \% ($2\sigma$)} \\
\midrule
"""
    for _, row in df_compare.iterrows():
        name = row['Configuration']
        r = f"{row['Pearson R_z']:.4f}"
        rmse = f"{row['RMSE (Dz)']:.4f}"
        out_f = f"{row['Outlier % (Fixed)']:.2f}\\%"
        out_2s = f"{row['Outlier % (2sigma)']:.2f}\\%"
        latex_table += f"{name} & {r} & {rmse} & {out_f} & {out_2s} \\\\\n"
        
    latex_table += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    tex_path = OUTPUT / "selection_cuts_comparison.tex"
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print("LaTeX table saved to selection_cuts_comparison.tex")

if __name__ == '__main__':
    main()
