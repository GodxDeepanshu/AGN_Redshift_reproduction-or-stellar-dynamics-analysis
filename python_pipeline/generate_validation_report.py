import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from scipy.stats import pearsonr, spearmanr

def main():
    print("=== STARTING GENERATION OF EXTERNAL VALIDATION REPORT ===")
    
    # Paths
    artifact_dir = r"C:\Users\deepa\.gemini\antigravity-ide\brain\5b6cb9f2-73fd-4e3a-8e4b-9c85f83199f7"
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 1. Load data
    df_rel = pd.read_csv('data/DR3_New_AGN_Redshift_Catalog_Reliability.csv')
    df_desi = pd.read_csv('data/desi_matches_counterparts.csv')
    df_ero = pd.read_csv('data/erosita_matches_counterparts.csv')
    
    # 2. Process DESI matches
    # Filter high-confidence: ZWARN == 0
    df_desi_clean = df_desi[df_desi['DESI_zwarn'] == 0].copy()
    # De-duplicate: Keep closest match for each source
    df_desi_clean = df_desi_clean.sort_values('DESI_Separation_arcsec')
    df_desi_unique = df_desi_clean.drop_duplicates(subset='Source_Name', keep='first').copy()
    
    # 3. Process eROSITA matches
    # De-duplicate: Keep closest match for each source
    df_ero = df_ero.sort_values('eRASS1_Separation_arcsec')
    df_ero_unique = df_ero.drop_duplicates(subset='Source_Name', keep='first').copy()
    
    # 4. Merge validation data with predicted catalog
    # Merge DESI
    merged = df_desi_unique.merge(df_rel, left_on='Source_Name', right_on='source_name', how='inner')
    # Merge eROSITA (optional join to check how many have both, but we validate photo-z vs spectroscopic-z)
    
    # Save the cleaned validation catalog
    validation_cat = merged[['Source_Name', 'Target_RA', 'Target_Dec', 'class', 'reliability_grade', 
                             'ensemble_z', 'DESI_z', 'DESI_zerr', 'DESI_spectype', 'DESI_Separation_arcsec']].copy()
    validation_cat.to_csv('data/DR3_External_Validation_Catalog.csv', index=False)
    if os.path.exists(artifact_dir):
        shutil.copy('data/DR3_External_Validation_Catalog.csv', os.path.join(artifact_dir, 'DR3_External_Validation_Catalog.csv'))
    print(f"Saved validation catalog to data/DR3_External_Validation_Catalog.csv. N = {len(validation_cat)}")
    
    # 5. Calculate Metrics
    pred_z = merged['ensemble_z'].values
    spec_z = merged['DESI_z'].values
    grades = merged['reliability_grade'].values
    
    def compute_stats(p_z, s_z):
        if len(p_z) < 1:
            return {k: np.nan for k in ['R', 'rho', 'rmse', 'mae', 'medae', 'outliers', 'N']}
        r_val, _ = pearsonr(p_z, s_z) if len(p_z) >= 2 else (np.nan, None)
        rho_val, _ = spearmanr(p_z, s_z) if len(p_z) >= 2 else (np.nan, None)
        rmse_val = np.sqrt(np.mean((p_z - s_z)**2))
        mae_val = np.mean(np.abs(p_z - s_z))
        medae_val = np.median(np.abs(p_z - s_z))
        outlier_frac = np.mean(np.abs(p_z - s_z) / (1.0 + s_z) > 0.15)
        return {
            'R': r_val,
            'rho': rho_val,
            'rmse': rmse_val,
            'mae': mae_val,
            'medae': medae_val,
            'outliers': outlier_frac,
            'N': len(p_z)
        }
    
    stats_overall = compute_stats(pred_z, spec_z)
    stats_grade_a = compute_stats(pred_z[grades == 'Grade A'], spec_z[grades == 'Grade A'])
    stats_grade_b = compute_stats(pred_z[grades == 'Grade B'], spec_z[grades == 'Grade B'])
    stats_grade_c = compute_stats(pred_z[grades == 'Grade C'], spec_z[grades == 'Grade C'])
    
    # 6. Generate LaTeX Table 2 (Performance Metrics)
    table2_tex = """\\begin{table}[htbp]
\\centering
\\caption{Statistical metrics for the external validation sample ($N=47$) against DESI DR1 spectroscopic redshifts, overall and partitioned by OOD Reliability Grade.}
\\label{tab:external_validation_metrics}
\\begin{tabular}{lcccc}
\\hline\\hline
\\textbf{Metric} & \\textbf{Overall} & \\textbf{Grade A (Reliable)} & \\textbf{Grade B (Intermediate)} & \\textbf{Grade C (Unreliable)} \\\\
\\hline
Sample Size ($N$) & $47$ & $30$ & $16$ & $1$ \\\\
Pearson $R$ & $""" + f"{stats_overall['R']:.4f}" + """$ & $""" + f"{stats_grade_a['R']:.4f}" + """$ & $""" + f"{stats_grade_b['R']:.4f}" + """$ & -- \\\\
Spearman $\\rho$ & $""" + f"{stats_overall['rho']:.4f}" + """$ & $""" + f"{stats_grade_a['rho']:.4f}" + """$ & $""" + f"{stats_grade_b['rho']:.4f}" + """$ & -- \\\\
RMSE & $""" + f"{stats_overall['rmse']:.4f}" + """$ & $""" + f"{stats_grade_a['rmse']:.4f}" + """$ & $""" + f"{stats_grade_b['rmse']:.4f}" + """$ & $""" + f"{stats_grade_c['rmse']:.4f}" + """$ \\\\
MAE & $""" + f"{stats_overall['mae']:.4f}" + """$ & $""" + f"{stats_grade_a['mae']:.4f}" + """$ & $""" + f"{stats_grade_b['mae']:.4f}" + """$ & $""" + f"{stats_grade_c['mae']:.4f}" + """$ \\\\
Median AE & $""" + f"{stats_overall['medae']:.4f}" + """$ & $""" + f"{stats_grade_a['medae']:.4f}" + """$ & $""" + f"{stats_grade_b['medae']:.4f}" + """$ & $""" + f"{stats_grade_c['medae']:.4f}" + """$ \\\\
Outlier Fraction & $""" + f"{stats_overall['outliers']*100:.1f}\\%" + """$ & $""" + f"{stats_grade_a['outliers']*100:.1f}\\%" + """$ & $""" + f"{stats_grade_b['outliers']*100:.1f}\\%" + """$ & $""" + f"{stats_grade_c['outliers']*100:.1f}\\%" + """$ \\\\
\\hline
\\end{tabular}
\\end{table}
"""
    with open('data/validation_table2.tex', 'w') as f:
        f.write(table2_tex)
    print("Generated Table 2 LaTeX.")
    
    # 7. Generate LaTeX Table 1 (Validated Sources List)
    # Sort by RA
    merged_sorted = merged.sort_values('Target_RA')
    
    table1_rows = []
    for _, row in merged_sorted.iterrows():
        name = row['Source_Name']
        ra = row['Target_RA']
        dec = row['Target_Dec']
        pred = row['ensemble_z']
        true = row['DESI_z']
        diff = pred - true
        sep = row['DESI_Separation_arcsec']
        grade = row['reliability_grade']
        
        # Determine class name formatting
        src_class = row['class'].upper()
        
        row_str = f"{name} & {ra:.4f} & {dec:.4f} & {src_class} & {pred:.3f} & {true:.4f} & {diff:+.3f} & {sep:.3f} & {grade} \\\\"
        table1_rows.append(row_str)
        
    table1_tex = """\\begin{table*}[t]
\\centering
\\caption{Complete list of the 47 Fermi 4LAC-DR3 newly identified AGNs validated against high-confidence ($ZWARN=0$) spectroscopic redshifts in DESI DR1.}
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
    print("Generated Table 1 LaTeX.")
    
    # 8. Generate Validation Figures
    # Color palette for Grades
    colors = {'Grade A': '#2b5c8f', 'Grade B': '#d95f02', 'Grade C': '#7570b3'}
    
    # 1. Predicted vs True (Spectroscopic) Redshift Plot
    plt.figure(figsize=(7, 6))
    for grade in ['Grade A', 'Grade B', 'Grade C']:
        mask = (grades == grade)
        if np.sum(mask) > 0:
            plt.scatter(spec_z[mask], pred_z[mask], c=colors[grade], label=grade, alpha=0.85, edgecolors='k', s=60, zorder=3)
            
    # Line of equality
    lims = [0, max(max(pred_z), max(spec_z)) + 0.1]
    plt.plot(lims, lims, 'k--', alpha=0.6, label='1:1 Line', zorder=2)
    
    plt.xlabel(r'Spectroscopic Redshift ($z_{\mathrm{spec}}$)', fontsize=12)
    plt.ylabel(r'Predicted Redshift ($z_{\mathrm{phot}}$)', fontsize=12)
    plt.title('Photometric vs. Spectroscopic Redshift (DESI DR1)', fontsize=13)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
    plt.xlim(0, max(spec_z) + 0.1)
    plt.ylim(0, max(pred_z) + 0.1)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_predicted_vs_true.png', dpi=300)
    plt.close()
    
    # 2. Residual Plot (Delta z vs z_spec)
    plt.figure(figsize=(7, 5))
    delta_z = pred_z - spec_z
    norm_delta_z = delta_z / (1.0 + spec_z)
    
    for grade in ['Grade A', 'Grade B', 'Grade C']:
        mask = (grades == grade)
        if np.sum(mask) > 0:
            plt.scatter(spec_z[mask], norm_delta_z[mask], c=colors[grade], label=grade, alpha=0.85, edgecolors='k', s=60, zorder=3)
            
    plt.axhline(0, color='r', linestyle='--', alpha=0.8, zorder=2)
    plt.axhline(0.15, color='gray', linestyle=':', alpha=0.6, zorder=2)
    plt.axhline(-0.15, color='gray', linestyle=':', alpha=0.6, zorder=2)
    
    plt.xlabel(r'Spectroscopic Redshift ($z_{\mathrm{spec}}$)', fontsize=12)
    plt.ylabel(r'Normalized Residual $\Delta z / (1 + z_{\mathrm{spec}})$', fontsize=12)
    plt.title('Normalized Residuals vs. Spectroscopic Redshift', fontsize=13)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
    plt.xlim(0, max(spec_z) + 0.1)
    plt.ylim(-0.8, 0.8)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_residuals.png', dpi=300)
    plt.close()
    
    # 3. Histogram of Normalized Residuals
    plt.figure(figsize=(7, 5))
    for grade in ['Grade A', 'Grade B']:
        mask = (grades == grade)
        if np.sum(mask) > 0:
            sns.histplot(norm_delta_z[mask], bins=12, kde=True, color=colors[grade], label=grade, alpha=0.5, element='step')
            
    mask_c = (grades == 'Grade C')
    if np.sum(mask_c) > 0:
        plt.axvline(norm_delta_z[mask_c][0], color=colors['Grade C'], linestyle='-.', label='Grade C')
        
    plt.axvline(0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel(r'Normalized Residual $\Delta z / (1 + z_{\mathrm{spec}})$', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Normalized Residuals', fontsize=13)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/dr3_validation_residuals_hist.png', dpi=300)
    plt.close()
    
    print("Figures generated successfully.")
    
    # Copy plots to artifact directory
    if os.path.exists(artifact_dir):
        shutil.copy('plots/dr3_validation_predicted_vs_true.png', os.path.join(artifact_dir, 'dr3_validation_predicted_vs_true.png'))
        shutil.copy('plots/dr3_validation_residuals.png', os.path.join(artifact_dir, 'dr3_validation_residuals.png'))
        shutil.copy('plots/dr3_validation_residuals_hist.png', os.path.join(artifact_dir, 'dr3_validation_residuals_hist.png'))
        print("Plots copied to artifact directory.")
        
    print("=== EXTERNAL VALIDATION REPORT COMPLETED ===")

if __name__ == '__main__':
    main()
