import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    print("=== Generating Improved Visualizations ===")
    os.makedirs("plots/dr3_plots", exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 18
    })
    
    # 1. Load catalog
    catalog = pd.read_csv("data/DR3_New_AGN_Redshift_Catalog_Improved.csv")
    print(f"Loaded {len(catalog)} predicted sources.")
    
    # 2. Histogram of predicted redshifts for all 319 new DR3 sources
    plt.figure(figsize=(10, 6))
    sns.histplot(catalog['ensemble_z'], bins=20, kde=True, color='#2C3E50', edgecolor='black', alpha=0.7)
    plt.xlabel("Predicted Redshift ($z_{\\mathrm{phot}}$)")
    plt.ylabel("Number of Sources")
    plt.title("Photometric Redshift Distribution of the Full 4LAC-DR3 New Catalog (n=319)")
    plt.savefig("plots/dr3_plots/dr3_new_redshifts_hist_improved.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class-specific redshift distribution (BLL, FSRQ, BCU)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=catalog, x='ensemble_z', hue='agn_type_est', kde=True,
                 bins=20, multiple='layer', palette={'BLL': '#C0392B', 'FSRQ': '#27AE60', 'BCU': '#F39C12'},
                 edgecolor='black', alpha=0.5)
    plt.xlabel("Predicted Redshift ($z_{\\mathrm{phot}}$)")
    plt.ylabel("Number of Sources")
    plt.title("Redshift Distribution by AGN Class (Including 221 BCUs)")
    plt.legend(title="AGN Class", labels=["BCU (Uncertain)", "FSRQ", "BL Lac"])
    plt.savefig("plots/dr3_plots/dr3_redshift_distribution_by_class_improved.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Predicted vs. Spectroscopic Redshift for the independent test set (n=89)
    # Load raw sources to get actual spectroscopic redshifts
    new_sources = pd.read_csv("data/DR3_New_Sources_mw.csv")
    spec_subset = new_sources[new_sources['Redshift'] > 0].copy()
    
    # Align catalog predictions with spec_subset
    spec_catalog = catalog[catalog['source_name'].isin(spec_subset['Source_Name'])].copy()
    spec_catalog['actual_z'] = spec_catalog['source_name'].map(lambda x: spec_subset[spec_subset['Source_Name'] == x]['Redshift'].values[0])
    
    # Calculate metrics for the ensemble
    r_val = np.corrcoef(spec_catalog['ensemble_z'], spec_catalog['actual_z'])[0, 1]
    rmse_val = np.sqrt(np.mean((spec_catalog['ensemble_z'] - spec_catalog['actual_z'])**2))
    
    plt.figure(figsize=(8, 8))
    # Color points by class
    sns.scatterplot(data=spec_catalog, x='actual_z', y='ensemble_z', hue='agn_type_est',
                    palette={'BLL': '#C0392B', 'FSRQ': '#27AE60', 'BCU': '#F39C12'},
                    s=80, alpha=0.8, edgecolor='black')
    
    # 1-to-1 line
    lims = [0, max(spec_catalog['actual_z'].max(), spec_catalog['ensemble_z'].max()) + 0.2]
    plt.plot(lims, lims, '--r', alpha=0.75, label='1-to-1 Line')
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Spectroscopic Redshift ($z_{\\mathrm{spec}}$)")
    plt.ylabel("Photometric Redshift ($z_{\\mathrm{phot}}$)")
    plt.title(f"Independent Spectroscopic Validation (n={len(spec_catalog)})\n$R$ = {r_val:.4f}, RMSE = {rmse_val:.4f}")
    plt.legend(title="AGN Class")
    plt.savefig("plots/dr3_plots/predicted_vs_spectroscopic_improved.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Conformal Prediction Intervals Visual
    # Sort spec_catalog by actual_z for visual representation
    sorted_spec = spec_catalog.sort_values(by='actual_z').reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(sorted_spec.index, sorted_spec['ensemble_z'],
                 yerr=[sorted_spec['ensemble_z'] - sorted_spec['z_low_68'], sorted_spec['z_high_68'] - sorted_spec['ensemble_z']],
                 fmt='o', color='#2980B9', ecolor='#AED6F1', elinewidth=1.5, capsize=2, alpha=0.8,
                 label='Photometric $z$ (68% Conformal Interval)')
    plt.scatter(sorted_spec.index, sorted_spec['actual_z'], color='red', s=30, zorder=5, label='Actual Spectroscopic $z$')
    
    plt.xlabel("Sources (ordered by spectroscopic redshift)")
    plt.ylabel("Redshift ($z$)")
    plt.title("Photometric Redshift Predictions with 68% Conformal Prediction Intervals")
    plt.legend()
    plt.savefig("plots/dr3_plots/conformal_prediction_intervals.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All improved plots generated successfully in plots/dr3_plots/.")

if __name__ == "__main__":
    generate_plots()
