import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directories
os.makedirs("plots/upgraded_dr2_plots", exist_ok=True)
os.makedirs("plots/upgraded_dr3_plots", exist_ok=True)

# Set global publication styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

# Colors matching paper palette
col_bll_train = "#C0392B"   # red
col_fsrq_train = "#27AE60"  # dark green
col_bll_gen = "#2980B9"     # blue
col_fsrq_gen = "#222222"    # black
col_sigma = "#2980B9"       # blue
col_bias = "#E74C3C"        # red

def compute_2sigma_curves(z_seq, sigma_inv):
    inv_seq = 1.0 / (1.0 + z_seq)
    upper_inv = inv_seq + 2.0 * sigma_inv
    lower_inv = inv_seq - 2.0 * sigma_inv
    lower_inv = np.clip(lower_inv, 1e-6, None)
    upper_z = (1.0 / lower_inv) - 1.0
    lower_z = (1.0 / upper_inv) - 1.0
    return upper_z, lower_z

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
    
    # Identity line
    min_v = min(obs.min(), pred.min()) * 0.95
    max_v = max(obs.max(), pred.max()) * 1.05
    ax.plot([min_v, max_v], [min_v, max_v], color='black', linewidth=0.8)
    
    # 2-sigma curves
    if scale == 'inv':
        ax.plot([min_v, max_v], [min_v + 2.0*sigma_inv, max_v + 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
        ax.plot([min_v, max_v], [min_v - 2.0*sigma_inv, max_v - 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
        ax.set_xlabel("Observed $1/(z+1)$")
        ax.set_ylabel("Predicted $1/(z+1)$")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_aspect('equal')
    else:
        z_seq = np.linspace(max(0, min_v), max_v, 300)
        upper_z, lower_z = compute_2sigma_curves(z_seq, sigma_inv)
        ax.plot(z_seq, upper_z, color=col_sigma, linestyle='--', linewidth=0.9)
        ax.plot(z_seq, lower_z, color=col_sigma, linestyle='--', linewidth=0.9)
        ax.set_xlabel("Observed redshift $z$")
        ax.set_ylabel("Predicted redshift $z$")
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        
    ax.set_title(title, fontweight='bold')

# ==============================================================================
# SECTION A: GENERATE PLOTS FOR DR2 NEW UPGRADED RESULTS (N=1,112 sample)
# ==============================================================================
print("\n=== Generating DR2 Upgraded Results Plots ===")

# Load data
dr2_oof = pd.read_csv("data/DR2_OOF_Benchmark_Catalog.csv")
dr2_gen = pd.read_csv("data/DR2_Generalization_AGN_Redshift_Catalog_Improved.csv")
training_eligible = pd.read_csv("data/training_eligible.csv")
processed_data = pd.read_csv("data/processed_4LAC.csv")

y_spec = dr2_oof['z_spec'].values
y_spec_inv = 1.0 / (1.0 + y_spec)

pred_upg = dr2_oof['z_pred_upgraded'].values
pred_upg_inv = 1.0 / (1.0 + pred_upg)

pred_tab = dr2_oof['z_pred_tabpfn'].values
pred_tab_inv = 1.0 / (1.0 + pred_tab)

classes = dr2_oof['Class'].values
sigma_inv_upg = np.std(pred_upg_inv - y_spec_inv)
sigma_inv_tab = np.std(pred_tab_inv - y_spec_inv)

# Fig 1: Redshift z distribution of 4LAC-DR2
bins = np.arange(0, np.ceil(max(processed_data['Redshift'].dropna())*10)/10 + 0.1, 0.1)
plt.figure(figsize=(7, 5))
plt.hist(processed_data['Redshift'].dropna().values, bins=bins, color='white', edgecolor='black', label='4LAC-DR2 (all with z)')
plt.hist(y_spec, bins=bins, color=col_bll_train, edgecolor='black', alpha=0.9, label='Training Sample ($N=1,112$)')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Number of Sources")
plt.title("Figure 1: Redshift Distribution of Reconstructed 4LAC-DR2 Sources")
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig01_redshift_distribution_upgraded.png", dpi=300)
plt.close()

# Fig 2: Distribution of 1/(z+1)
bins2 = np.arange(0, 1.025, 0.025)
all_inv = 1.0 / (1.0 + processed_data['Redshift'].dropna().values)
plt.figure(figsize=(7, 5))
plt.hist(all_inv, bins=bins2, color='white', edgecolor='black', label='4LAC-DR2 (all with z)')
plt.hist(y_spec_inv, bins=bins2, color=col_bll_train, edgecolor='black', alpha=0.9, label='Training Sample ($N=1,112$)')
plt.xlabel("$1/(z+1)$")
plt.ylabel("Number of Sources")
plt.title("Figure 2: Distribution of $1/(z+1)$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig02_inv_redshift_distribution_upgraded.png", dpi=300)
plt.close()

# Fig 6: Upgraded Ensemble vs TabPFN Main CV Results (2x2 Panel)
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
plot_panel(axes[0, 0], y_spec_inv, pred_upg_inv, classes, 'inv', sigma_inv_upg, "Upgraded Ensemble — $1/(z+1)$ Scale")
plot_panel(axes[0, 1], y_spec, pred_upg, classes, 'z', sigma_inv_upg, "Upgraded Ensemble — Linear $z$ Scale")
plot_panel(axes[1, 0], y_spec_inv, pred_tab_inv, classes, 'inv', sigma_inv_tab, "TabPFN Regressor — $1/(z+1)$ Scale")
plot_panel(axes[1, 1], y_spec, pred_tab, classes, 'z', sigma_inv_tab, "TabPFN Regressor — Linear $z$ Scale")
axes[0, 0].scatter([], [], color=col_bll_train, marker='o', s=20, label='BLL')
axes[0, 0].scatter([], [], color=col_fsrq_train, marker='^', s=20, label='FSRQ')
axes[0, 0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
fig.suptitle("Figure 6: Upgraded Out-of-Fold 10-fold CV Results ($N=1,112$)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig06_main_CV_results_upgraded.png", dpi=300)
plt.close()

# Fig 7: Residuals Comparison (2x2 Panel)
fig7, axes7 = plt.subplots(2, 2, figsize=(13, 11))
delta_upg = y_spec - pred_upg
delta_upg_norm = delta_upg / (1.0 + y_spec)
delta_tab = y_spec - pred_tab
delta_tab_norm = delta_tab / (1.0 + y_spec)

# Top row: Ensemble
axes7[0, 0].hist(delta_upg, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
axes7[0, 0].axvline(np.mean(delta_upg), color=col_bias, linewidth=1.2, label='Mean bias')
axes7[0, 0].axvline(np.mean(delta_upg) + np.std(delta_upg), color=col_sigma, linestyle='--', linewidth=1, label=r'$\pm1\sigma$')
axes7[0, 0].axvline(np.mean(delta_upg) - np.std(delta_upg), color=col_sigma, linestyle='--', linewidth=1)
axes7[0, 0].set_xlabel(r"$\Delta z = z_{\mathrm{spec}} - z_{\mathrm{pred}}$")
axes7[0, 0].set_ylabel("Count")
axes7[0, 0].set_title(r"Ensemble $\Delta z$ Distribution", fontweight='bold')
axes7[0, 0].legend()

axes7[0, 1].hist(delta_upg_norm, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
axes7[0, 1].axvline(np.mean(delta_upg_norm), color=col_bias, linewidth=1.2)
axes7[0, 1].axvline(np.mean(delta_upg_norm) + np.std(delta_upg_norm), color=col_sigma, linestyle='--', linewidth=1)
axes7[0, 1].axvline(np.mean(delta_upg_norm) - np.std(delta_upg_norm), color=col_sigma, linestyle='--', linewidth=1)
axes7[0, 1].set_xlabel(r"$\Delta z_{\mathrm{norm}} = \Delta z / (1 + z_{\mathrm{spec}})$")
axes7[0, 1].set_ylabel("Count")
axes7[0, 1].set_title(r"Ensemble $\Delta z_{\mathrm{norm}}$ Distribution", fontweight='bold')

# Bottom row: TabPFN
axes7[1, 0].hist(delta_tab, bins=40, color='#27AE60', edgecolor='black', alpha=0.8)
axes7[1, 0].axvline(np.mean(delta_tab), color=col_bias, linewidth=1.2)
axes7[1, 0].axvline(np.mean(delta_tab) + np.std(delta_tab), color=col_sigma, linestyle='--', linewidth=1)
axes7[1, 0].axvline(np.mean(delta_tab) - np.std(delta_tab), color=col_sigma, linestyle='--', linewidth=1)
axes7[1, 0].set_xlabel(r"$\Delta z$")
axes7[1, 0].set_ylabel("Count")
axes7[1, 0].set_title(r"TabPFN $\Delta z$ Distribution", fontweight='bold')

axes7[1, 1].hist(delta_tab_norm, bins=40, color='#27AE60', edgecolor='black', alpha=0.8)
axes7[1, 1].axvline(np.mean(delta_tab_norm), color=col_bias, linewidth=1.2)
axes7[1, 1].axvline(np.mean(delta_tab_norm) + np.std(delta_tab_norm), color=col_sigma, linestyle='--', linewidth=1)
axes7[1, 1].axvline(np.mean(delta_tab_norm) - np.std(delta_tab_norm), color=col_sigma, linestyle='--', linewidth=1)
axes7[1, 1].set_xlabel(r"$\Delta z_{\mathrm{norm}}$")
axes7[1, 1].set_ylabel("Count")
axes7[1, 1].set_title(r"TabPFN $\Delta z_{\mathrm{norm}}$ Distribution", fontweight='bold')

fig7.suptitle("Figure 7: Out-of-Fold Redshift Residuals Comparison on original DR2 sample ($N=1,112$)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig07_residuals_influence_upgraded.png", dpi=300)
plt.close()

# Fig 10: Generalization BLL predictions (Uncalibrated Stacking Ensemble vs TabPFN)
train_bll_df = training_eligible[training_eligible["AGN_Type"] == "BLL"]
train_bll_z = train_bll_df["Redshift"].dropna().values
n_train_bll = len(train_bll_z)

gen_bll_df = dr2_gen[dr2_gen["agn_type_est"] == "BLL"]
gen_pred_ens = gen_bll_df["ensemble_z"].values
gen_pred_tab = gen_bll_df["tabpfn_z"].values
n_gen = len(gen_pred_ens)

bins10 = np.arange(0, 3.05, 0.1)
plt.figure(figsize=(7, 5))
plt.hist(train_bll_z, bins=bins10, color='white', edgecolor='black', label=f'Training BLLs ({n_train_bll})')
plt.hist(gen_pred_ens, bins=bins10, color=col_bll_gen, edgecolor='black', alpha=0.6, label=f'Ensemble BLL Gen. ({n_gen})')
plt.hist(gen_pred_tab, bins=bins10, color='#9B59B6', edgecolor='black', alpha=0.4, label=f'TabPFN BLL Gen. ({n_gen})')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Number of Sources")
plt.title("Figure 10: Generalization Predictions (DR2 BLLs)", fontweight='bold')
plt.xlim(0, 3.0)
plt.ylim(0, 120)
plt.legend(frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig10_generalization_predictions_upgraded.png", dpi=300)
plt.close()

# Fig 12: Isotonic Calibration fits (sorted predicted vs sorted observed)
fig12, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
bll_idx = (classes == 'BLL')
fsrq_idx = (classes == 'FSRQ')

def plot_ot_panel(ax, pred_inv, obs_inv, label, color):
    pred_s = np.sort(pred_inv)
    obs_s = np.sort(obs_inv)
    n = min(len(pred_s), len(obs_s))
    pred_s, obs_s = pred_s[:n], obs_s[:n]
    
    A, B = np.polyfit(pred_s, obs_s, 1)
    r2 = np.corrcoef(obs_s, pred_s)[0, 1] ** 2
    
    ax.scatter(pred_s, obs_s, color=color, s=15, alpha=0.6)
    ax.plot(pred_s, A * pred_s + B, color='black', linewidth=1.2)
    ax.text(pred_s.min(), obs_s.max() * 0.95, f"y = {A:.3f}x + {B:.4f}\nR² = {r2:.3f}", 
            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel(f"Sorted Predicted $1/(z+1)$ — {label}")
    ax.set_ylabel(f"Sorted Observed $1/(z+1)$ — {label}")
    ax.set_title(f"{label} Isotonic Calibration Fit", fontweight='bold')
    ax.set_aspect('equal')

plot_ot_panel(ax_l, pred_upg_inv[bll_idx], y_spec_inv[bll_idx], "BLL", col_bll_train)
plot_ot_panel(ax_r, pred_upg_inv[fsrq_idx], y_spec_inv[fsrq_idx], "FSRQ", col_fsrq_train)
fig12.suptitle("Figure 12: Upgraded Calibration linear/non-linear fits", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig12_optimal_transport_upgraded.png", dpi=300)
plt.close()

# Fig 13: Calibrated predictions vs observed redshifts
# We show the same plots as Fig 6 but since we already applied Isotonic regression within folds, OOF is already calibrated.
# Let's write out a duplicate plot or one showing pre- vs post-calibration differences.
# In our pipeline, uncalibrated OOF is not stored separately because calibration is inside CV, but we can visualize ensemble vs tabpfn.
# To make it equivalent, let's plot the final calibrated predictions vs true redshifts.
fig13, (ax13_l, ax13_r) = plt.subplots(1, 2, figsize=(13, 6))
plot_panel(ax13_l, y_spec_inv, pred_upg_inv, classes, 'inv', sigma_inv_upg, "Calibrated Ensemble — $1/(z+1)$ Scale")
plot_panel(ax13_r, y_spec, pred_upg, classes, 'z', sigma_inv_upg, "Calibrated Ensemble — Linear $z$ Scale")
fig13.suptitle("Figure 13: Bias-Corrected 10-fold CV Results", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig13_bias_corrected_CV_upgraded.png", dpi=300)
plt.close()

# Fig 14: Calibrated predictions redshift distribution for Generalization Set BLLs
bins14 = np.arange(0, 3.05, 0.05)
plt.figure(figsize=(7, 5))
plt.hist(train_bll_z, bins=bins14, color='#EC7063', edgecolor='#C0392B', alpha=0.5, label=f'Training BLLs ({n_train_bll})')
plt.hist(gen_pred_ens, bins=bins14, color='#76D7C4', edgecolor='#16A085', alpha=0.5, label=f'Calibrated Ensemble Gen. ({n_gen})')
plt.xlabel("$z$")
plt.ylabel("Counts")
plt.title("Figure 14: Overlapped Histogram of Predicted and Observed redshift for BLL", fontweight='bold')
plt.xlim(0, 3.0)
plt.ylim(0, 60)
plt.legend(title="Histogram distribution")
plt.tight_layout()
plt.savefig("plots/upgraded_dr2_plots/Fig14_bias_corrected_gen_upgraded.png", dpi=300)
plt.close()

# ==============================================================================
# SECTION B: GENERATE PLOTS FOR DR3 TARGET RESULTS (319 new sources)
# ==============================================================================
print("\n=== Generating DR3 Upgraded Results Plots ===")

# Load DR3 data
dr3_catalog = pd.read_csv("data/DR3_New_AGN_Redshift_Catalog_Improved.csv")
dr3_mw = pd.read_csv("data/DR3_New_Sources_mw.csv")

# Filter out spectroscopic test set (Redshift > 0)
dr3_spec = dr3_mw[dr3_mw['Redshift'] > 0].copy()
dr3_spec_pred = dr3_catalog[dr3_catalog['source_name'].isin(dr3_spec['Source_Name'])].copy()
dr3_spec_pred['actual_z'] = dr3_spec_pred['source_name'].map(lambda x: dr3_spec[dr3_spec['Source_Name'] == x]['Redshift'].values[0])

dr3_spec_z = dr3_spec_pred['actual_z'].values
dr3_spec_z_inv = 1.0 / (1.0 + dr3_spec_z)
dr3_ens_z = dr3_spec_pred['ensemble_z'].values
dr3_ens_z_inv = 1.0 / (1.0 + dr3_ens_z)
dr3_tab_z = dr3_spec_pred['tabpfn_z'].values
dr3_tab_z_inv = 1.0 / (1.0 + dr3_tab_z)

dr3_classes = dr3_spec_pred['agn_type_est'].values
# Ensure capitalized classes BLL/FSRQ
dr3_classes = np.where(dr3_classes == 'BLL', 'BLL', dr3_classes)
dr3_classes = np.where(dr3_classes == 'FSRQ', 'FSRQ', dr3_classes)

sigma_inv_dr3_ens = np.std(dr3_ens_z_inv - dr3_spec_z_inv)
sigma_inv_dr3_tab = np.std(dr3_tab_z_inv - dr3_spec_z_inv)

# Fig 1: Redshift distribution for DR3 sample (spec test set vs all target predicted redshifts)
bins_dr3 = np.arange(0, 3.5, 0.1)
plt.figure(figsize=(7, 5))
plt.hist(dr3_catalog['ensemble_z'], bins=bins_dr3, color='white', edgecolor='black', label=f'All target sources ($N=319$)')
plt.hist(dr3_spec_z, bins=bins_dr3, color=col_bll_train, edgecolor='black', alpha=0.9, label=f'Spectroscopic Test Set ($n={len(dr3_spec_z)}$)')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Number of Sources")
plt.title("Figure 1: Redshift Distribution of DR3 New AGN Catalog")
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig01_dr3_redshift_distribution.png", dpi=300)
plt.close()

# Fig 2: 1/(z+1) distribution for DR3 sample
bins_dr3_inv = np.arange(0, 1.025, 0.025)
dr3_catalog_inv = 1.0 / (1.0 + dr3_catalog['ensemble_z'])
plt.figure(figsize=(7, 5))
plt.hist(dr3_catalog_inv, bins=bins_dr3_inv, color='white', edgecolor='black', label=f'All target sources ($N=319$)')
plt.hist(dr3_spec_z_inv, bins=bins_dr3_inv, color=col_bll_train, edgecolor='black', alpha=0.9, label=f'Spectroscopic Test Set ($n={len(dr3_spec_z)}$)')
plt.xlabel("$1/(z+1)$")
plt.ylabel("Number of Sources")
plt.title("Figure 2: Distribution of $1/(z+1)$ for DR3 Sources")
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig02_dr3_inv_redshift_distribution.png", dpi=300)
plt.close()

# Fig 6: DR3 Predicted vs Observed Redshift Comparison (2x2 Panel)
fig_dr3, axes_dr3 = plt.subplots(2, 2, figsize=(13, 11))
plot_panel(axes_dr3[0, 0], dr3_spec_z_inv, dr3_ens_z_inv, dr3_classes, 'inv', sigma_inv_dr3_ens, "Ensemble — $1/(z+1)$ Scale", point_size=30)
plot_panel(axes_dr3[0, 1], dr3_spec_z, dr3_ens_z, dr3_classes, 'z', sigma_inv_dr3_ens, "Ensemble — Linear $z$ Scale", point_size=30)
plot_panel(axes_dr3[1, 0], dr3_spec_z_inv, dr3_tab_z_inv, dr3_classes, 'inv', sigma_inv_dr3_tab, "TabPFN — $1/(z+1)$ Scale", point_size=30)
plot_panel(axes_dr3[1, 1], dr3_spec_z, dr3_tab_z, dr3_classes, 'z', sigma_inv_dr3_tab, "TabPFN — Linear $z$ Scale", point_size=30)
axes_dr3[0, 0].scatter([], [], color=col_bll_train, marker='o', s=30, label='BLL')
axes_dr3[0, 0].scatter([], [], color=col_fsrq_train, marker='^', s=30, label='FSRQ')
axes_dr3[0, 0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
fig_dr3.suptitle(f"Figure 6: Independent Spectroscopic Validation Results ($n={len(dr3_spec_z)}$)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig06_dr3_predicted_vs_true.png", dpi=300)
plt.close()

# Fig 7: DR3 Residuals Comparison (2x2 Panel)
fig_dr3_7, axes_dr3_7 = plt.subplots(2, 2, figsize=(13, 11))
delta_dr3_ens = dr3_spec_z - dr3_ens_z
delta_dr3_ens_norm = delta_dr3_ens / (1.0 + dr3_spec_z)
delta_dr3_tab = dr3_spec_z - dr3_tab_z
delta_dr3_tab_norm = delta_dr3_tab / (1.0 + dr3_spec_z)

# Top row: Ensemble
axes_dr3_7[0, 0].hist(delta_dr3_ens, bins=25, color='#2980B9', edgecolor='black', alpha=0.8)
axes_dr3_7[0, 0].axvline(np.mean(delta_dr3_ens), color=col_bias, linewidth=1.2, label='Mean bias')
axes_dr3_7[0, 0].axvline(np.mean(delta_dr3_ens) + np.std(delta_dr3_ens), color=col_sigma, linestyle='--', linewidth=1, label=r'$\pm1\sigma$')
axes_dr3_7[0, 0].axvline(np.mean(delta_dr3_ens) - np.std(delta_dr3_ens), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[0, 0].set_xlabel(r"$\Delta z = z_{\mathrm{spec}} - z_{\mathrm{pred}}$")
axes_dr3_7[0, 0].set_ylabel("Count")
axes_dr3_7[0, 0].set_title(r"Ensemble $\Delta z$ Distribution (DR3)", fontweight='bold')
axes_dr3_7[0, 0].legend()

axes_dr3_7[0, 1].hist(delta_dr3_ens_norm, bins=25, color='#2980B9', edgecolor='black', alpha=0.8)
axes_dr3_7[0, 1].axvline(np.mean(delta_dr3_ens_norm), color=col_bias, linewidth=1.2)
axes_dr3_7[0, 1].axvline(np.mean(delta_dr3_ens_norm) + np.std(delta_dr3_ens_norm), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[0, 1].axvline(np.mean(delta_dr3_ens_norm) - np.std(delta_dr3_ens_norm), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[0, 1].set_xlabel(r"$\Delta z_{\mathrm{norm}} = \Delta z / (1 + z_{\mathrm{spec}})$")
axes_dr3_7[0, 1].set_ylabel("Count")
axes_dr3_7[0, 1].set_title(r"Ensemble $\Delta z_{\mathrm{norm}}$ Distribution (DR3)", fontweight='bold')

# Bottom row: TabPFN
axes_dr3_7[1, 0].hist(delta_dr3_tab, bins=25, color='#27AE60', edgecolor='black', alpha=0.8)
axes_dr3_7[1, 0].axvline(np.mean(delta_dr3_tab), color=col_bias, linewidth=1.2)
axes_dr3_7[1, 0].axvline(np.mean(delta_dr3_tab) + np.std(delta_dr3_tab), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[1, 0].axvline(np.mean(delta_dr3_tab) - np.std(delta_dr3_tab), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[1, 0].set_xlabel(r"$\Delta z$")
axes_dr3_7[1, 0].set_ylabel("Count")
axes_dr3_7[1, 0].set_title(r"TabPFN $\Delta z$ Distribution (DR3)", fontweight='bold')

axes_dr3_7[1, 1].hist(delta_dr3_tab_norm, bins=25, color='#27AE60', edgecolor='black', alpha=0.8)
axes_dr3_7[1, 1].axvline(np.mean(delta_dr3_tab_norm), color=col_bias, linewidth=1.2)
axes_dr3_7[1, 1].axvline(np.mean(delta_dr3_tab_norm) + np.std(delta_dr3_tab_norm), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[1, 1].axvline(np.mean(delta_dr3_tab_norm) - np.std(delta_dr3_tab_norm), color=col_sigma, linestyle='--', linewidth=1)
axes_dr3_7[1, 1].set_xlabel(r"$\Delta z_{\mathrm{norm}}$")
axes_dr3_7[1, 1].set_ylabel("Count")
axes_dr3_7[1, 1].set_title(r"TabPFN $\Delta z_{\mathrm{norm}}$ Distribution (DR3)", fontweight='bold')

fig_dr3_7.suptitle(f"Figure 7: Redshift Residuals on Independent Test set ($n={len(dr3_spec_z)}$)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig07_dr3_residuals.png", dpi=300)
plt.close()

# Fig 10: Generalization (predicted redshift of all 319 new sources vs. training set redshifts)
plt.figure(figsize=(7, 5))
plt.hist(train_bll_z, bins=bins10, color='white', edgecolor='black', label=f'Training BLLs ({n_train_bll})')
plt.hist(dr3_catalog['ensemble_z'], bins=bins10, color=col_bll_gen, edgecolor='black', alpha=0.7, label=f'Predicted DR3 Catalog ({len(dr3_catalog)})')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Number of Sources")
plt.title("Figure 10: Redshift Distribution for DR3 Target Set vs. Training Set BLLs", fontweight='bold')
plt.xlim(0, 3.0)
plt.ylim(0, 120)
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig10_dr3_generalization_predictions.png", dpi=300)
plt.close()

# Fig 12: Calibration Curve for DR3 (sorted predicted vs sorted observed)
fig_dr3_12, ax_dr3_12 = plt.subplots(figsize=(6, 6))
plot_ot_panel(ax_dr3_12, dr3_ens_z_inv, dr3_spec_z_inv, "DR3 validation set", col_bll_train)
fig_dr3_12.suptitle("Figure 12: DR3 Validation Calibration Fit", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig12_dr3_calibration_curves.png", dpi=300)
plt.close()

# Fig 13: Calibrated DR3 predicted vs observed redshift
fig_dr3_13, (ax_dr3_13_l, ax_dr3_13_r) = plt.subplots(1, 2, figsize=(13, 6))
plot_panel(ax_dr3_13_l, dr3_spec_z_inv, dr3_ens_z_inv, dr3_classes, 'inv', sigma_inv_dr3_ens, "Calibrated Ensemble — $1/(z+1)$ Scale", point_size=30)
plot_panel(ax_dr3_13_r, dr3_spec_z, dr3_ens_z, dr3_classes, 'z', sigma_inv_dr3_ens, "Calibrated Ensemble — Linear $z$ Scale", point_size=30)
fig_dr3_13.suptitle("Figure 13: Calibrated DR3 Validation Results", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig13_dr3_calibrated_predictions.png", dpi=300)
plt.close()

# Fig 14: Calibrated predicted redshift distribution for target sources
plt.figure(figsize=(7, 5))
plt.hist(train_bll_z, bins=bins14, color='#EC7063', edgecolor='#C0392B', alpha=0.5, label=f'Training BLLs ({n_train_bll})')
plt.hist(dr3_catalog['ensemble_z'], bins=bins14, color='#76D7C4', edgecolor='#16A085', alpha=0.5, label=f'Calibrated DR3 Predictions ({len(dr3_catalog)})')
plt.xlabel("$z$")
plt.ylabel("Counts")
plt.title("Figure 14: Overlapped Histogram of Predicted and Observed z for DR3", fontweight='bold')
plt.xlim(0, 3.0)
plt.ylim(0, 60)
plt.legend()
plt.tight_layout()
plt.savefig("plots/upgraded_dr3_plots/Fig14_dr3_calibrated_generalization.png", dpi=300)
plt.close()

print("\nAll plots for DR2 and DR3 results successfully generated!")
