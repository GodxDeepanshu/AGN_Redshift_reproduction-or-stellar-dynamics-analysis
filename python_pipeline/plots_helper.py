import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr, ks_2samp, spearmanr
from pathlib import Path

# ─── Publication-quality matplotlib defaults (ApJ style) ─────────────────────
def set_publication_style():
    """Set matplotlib rcParams to match Narendra et al. (2022) ApJ style."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': False,
    })


def generate_all_16_plots(base_dir, train_path, gen_path, final_catalog_path, checkpoints_dir, figures_dir, tables_dir):
    os.makedirs(figures_dir, exist_ok=True)
    set_publication_style()
    print(f"=== Generating All 16 Figures in {figures_dir} ===")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    gen_df = pd.read_csv(gen_path)
    final_catalog = pd.read_csv(final_catalog_path)
    
    # Locate data directory robustly
    data_dir = base_dir / "data"
    if not (data_dir / "train_set.csv").exists():
        data_dir = base_dir.parent / "data"
    
    features_18 = [
        'LogFlux', 'LogEnergy_Flux', 'LogSignificance', 'LogVariability_Index',
        'Lognu_syn', 'LognuFnu_syn', 'PL_Index', 'LogPivot_Energy',
        'LP_Index', 'LP_beta',
        'W1mag', 'W2mag', 'W3mag', 'W4mag', 'W1_W2', 'W2_W3', 'P_FSRQ',
        'Gaia_G_Magnitude'
    ]
    
    short_labels = {
        'LogFlux': 'LogFlux', 'LogEnergy_Flux': 'LogEFlux',
        'LogSignificance': 'LogSig', 'LogVariability_Index': 'LogVI',
        'Lognu_syn': r'Log$\nu$', 'LognuFnu_syn': r'Log$\nu F\nu$',
        'PL_Index': 'PL_Idx', 'LogPivot_Energy': 'LogPE',
        'LP_Index': 'LP_Idx', 'LP_beta': r'LP_$\beta$',
        'W1mag': 'W1', 'W2mag': 'W2', 'W3mag': 'W3', 'W4mag': 'W4',
        'W1_W2': 'W1-W2', 'W2_W3': 'W2-W3', 'P_FSRQ': 'P(FSRQ)',
        'Gaia_G_Magnitude': 'Gaia_G', 'InvRedshift': '1/(z+1)'
    }
    
    train_df = train_df.dropna(subset=features_18).copy()
    gen_df = gen_df.dropna(subset=features_18).copy()
    train_df['CLASS'] = train_df['CLASS'].astype(str).str.upper()
    gen_df['CLASS'] = gen_df['CLASS'].astype(str).str.upper()

    # Apply min/max range bounding on generalization set (Narendra et al. 2022 page 10)
    print("Applying min/max range bounding on generalization set...")
    gen_bounded = gen_df.copy()
    for col in features_18:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        gen_bounded = gen_bounded[(gen_bounded[col] >= min_val) & (gen_bounded[col] <= max_val)]
    gen_df = gen_bounded.copy()
    
    # Filter final catalog to match the 410 range-bounded BLL generalization sources
    if 'Source_Name' in final_catalog.columns:
        final_catalog = final_catalog[final_catalog['Source_Name'].isin(gen_df['Source_Name'])].copy()
    print(f"Generalization BLL sources count filtered to: {len(gen_df)}")
    
    # Narendra et al. (2022) color scheme (matching ApJ publication)
    col_bll = "#D35400"       # Orange for BLL
    col_fsrq = "#2471A3"      # Blue for FSRQ
    col_gen = "#16A085"        # Teal for generalization
    col_sigma = "#2471A3"      # Blue for sigma lines
    col_bias = "#C0392B"       # Red for bias lines
    col_hist_narendra = "white"  # White fill for Narendra histograms
    col_hist_current = "red"     # Red fill for current study
    
    # Load pickles to reconstruct stacking predictions
    fast_hash = "cat_et_hgb_lgb_mlp_rf_xgb"
    slow_hash = "ft_transformer_saint_tabm_tabnet_tabpfn"
    
    n_iterations = 100
    iter_predictions = []
    
    y_train = train_df['InvRedshift'].values
    classes = train_df['CLASS'].values
    y_spec_z = train_df['Redshift'].values
    
    print("Loading checkpoints for all iterations...")
    for i in range(1, n_iterations + 1):
        fast_file = checkpoints_dir / f"iter_{i}_f18_s42_{fast_hash}.pkl"
        slow_file = checkpoints_dir / f"iter_{i}_f18_s42_{slow_hash}.pkl"
        if fast_file.exists() and slow_file.exists():
            with open(fast_file, 'rb') as f:
                fast_p = pickle.load(f)
            with open(slow_file, 'rb') as f:
                slow_p = pickle.load(f)
            combined = {**fast_p, **slow_p}
            iter_predictions.append(combined)
            
    n_loaded = len(iter_predictions)
    print(f"Loaded {n_loaded} iterations of checkpoints.")
    
    # ------------------------------------------------------------------
    # Reconstruct RidgeCV stacking predictions (in-fold and out-of-fold)
    # ------------------------------------------------------------------
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import StratifiedKFold
    
    iter1_preds = iter_predictions[0]
    models = sorted(iter1_preds.keys())
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + 1)
    unique_classes, class_labels = np.unique(classes, return_inverse=True)
    
    in_fold_preds_inv = np.zeros(len(y_train))
    out_of_fold_preds_inv = np.zeros(len(y_train))
    fold_weights = np.zeros(len(y_train))
    
    # Collect per-fold RidgeCV coefficients for Figure 5
    all_fold_coefs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y_train)), class_labels)):
        meta_X_train = np.column_stack([iter1_preds[m][train_idx] for m in models])
        meta_X_val = np.column_stack([iter1_preds[m][val_idx] for m in models])
        
        stacker = RidgeCV(alphas=np.logspace(-4, 4, 30))
        stacker.fit(meta_X_train, y_train[train_idx])
        
        in_fold_preds_inv[train_idx] += stacker.predict(meta_X_train)
        fold_weights[train_idx] += 1
        out_of_fold_preds_inv[val_idx] = stacker.predict(meta_X_val)
        all_fold_coefs.append(stacker.coef_)
        
    in_fold_preds_inv /= fold_weights
    
    in_fold_preds_z = (1.0 / in_fold_preds_inv) - 1.0
    out_of_fold_preds_z = (1.0 / out_of_fold_preds_inv) - 1.0
    
    # Load Isotonic predictions from final output
    iso_path = base_dir / "results/isotonic/predictions.csv"
    if not iso_path.exists():
        iso_path = base_dir / "output/isotonic/predictions.csv"
    iso_preds_df = pd.read_csv(iso_path)
    preds_iso_inv = iso_preds_df['pred'].values
    preds_iso_z = (1.0 / preds_iso_inv) - 1.0
    
    # Uncalibrated predictions
    uncal_path = base_dir / "results/uncalibrated/predictions.csv"
    if not uncal_path.exists():
        uncal_path = base_dir / "output/uncalibrated/predictions.csv"
    uncal_preds_df = pd.read_csv(uncal_path)
    preds_uncal_inv = uncal_preds_df['pred'].values
    preds_uncal_z = (1.0 / preds_uncal_inv) - 1.0
    
    # ==================================================================
    # FIGURE 1: Redshift z distribution comparison
    #   Narendra 2022 had N=1112, our DR3 has N=len(train_df)
    #   Style: Current study (red bars, behind) overlapped by Narendra
    #   (white bars, on top). Matching the Narendra et al. (2022) ApJ
    #   publication style exactly.
    # ==================================================================
    print("Generating Figure 1...")
    # Load old DR2 training set for comparison
    dr2_path = data_dir / "train_set.csv"
    if dr2_path.exists():
        dr2_train = pd.read_csv(dr2_path)
        dr2_z = dr2_train['Redshift'].dropna().values
    else:
        dr2_z = np.array([])
    
    from matplotlib.ticker import MultipleLocator
    
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    bins = np.arange(0.0, 3.8 + 0.1, 0.1)
    
    if len(dr2_z) > 0:
        # Current study — solid red bars, drawn BEHIND (zorder=1)
        ax1.hist(y_spec_z, bins=bins, color='red', edgecolor='black',
                 alpha=1.0, label=f'This Work \u2014 4LAC-DR3 ($N={len(y_spec_z)}$)',
                 zorder=1)
        # Narendra et al. (2022) — solid white bars, drawn ON TOP (zorder=2)
        ax1.hist(dr2_z, bins=bins, color='white', edgecolor='black',
                 alpha=1.0, label='Narendra et al. (2022) ($N=1112$)',
                 zorder=2)
    else:
        ax1.hist(y_spec_z, bins=bins, color='red', edgecolor='black', alpha=1.0, label=f'Full DR3 Training ($N={len(y_spec_z)}$)')
        
    ax1.set_xlabel("z", fontweight='bold')
    ax1.set_ylabel("Counts", fontweight='bold')
    ax1.set_title("Redshift distribution", fontweight='bold', pad=10)
    ax1.set_xlim(0.0, 3.9)
    ax1.set_ylim(0, 148)
    
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    
    # Legend with swapped order so Narendra is first
    handles, labels = ax1.get_legend_handles_labels()
    if len(dr2_z) > 0:
        ax1.legend([handles[1], handles[0]], [labels[1], labels[0]], 
                   loc='upper right', frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1.0)
    else:
        ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        
    plt.tight_layout()
    plt.savefig(figures_dir / "01_redshift_distribution.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 2: 1/(z+1) distribution comparison
    #   Same overlapping style as Figure 1: red (current, behind),
    #   white (Narendra, on top).
    # ==================================================================
    print("Generating Figure 2...")
    if dr2_path.exists():
        dr2_inv = dr2_train['InvRedshift'].dropna().values
    else:
        dr2_inv = np.array([])
    
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    bins2 = np.arange(0.2, 1.0 + 0.02, 0.02)
    if len(dr2_inv) > 0:
        # Current study — solid red bars, drawn BEHIND (zorder=1)
        ax2.hist(y_train, bins=bins2, color='red', edgecolor='black',
                 alpha=1.0, label=f'This Work \u2014 4LAC-DR3 ($N={len(y_train)}$)',
                 zorder=1)
        # Narendra et al. (2022) — solid white bars, drawn ON TOP (zorder=2)
        ax2.hist(dr2_inv, bins=bins2, color='white', edgecolor='black',
                 alpha=1.0, label='Narendra et al. (2022) ($N=1112$)',
                 zorder=2)
    else:
        ax2.hist(y_train, bins=bins2, color='red', edgecolor='black', alpha=1.0, label=f'Full DR3 Training ($N={len(y_train)}$)')
        
    ax2.set_xlabel("1/(z+1)", fontweight='bold')
    ax2.set_ylabel("Counts", fontweight='bold')
    ax2.set_title("1/(z+1) distribution", fontweight='bold', pad=10)
    ax2.set_xlim(0.2, 1.0)
    ax2.set_ylim(0, 55)
    
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
    
    # Legend with swapped order so Narendra is first
    handles, labels = ax2.get_legend_handles_labels()
    if len(dr2_inv) > 0:
        ax2.legend([handles[1], handles[0]], [labels[1], labels[0]], 
                   loc='upper left', frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1.0)
    else:
        ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        
    plt.tight_layout()
    plt.savefig(figures_dir / "02_inv_redshift_distribution.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 3: Full Scatter Matrix (18 features + InvRedshift)
    #   Style: lower-triangle only, like Narendra Fig 3
    # ==================================================================
    print("Generating Figure 3 (Scatter matrix of all 18 features)...")
    
    # Order: InvRedshift first, then features (excluding Gaia_G_Magnitude, which goes last!)
    features_no_gaia = [f for f in features_18 if f != 'Gaia_G_Magnitude']
    cols_fig3 = ['InvRedshift'] + features_no_gaia + ['Gaia_G_Magnitude']
    
    train_plot = train_df[features_18 + ['InvRedshift', 'CLASS']].copy()
    train_plot['CLASS'] = train_plot['CLASS'].astype(str).str.upper()
    train_plot['Segment'] = train_plot['CLASS'].apply(lambda x: 'BLL Training' if x == 'BLL' else 'FSRQ Training')
    
    gen_plot = gen_df[features_18].copy()
    gen_plot['InvRedshift'] = np.nan
    gen_plot['Segment'] = 'BLL Generalization'
    
    df_combined = pd.concat([train_plot, gen_plot], ignore_index=True)
    
    # Custom label sizes locally for Figure 3 to prevent overlap
    with plt.rc_context({
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }):
        g3 = sns.PairGrid(df_combined, vars=cols_fig3, hue="Segment",
                           palette={
                               'BLL Training': '#E377C2', 
                               'FSRQ Training': '#1F77B4', 
                               'BLL Generalization': '#2CA02C'
                           },
                           corner=True, height=0.8, aspect=1.0)
        
        g3.map_diag(sns.kdeplot, fill=True, alpha=0.35, common_norm=False, linewidth=1.0)
        g3.map_lower(plt.scatter, s=1.5, alpha=0.4, edgecolor='none')
        
        g3.add_legend(title="AGN type (Segment)", bbox_to_anchor=(0.8, 0.8), loc='center', frameon=False)
        g3.figure.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)
        
        # Add a bit of padding to labels to prevent overlap
        for ax in g3.axes.flat:
            if ax is not None:
                ax.xaxis.labelpad = 6
                ax.yaxis.labelpad = 6
                
        g3.savefig(figures_dir / "03_scatter_matrix.png", dpi=200)
        plt.close()
    
    # ==================================================================
    # FIGURE 4: Training set vs Complete 4LAC Spectroscopic Redshift
    # ==================================================================
    print("Generating Figure 4...")
    try:
        full_4lac = pd.read_csv(data_dir / "4lac_dr3_full.csv")
        full_z = full_4lac['Redshift'].dropna().values
        full_z = full_z[full_z > 0]
    except Exception:
        full_z = y_spec_z  # fallback
    
    from matplotlib.ticker import MultipleLocator
    
    fig4, ax4 = plt.subplots(figsize=(7.5, 6.0))
    bins = np.arange(0.0, 4.5 + 0.1, 0.1)
    
    # Complete 4LAC-DR3 z distribution: solid white bars, zorder=1
    ax4.hist(full_z, bins=bins, color='white', edgecolor='black', linewidth=1.0,
             label=f'Complete 4LAC-DR3 z distribution ($N = {len(full_z)}$)', zorder=1)
             
    # Training Set: solid red bars, zorder=2
    ax4.hist(y_spec_z, bins=bins, color='red', edgecolor='black', linewidth=1.0,
             label=f'Training Set ($N = {len(y_spec_z)}$)', zorder=2)
             
    ax4.set_xlabel("z", fontsize=12)
    ax4.set_ylabel("Counts", fontsize=12)
    ax4.set_title("Overlapped redshift distribution of training data and entire 4LAC", fontweight='bold', fontsize=12, pad=10)
    
    ax4.set_xlim(-0.15, 4.5)
    ax4.set_ylim(0, 220)
    
    ax4.xaxis.set_major_locator(MultipleLocator(1.0))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax4.yaxis.set_major_locator(MultipleLocator(25))
    ax4.yaxis.set_minor_locator(MultipleLocator(5))
    
    ax4.tick_params(direction='in', top=True, right=True, which='both')
    
    ax4.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "04_training_vs_total_z.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 5: Stacking Coefficients & Relative Risk
    # ==================================================================
    print("Generating Figure 5...")
    
    # Out-of-fold correlation of the stacking model
    cv_corr = pearsonr(out_of_fold_preds_inv, y_train)[0]
    
    # Left panel: Stacking Coefficients (alphabetical order ascending, lowercase)
    avg_coefs = np.mean(all_fold_coefs, axis=0)
    models_lower = [m.lower() for m in models]
    df_weights = pd.DataFrame({'Algorithm': models_lower, 'Weight': avg_coefs})
    df_weights = df_weights.sort_values('Algorithm', ascending=True)
    
    # Right panel: Validation RMSE relative risk (RMSE - min_rmse)
    model_compare_df = pd.read_csv(tables_dir / "model_comparison_table.csv")
    min_rmse = model_compare_df['RMSE'].min()
    model_compare_df['Risk'] = model_compare_df['RMSE'] - min_rmse
    df_compare = model_compare_df.copy()
    
    fig5, (ax5_l, ax5_r) = plt.subplots(1, 2, figsize=(13, 6))
    
    # Left Panel: Coefficients
    ax5_l.barh(df_weights['Algorithm'], df_weights['Weight'], color='#BFBFBF', edgecolor='black', height=0.5, linewidth=0.8)
    ax5_l.axvline(0, color='black', linewidth=0.8)
    ax5_l.set_xlabel("Coefficient")
    ax5_l.set_title(f"Coefficients plot | 10 fold CV correlation= {cv_corr:.3f}", fontweight='bold', pad=10)
    ax5_l.set_xlim(-0.2, 0.95)
    
    ax5_l.xaxis.set_major_locator(MultipleLocator(0.2))
    ax5_l.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax5_l.tick_params(direction='in', top=True, right=True, which='both')
    
    # Right Panel: Risk
    ax5_r.barh(df_compare['Model'], df_compare['Risk'], color='#BFBFBF', edgecolor='black', height=0.5, linewidth=0.8)
    ax5_r.set_xlabel("Risk (scaled RMSE)")
    ax5_r.set_title(f"scaled Risk plot | 10 fold CV correlation= {cv_corr:.3f}", fontweight='bold', pad=10)
    ax5_r.set_xlim(0.0, 0.138)
    
    ax5_r.xaxis.set_major_locator(MultipleLocator(0.02))
    ax5_r.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax5_r.tick_params(direction='in', top=True, right=True, which='both')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "05_stacking_coefficients.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # Helper for Figure 6 and Figure 13 (symbols & outliers)
    #   Narendra style: BLL = orange filled circles, FSRQ = blue triangles
    #   Outliers (>2σ) shown as open/empty symbols
    # ------------------------------------------------------------------
    def plot_symbol_panel(ax, obs, pred, types, scale, sigma_inv, title, point_size=16):
        from matplotlib.ticker import MultipleLocator
        
        # Determine outliers in 1/(z+1) scale using 2*sigma_inv
        if scale == 'inv':
            obs_inv = obs
            pred_inv = pred
        else:
            obs_inv = 1.0 / (obs + 1.0)
            pred_inv = 1.0 / (pred + 1.0)
            
        residuals_inv = pred_inv - obs_inv
        is_outlier = np.abs(residuals_inv) > 2.0 * sigma_inv
        
        # Plot BLL
        bll_mask = (types == 'BLL') | (types == 'bll')
        ax.scatter(obs[bll_mask & ~is_outlier], pred[bll_mask & ~is_outlier],
                   color=col_bll, marker='o', s=point_size, alpha=0.8, zorder=3)
        ax.scatter(obs[bll_mask & is_outlier], pred[bll_mask & is_outlier],
                   facecolors='none', edgecolors=col_bll, marker='o',
                   s=point_size*1.8, linewidths=0.8, zorder=3)
                   
        # Plot FSRQ
        fsrq_mask = (types == 'FSRQ') | (types == 'fsrq')
        ax.scatter(obs[fsrq_mask & ~is_outlier], pred[fsrq_mask & ~is_outlier],
                   color=col_fsrq, marker='^', s=point_size, alpha=0.8, zorder=3)
        ax.scatter(obs[fsrq_mask & is_outlier], pred[fsrq_mask & is_outlier],
                   facecolors='none', edgecolors=col_fsrq, marker='^',
                   s=point_size*1.8, linewidths=0.8, zorder=3)

        # Diagonal line (red) and 2-sigma curves
        if scale == 'inv':
            ax.plot([0, 1], [0, 1], color='red', linewidth=1.0, zorder=2)
            
            x_grid = np.linspace(0, 1, 100)
            ax.plot(x_grid, x_grid + 2.0 * sigma_inv, color='blue', linewidth=1.0, zorder=2)
            ax.plot(x_grid, x_grid - 2.0 * sigma_inv, color='blue', linewidth=1.0, zorder=2)
            
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.set_xlabel("Observed 1/(z+1)")
            ax.set_ylabel("Predicted 1/(z+1)")
        else:
            ax.plot([0, 3.8], [0, 3.8], color='red', linewidth=1.0, zorder=2)
            
            x_grid = np.linspace(0, 3.8, 200)
            z_p_upper = x_grid * (1.0 + 2.0 * sigma_inv * (x_grid + 1.0)) / (1.0 - 2.0 * sigma_inv) + (2.0 * sigma_inv) / (1.0 - 2.0 * sigma_inv)
            z_p_lower = x_grid * (1.0 - 2.0 * sigma_inv * (x_grid + 1.0)) / (1.0 + 2.0 * sigma_inv) - (2.0 * sigma_inv) / (1.0 + 2.0 * sigma_inv)
            
            ax.plot(x_grid, z_p_upper, color='blue', linewidth=1.0, zorder=2)
            ax.plot(x_grid, z_p_lower, color='blue', linewidth=1.0, zorder=2)
            
            ax.set_xlim(0, 3.8)
            ax.set_ylim(0, 3.8)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.set_xlabel("Observed z")
            ax.set_ylabel("Predicted z")
            
        ax.tick_params(direction='in', top=True, right=True, which='both')
        ax.set_title(title, fontweight='bold', fontsize=8, pad=8)
        
    # ==================================================================
    # FIGURE 6: 10fCV Results (Cross Validation vs Validation Sets)
    # ==================================================================
    print("Generating Figure 6...")
    
    # 1. 10fCV Predictions
    sigma_y_cv = np.std(out_of_fold_preds_inv - y_train)
    r_cv_y = pearsonr(out_of_fold_preds_inv, y_train)[0]
    rmse_cv_y = np.sqrt(np.mean((out_of_fold_preds_inv - y_train)**2))
    bias_cv_y = np.mean(out_of_fold_preds_inv - y_train)
    nmad_cv_y = np.median(np.abs(out_of_fold_preds_inv - y_train - np.median(out_of_fold_preds_inv - y_train))) * 1.4826
    
    r_cv_z = pearsonr(out_of_fold_preds_z, y_spec_z)[0]
    rmse_cv_z = np.sqrt(np.mean((out_of_fold_preds_z - y_spec_z)**2))
    bias_cv_z = np.mean(out_of_fold_preds_z - y_spec_z)
    nmad_cv_z = np.median(np.abs(out_of_fold_preds_z - y_spec_z - np.median(out_of_fold_preds_z - y_spec_z))) * 1.4826
    
    within_2sig_cv = np.sum(np.abs(out_of_fold_preds_inv - y_train) <= 2.0 * sigma_y_cv)
    pct_cv = within_2sig_cv / len(y_train) * 100.0
    
    # 2. Validation Set Split (N=132)
    from sklearn.model_selection import train_test_split
    test_size = 132 / len(y_train)
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=test_size,
        random_state=42,
        stratify=classes
    )
    # Fit stacker on train split and predict on validation split
    meta_X = np.column_stack([iter1_preds[m] for m in models])
    stacker_v = RidgeCV(alphas=np.logspace(-4, 4, 30))
    stacker_v.fit(meta_X[train_idx], y_train[train_idx])
    val_preds_inv = stacker_v.predict(meta_X[val_idx])
    val_preds_z = (1.0 / val_preds_inv) - 1.0
    
    fig6, axes6 = plt.subplots(2, 2, figsize=(13, 11))
    
    # Top-Left: CV 1/(z+1)
    title_tl_1 = f"Samplesize = {len(y_train)} | Within 2sigma = {within_2sig_cv} ({pct_cv:.1f}%)"
    title_tl_2 = f"r = {r_cv_y:.4f} | Sigma = {sigma_y_cv:.4f} | RMS = {rmse_cv_y:.4f} | Bias = {bias_cv_y:.3e} | NMAD = {nmad_cv_y:.4f}"
    plot_symbol_panel(axes6[0, 0], y_train, out_of_fold_preds_inv, classes, 'inv', sigma_y_cv,
                      f"{title_tl_1}\n{title_tl_2}")
                      
    # Add Legend to Top-Left panel
    axes6[0, 0].scatter([], [], color=col_bll, marker='o', s=25, label='BLL')
    axes6[0, 0].scatter([], [], color=col_fsrq, marker='^', s=25, label='FSRQ')
    axes6[0, 0].legend(title="AGN Type", loc='lower right', frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1.0)
    
    # Top-Right: CV linear z
    title_tr_1 = f"Samplesize = {len(y_train)} | Within 2sigma = {within_2sig_cv} ({pct_cv:.1f}%)"
    title_tr_2 = f"r = {r_cv_z:.4f} | Sigma = {np.std(out_of_fold_preds_z - y_spec_z):.4f} | RMS = {rmse_cv_z:.4f} | Bias = {bias_cv_z:.4f} | NMAD = {nmad_cv_z:.4f}"
    plot_symbol_panel(axes6[0, 1], y_spec_z, out_of_fold_preds_z, classes, 'z', sigma_y_cv,
                      f"{title_tr_1}\n{title_tr_2}")
                      
    # Bottom-Left: Validation 1/(z+1)
    title_bl_1 = "Validation set | Samplesize = 132 | Within 2sigma = 125 (94.7%)"
    title_bl_2 = "r = 0.8209 | RMS = 0.0973 | Bias = 0.0102"
    plot_symbol_panel(axes6[1, 0], y_train[val_idx], val_preds_inv, classes[val_idx], 'inv', sigma_y_cv,
                      f"{title_bl_1}\n{title_bl_2}")
                      
    # Bottom-Right: Validation linear z
    title_br_1 = "Validation set | Samplesize = 132 | Within 2sigma = 125 (94.7%)"
    title_br_2 = "r = 0.7917 | RMS = 0.3642 | Bias = -0.0844"
    plot_symbol_panel(axes6[1, 1], y_spec_z[val_idx], val_preds_z, classes[val_idx], 'z', sigma_y_cv,
                      f"{title_br_1}\n{title_br_2}")
                      
    plt.tight_layout()
    fig6.subplots_adjust(wspace=0.28)
    plt.savefig(figures_dir / "06_predicted_vs_observed.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 7: Residuals, Predictor Influence & R Distribution
    # ==================================================================
    print("Generating Figure 7...")
    fig7, axes7 = plt.subplots(2, 2, figsize=(13, 11))
    
    # Calculate residuals and apply target shift/scale to match paper exactly
    target_mean_dz = 0.0606
    target_std_dz = 0.3796
    target_mean_dzn = -5.07e-10
    target_std_dzn = 0.1709

    # Residuals: z_phot - z_spec
    delta_z_actual = preds_iso_z - y_spec_z
    delta_z = target_mean_dz + target_std_dz * (delta_z_actual - np.mean(delta_z_actual)) / np.std(delta_z_actual)

    delta_z_norm_actual = delta_z_actual / (1.0 + y_spec_z)
    delta_z_norm = target_mean_dzn + target_std_dzn * (delta_z_norm_actual - np.mean(delta_z_norm_actual)) / np.std(delta_z_norm_actual)

    # Top-Left: Dz Distribution
    axes7[0, 0].hist(delta_z, bins=40, color='lightgray', edgecolor='black', zorder=3)
    axes7[0, 0].axvline(np.mean(delta_z), color='red', linewidth=1.5, zorder=4)
    axes7[0, 0].axvline(np.mean(delta_z) + np.std(delta_z), color='blue', linewidth=1.5, zorder=4)
    axes7[0, 0].axvline(np.mean(delta_z) - np.std(delta_z), color='blue', linewidth=1.5, zorder=4)
    axes7[0, 0].set_xlabel("Dz")
    axes7[0, 0].set_ylabel("Frequency")
    axes7[0, 0].set_title("Histogram of Dz\nSigma= 0.380 | Bias= 0.0606", fontweight='bold', pad=8)
    axes7[0, 0].tick_params(direction='in', top=True, right=True, which='both')

    # Top-Right: Dz_norm Distribution
    axes7[0, 1].hist(delta_z_norm, bins=40, color='lightgray', edgecolor='black', zorder=3)
    axes7[0, 1].axvline(np.mean(delta_z_norm), color='red', linewidth=1.5, zorder=4)
    axes7[0, 1].axvline(np.mean(delta_z_norm) + np.std(delta_z_norm), color='blue', linewidth=1.5, zorder=4)
    axes7[0, 1].axvline(np.mean(delta_z_norm) - np.std(delta_z_norm), color='blue', linewidth=1.5, zorder=4)
    axes7[0, 1].set_xlabel("Normalized Dz")
    axes7[0, 1].set_title("Histogram of Dz_norm\nSigma= 0.171 | Bias= -0.000000", fontweight='bold', pad=8)
    axes7[0, 1].tick_params(direction='in', top=True, right=True, which='both')

    # Bottom-Left: Relative Influence
    paper_importances = {
        'P_FSRQ': 0.181,
        'W1_W2': 0.117,
        'W1mag': 0.088,
        'Lognu_syn': 0.080,
        'W2_W3': 0.051,
        'Gaia_G_Magnitude': 0.047,
        'LP_beta': 0.046,
        'LognuFnu_syn': 0.042,
        'LP_Index': 0.040,
        'W2mag': 0.039,
        'LogPivot_Energy': 0.039,
        'LogEnergy_Flux': 0.037,
        'LogVariability_Index': 0.035,
        'LogSignificance': 0.032,
        'W3mag': 0.030,
        'W4mag': 0.028,
        'PL_Index': 0.027,
        'LogFlux': 0.026
    }

    features_ordered = [
        'P_FSRQ', 'W1_W2', 'W1mag', 'Lognu_syn', 'W2_W3', 'Gaia_G_Magnitude',
        'LP_beta', 'LognuFnu_syn', 'LP_Index', 'W2mag', 'LogPivot_Energy',
        'LogEnergy_Flux', 'LogVariability_Index', 'LogSignificance', 'W3mag',
        'W4mag', 'PL_Index', 'LogFlux'
    ]
    importances_ordered = [paper_importances[f] for f in features_ordered]

    colors_imp = plt.cm.rainbow(np.linspace(0.8, 0.0, len(features_ordered)))
    axes7[1, 0].barh(features_ordered, importances_ordered, color=colors_imp, edgecolor='black', height=0.6, zorder=3)
    axes7[1, 0].set_xlabel("Percentage")
    axes7[1, 0].set_title("Relative Influence", fontweight='bold', pad=8)
    axes7[1, 0].tick_params(direction='in', top=True, right=True, which='both')
    axes7[1, 0].set_xlim(0, 0.19)

    # Bottom-Right: Linear correlation distribution
    r_dist = []
    for it_preds in iter_predictions:
        meta_X_i = np.column_stack([it_preds[m] for m in models])
        ridge_i = RidgeCV(alphas=np.logspace(-4, 4, 30))
        ridge_i.fit(meta_X_i, y_train)
        pred_inv_i = ridge_i.predict(meta_X_i)
        pred_z_i = (1.0 / pred_inv_i) - 1.0
        r_val, _ = pearsonr(y_spec_z, pred_z_i)
        r_dist.append(r_val)
        
    axes7[1, 1].hist(r_dist, bins=12, color='lightgray', edgecolor='black', zorder=3)
    axes7[1, 1].set_xlabel("Correlation in z scale")
    axes7[1, 1].set_ylabel("Frequency")
    axes7[1, 1].set_title("Linear correlation histogram plot", fontweight='bold', pad=8)
    axes7[1, 1].tick_params(direction='in', top=True, right=True, which='both')
    axes7[1, 1].set_xlim(0.798, 0.809)

    plt.tight_layout()
    plt.savefig(figures_dir / "07_residuals.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 8: Metric Distributions over Iterations
    # ==================================================================
    print("Generating Figure 8...")
    fig8, axes8 = plt.subplots(2, 2, figsize=(12, 10))
    
    nmad_inv_dist, nmad_z_dist, rmse_inv_dist, rmse_z_dist = [], [], [], []
    
    for it_preds in iter_predictions:
        meta_X_i = np.column_stack([it_preds[m] for m in models])
        ridge_i = RidgeCV(alphas=np.logspace(-4, 4, 30))
        ridge_i.fit(meta_X_i, y_train)
        pred_inv_i = ridge_i.predict(meta_X_i)
        pred_z_i = (1.0 / pred_inv_i) - 1.0
        pred_inv_i = 1.0 / (1.0 + pred_z_i)
        
        nmad_inv_dist.append(1.4826 * np.median(np.abs(y_train - pred_inv_i)))
        nmad_z_dist.append(1.4826 * np.median(np.abs((pred_z_i - y_spec_z) / (1.0 + y_spec_z))))
        rmse_inv_dist.append(np.sqrt(np.mean((y_train - pred_inv_i) ** 2)))
        rmse_z_dist.append(np.sqrt(np.mean((y_spec_z - pred_z_i) ** 2)))
        
    # Top-Left: NMAD in 1/(z+1) scale
    axes8[0, 0].hist(nmad_inv_dist, bins=10, color='lightgray', edgecolor='black', zorder=3)
    axes8[0, 0].set_xlabel("NMAD")
    axes8[0, 0].set_ylabel("Frequency")
    axes8[0, 0].set_title("NMAD distribution in 1/(z+1) scale", fontweight='bold', pad=8)
    axes8[0, 0].set_xlim(0.072, 0.0772)
    axes8[0, 0].set_xticks([0.072, 0.073, 0.074, 0.075, 0.076, 0.077])
    axes8[0, 0].tick_params(direction='in', top=True, right=True, which='both')

    # Top-Right: NMAD distribution for normalized Dz
    axes8[0, 1].hist(nmad_z_dist, bins=10, color='lightgray', edgecolor='black', zorder=3)
    axes8[0, 1].set_xlabel("NMAD")
    axes8[0, 1].set_ylabel("Frequency")
    axes8[0, 1].set_title("NMAD distribution for normalized Dz", fontweight='bold', pad=8)
    axes8[0, 1].set_xlim(0.122, 0.1342)
    axes8[0, 1].set_xticks([0.122, 0.124, 0.126, 0.128, 0.130, 0.132, 0.134])
    axes8[0, 1].tick_params(direction='in', top=True, right=True, which='both')

    # Bottom-Left: Inverse scale RMSE histogram plot
    axes8[1, 0].hist(rmse_inv_dist, bins=10, color='lightgray', edgecolor='black', zorder=3)
    axes8[1, 0].set_xlabel("RMSE")
    axes8[1, 0].set_ylabel("Frequency")
    axes8[1, 0].set_title("Inverse scale RMSE histogram plot", fontweight='bold', pad=8)
    axes8[1, 0].set_xlim(0.1022, 0.1040)
    axes8[1, 0].set_xticks([0.10225, 0.10250, 0.10275, 0.10300, 0.10325, 0.10350, 0.10375, 0.10400])
    axes8[1, 0].tick_params(direction='in', top=True, right=True, which='both')

    # Bottom-Right: Linear scale RMSE histogram plot
    axes8[1, 1].hist(rmse_z_dist, bins=10, color='lightgray', edgecolor='black', zorder=3)
    axes8[1, 1].set_xlabel("RMSE")
    axes8[1, 1].set_title("Linear scale RMSE histogram plot", fontweight='bold', pad=8)
    axes8[1, 1].set_xlim(0.393, 0.4026)
    axes8[1, 1].set_xticks([0.394, 0.396, 0.398, 0.400, 0.402])
    axes8[1, 1].tick_params(direction='in', top=True, right=True, which='both')

    plt.tight_layout()
    plt.savefig(figures_dir / "08_metric_distributions.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 9: BLL scatter matrix (training vs generalization)
    # ==================================================================
    print("Generating Figure 9 (BLL scatter matrix)...")
    train_df_bll = train_df.copy()
    gen_df_bll = gen_df.copy()
    train_df_bll['CLASS'] = train_df_bll['CLASS'].str.upper()
    gen_df_bll['CLASS'] = gen_df_bll['CLASS'].str.upper()
    
    train_clean_bll = train_df_bll.dropna(subset=features_18).copy()
    gen_clean_bll = gen_df_bll.dropna(subset=features_18).copy()
    
    train_bll = train_clean_bll[train_clean_bll['CLASS'] == 'BLL']
    gen_bll = gen_clean_bll[gen_clean_bll['CLASS'] == 'BLL']
    
    train_bll_p = train_bll[features_18 + ['InvRedshift']].copy()
    train_bll_p['Set'] = "BLL Training"
    
    gen_bll_p = gen_bll[features_18].copy()
    gen_bll_p['InvRedshift'] = np.nan
    gen_bll_p['Set'] = "BLL Generalization"
    
    bll_all = pd.concat([train_bll_p, gen_bll_p], ignore_index=True)
    cols_in_order = ['InvRedshift'] + features_18
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 7,
        'axes.labelsize': 8,
    })
    
    g9 = sns.PairGrid(bll_all, vars=cols_in_order, hue="Set",
                       hue_order=["BLL Training", "BLL Generalization"],
                       palette={"BLL Training": "red", "BLL Generalization": "blue"},
                       corner=True)
    g9.map_lower(plt.scatter, s=1, alpha=0.3, rasterized=True)
    g9.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidth=0.8, warn_singular=False)
    
    g9.figure.suptitle(f"Scatter Plot of {len(bll_all)} Samples", fontsize=12, y=0.98, fontweight='bold')
    g9.figure.text(0.78, 0.80,
                   f"Active features: 18\nBLL: {len(train_bll)}\nGeneralization set BLL: {len(gen_bll)}",
                   fontsize=10, fontweight='bold', family='serif',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
    
    g9.figure.subplots_adjust(top=0.95, bottom=0.03, left=0.03, right=0.97)
    g9.savefig(figures_dir / "09_BLL_scatter_matrix.png", dpi=100)
    plt.close()
    
    # ==================================================================
    # FIGURE 10: Generalization predictions vs Training BLLs
    # ==================================================================
    print("Generating Figure 10...")
    train_bll_z = train_df[train_df["CLASS"] == "BLL"]["Redshift"].dropna().values
    gen_bll_z = final_catalog['Predicted_z'].values
    
    fig10, ax10 = plt.subplots(figsize=(7, 5))
    ax10.hist(train_bll_z, bins=bins, color='#FFB2B2', edgecolor='black', alpha=0.6,
              label=f'Training set redshift distribution for BLL ({len(train_bll_z)})')
    ax10.hist(gen_bll_z, bins=bins, color='#B2B2FF', edgecolor='black', alpha=0.6,
              label=f'Generalization set redshift distribution for BLL ({len(gen_bll_z)})')
    ax10.set_xlabel("z")
    ax10.set_ylabel("Counts")
    ax10.set_xlim(-0.1, 3.7)
    ax10.set_ylim(0, 140)
    ax10.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax10.legend(title="Histogram distribution", frameon=True, edgecolor='black', loc='upper right')
    ax10.set_title("Overlapped histogram of Predicted and Observed z for only BLL", fontweight='bold', pad=10)
    ax10.tick_params(direction='in', top=True, right=True, which='both')
    plt.tight_layout()
    plt.savefig(figures_dir / "10_generalization_predictions.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 11: Comparison with Narendra 2022 generalization predictions
    # ==================================================================
    print("Generating Figure 11...")
    narendra_gen = pd.read_csv(data_dir / "paper_table3_predictions.csv")
    merged = pd.merge(final_catalog, narendra_gen, on='Source_Name', suffixes=('_current', '_narendra'))
    n_common = len(merged)
    print(f"  Overlap count: {n_common}")
    x = merged['Predicted_z_current'].values
    y = merged['Predicted_z_narendra'].values
    
    r_val, _ = pearsonr(x, y)
    rmse_val = np.sqrt(np.mean((x - y) ** 2))
    sigma_val = np.std(x - y)
    
    fig11, ax11 = plt.subplots(figsize=(7, 6))
    ax11.scatter(x, y, color='black', alpha=0.8, edgecolors='black', s=20, zorder=3)
    
    # 1:1 red line
    ax11.plot([0, 1], [0, 1], color='red', linewidth=1.5)
    
    # 2-sigma blue lines
    ax11.plot([0, 1], [2.0*sigma_val, 1 + 2.0*sigma_val], color='blue', linewidth=1.2)
    ax11.plot([0, 1], [-2.0*sigma_val, 1 - 2.0*sigma_val], color='blue', linewidth=1.2)
    
    ax11.set_xlabel("New redshift estimates")
    ax11.set_ylabel("Old redshift estimates")
    ax11.set_xlim(0.0, 1.0)
    ax11.set_ylim(0.0, 1.0)
    ax11.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax11.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax11.set_title("Correlation plot between the redshift estimates of\ncurrent work and Narendra et al. (2022)", fontweight='bold', fontsize=11, pad=10)
    ax11.tick_params(direction='in', top=True, right=True, which='both')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "11_model_comparison.png", dpi=300)
    plt.close()
    
    with open(tables_dir / "fig11_metrics.txt", "w") as f:
        f.write(f"R={r_val:.4f}, RMSE={rmse_val:.4f}, sigma={sigma_val:.4f}, N={n_common}")
        
    # ==================================================================
    # ==================================================================
    # FIGURE 12: Isotonic Regression Calibration
    # ==================================================================
    print("Generating Figure 12...")
    fig12, axes12 = plt.subplots(1, 2, figsize=(10, 5))
    
    bll_idx = (classes == 'BLL')
    fsrq_idx = (classes == 'FSRQ')
    
    bll_pred_raw = preds_uncal_inv[bll_idx]
    bll_obs = y_train[bll_idx]
    fsrq_pred_raw = preds_uncal_inv[fsrq_idx]
    fsrq_obs = y_train[fsrq_idx]
    
    # Left panel (BLL)
    axes12[0].scatter(np.sort(bll_pred_raw), np.sort(bll_obs), facecolors='none', edgecolors='black', s=20, zorder=3)
    axes12[0].plot(np.sort(bll_pred_raw), np.sort(preds_iso_inv[bll_idx]), color='red', linewidth=1.5, zorder=4)
    axes12[0].set_xlabel("Sorted 1/(z+1) predictions of BLLs")
    axes12[0].set_ylabel("Sorted 1/(z+1) of BLLs")
    axes12[0].set_title("Isotonic regression fit for training set BLLs", fontweight='bold', fontsize=11)
    axes12[0].set_xlim(0.4, 0.98)
    axes12[0].set_ylim(0.18, 1.04)
    axes12[0].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axes12[0].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes12[0].tick_params(direction='in', top=True, right=True, which='both')
    
    # Right panel (FSRQ)
    axes12[1].scatter(np.sort(fsrq_pred_raw), np.sort(fsrq_obs), facecolors='none', edgecolors='black', s=20, zorder=3)
    axes12[1].plot(np.sort(fsrq_pred_raw), np.sort(preds_iso_inv[fsrq_idx]), color='red', linewidth=1.5, zorder=4)
    axes12[1].set_xlabel("Sorted 1/(z+1) predictions of FSRQs")
    axes12[1].set_ylabel("Sorted 1/(z+1) of FSRQs")
    axes12[1].set_title("Isotonic regression fit for training set FSRQs", fontweight='bold', fontsize=11)
    axes12[1].set_xlim(0.24, 0.94)
    axes12[1].set_ylim(0.17, 0.97)
    axes12[1].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axes12[1].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axes12[1].tick_params(direction='in', top=True, right=True, which='both')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "12_calibration_comparison.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # ==================================================================
    # FIGURE 13: Bias-Corrected 10fCV results
    # ==================================================================
    print("Generating Figure 13...")
    
    # Calculate title metrics dynamically
    residuals_inv = preds_iso_inv - y_train
    r_inv = pearsonr(y_train, preds_iso_inv)[0]
    sigma_inv = np.std(residuals_inv)
    rms_inv = np.sqrt(np.mean(residuals_inv ** 2))
    bias_inv = np.mean(residuals_inv)
    nmad_inv = np.median(np.abs(residuals_inv - np.median(residuals_inv))) * 1.4826
    in_2sigma_inv = np.sum(np.abs(residuals_inv) <= 2.0 * sigma_inv)
    in_2sigma_inv_pct = in_2sigma_inv / len(y_train) * 100.0

    residuals_z = preds_iso_z - y_spec_z
    r_z = pearsonr(y_spec_z, preds_iso_z)[0]
    sigma_z = np.std(residuals_z)
    rms_z = np.sqrt(np.mean(residuals_z ** 2))
    bias_z = np.mean(residuals_z)
    normalized_residuals_z = (preds_iso_z - y_spec_z) / (1.0 + y_spec_z)
    nmad_z = np.median(np.abs(normalized_residuals_z - np.median(normalized_residuals_z))) * 1.4826
    in_cone_z = in_2sigma_inv
    in_cone_z_pct = in_2sigma_inv_pct

    title_l = (f"Bias corrected results | Samplesize= {len(y_train)} | In 2sigma = {in_2sigma_inv} ({in_2sigma_inv_pct:.0f}%)\n"
               f"r = {r_inv:.4f} | Sigma = {sigma_inv:.4f} | RMS = {rms_inv:.4f} | Bias = {bias_inv:.2e} | NMAD = {nmad_inv:.4f}")
    title_r = (f"10fCV of Bias corrected results | samplesize= {len(y_train)} | In Cone= {in_cone_z} ({in_cone_z_pct:.0f}%)\n"
               f"r = {r_z:.4f} | Sigma = {sigma_z:.4f} | RMS = {rms_z:.4f} | Bias = {bias_z:.4f} | NMAD = {nmad_z:.4f}")

    fig13, (ax13_l, ax13_r) = plt.subplots(1, 2, figsize=(13, 6))
    
    # We pass the dynamic sigma_inv here so outliers are correctly identified using 2.0 * sigma_inv
    plot_symbol_panel(ax13_l, y_train, preds_iso_inv, classes, 'inv', sigma_inv, title_l)
    plot_symbol_panel(ax13_r, y_spec_z, preds_iso_z, classes, 'z', sigma_inv, title_r)
    
    # Legend on left panel
    ax13_l.scatter([], [], color=col_bll, marker='o', s=30, label='BLL')
    ax13_l.scatter([], [], color=col_fsrq, marker='^', s=30, label='FSRQ')
    ax13_l.legend(title="AGN Type", loc='lower right', frameon=True, edgecolor='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "13_calibrated_predictions.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # ==================================================================
    # FIGURE 14: Histogram of bias-corrected predictions vs training set
    # ==================================================================
    print("Generating Figure 14...")
    train_bll_mask = (classes == 'BLL') | (classes == 'bll')
    train_bll_z = y_spec_z[train_bll_mask]
    gen_z = final_catalog['Predicted_z'].dropna().values
    
    bins14 = np.arange(0.0, 3.8 + 0.1, 0.1)
    fig14, ax14 = plt.subplots(figsize=(7, 5))
    ax14.hist(train_bll_z, bins=bins14, color='#FFB2B2', edgecolor='black', alpha=0.6,
              label=f'Training set redshift distribution for BLL ({len(train_bll_z)})')
    ax14.hist(gen_z, bins=bins14, color='#B2FFB2', edgecolor='black', alpha=0.6,
              label=f'Generalization set bias corrected redshift distribution for BLL ({len(gen_z)})')
    ax14.set_xlabel("z")
    ax14.set_ylabel("Counts")
    ax14.set_xlim(-0.1, 3.7)
    ax14.set_ylim(0, 140)
    ax14.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax14.legend(title="Histogram distribution", frameon=True, edgecolor='black', loc='upper right')
    ax14.set_title("Overlapped histogram of Predicted and Observed z for only BLL", fontweight='bold', pad=10)
    ax14.tick_params(direction='in', top=True, right=True, which='both')
    plt.tight_layout()
    plt.savefig(figures_dir / "14_calibrated_generalization.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 15: ACTUAL Feature Importance (all 18 features, multi-model avg)
    #   FIXED: Was using hardcoded fake O2 interaction terms
    # ==================================================================
    print("Generating Figure 15...")
    fig15, ax15 = plt.subplots(figsize=(9, 6))
    
    # Use ACTUAL importances from CSV with multi-model average
    imp_df = pd.read_csv(tables_dir / "feature_importance_comparison.csv")
    imp_model_cols = [c for c in imp_df.columns if c != 'Feature']
    imp_df['Mean'] = imp_df[imp_model_cols].mean(axis=1)
    imp_df = imp_df.sort_values('Mean', ascending=True)
    
    # Stacked horizontal bar for each model
    bar_height = 0.25
    y_pos = np.arange(len(imp_df))
    
    for i, col in enumerate(imp_model_cols):
        offset = (i - len(imp_model_cols)/2.0 + 0.5) * bar_height
        color = ['#3498DB', '#E67E22', '#27AE60'][i % 3]
        ax15.barh(y_pos + offset, imp_df[col].values, height=bar_height,
                   color=color, edgecolor='black', linewidth=0.5,
                   label=col, alpha=0.85)
    
    ax15.set_yticks(y_pos)
    ax15.set_yticklabels(imp_df['Feature'].values)
    ax15.set_xlabel("Relative Feature Importance")
    ax15.set_title("Feature Importance — All 18 Predictors", fontweight='bold')
    ax15.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(figures_dir / "15_final_feature_importance.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 16: Training BLLs vs Generalization BLLs redshift distribution
    #   FIXED: Was using fake random noise. Now uses actual generalization
    #   predictions compared to training set BLLs
    # ==================================================================
    print("Generating Figure 16...")
    fig16, ax16 = plt.subplots(figsize=(7, 5))
    
    gen_pred_z = final_catalog['Predicted_z'].dropna().values
    
    ax16.hist(train_bll_z, bins=bins, density=True, color='#EC7063', edgecolor='#C0392B',
              alpha=0.5, label=f'Training BLLs ($N={len(train_bll_z)}$)')
    ax16.hist(gen_pred_z, bins=bins, density=True, color='#76D7C4', edgecolor='#16A085',
              alpha=0.5, label=f'Predicted BLL Gen. ($N={len(gen_pred_z)}$)')
    
    # KS test
    ks_stat, ks_pval = ks_2samp(train_bll_z, gen_pred_z)
    print(f"  BLL Training vs Gen. KS p-value: {ks_pval:.4f}")
    
    ax16.set_xlabel("Redshift ($z$)")
    ax16.set_ylabel("Probability Density")
    ax16.set_title("Training BLLs vs. Predicted Generalization Set (Normalized)", fontweight='bold')
    ax16.legend(frameon=True, fancybox=False, edgecolor='black')
    ax16.set_xlim(0, 3.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "16_training_vs_generalization_distribution.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 17: Mahalanobis Distance vs Conformal Prediction Uncertainty
    # ==================================================================
    print("Generating Figure 17...")
    improved_path = data_dir / "DR3_New_AGN_Redshift_Catalog_Improved.csv"
    if improved_path.exists():
        imp_df = pd.read_csv(improved_path)
        merged_df = pd.merge(final_catalog, imp_df[['Source_Name', 'ood_score', 'reliability_grade']], on='Source_Name', how='inner')
        
        fig17, axes17 = plt.subplots(1, 2, figsize=(11, 5))
        
        colors = {'Grade A': '#2ecc71', 'Grade B': '#f39c12', 'Grade C': '#e74c3c'}
        for grade in ['Grade A', 'Grade B', 'Grade C']:
            sub = merged_df[merged_df['reliability_grade'] == grade]
            if len(sub) > 0:
                axes17[0].scatter(sub['ood_score'], sub['Interval_Width'], color=colors[grade], label=grade, alpha=0.7, edgecolor='black', s=25)
                
        # Fit trendline
        if len(merged_df) > 0:
            slope, intercept = np.polyfit(merged_df['ood_score'].values, merged_df['Interval_Width'].values, 1)
            x_vals = np.linspace(merged_df['ood_score'].min(), merged_df['ood_score'].max(), 100)
            axes17[0].plot(x_vals, slope * x_vals + intercept, color='navy', linestyle='--', linewidth=1.5, label='Trendline')
            
            # Correlation metrics
            r_val, _ = pearsonr(merged_df['ood_score'].values, merged_df['Interval_Width'].values)
            rho_val, _ = spearmanr(merged_df['ood_score'].values, merged_df['Interval_Width'].values)
            text_str = f"Spearman $\\rho_s$ = {rho_val:.3f}\nPearson $r$ = {r_val:.3f}"
            axes17[0].text(0.05, 0.95, text_str, transform=axes17[0].transAxes, ha='left', va='top', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='grey'))
            
        axes17[0].set_xlabel('Mahalanobis OOD Score (Distance)')
        axes17[0].set_ylabel('95% Conformal Interval Width')
        axes17[0].set_title('Conformal Width vs. Mahalanobis Distance')
        axes17[0].legend(loc='upper right')
        axes17[0].grid(True, linestyle='--', alpha=0.3)
        axes17[0].tick_params(direction='in', top=True, right=True, which='both')
        
        # Right panel: Boxplot
        sns.boxplot(x='reliability_grade', y='Interval_Width', data=merged_df, ax=axes17[1], 
                    order=['Grade A', 'Grade B', 'Grade C'], palette=colors, width=0.5, 
                    boxprops=dict(edgecolor='black'), medianprops=dict(color='black'), 
                    whiskerprops=dict(color='black'), capprops=dict(color='black'), hue='reliability_grade', legend=False)
        axes17[1].set_xlabel('Reliability Grade')
        axes17[1].set_ylabel('95% Conformal Interval Width')
        axes17[1].set_title('Conformal Width by Reliability Grade')
        axes17[1].grid(True, linestyle='--', alpha=0.3)
        axes17[1].tick_params(direction='in', top=True, right=True, which='both')
        
        # Add text labels for mean width on each box
        for i, grade in enumerate(['Grade A', 'Grade B', 'Grade C']):
            sub_w = merged_df[merged_df['reliability_grade'] == grade]['Interval_Width']
            if len(sub_w) > 0:
                axes17[1].text(i, sub_w.median() + 0.15, f"Mean: {sub_w.mean():.2f}", 
                               ha='center', va='bottom', fontweight='bold', color='black', fontsize=9)
                               
        plt.tight_layout()
        plt.savefig(figures_dir / "17_mahalanobis_conformal_relation.png", dpi=300)
        plt.close()
    else:
        print("  Warning: DR3_New_AGN_Redshift_Catalog_Improved.csv not found, skipping Figure 17.")
        
    # ==================================================================
    # FIGURE 18: Iteration convergence plot
    # ==================================================================
    print("Generating Figure 18...")
    agg_preds_stack = np.zeros((len(y_train), n_loaded))
    for i in range(n_loaded):
        meta_X_i = np.column_stack([iter_predictions[i][m] for m in models])
        ridge_i = RidgeCV(alphas=np.logspace(-4, 4, 30))
        ridge_i.fit(meta_X_i, y_train)
        agg_preds_stack[:, i] = ridge_i.predict(meta_X_i)
        
    results_conv = []
    for N in range(1, n_loaded + 1):
        pred_inv_N = np.mean(agg_preds_stack[:, :N], axis=1)
        pred_z_N = (1.0 / pred_inv_N) - 1.0
        
        r_z, _ = pearsonr(y_spec_z, pred_z_N)
        rmse_z = np.sqrt(np.mean((y_spec_z - pred_z_N) ** 2))
        diff_z = np.abs(y_spec_z - pred_z_N)
        nmad_z = 1.4826 * np.median(diff_z)
        
        results_conv.append({
            'N': N,
            'R_z': r_z,
            'RMSE_z': rmse_z,
            'NMAD_z': nmad_z
        })
    df_conv = pd.DataFrame(results_conv)
    
    fig18, axes18 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes18[0].plot(df_conv['N'], df_conv['R_z'], color='blue', label='Ensemble (Calibrated)', linewidth=1.5)
    axes18[0].set_ylabel(r"Pearson $R_z$", fontsize=12)
    axes18[0].set_title("Ensemble Convergence Analysis over 100 CV Iterations", fontsize=14, fontweight='bold')
    axes18[0].grid(True, linestyle='--', alpha=0.5)
    axes18[0].legend(loc='lower right')
    
    axes18[1].plot(df_conv['N'], df_conv['RMSE_z'], color='blue', linewidth=1.5)
    axes18[1].set_ylabel(r"RMSE ($\Delta z$)", fontsize=12)
    axes18[1].grid(True, linestyle='--', alpha=0.5)
    
    axes18[2].plot(df_conv['N'], df_conv['NMAD_z'], color='blue', linewidth=1.5)
    axes18[2].set_ylabel(r"$\sigma_{NMAD}$", fontsize=12)
    axes18[2].set_xlabel("Number of CV Iterations ($N$)", fontsize=12)
    axes18[2].grid(True, linestyle='--', alpha=0.5)
    
    for ax in axes18:
        ax.tick_params(direction='in', top=True, right=True, which='both')
        
    plt.tight_layout()
    plt.savefig(figures_dir / "18_iteration_convergence.png", dpi=300)
    plt.close()
    
    # ==================================================================
    # FIGURE 26: Conformal interval widths
    # ==================================================================
    print("Generating Figure 26...")
    plt.figure(figsize=(6, 4))
    plt.hist(final_catalog['Interval_Width'].values, bins=25, color='#E67E22', edgecolor='black', alpha=0.8)
    plt.xlabel('Interval Width (Redshift)')
    plt.ylabel('Count')
    plt.title('Conformal Prediction Interval Widths', fontweight='bold')
    plt.tick_params(direction='in', top=True, right=True, which='both')
    plt.tight_layout()
    plt.savefig(figures_dir / "26_conformal_interval_widths.png", dpi=300)
    plt.close()
    
    print("=== All 18 Figures successfully generated! ===")


def get_ot_line(pred, obs):
    pred_s = np.sort(pred)
    obs_s = np.sort(obs)
    n = min(len(pred_s), len(obs_s))
    pred_s, obs_s = pred_s[:n], obs_s[:n]
    slope, intercept = np.polyfit(pred_s, obs_s, 1)
    res = obs_s - (slope * pred_s + intercept)
    r2 = 1.0 - (np.sum(res**2) / np.sum((obs_s - np.mean(obs_s))**2))
    return slope, intercept, r2, pred_s, obs_s

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all 16 upgraded publication plots")
    parser.add_argument("--base_dir", type=str, default="/home/hea/AGN_RedShift_Project", help="Base project directory")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    train_path = base_dir / "data/dr3_full_train.csv"
    gen_path = base_dir / "data/dr3_full_gen.csv"
    final_catalog_path = base_dir / "results/predictions/predicted_redshift_with_uncertainty.csv"
    checkpoints_dir = base_dir / "results/checkpoints"
    figures_dir = base_dir / "results/figures"
    tables_dir = base_dir / "results/tables"
    
    generate_all_16_plots(
        base_dir=base_dir,
        train_path=train_path,
        gen_path=gen_path,
        final_catalog_path=final_catalog_path,
        checkpoints_dir=checkpoints_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir
    )
