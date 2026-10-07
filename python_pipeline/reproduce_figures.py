import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV

# Add current folder to path to import local helpers
sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics, compute_nmad, compute_pearson, compute_rmse
from py_bias_correction import fit_optimal_transport, apply_optimal_transport
from py_sequential_runner import get_model_instance, lasso_feature_selection_1se

def compute_2sigma_curves(z_seq, sigma_inv):
    inv_seq = 1.0 / (1.0 + z_seq)
    upper_inv = inv_seq + 2.0 * sigma_inv
    lower_inv = inv_seq - 2.0 * sigma_inv
    lower_inv = np.clip(lower_inv, 1e-6, None)
    upper_z = (1.0 / lower_inv) - 1.0
    lower_z = (1.0 / upper_inv) - 1.0
    return upper_z, lower_z

def main():
    print("=== Phase 9: Generating All Reproduced Figures in Python ===")
    output_dir = "plots/python_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------
    training_eligible = pd.read_csv("data/training_eligible.csv")
    train_set = pd.read_csv("data/train_set.csv")
    val_set = pd.read_csv("data/validation_set.csv")
    gen_set = pd.read_csv("data/generalization_set.csv")
    processed_data = pd.read_csv("data/processed_4LAC.csv")
    
    with open("data/predictor_columns.txt", "r") as f:
        predictors = [line.strip() for line in f if line.strip()]
        
    with open("output/python_results/python_results.pkl", "rb") as f:
        results = pickle.load(f)
        
    mean_predictions = results['mean_predictions']
    metrics_report = results['metrics_report']
    weights_matrix = results['weights_matrix']
    feature_counts = results['feature_counts']
    ot_params = results['ot_params']
    
    # Style constants
    col_bll_train = "#C0392B"   # red
    col_fsrq_train = "#27AE60"  # dark green
    col_bll_gen = "#2980B9"     # blue
    col_fsrq_gen = "#222222"    # black
    col_sigma = "#2980B9"       # blue
    col_bias = "#E74C3C"        # red
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13
    })
    
    # ------------------------------------------------------------------
    # FIGURE 1: Distribution of Redshift z
    # ------------------------------------------------------------------
    print("Generating Figure 1 (Redshift z distribution)...")
    all_z = processed_data['Redshift'].dropna().values
    train_z = training_eligible['Redshift'].dropna().values
    
    bins = np.arange(0, np.ceil(max(all_z)*10)/10 + 0.1, 0.1)
    
    plt.figure(figsize=(7, 5))
    plt.hist(all_z, bins=bins, color='white', edgecolor='black', label='4LAC-DR2 (all with z)')
    plt.hist(train_z, bins=bins, color=col_bll_train, edgecolor='black', alpha=0.9, label='Training Sample')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 1: Redshift Distribution of 4LAC-DR2 Sources")
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig01_redshift_distribution.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 2: Distribution of 1/(z+1)
    # ------------------------------------------------------------------
    print("Generating Figure 2 (1/(z+1) distribution)...")
    all_inv = 1.0 / (1.0 + all_z)
    train_inv = 1.0 / (1.0 + train_z)
    
    bins2 = np.arange(0, 1.025, 0.025)
    
    plt.figure(figsize=(7, 5))
    plt.hist(all_inv, bins=bins2, color='white', edgecolor='black', label='4LAC-DR2 (all with z)')
    plt.hist(train_inv, bins=bins2, color=col_bll_train, edgecolor='black', alpha=0.9, label='Training Sample')
    plt.xlabel("1/(z+1)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 2: Distribution of 1/(z+1)")
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig02_inv_redshift_distribution.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 3: Scatter matrix with 4 AGN categories
    # ------------------------------------------------------------------
    print("Generating Figure 3 (Scatter matrix with 4 categories)...")
    key_preds = ["LogFlux", "Lognu_syn", "LP_beta", "PL_Index", "Gaia_G_Magnitude"]
    key_preds = [p for p in key_preds if p in training_eligible.columns and p in gen_set.columns]
    
    if len(key_preds) >= 3:
        df_train_plot = training_eligible.copy()
        df_train_plot['Category'] = df_train_plot['AGN_Type'] + " Training"
        
        df_gen_plot = gen_set.copy()
        df_gen_plot['Category'] = df_gen_plot['AGN_Type'] + " Generalization"
        
        scatter_data = pd.concat([df_train_plot[key_preds + ['Category']], df_gen_plot[key_preds + ['Category']]], ignore_index=True)
        
        g = sns.PairGrid(scatter_data, hue="Category", hue_order=[
            "BLL Training", "FSRQ Training", "BLL Generalization", "FSRQ Generalization"
        ], palette={
            "BLL Training": col_bll_train,
            "FSRQ Training": col_fsrq_train,
            "BLL Generalization": col_bll_gen,
            "FSRQ Generalization": col_fsrq_gen
        })
        
        # Plot styles: Training = filled marker, Gen = open marker
        g.map_diag(sns.kdeplot, alpha=0.4, fill=True)
        
        # Lower diagonal: scatter plot
        # We can implement a custom plot function to handle the different markers
        def custom_scatter(x, y, **kwargs):
            label = kwargs.get('label', '')
            if 'Training' in label:
                marker = 'o' if 'BLL' in label else '^'
                facecolors = kwargs.get('color')
                edgecolors = 'none'
            else:
                marker = 'o' if 'BLL' in label else '^'
                facecolors = 'none'
                edgecolors = kwargs.get('color')
            plt.scatter(x, y, marker=marker, facecolors=facecolors, edgecolors=edgecolors, s=12, alpha=0.6)
            
        g.map_offdiag(custom_scatter)
        g.add_legend(title="", bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=4)
        g.figure.subplots_adjust(top=0.94, bottom=0.1)
        g.figure.suptitle("Figure 3: Scatter Matrix — 4 AGN Categories", fontweight='bold')
        g.savefig(f"{output_dir}/Fig03_scatter_matrix.png", dpi=180)
        plt.close()
        
    # ------------------------------------------------------------------
    # FIGURE 4: Training set vs complete 4LAC redshift
    # ------------------------------------------------------------------
    print("Generating Figure 4 (Training vs complete 4LAC redshift)...")
    plt.figure(figsize=(7, 5))
    plt.hist(all_z, bins=bins, color='white', edgecolor='black', label='4LAC-DR2 (complete)')
    plt.hist(train_z, bins=bins, color=col_bll_train, edgecolor='black', alpha=0.9, label='Training Sample')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Number of Sources")
    plt.title("Figure 4: Training Set vs Complete 4LAC-DR2 Redshift")
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig04_training_vs_total_z.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 5: Stacking weights bar chart
    # ------------------------------------------------------------------
    print("Generating Figure 5 (Stacking Weights)...")
    # Identify which classical models were trained
    models_active = ['xgb', 'lgb', 'cat', 'et', 'hgb', 'ngb']
            
    avg_weights = np.mean(weights_matrix, axis=0)
    
    df_weights = pd.DataFrame({
        'Algorithm': [m.upper() for m in models_active],
        'Weight': avg_weights
    }).sort_values(by='Weight', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Weight', y='Algorithm', data=df_weights, palette='viridis')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Stacking Weight Coefficient")
    plt.ylabel("Algorithm")
    plt.title("Figure 5: Average Stacking Ensemble Weights (Python)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig05_SL_coefficients.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # Train validation model to compute validation results for Fig 6
    # ------------------------------------------------------------------
    print("Evaluating validation set predictions...")
    X_train_v = train_set[predictors]
    y_train_v = train_set['InvRedshift']
    X_val_v = val_set[predictors]
    y_val_v = val_set['InvRedshift']
    
    # Lasso feature selection
    try:
        selected_features = lasso_feature_selection_1se(X_train_v, y_train_v, cv=5, random_state=42)
    except:
        selected_features = list(predictors)
    if len(selected_features) == 0:
        selected_features = list(predictors)
        
    # Fit individual base models on train_set
    val_preds = {}
    train_preds = {}
    for m in models_active:
        model = get_model_instance(m, seed=42)
        model.fit(X_train_v[selected_features], y_train_v)
        val_preds[m] = model.predict(X_val_v[selected_features])
        train_preds[m] = model.predict(X_train_v[selected_features])
        
    # Stack validation predictions
    val_stack_pred_inv = np.zeros(len(val_set))
    train_stack_pred_inv = np.zeros(len(train_set))
    for idx, m in enumerate(models_active):
        val_stack_pred_inv += avg_weights[idx] * val_preds[m]
        train_stack_pred_inv += avg_weights[idx] * train_preds[m]
        
    # OT bias correction for validation
    val_ot_params = fit_optimal_transport(train_stack_pred_inv, y_train_v.values, train_set['AGN_Type'].values, verbose=False)
    val_corrected_inv = apply_optimal_transport(val_stack_pred_inv, val_set['AGN_Type'].values, val_ot_params, verbose=False)
    
    # ------------------------------------------------------------------
    # FIGURE 6: 10fCV and Validation Results — 2x2 panel
    # ------------------------------------------------------------------
    print("Generating Figure 6 (Main CV & Validation Results)...")
    
    obs_inv_cv = training_eligible['InvRedshift'].values
    pred_inv_cv = mean_predictions['stack']
    if len(pred_inv_cv) != len(obs_inv_cv):
        pred_inv_cv = pred_inv_cv[:len(obs_inv_cv)]
    agn_types_cv = training_eligible['AGN_Type'].values
    
    obs_z_cv = training_eligible['Redshift'].values
    pred_z_cv = (1.0 / pred_inv_cv) - 1.0
    
    obs_inv_val = val_set['InvRedshift'].values
    pred_inv_val = val_stack_pred_inv
    agn_types_val = val_set['AGN_Type'].values
    
    obs_z_val = val_set['Redshift'].values
    pred_z_val = (1.0 / pred_inv_val) - 1.0
    
    # Standard deviation of residuals for 2-sigma curves
    sigma_inv_cv = np.std(pred_inv_cv - obs_inv_cv)
    sigma_inv_val = np.std(pred_inv_val - obs_inv_val)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Helper plotting panel function
    def plot_panel(ax, obs, pred, types, scale, sigma_inv, title):
        residuals_inv = (pred - obs) if scale == 'inv' else (1.0/(pred+1.0) - 1.0/(obs+1.0))
        is_outlier = np.abs(residuals_inv) > 2.0 * sigma_inv
        
        # Plot BLLs (red)
        bll_mask = (types == 'BLL')
        ax.scatter(obs[bll_mask & ~is_outlier], pred[bll_mask & ~is_outlier], color=col_bll_train, marker='o', s=15, alpha=0.7)
        ax.scatter(obs[bll_mask & is_outlier], pred[bll_mask & is_outlier], facecolors='none', edgecolors=col_bll_train, marker='o', s=25, linewidths=0.8)
        
        # Plot FSRQs (green)
        fsrq_mask = (types == 'FSRQ')
        ax.scatter(obs[fsrq_mask & ~is_outlier], pred[fsrq_mask & ~is_outlier], color=col_fsrq_train, marker='^', s=15, alpha=0.7)
        ax.scatter(obs[fsrq_mask & is_outlier], pred[fsrq_mask & is_outlier], facecolors='none', edgecolors=col_fsrq_train, marker='^', s=25, linewidths=0.8)
        
        # Identity line
        min_v = min(obs.min(), pred.min()) * 0.95
        max_v = max(obs.max(), pred.max()) * 1.05
        ax.plot([min_v, max_v], [min_v, max_v], color='black', linewidth=0.8)
        
        # 2-sigma curves
        if scale == 'inv':
            ax.plot([min_v, max_v], [min_v + 2.0*sigma_inv, max_v + 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
            ax.plot([min_v, max_v], [min_v - 2.0*sigma_inv, max_v - 2.0*sigma_inv], color=col_sigma, linestyle='--', linewidth=0.9)
            ax.set_xlabel("Observed 1/(z+1)")
            ax.set_ylabel("Predicted 1/(z+1)")
            ax.set_xlim(min_v, max_v)
            ax.set_ylim(min_v, max_v)
            ax.set_aspect('equal')
        else:
            z_seq = np.linspace(max(0, min_v), max_v, 300)
            upper_z, lower_z = compute_2sigma_curves(z_seq, sigma_inv)
            ax.plot(z_seq, upper_z, color=col_sigma, linestyle='--', linewidth=0.9)
            ax.plot(z_seq, lower_z, color=col_sigma, linestyle='--', linewidth=0.9)
            ax.set_xlabel("Observed z")
            ax.set_ylabel("Predicted z")
            ax.set_xlim(0, max_v)
            ax.set_ylim(0, max_v)
            
        ax.set_title(title, fontweight='bold')
        
    plot_panel(axes[0, 0], obs_inv_cv, pred_inv_cv, agn_types_cv, 'inv', sigma_inv_cv, "10fCV — 1/(z+1) Scale")
    plot_panel(axes[0, 1], obs_z_cv, pred_z_cv, agn_types_cv, 'z', sigma_inv_cv, "10fCV — Linear z Scale")
    plot_panel(axes[1, 0], obs_inv_val, pred_inv_val, agn_types_val, 'inv', sigma_inv_val, "Validation — 1/(z+1) Scale")
    plot_panel(axes[1, 1], obs_z_val, pred_z_val, agn_types_val, 'z', sigma_inv_val, "Validation — Linear z Scale")
    
    # Legend labels
    axes[0, 0].scatter([], [], color=col_bll_train, marker='o', s=20, label='BLL')
    axes[0, 0].scatter([], [], color=col_fsrq_train, marker='^', s=20, label='FSRQ')
    axes[0, 0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    fig.suptitle("Figure 6: 10-fold CV and Validation Results (Python Stacking)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig06_main_CV_results.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 7: Residuals, Predictor Influence & R Distribution
    # ------------------------------------------------------------------
    print("Generating Figure 7 (Residuals & Influence)...")
    delta_z = obs_z_cv - pred_z_cv
    delta_z_norm = delta_z / (1.0 + obs_z_cv)
    
    fig7, axes7 = plt.subplots(2, 2, figsize=(13, 11))
    
    # 7a: Delta z histogram
    axes7[0, 0].hist(delta_z, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
    axes7[0, 0].axvline(np.mean(delta_z), color=col_bias, linewidth=1.2, label='Mean bias')
    axes7[0, 0].axvline(np.mean(delta_z) + np.std(delta_z), color=col_sigma, linestyle='--', linewidth=1, label=r'$\pm1\sigma$')
    axes7[0, 0].axvline(np.mean(delta_z) - np.std(delta_z), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 0].set_xlabel(r"$\Delta z = z_{obs} - z_{pred}$")
    axes7[0, 0].set_ylabel("Count")
    axes7[0, 0].set_title(r"$\Delta z$ Distribution (Python)", fontweight='bold')
    axes7[0, 0].legend()
    
    # 7b: Delta z_norm histogram
    axes7[0, 1].hist(delta_z_norm, bins=40, color='#2980B9', edgecolor='black', alpha=0.8)
    axes7[0, 1].axvline(np.mean(delta_z_norm), color=col_bias, linewidth=1.2)
    axes7[0, 1].axvline(np.mean(delta_z_norm) + np.std(delta_z_norm), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 1].axvline(np.mean(delta_z_norm) - np.std(delta_z_norm), color=col_sigma, linestyle='--', linewidth=1)
    axes7[0, 1].set_xlabel(r"$\Delta z_{norm} = \Delta z / (1 + z_{obs})$")
    axes7[0, 1].set_ylabel("Count")
    axes7[0, 1].set_title(r"$\Delta z_{norm}$ Distribution (Python)", fontweight='bold')
    
    # 7c: Predictor influence
    df_inf = pd.DataFrame({
        'Predictor': list(feature_counts.keys()),
        'Influence': [float(feature_counts[p]) / (len(models_active) * 100.0) * 100.0 for p in feature_counts.keys()] # in percentage
    }).sort_values(by='Influence', ascending=False)
    
    sns.barplot(x='Influence', y='Predictor', data=df_inf.head(10), ax=axes7[1, 0], palette='coolwarm')
    axes7[1, 0].set_xlabel("LASSO Selection Frequency (%)")
    axes7[1, 0].set_ylabel("Predictor")
    axes7[1, 0].set_title("Relative Predictor Influence (Python)", fontweight='bold')
    
    # 7d: Pearson R distribution across iterations
    # Compute R for each iteration
    # Load xgb or stack results across iterations
    # In python_results.pkl, results['all_stack_pred'] has shape (n_samples, n_iterations)
    all_stack_pred = results.get('all_stack_pred')
    if all_stack_pred is not None:
        if all_stack_pred.shape[0] != len(obs_z_cv):
            all_stack_pred = all_stack_pred[:len(obs_z_cv), :]
        r_dist = []
        for i in range(all_stack_pred.shape[1]):
            pred_iter_inv = all_stack_pred[:, i]
            pred_iter_z = (1.0 / pred_iter_inv) - 1.0
            r_dist.append(compute_pearson(obs_z_cv, pred_iter_z))
            
        axes7[1, 1].hist(r_dist, bins=25, color='#8E44AD', edgecolor='black', alpha=0.8)
        axes7[1, 1].axvline(np.mean(r_dist), color=col_bias, linewidth=1.2)
        axes7[1, 1].set_xlabel("Pearson Correlation R (z-scale)")
        axes7[1, 1].set_ylabel("Count")
        axes7[1, 1].set_title(f"R Distribution Across {all_stack_pred.shape[1]} Iterations\n(Mean R = {np.mean(r_dist):.3f})", fontweight='bold')
        
    fig7.suptitle("Figure 7: Residuals, Predictor Influence & R Distribution", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig07_residuals_influence.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 8: Metric Distributions over 100 Iterations
    # ------------------------------------------------------------------
    print("Generating Figure 8 (Metric distributions)...")
    if all_stack_pred is not None:
        nmad_inv_dist = []
        nmad_z_dist = []
        rmse_inv_dist = []
        rmse_z_dist = []
        
        for i in range(all_stack_pred.shape[1]):
            pred_iter_inv = all_stack_pred[:, i]
            pred_iter_z = (1.0 / pred_iter_inv) - 1.0
            
            nmad_inv_dist.append(compute_nmad(obs_inv_cv, pred_iter_inv, normalized=False))
            nmad_z_dist.append(compute_nmad(obs_z_cv, pred_iter_z, normalized=False))
            rmse_inv_dist.append(compute_rmse(obs_inv_cv, pred_iter_inv))
            rmse_z_dist.append(compute_rmse(obs_z_cv, pred_iter_z))
            
        fig8, axes8 = plt.subplots(2, 2, figsize=(12, 10))
        
        def plot_metric_hist(ax, vals, xlab, title, color):
            ax.hist(vals, bins=25, color=color, edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(vals), color=col_bias, linewidth=1.2)
            ax.set_xlabel(xlab)
            ax.set_ylabel("Count")
            ax.set_title(f"{title}\n(Mean = {np.mean(vals):.4f})", fontweight='bold')
            
        plot_metric_hist(axes8[0, 0], nmad_inv_dist, r"$\sigma_{NMAD}$", r"$\sigma_{NMAD}$ (1/(z+1) scale)", "#2ECC71")
        plot_metric_hist(axes8[0, 1], nmad_z_dist, r"$\sigma_{NMAD}$", r"$\sigma_{NMAD}$ (Linear z scale)", "#2ECC71")
        plot_metric_hist(axes8[1, 0], rmse_inv_dist, "RMSE", "RMSE (1/(z+1) scale)", "#E74C3C")
        plot_metric_hist(axes8[1, 1], rmse_z_dist, "RMSE(z)", "RMSE (Linear z scale)", "#E74C3C")
        
        fig8.suptitle("Figure 8: Metric Distributions over 100 Iterations (Python Stacking)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig08_metric_distributions.png", dpi=300)
        plt.close()
        
    # ------------------------------------------------------------------
    # FIGURE 9: BLL scatter matrix — Training vs Generalization
    # ------------------------------------------------------------------
    print("Generating Figure 9 (BLL scatter matrix)...")
    train_bll = training_eligible[training_eligible['AGN_Type'] == 'BLL']
    gen_bll = gen_set[gen_set['AGN_Type'] == 'BLL']
    
    if len(key_preds) >= 3:
        train_bll_p = train_bll[key_preds].copy(); train_bll_p['Set'] = "BLL Training"
        gen_bll_p = gen_bll[key_preds].copy(); gen_bll_p['Set'] = "BLL Generalization"
        bll_all = pd.concat([train_bll_p, gen_bll_p], ignore_index=True)
        
        g9 = sns.PairGrid(bll_all, hue="Set", hue_order=["BLL Training", "BLL Generalization"], palette={
            "BLL Training": col_bll_train,
            "BLL Generalization": col_bll_gen
        })
        
        g9.map_diag(sns.kdeplot, alpha=0.4, fill=True)
        
        def bll_scatter(x, y, **kwargs):
            label = kwargs.get('label', '')
            if 'Training' in label:
                plt.scatter(x, y, marker='o', color=col_bll_train, s=12, alpha=0.6)
            else:
                plt.scatter(x, y, marker='o', facecolors='none', edgecolors=col_bll_gen, s=12, alpha=0.6)
                
        g9.map_offdiag(bll_scatter)
        g9.add_legend(title="", bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2)
        g9.figure.subplots_adjust(top=0.94, bottom=0.1)
        g9.figure.suptitle("Figure 9: BLL — Training (red) vs Generalization (blue)", fontweight='bold')
        g9.savefig(f"{output_dir}/Fig09_BLL_scatter_matrix.png", dpi=180)
        plt.close()
        
    # ------------------------------------------------------------------
    # FIGURE 10: Predicted BLL Generalization Redshifts vs. Training BLLs
    # Paper Figure 10: Training BLL redshifts (white) + predicted gen BLLs (blue)
    # ------------------------------------------------------------------
    print("Generating Figure 10 (Generalization predictions)...")
    train_set_df = pd.read_csv("data/train_set.csv")
    train_bll_df = train_set_df[train_set_df["AGN_Type"] == "BLL"]
    # Use ALL training BLLs (paper had 501 from their split; we have 552)
    train_bll_z = train_bll_df["Redshift"].dropna().values
    n_train_bll_10 = len(train_bll_z)
    
    # Load Table 3 predictions from python results
    paper_table3_path = "output/python_results/Table3_python_predictions.csv"
    if os.path.exists(paper_table3_path):
        paper_table3 = pd.read_csv(paper_table3_path)
        gen_pred_z = paper_table3['Predicted_z'].dropna().values
        n_gen_10 = len(gen_pred_z)
        
        print(f"  Fig10: Training BLLs = {n_train_bll_10}, Gen BLLs = {n_gen_10}")
        
        # Bins: width 0.1, range 0 to 3.0 matching paper
        bins10 = np.arange(0, 3.05, 0.1)
        
        plt.figure(figsize=(7, 5))
        plt.hist(train_bll_z, bins=bins10, color='white', edgecolor='black',
                 label=f'Training BLLs ({n_train_bll_10})')
        plt.hist(gen_pred_z, bins=bins10, color=col_bll_gen, edgecolor='black', alpha=0.7,
                 label=f'Predicted BLL Gen. ({n_gen_10})')
        plt.xlabel("Redshift (z)")
        plt.ylabel("Number of Sources")
        plt.title("Predicted Redshifts: BLL Generalization vs Training Set", fontweight='bold')
        plt.xlim(0, 3.0)
        plt.ylim(0, 120)
        plt.xticks(np.arange(0, 3.1, 0.5))
        plt.yticks(np.arange(0, 121, 20))
        plt.legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right')
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().grid(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig10_generalization_predictions.png", dpi=300)
        plt.close()
        
    # ------------------------------------------------------------------
    # FIGURE 11: Alternative models (Python Stacking vs Baseline)
    # ------------------------------------------------------------------
    print("Generating Figure 11 (Python model comparison bar plot)...")
    # Compare Pearson R and RMSE of all trained models in a 2-panel chart
    model_keys = [k for k in metrics_report.keys() if not k.endswith('_corrected')]
    model_r = [metrics_report[k]['z']['R'] for k in model_keys]
    model_rmse = [metrics_report[k]['z']['RMSE'] for k in model_keys]
    
    df_compare = pd.DataFrame({
        'Model': [m.upper() for m in model_keys],
        'Pearson R': model_r,
        'RMSE': model_rmse
    }).sort_values(by='Pearson R', ascending=False)
    
    fig11, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    sns.barplot(x='Pearson R', y='Model', data=df_compare, palette='viridis', ax=ax1)
    ax1.set_xlim(0.65, 0.76)
    ax1.set_title("Pearson Correlation (R) Comparison", fontweight='bold')
    
    sns.barplot(x='RMSE', y='Model', data=df_compare.sort_values(by='RMSE'), palette='rocket', ax=ax2)
    ax2.set_xlim(0.35, 0.50)
    ax2.set_title("RMSE Comparison (Linear z Scale)", fontweight='bold')
    
    fig11.suptitle("Figure 11: Performance Comparisons for Alternative Python Models", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig11_SLO2_SLOPE_CV_results.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 12: Optimal Transport linear fits — BLL and FSRQ
    # ------------------------------------------------------------------
    print("Generating Figure 12 (Optimal Transport fits)...")
    fig12, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
    def plot_ot_panel(ax, pred_inv, obs_inv, label, color):
        pred_s = np.sort(pred_inv)
        obs_s = np.sort(obs_inv)
        n = min(len(pred_s), len(obs_s))
        pred_s, obs_s = pred_s[:n], obs_s[:n]
        
        # Fit linear model
        A, B = np.polyfit(pred_s, obs_s, 1)
        r2 = compute_pearson(obs_s, pred_s) ** 2
        
        ax.scatter(pred_s, obs_s, color=color, s=15, alpha=0.6)
        # Plot fit line
        ax.plot(pred_s, A * pred_s + B, color='black', linewidth=1.2)
        
        # Text annotation
        ax.text(pred_s.min(), obs_s.max() * 0.95, f"y = {A:.3f}x + {B:.4f}\nR² = {r2:.3f}", 
                ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_xlabel(f"Sorted Predicted 1/(z+1) — {label}")
        ax.set_ylabel(f"Sorted Observed 1/(z+1) — {label}")
        ax.set_title(f"{label} Optimal Transport Fit", fontweight='bold')
        ax.set_aspect('equal')
        
    bll_idx = (agn_types_cv == 'BLL')
    fsrq_idx = (agn_types_cv == 'FSRQ')
    
    plot_ot_panel(ax_l, pred_inv_cv[bll_idx], obs_inv_cv[bll_idx], "BLL", col_bll_train)
    plot_ot_panel(ax_r, pred_inv_cv[fsrq_idx], obs_inv_cv[fsrq_idx], "FSRQ", col_fsrq_train)
    
    fig12.suptitle("Figure 12: Optimal Transport Linear Fits", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig12_optimal_transport.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 13: Bias-corrected 10fCV plots (left = 1/(z+1), right = z)
    # ------------------------------------------------------------------
    print("Generating Figure 13 (Bias-corrected CV)...")
    pred_inv_bc = mean_predictions['stack_corrected']
    if len(pred_inv_bc) != len(obs_inv_cv):
        pred_inv_bc = pred_inv_bc[:len(obs_inv_cv)]
    pred_z_bc = (1.0 / pred_inv_bc) - 1.0
    sigma_inv_bc = np.std(pred_inv_bc - obs_inv_cv)
    
    fig13, (ax13_l, ax13_r) = plt.subplots(1, 2, figsize=(13, 6))
    
    plot_panel(ax13_l, obs_inv_cv, pred_inv_bc, agn_types_cv, 'inv', sigma_inv_bc, "Bias-Corrected CV — 1/(z+1) Scale")
    plot_panel(ax13_r, obs_z_cv, pred_z_bc, agn_types_cv, 'z', sigma_inv_bc, "Bias-Corrected CV — Linear z Scale")
    
    # Add legend to left panel
    ax13_l.scatter([], [], color=col_bll_train, marker='o', s=20, label='BLL')
    ax13_l.scatter([], [], color=col_fsrq_train, marker='^', s=20, label='FSRQ')
    ax13_l.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    fig13.suptitle("Figure 13: Bias-Corrected 10-fold CV Results", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig13_bias_corrected_CV.png", dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------
    # FIGURE 14: Bias-corrected BLL predicted vs observed redshift distribution
    # Matches paper's Figure 16:
    #   - Red (alpha=0.5): training BLL redshifts
    #   - Turquoise (alpha=0.5): bias-corrected gen BLL predictions  
    #   - X-axis: [0, 2.5], ticks every 0.5
    #   - Y-axis: [0, 60], ticks every 10
    #   - Bins: 0.05 width from 0 to 2.5
    # ------------------------------------------------------------------
    print("Generating Figure 14 (Bias-corrected generalization - matching paper Fig 16)...")
    if os.path.exists(paper_table3_path):
        paper_table3 = pd.read_csv(paper_table3_path)
        gen_bll_bc_z = paper_table3['Bias_Corrected_z'].dropna().values
        n_gen_bc = len(gen_bll_bc_z)
        
        print(f"  Fig14: Training BLLs = {n_train_bll_10}, Gen BLLs (bias-corrected) = {n_gen_bc}")
        
        # Bins: width 0.05, range 0 to 3.0
        bins14 = np.arange(0, 3.05, 0.05)
        
        # Clip negative bias-corrected z to 0 (same as paper display)
        gen_bll_bc_z_clipped = np.clip(gen_bll_bc_z, 0.001, 2.999)
        train_bll_z_clipped  = np.clip(train_bll_z, 0.001, 2.999)
        
        label_train = f"Training set redshift distribution for BLL ({n_train_bll_10})"
        label_gen   = f"Generalization set redshfit distribution for BLL ({n_gen_bc})"
        
        plt.figure(figsize=(7, 5))
        plt.hist(train_bll_z_clipped,   bins=bins14, color='#EC7063', edgecolor='#C0392B',
                 alpha=0.5, label=label_train, density=False)
        plt.hist(gen_bll_bc_z_clipped, bins=bins14, color='#76D7C4', edgecolor='#16A085',
                 alpha=0.5, label=label_gen, density=False)
        plt.xlabel("z")
        plt.ylabel("Counts")
        plt.title("Overlapped Histogram of Predicted and Observed z for only BLL", fontweight='bold')
        plt.xlim(0, 3.0)
        plt.ylim(0, 60)
        plt.xticks(np.arange(0, 2.6, 0.5))
        plt.yticks(np.arange(0, 61, 10))
        # Legend box inside upper right (matching paper's layout)
        plt.legend(title="Histogram distribution", frameon=True, facecolor='white',
                   edgecolor='black', loc='upper right', fontsize=8, title_fontsize=9)
        plt.gca().grid(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig14_bias_corrected_gen.png", dpi=300)
        plt.close()
        
    print("\nAll 14 figures successfully reproduced and saved in 'plots/python_plots/'!")

if __name__ == "__main__":
    main()
