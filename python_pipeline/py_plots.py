import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparison_plots(metrics_report, model_runtimes, r_metrics, output_dir="plots/python_plots"):
    """
    Generates comparison figures for the newly trained Python models against the R baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # ----------------------------------------------------
    # 1. Performance Comparisons (R and RMSE)
    # ----------------------------------------------------
    model_names = list(metrics_report.keys())
    
    # Extract linear z scale metrics
    pearsons_z = [metrics_report[m]['z']['R'] for m in model_names]
    rmses_z = [metrics_report[m]['z']['RMSE'] for m in model_names]
    
    # Add R-baseline for comparison (approximate values from Narendra et al. 2022)
    # SL-O1: R=0.74, RMSE=0.467. SLOPE: R=0.72, RMSE=0.479
    model_names_all = model_names + ['R_SL_O1', 'R_SLOPE']
    pearsons_z_all = pearsons_z + [0.742, 0.721]
    rmses_z_all = rmses_z + [0.457, 0.473]
    
    perf_df = pd.DataFrame({
        'Model': model_names_all,
        'Pearson R': pearsons_z_all,
        'RMSE': rmses_z_all
    }).sort_values(by='Pearson R', ascending=False)
    
    # Plot Pearson R Comparison
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='Pearson R', y='Model', data=perf_df, palette='viridis')
    plt.title('Redshift Prediction Pearson R Comparison (Linear z Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Pearson Correlation (R)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0.65, 0.80)
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.002, p.get_y() + p.get_height()/2 + 0.05, f'{width:.4f}', ha="left", va="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_pearson_comparison.png", dpi=300)
    plt.close()
    
    # Plot RMSE Comparison
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='RMSE', y='Model', data=perf_df.sort_values(by='RMSE'), palette='rocket')
    plt.title('Redshift Prediction RMSE Comparison (Linear z Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('RMSE (Dz)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0.35, 0.55)
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.002, p.get_y() + p.get_height()/2 + 0.05, f'{width:.4f}', ha="left", va="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_rmse_comparison.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 2. Runtime / Computational Efficiency Comparison
    # ----------------------------------------------------
    # R baseline took ~3594 seconds (59.9 mins) for SuperLearner O1 + O2 + SLOPE
    # We compare model training times. Since individual python models run inside parallelized cv loop,
    # we can show the total loop runtime.
    python_total_time = model_runtimes.get('total_pipeline', 0)
    r_total_time = 3594.0 # 59.9 mins in seconds
    
    speed_df = pd.DataFrame({
        'Pipeline': ['R Pipeline (SuperLearner+SLOPE)', 'Python Pipeline (9 Models+Stacking)'],
        'Total Runtime (seconds)': [r_total_time, python_total_time],
        'Total Runtime (minutes)': [r_total_time / 60.0, python_total_time / 60.0]
    })
    
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='Total Runtime (minutes)', y='Pipeline', data=speed_df, palette='coolwarm')
    plt.title('Execution Efficiency Comparison (100 Iterations × 10fCV)', fontsize=14, fontweight='bold')
    plt.xlabel('Total Runtime (minutes)', fontsize=12)
    plt.ylabel('', fontsize=12)
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5, p.get_y() + p.get_height()/2 + 0.05, f'{width:.2f} mins', ha="left", va="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pipeline_efficiency_comparison.png", dpi=300)
    plt.close()
    
    print(f"Comparison figures successfully saved to '{output_dir}/'!")

def plot_predicted_vs_observed(observed_z, predicted_z, model_name, output_dir="plots/python_plots"):
    """
    Plots predicted vs observed redshifts with 2-sigma curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate inverse standard deviation of residuals for 2-sigma curves
    obs_inv = 1.0 / (observed_z + 1.0)
    pred_inv = 1.0 / (predicted_z + 1.0)
    residuals_inv = pred_inv - obs_inv
    mask = np.isfinite(residuals_inv)
    if np.sum(mask) > 1:
        sigma_inv = np.std(residuals_inv[mask], ddof=1)
    else:
        sigma_inv = 0.12 # fallback average
        
    plt.figure(figsize=(7, 6))
    
    # Plot scatter
    plt.scatter(observed_z, predicted_z, alpha=0.5, c='#2C3E50', edgecolors='none', s=15)
    
    # Diagonal y = x line
    max_val = max(np.max(observed_z), np.max(predicted_z))
    plt.plot([0, max_val], [0, max_val], color='#E74C3C', linestyle='-', linewidth=2, label='1:1 Line')
    
    # Compute and plot 2-sigma curves
    z_seq = np.linspace(0.0, max_val, 300)
    inv_seq = 1.0 / (z_seq + 1.0)
    upper_inv = inv_seq + 2.0 * sigma_inv
    lower_inv = inv_seq - 2.0 * sigma_inv
    lower_inv = np.clip(lower_inv, 1e-6, None)
    
    upper_z = (1.0 / lower_inv) - 1.0
    lower_z = (1.0 / upper_inv) - 1.0
    
    plt.plot(z_seq, upper_z, color='#2980B9', linestyle='--', linewidth=1.5, label=r'2$\sigma$ Band')
    plt.plot(z_seq, lower_z, color='#2980B9', linestyle='--', linewidth=1.5)
    
    plt.title(f'Predicted vs. Observed Redshift - {model_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Spectroscopic Redshift (z_spectroscopic)', fontsize=10)
    plt.ylabel('Predicted Redshift (z_predicted)', fontsize=10)
    plt.xlim(0, max_val + 0.1)
    plt.ylim(0, max_val + 0.1)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/{model_name}_pred_vs_obs.png", dpi=300)
    plt.close()
