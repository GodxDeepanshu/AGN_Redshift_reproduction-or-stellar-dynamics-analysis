"""
Regenerate the model comparison bar charts including ALL neural models + TabPFN.
Reads from: python_results.pkl (classical) + neural_metrics.json (neural+TabPFN)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import json
import pickle

output_dir = "plots/python_plots"
os.makedirs(output_dir, exist_ok=True)
sns.set_theme(style="whitegrid")

# ── 1. Load classical model metrics from pickle ─────────────────────────
pkl_path = "output/python_results/python_results.pkl"
with open(pkl_path, "rb") as f:
    results = pickle.load(f)

metrics_report = results.get('metrics_report', {})

# ── 2. Load neural model metrics from JSON ───────────────────────────────
json_path = "output/python_results/neural_metrics.json"
with open(json_path, "r") as f:
    neural_metrics = json.load(f)

# ── 3. Build unified table ───────────────────────────────────────────────
rows = []

# Classical models (100x10fCV) 
display_names = {
    'xgb': 'XGBoost', 'lgb': 'LightGBM', 'cat': 'CatBoost',
    'et': 'ExtraTrees', 'hgb': 'HistGradientBoosting',
    'xgb_opt': 'XGBoost-Optuna', 'lgb_opt': 'LightGBM-Optuna',
    'cat_opt': 'CatBoost-Optuna', 'lgb_es': 'LightGBM-ES',
    'stacking': 'Stacking Ensemble'
}
for model_key, model_data in metrics_report.items():
    name = display_names.get(model_key, model_key)
    try:
        rows.append({
            'Model': name,
            'Pearson R': model_data['z']['R'],
            'RMSE': model_data['z']['RMSE'],
            'Outlier %': model_data.get('outlier_pct', np.nan),
            'Type': 'Classical (100x10fCV)'
        })
    except (KeyError, TypeError):
        pass

# Neural models (single split)
neural_display = {
    'ft_transformer': 'FT-Transformer',
    'saint': 'SAINT',
    'tabm': 'TabM',
    'tabnet': 'TabNet',
    'tabpfn': 'TabPFN'
}
for model_key, model_data in neural_metrics.items():
    name = neural_display.get(model_key, model_key)
    rows.append({
        'Model': name,
        'Pearson R': model_data['R_z'],
        'RMSE': model_data['RMSE_z'],
        'Outlier %': model_data['outlier_pct_fixed'],
        'Type': 'Neural (Single Split)'
    })

# R baseline references
rows.append({'Model': 'R Baseline (SL-O1)', 'Pearson R': 0.742, 'RMSE': 0.457, 'Outlier %': 43.5, 'Type': 'R Baseline'})
rows.append({'Model': 'R Baseline (SLOPE)', 'Pearson R': 0.721, 'RMSE': 0.473, 'Outlier %': 45.0, 'Type': 'R Baseline'})

df = pd.DataFrame(rows)
df_sorted_r = df.sort_values('Pearson R', ascending=True)
df_sorted_rmse = df.sort_values('RMSE', ascending=False)
df_sorted_outlier = df.dropna(subset=['Outlier %']).sort_values('Outlier %', ascending=False)

# ── Color palette by type ────────────────────────────────────────────────
type_colors = {
    'Classical (100x10fCV)': '#3498DB',
    'Neural (Single Split)': '#E74C3C',
    'R Baseline': '#95A5A6'
}

def get_colors(df_sorted, col='Type'):
    return [type_colors.get(t, '#555555') for t in df_sorted[col]]

# ── 4. PLOT 1: Pearson R Comparison ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
colors = get_colors(df_sorted_r)
bars = ax.barh(df_sorted_r['Model'], df_sorted_r['Pearson R'], color=colors, edgecolor='white', linewidth=0.5)

# Highlight best
best_idx = df_sorted_r['Pearson R'].idxmax()
best_model = df_sorted_r.loc[best_idx, 'Model']

for bar, (_, row) in zip(bars, df_sorted_r.iterrows()):
    width = bar.get_width()
    fontweight = 'bold' if row['Model'] == best_model else 'normal'
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
            ha='left', va='center', fontsize=9, fontweight=fontweight)

ax.set_xlim(0.65, 0.82)
ax.set_xlabel('Pearson Correlation (R) in Linear z', fontsize=12)
ax.set_title('AGN Photometric Redshift: Model Performance Comparison\n(Pearson R - Higher is Better)', 
             fontsize=14, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/model_pearson_comparison.png", dpi=300)
plt.close()
print("  Saved: model_pearson_comparison.png")

# ── 5. PLOT 2: RMSE Comparison ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
colors = get_colors(df_sorted_rmse)
bars = ax.barh(df_sorted_rmse['Model'], df_sorted_rmse['RMSE'], color=colors, edgecolor='white', linewidth=0.5)

best_idx_rmse = df_sorted_rmse['RMSE'].idxmin()
best_model_rmse = df_sorted_rmse.loc[best_idx_rmse, 'Model']

for bar, (_, row) in zip(bars, df_sorted_rmse.iterrows()):
    width = bar.get_width()
    fontweight = 'bold' if row['Model'] == best_model_rmse else 'normal'
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
            ha='left', va='center', fontsize=9, fontweight=fontweight)

ax.set_xlim(0.30, 0.55)
ax.set_xlabel('RMSE (Dz) in Linear z', fontsize=12)
ax.set_title('AGN Photometric Redshift: Model Performance Comparison\n(RMSE - Lower is Better)',
             fontsize=14, fontweight='bold')
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_rmse_comparison.png", dpi=300)
plt.close()
print("  Saved: model_rmse_comparison.png")

# ── 6. PLOT 3: Outlier % Comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
colors = get_colors(df_sorted_outlier)
bars = ax.barh(df_sorted_outlier['Model'], df_sorted_outlier['Outlier %'], color=colors, edgecolor='white', linewidth=0.5)

best_idx_out = df_sorted_outlier['Outlier %'].idxmin()
best_model_out = df_sorted_outlier.loc[best_idx_out, 'Model']

for bar, (_, row) in zip(bars, df_sorted_outlier.iterrows()):
    width = bar.get_width()
    fontweight = 'bold' if row['Model'] == best_model_out else 'normal'
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
            ha='left', va='center', fontsize=9, fontweight=fontweight)

ax.set_xlim(30, 55)
ax.set_xlabel('Catastrophic Outlier Rate (|Dz_norm| > 0.15) %', fontsize=12)
ax.set_title('AGN Photometric Redshift: Catastrophic Outlier Comparison\n(Lower is Better)',
             fontsize=14, fontweight='bold')
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_outlier_comparison.png", dpi=300)
plt.close()
print("  Saved: model_outlier_comparison.png")

# ── 7. Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FINAL MODEL RANKING (by Pearson R)")
print("=" * 80)
df_final = df.sort_values('Pearson R', ascending=False)
for i, (_, row) in enumerate(df_final.iterrows(), 1):
    star = " ***BEST***" if i == 1 else ""
    print(f"  {i:2d}. {row['Model']:25s}  R={row['Pearson R']:.4f}  RMSE={row['RMSE']:.4f}  Outlier={row['Outlier %']:.1f}%  [{row['Type']}]{star}")

print("\n  Done! All comparison plots updated with TabPFN.")
