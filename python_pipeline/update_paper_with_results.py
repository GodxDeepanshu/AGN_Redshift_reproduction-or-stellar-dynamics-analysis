import os
import re
import pickle
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUTPUT = BASE / "output" / "dr3_results"
PAPER_TEX = BASE / "dr3_results_paper.tex"

def main():
    csv_path = OUTPUT / "selection_cuts_comparison.csv"
    tex_table_path = OUTPUT / "selection_cuts_comparison.tex"
    pkl_path = OUTPUT / "dr3_python_results.pkl"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found! Run the comparison study script first.")
        return
        
    df = pd.read_csv(csv_path)
    print("Loaded comparison results:")
    print(df.to_string(index=False))
    
    # Extract values for final configuration (Config 4) and baseline (Config 2)
    # Config 2: "Config 2: Full-Lat BLL+FSRQ, 12 feat, 5 trees, OT Calib (inv)"
    # Config 4: "Config 4: Full-Lat BLL+FSRQ, 20 feat, 10 models, Isotonic (inv)"
    
    row_c2 = df[df['Configuration'].str.contains('Config 2:')].iloc[0]
    row_c4 = df[df['Configuration'].str.contains('Config 4:')].iloc[0]
    
    r_c2 = row_c2['Pearson R_z']
    rmse_c2 = row_c2['RMSE (Dz)']
    
    r_c4 = row_c4['Pearson R_z']
    rmse_c4 = row_c4['RMSE (Dz)']
    
    # Calculate improvements
    r_imp = (r_c4 - r_c2) / r_c2 * 100.0
    rmse_imp = (rmse_c2 - rmse_c4) / rmse_c2 * 100.0
    
    print(f"\nBaseline R_z: {r_c2:.4f}, Upgraded R_z: {r_c4:.4f} ({r_imp:+.2f}% relative)")
    print(f"Baseline RMSE: {rmse_c2:.4f}, Upgraded RMSE: {rmse_c4:.4f} ({rmse_imp:+.2f}% relative)")
    
    # Dynamically extract iterations and samples
    n_iterations = 5
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                res_dict = pickle.load(f)
            n_iterations = res_dict['upgraded_results']['agg_predictions']['stack_corrected'].shape[1]
            print(f"Dynamically detected iterations: {n_iterations}")
        except Exception as e:
            print(f"Warning loading pickle iterations: {e}")
            
    n_samples = 1389
    train_csv_path = DATA / "dr3_full_train.csv"
    if train_csv_path.exists():
        try:
            train_df = pd.read_csv(train_csv_path)
            n_samples = len(train_df)
            print(f"Dynamically detected training samples: {n_samples}")
        except Exception as e:
            print(f"Warning loading training samples: {e}")
            
    # Read the paper text
    with open(PAPER_TEX, 'r', encoding='utf-8') as f:
        paper_text = f.read()
        
    # 1. Update Abstract
    n_samples_str = f"{n_samples:,}"
    old_abstract_pattern = r"Using \d+ iterations of 10-fold cross-validation on \d+,?\d* training sources, we achieve a Pearson correlation of \$R_z = \d+\.\d+\$ and RMSE\$_z = \d+\.\d+\$---a \d+\.\d+\\% relative improvement in \$R_z\$ over the published Optimal Transport calibration \(\$R_z = \d+\.\d+\$\)\."
    new_abstract_replacement = f"Using {n_iterations} iterations of 10-fold cross-validation on {n_samples_str} training sources, we achieve a Pearson correlation of $R_z = {r_c4:.4f}$ and RMSE$_z = {rmse_c4:.4f}$---a {r_imp:.1f}\\% relative improvement in $R_z$ over the published Optimal Transport calibration ($R_z = {r_c2:.4f}$)."
    
    paper_text = re.sub(old_abstract_pattern, new_abstract_replacement, paper_text)
    
    # 2. Replace Table (tab:selection_cuts_comparison) with the new selection_cuts_comparison table
    with open(tex_table_path, 'r') as tf:
        new_table_tex = tf.read().strip()
        
    # Find the table block in the paper text and replace using string replacement
    table_pattern = re.compile(r"\\begin\{table\*\}\[t\].*?\\label\{tab:selection_cuts_comparison\}.*?\\end\{table\*\}", re.DOTALL)
    match = table_pattern.search(paper_text)
    if match:
        old_table_text = match.group(0)
        paper_text = paper_text.replace(old_table_text, new_table_tex)
        print("Replaced Table 2 (tab:selection_cuts_comparison) with new selection cuts comparison table.")
    else:
        # Fallback: insert the table before the top30_generalization table
        top30_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\caption\{Top 30 Highest-Redshift Predictions"
        match_top30 = re.search(top30_pattern, paper_text)
        if match_top30:
            insert_idx = match_top30.start()
            paper_text = paper_text[:insert_idx] + new_table_tex + "\n\n" + paper_text[insert_idx:]
            print("Inserted Table 2 (tab:selection_cuts_comparison) before Table 3 (tab:top30_generalization).")
        else:
            print("Warning: Table with label 'tab:selection_cuts_comparison' and 'tab:top30_generalization' not found.")
        
    # Save the updated paper
    with open(PAPER_TEX, 'w', encoding='utf-8') as f:
        f.write(paper_text)
    rel_path = PAPER_TEX.name
    print(f"\nLaTeX paper successfully updated in {rel_path}!")

if __name__ == '__main__':
    main()
