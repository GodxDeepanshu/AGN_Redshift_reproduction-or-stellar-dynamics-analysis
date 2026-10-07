#!/usr/bin/env python3
import time
import subprocess
import pickle
import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE / "output" / "dr3_results"
PAPER_TEX = BASE / "dr3_results_paper.tex"
PDFLATEX = "/home/hea/.TinyTeX/bin/x86_64-linux/pdflatex"

def is_pipeline_running():
    # Check if run_full_dr3_pipeline.py is running
    cmd = "ps aux | grep 'python.*run_full_dr3_pipeline.py' | grep -v grep || true"
    out = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return len(out) > 0

def wait_for_pipeline():
    print("Waiting for run_full_dr3_pipeline.py to complete...")
    while is_pipeline_running():
        time.sleep(15)
    print("Pipeline completed or stopped.")

def run_comparison_study():
    print("Running comparative study for 30 iterations on 6 cores...")
    cmd = ["python", "python_pipeline/run_comparison_study.py", "--iterations", "30", "--n_jobs", "6"]
    subprocess.run(cmd, check=True)
    print("Comparative study complete.")

def update_latex_paper():
    print("Updating LaTeX paper text and tables with new metrics...")
    
    # 1. Load data
    pkl_path = OUTPUT_DIR / "dr3_python_results.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    csv_path = OUTPUT_DIR / "selection_cuts_comparison.csv"
    df_compare = pd.read_csv(csv_path)
    
    upg = data['upgraded_results']
    base = data['baseline_results']
    
    # Extract baseline (Config 2) and upgraded (Config 4) values
    row_c2 = df_compare[df_compare['Configuration'].str.contains('Config 2:')].iloc[0]
    row_c4 = df_compare[df_compare['Configuration'].str.contains('Config 4:')].iloc[0]
    
    r_c2 = row_c2['Pearson R_z']
    rmse_c2 = row_c2['RMSE (Dz)']
    
    r_c4 = row_c4['Pearson R_z']
    rmse_c4 = row_c4['RMSE (Dz)']
    
    r_imp = (r_c4 - r_c2) / r_c2 * 100.0
    rmse_imp = (rmse_c2 - rmse_c4) / rmse_c2 * 100.0
    
    # Read paper tex
    with open(PAPER_TEX, 'r', encoding='utf-8') as f:
        paper_text = f.read()
        
    # --- UPDATE ABSTRACT ---
    n_iterations = 30
    n_samples = 1389
    old_abstract_pattern = r"Using \d+ iterations of 10-fold cross-validation on \d+,?\d* training sources, we achieve a Pearson correlation of \$R_z = \d+\.\d+\$ and RMSE\$_z = \d+\.\d+\$---a \d+\.\d+\\% relative improvement in \$R_z\$ over the published Optimal Transport calibration \(\$R_z = \d+\.\d+\$\)\."
    new_abstract = f"Using {n_iterations} iterations of 10-fold cross-validation on {n_samples:,} training sources, we achieve a Pearson correlation of $R_z = {r_c4:.4f}$ and RMSE$_z = {rmse_c4:.4f}$---a {r_imp:.1f}\\% relative improvement in $R_z$ over the published Optimal Transport calibration ($R_z = {r_c2:.4f}$)."
    paper_text = re.sub(old_abstract_pattern, new_abstract, paper_text)
    
    # --- UPDATE TABLE 1 (lit_comparison) ---
    # Stacking Ensemble
    stack_r = upg['metrics_report']['stack']['z']['R']
    stack_rmse = upg['metrics_report']['stack']['z']['RMSE']
    stack_out = upg['metrics_report']['stack']['outlier_pct']
    
    # TabPFN
    tabpfn_r = upg['metrics_report']['tabpfn']['z']['R']
    tabpfn_rmse = upg['metrics_report']['tabpfn']['z']['RMSE']
    tabpfn_out = upg['metrics_report']['tabpfn']['outlier_pct']
    
    old_t1_rows = [
        r"\textbf{This Work (Ensemble)} & \textbf{RidgeCV (1389 sources)} & \textbf{0.7994} & \textbf{0.4066} & \textbf{60.84\%} \\",
        r"\textbf{This Work (TabPFN)} & \textbf{TabPFN (1389 sources)} & \textbf{0.7985} & \textbf{0.4053} & \textbf{60.19\%} \\"
    ]
    new_t1_rows = [
        f"\\textbf{{This Work (Ensemble)}} & \\textbf{{RidgeCV (1389 sources)}} & \\textbf{{{stack_r:.4f}}} & \\textbf{{{stack_rmse:.4f}}} & \\textbf{{{stack_out:.2f}\\%}} \\\\",
        f"\\textbf{{This Work (TabPFN)}} & \\textbf{{TabPFN (1389 sources)}} & \\textbf{{{tabpfn_r:.4f}}} & \\textbf{{{tabpfn_rmse:.4f}}} & \\textbf{{{tabpfn_out:.2f}\\%}} \\\\"
    ]
    for old_row, new_row in zip(old_t1_rows, new_t1_rows):
        paper_text = paper_text.replace(old_row, new_row)
        
    # --- UPDATE TABLE 2 (selection_cuts_comparison) ---
    tex_table_path = OUTPUT_DIR / "selection_cuts_comparison.tex"
    with open(tex_table_path, 'r') as tf:
        new_table_tex = tf.read().strip()
        
    table_pattern = re.compile(r"\\begin\{table\*\}\[t\].*?\\label\{tab:selection_cuts_comparison\}.*?\\end\{table\*\}", re.DOTALL)
    paper_text = re.sub(table_pattern, new_table_tex, paper_text)
    
    # --- UPDATE TABLE 5 (unified_metrics) ---
    # We will build Table 5 content dynamically from pickle
    t5_stack = upg['metrics_report']['stack']
    t5_stack_c = upg['metrics_report']['stack_corrected']
    t5_tabpfn = upg['metrics_report']['tabpfn']
    t5_tabpfn_c = upg['metrics_report']['tabpfn_corrected']
    
    new_table_5_tex = r"""\begin{table*}[t]
\centering
\caption{Detailed Cross-Validation Performance Metrics (10-fold CV on 1,389 training sources)}
\label{tab:unified_metrics}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Model / Configuration}} & \multicolumn{3}{c}{\textbf{Inverse Scale $1/(z+1)$}} & \multicolumn{3}{c}{\textbf{Linear Redshift Scale $z$}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
 & \textbf{Pearson $R$} & \textbf{RMSE} & \textbf{$\sigma_{\text{NMAD}}$} & \textbf{Pearson $R_z$} & \textbf{RMSE$_z$} & \textbf{$\sigma_{\text{NMAD}, z}$} \\
\midrule
"""
    new_table_5_tex += f"Stacking Ensemble (Uncalibrated) & {t5_stack['inv']['R']:.4f} & {t5_stack['inv']['RMSE']:.4f} & {t5_stack['inv']['NMAD']:.4f} & {t5_stack['z']['R']:.4f} & {t5_stack['z']['RMSE']:.4f} & {t5_stack['z']['NMAD']:.4f} \\\\\n"
    new_table_5_tex += f"Stacking + Smooth OTQM (Calibrated) & {t5_stack_c['inv']['R']:.4f} & {t5_stack_c['inv']['RMSE']:.4f} & {t5_stack_c['inv']['NMAD']:.4f} & {t5_stack_c['z']['R']:.4f} & {t5_stack_c['z']['RMSE']:.4f} & {t5_stack_c['z']['NMAD']:.4f} \\\\\n"
    new_table_5_tex += f"TabPFN (Uncalibrated) & {t5_tabpfn['inv']['R']:.4f} & {t5_tabpfn['inv']['RMSE']:.4f} & {t5_tabpfn['inv']['NMAD']:.4f} & {t5_tabpfn['z']['R']:.4f} & {t5_tabpfn['z']['RMSE']:.4f} & {t5_tabpfn['z']['NMAD']:.4f} \\\\\n"
    new_table_5_tex += f"TabPFN + Smooth OTQM (Calibrated) & {t5_tabpfn_c['inv']['R']:.4f} & {t5_tabpfn_c['inv']['RMSE']:.4f} & {t5_tabpfn_c['inv']['NMAD']:.4f} & {t5_tabpfn_c['z']['R']:.4f} & {t5_tabpfn_c['z']['RMSE']:.4f} & {t5_tabpfn_c['z']['NMAD']:.4f} \\\\\n"
    new_table_5_tex += r"""\bottomrule
\end{tabular}
\end{table*}"""
    
    t5_pattern = re.compile(r"\\begin\{table\*\}\[t\].*?\\label\{tab:unified_metrics\}.*?\\end\{table\*\}", re.DOTALL)
    paper_text = re.sub(t5_pattern, new_table_5_tex, paper_text)
    
    # --- UPDATE TEXT MENTIONS ---
    # Update upgraded pipeline achieves $R_z = 0.8073$ and an RMSE$_z = 0.3993$
    old_text_pattern = r"our upgraded pipeline \(Config~4\) utilizing the \d+-feature set and 10 models calibrated via Smooth OTQM achieves a Pearson correlation of \$R_z = \d+\.\d+\$ and an RMSE\$_z = \d+\.\d+\$ on the Full-Latitude training set\."
    new_text_rep = f"our upgraded pipeline (Config~4) utilizing the 40-feature set and 10 models calibrated via Smooth OTQM achieves a Pearson correlation of $R_z = {r_c4:.4f}$ and an RMSE$_z = {rmse_c4:.4f}$ on the Full-Latitude training set."
    paper_text = re.sub(old_text_pattern, new_text_rep, paper_text)
    
    # Update relative improvements "+13.10\%" and "16.62\%"
    old_imp_pattern = r"significant \+\d+\.\d+\\% relative improvement in correlation and a \d+\.\d+\\% reduction in RMSE compared to the replicated baseline\."
    new_imp_rep = f"significant {r_imp:+.2f}\\% relative improvement in correlation and a {rmse_imp:.2f}\\% reduction in RMSE compared to the replicated baseline."
    paper_text = re.sub(old_imp_pattern, new_imp_rep, paper_text)
    
    # Update TabPFN achieves a remarkable Pearson correlation of $R_z = 0.7985$ and RMSE$_z = 0.4053$ (uncalibrated), and $R_z = 0.7973$ and RMSE$_z = 0.4046$ (Smooth OTQM calibrated)
    old_tabpfn_pattern = r"TabPFN achieves a remarkable Pearson correlation of \$R_z = \d+\.\d+\$ and RMSE\$_z = \d+\.\d+\$ \(uncalibrated\), and \$R_z = \d+\.\d+\$ and RMSE\$_z = \d+\.\d+\$ \(Smooth OTQM calibrated\)"
    new_tabpfn_rep = f"TabPFN achieves a remarkable Pearson correlation of $R_z = {t5_tabpfn['z']['R']:.4f}$ and RMSE$_z = {t5_tabpfn['z']['RMSE']:.4f}$ (uncalibrated), and $R_z = {t5_tabpfn_c['z']['R']:.4f}$ and RMSE$_z = {t5_tabpfn_c['z']['RMSE']:.4f}$ (Smooth OTQM calibrated)"
    paper_text = re.sub(old_tabpfn_pattern, new_tabpfn_rep, paper_text)
    
    # Save paper_text
    with open(PAPER_TEX, 'w', encoding='utf-8') as f:
        f.write(paper_text)
    print("LaTeX paper updated successfully.")

def run_other_latex_tables():
    print("Regenerating and replacing Top30 and Mismatched tables...")
    subprocess.run(["python", "scratch/gen_latex_tables.py"], check=True)
    subprocess.run(["python", "scratch/update_latex_tables_in_paper.py"], check=True)

def compile_pdf():
    print("Compiling LaTeX paper PDF...")
    for i in range(2):
        print(f"pdflatex compilation run {i+1}...")
        subprocess.run([PDFLATEX, "dr3_results_paper.tex"], check=True)
    
    # Copy results to latest_result/
    print("Copying results to latest_result/ directory...")
    subprocess.run(["cp", "dr3_results_paper.tex", "latest_result/latest_result.tex"], check=True)
    subprocess.run(["cp", "dr3_results_paper.pdf", "latest_result/latest_result.pdf"], check=True)
    print("PDF compilation and copy complete.")

def main():
    wait_for_pipeline()
    run_comparison_study()
    run_other_latex_tables()
    update_latex_paper()
    compile_pdf()
    print("=== ALL AUTOMATION TASKS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
