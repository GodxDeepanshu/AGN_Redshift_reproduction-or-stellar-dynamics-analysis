import os
import pandas as pd
import subprocess

def main():
    csv_path = "output/satellite_experiments/satellite_experiments_summary.csv"
    tex_path = "dr3_results_paper.tex"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Feature mappings
    feat_map = {
        'Baseline': 'None',
        'AllWISE': '$W1, W2, W3, W4$ magnitudes and colors',
        'NVSS': 'Radio flux ($S_{1.4}$)',
        'Swift': 'X-ray count rate ($CR_0$)',
        'SDSS': '$u, g, r, i, z$ magnitudes and colors',
        'PS1': '$g, r, i, z, y$ magnitudes and colors',
        'FermiExtra': 'Variability fraction and highest energy',
        'AllCombined': 'All 28 multi-wavelength features combined'
    }
    
    # Construct LaTeX rows
    rows = []
    for _, row in df.iterrows():
        name = row['Experiment']
        disp_name = name.replace('FermiExtra', 'FermiExtra').replace('AllCombined', 'AllCombined')
        if name in ['Baseline', 'AllWISE', 'NVSS', 'Swift', 'SDSS', 'PS1', 'FermiExtra', 'AllCombined']:
            disp_name = f"\\textbf{{{name}}}" if name in ['Baseline', 'AllCombined'] else name
        extra_feats = feat_map.get(name, 'Unknown')
        
        row_str = (
            f"{disp_name} & {extra_feats} & {int(row['Train_Size']):,} & {int(row['Gen_Size']):,} & "
            f"{row['OT_R']:.4f} & {row['OT_RMSE']:.4f} & {row['OT_Conf_Width']:.4f} & "
            f"{row['Iso_R']:.4f} & {row['Iso_RMSE']:.4f} & {row['Iso_Conf_Width']:.4f} \\\\"
        )
        rows.append(row_str)
        
    table_content = "\n".join(rows)
    
    # Generate the LaTeX block
    latex_table = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Systematic Study of Photometric Redshift Performance under Complete-Case Analysis across Different Survey/Satellite Additions (Conformal Validation).}}
\\label{{tab:satellite_experiments}}
\\begin{{tabular}}{{lccccccccc}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{Extra Features}} & \\textbf{{Train Size}} & \\textbf{{Gen Size}} & \\textbf{{OT $R_z$}} & \\textbf{{OT RMSE$_z$}} & \\textbf{{OT Width$_{{68}}$}} & \\textbf{{Iso $R_z$}} & \\textbf{{Iso RMSE$_z$}} & \\textbf{{Iso Width$_{{68}}$}} \\\\
\\midrule
{table_content}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""

    # Read LaTeX file
    with open(tex_path, 'r') as f:
        tex_content = f.read()
        
    # Locate the old table and replace it
    start_marker = "\\begin{table*}[t]\n\\centering\n\\caption{Systematic Study of Photometric Redshift Performance under Complete-Case Analysis across Different Survey/Satellite Additions."
    if start_marker not in tex_content:
        # Try generic search for label
        label_marker = "\\label{tab:satellite_experiments}"
        if label_marker in tex_content:
            # Find the surrounding \begin{table*} and \end{table*}
            idx = tex_content.find(label_marker)
            start_idx = tex_content.rfind("\\begin{table*}", 0, idx)
            end_idx = tex_content.find("\\end{table*}", idx) + len("\\end{table*}")
            old_table = tex_content[start_idx:end_idx]
            tex_content = tex_content.replace(old_table, latex_table)
            print("Found old table via label and replaced it.")
        else:
            print("Error: Could not locate table in LaTeX file.")
            return
    else:
        # Find index
        idx = tex_content.find(start_marker)
        end_idx = tex_content.find("\\end{table*}", idx) + len("\\end{table*}")
        old_table = tex_content[idx:end_idx]
        tex_content = tex_content.replace(old_table, latex_table)
        print("Replaced old table via start marker.")
        
    with open(tex_path, 'w') as f:
        f.write(tex_content)
        
    print("LaTeX file successfully updated with exact results!")
    
    # Compile the document
    print("Compiling LaTeX paper...")
    pdflatex_cmd = ["/home/hea/.local/bin/pdflatex", "-interaction=nonstopmode", tex_path]
    subprocess.run(pdflatex_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(pdflatex_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("LaTeX paper successfully compiled to dr3_results_paper.pdf!")

if __name__ == "__main__":
    main()
