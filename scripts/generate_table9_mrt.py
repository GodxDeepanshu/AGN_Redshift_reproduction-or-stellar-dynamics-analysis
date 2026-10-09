#!/usr/bin/env python3
"""
Generate AAS Journal Machine-Readable Table (MRT) and CSV for Table 9
Contains all 410 photometric redshift predictions in the Generalization Set.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load predicted_redshift_DR3
preds_path = PROJECT_DIR / "results" / "predictions" / "predicted_redshift_DR3.csv"
df_pred = pd.read_csv(preds_path)

# Load catalog
cat_path = PROJECT_DIR / "results" / "DR3_BLL_photometric_redshift_catalog.csv"
df_cat = pd.read_csv(cat_path)

# Merge coordinates and quality diagnostics
df = pd.merge(
    df_cat[['Source_Name', 'RA', 'DEC', 'CLASS', 'P_FSRQ', 'Mahalanobis_D_M', 'Reliability_Grade']],
    df_pred[['Source_Name', 'Predicted_z', 'Lower_95', 'Upper_95', 'Interval_Width']],
    on='Source_Name',
    how='left'
)

# Format clean DataFrame
df_clean = pd.DataFrame({
    'Source_Name': df['Source_Name'],
    'RAJ2000_deg': df['RA'].round(4),
    'DEJ2000_deg': df['DEC'].round(4),
    'CLASS': df['CLASS'],
    'P_FSRQ': df['P_FSRQ'].round(4),
    'z_phot': df['Predicted_z'].round(4),
    'z_low_95': df['Lower_95'].round(4),
    'z_upp_95': df['Upper_95'].round(4),
    'interval_width_95': df['Interval_Width'].round(4),
    'Mahalanobis_D_M': df['Mahalanobis_D_M'].round(3),
    'Reliability_Grade': df['Reliability_Grade']
})

# Save CSVs
csv_root = PROJECT_DIR / "table9_machine_readable.csv"
csv_tables = PROJECT_DIR / "results" / "tables" / "table9_machine_readable.csv"
df_clean.to_csv(csv_root, index=False)
df_clean.to_csv(csv_tables, index=False)
print(f"Saved: {csv_root} ({len(df_clean)} sources)")
print(f"Saved: {csv_tables} ({len(df_clean)} sources)")

# CDS / AAS MRT Header
mrt_header = """Title: Bridging the Redshift Gap in the Fermi-LAT Catalog Using Supervised Machine Learning
Authors: Deepanshu Kushwaha and Raj Prince
Table: Table 9. Photometric Redshift Predictions and 95% Conformal Uncertainty Intervals for the Complete Generalization Set of 410 BL Lacs
================================================================================
Byte-by-byte Description of file: table9.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label             Explanations
--------------------------------------------------------------------------------
   1- 18 A18    ---     Source_Name       4FGL source identifier (4FGL JHHMM.m+DDMM)
  20- 28 F9.4   deg     RAJ2000           Right Ascension (J2000, decimal degrees)
  30- 38 F9.4   deg     DEJ2000           Declination (J2000, decimal degrees)
  40- 44 A5     ---     CLASS             Optical blazar subclass (BLL)
  46- 51 F6.4   ---     P_FSRQ            Estimated Bayesian FSRQ subclass probability
  53- 58 F6.4   ---     z_phot            Calibrated photometric redshift
  60- 65 F6.4   ---     z_low_95          95% Conformal lower bound (physically clipped >= 0.0000)
  67- 72 F6.4   ---     z_upp_95          95% Conformal upper bound
  74- 79 F6.4   ---     interval_width_95 95% Conformal prediction interval width
  81- 87 F7.3   ---     Mahalanobis_D_M   Mahalanobis distance to training centroid
  89- 95 A7     ---     Reliability_Grade Reliability grade (Grade A, Grade B, Grade C)
--------------------------------------------------------------------------------
"""

lines = [mrt_header]
for _, r in df_clean.iterrows():
    line = (
        f"{r['Source_Name']:<18} "
        f"{r['RAJ2000_deg']:>9.4f} "
        f"{r['DEJ2000_deg']:>9.4f} "
        f"{r['CLASS']:<5} "
        f"{r['P_FSRQ']:>6.4f} "
        f"{r['z_phot']:>6.4f} "
        f"{r['z_low_95']:>6.4f} "
        f"{r['z_upp_95']:>6.4f} "
        f"{r['interval_width_95']:>6.4f} "
        f"{r['Mahalanobis_D_M']:>7.3f} "
        f"{r['Reliability_Grade']:<7}\n"
    )
    lines.append(line)

mrt_text = "".join(lines)

for fname in ["table9_machine_readable.mrt", "table9_machine_readable.txt"]:
    p_root = PROJECT_DIR / fname
    p_tbl = PROJECT_DIR / "results" / "tables" / fname
    with open(p_root, "w", encoding="utf-8") as f:
        f.write(mrt_text)
    with open(p_tbl, "w", encoding="utf-8") as f:
        f.write(mrt_text)
    print(f"Saved: {p_root}")
    print(f"Saved: {p_tbl}")

print("\nAll Table 9 Machine-Readable Table files successfully created!")
