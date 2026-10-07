import numpy as np
import pandas as pd
from scipy.stats import linregress
from py_metrics import compute_all_metrics, print_metrics

def fit_optimal_transport(predicted_inv, observed_inv, agn_types, verbose=True):
    """
    Fits Optimal Transport linear mapping: C_preds = B + A * U_phot
    separately for BLL and FSRQ classes.
    """
    ot_params = {}
    
    for agn_type in ["BLL", "FSRQ"]:
        idx = np.where(agn_types == agn_type)[0]
        if len(idx) < 5:
            if verbose:
                print(f"  {agn_type}: Too few samples ({len(idx)}), skipping bias correction.")
            continue
            
        pred_type = predicted_inv[idx]
        obs_type = observed_inv[idx]
        
        # Sort both in ascending order
        pred_sorted = np.sort(pred_type)
        obs_sorted = np.sort(obs_type)
        
        # Ensure equal lengths
        n = min(len(pred_sorted), len(obs_sorted))
        pred_sorted = pred_sorted[:n]
        obs_sorted = obs_sorted[:n]
        
        # Linear fit: observed = A * predicted + B
        slope, intercept, r_value, p_value, std_err = linregress(pred_sorted, obs_sorted)
        
        if verbose:
            print(f"  {agn_type} (n={len(idx)}):")
            print(f"    Slope (A):     {slope:.4f}")
            print(f"    Intercept (B): {intercept:.4f}")
            print(f"    R^2 of fit:    {r_value**2:.4f}")
        
        # Store parameters
        ot_params[agn_type] = {'A': slope, 'B': intercept}
        
    return ot_params

def apply_optimal_transport(predicted_inv, agn_types, ot_params, verbose=True):
    """
    Applies BLL and FSRQ bias correction parameters to predictions.
    """
    corrected = np.array(predicted_inv, dtype=float).copy()
    
    for agn_type in ["BLL", "FSRQ"]:
        if agn_type not in ot_params:
            continue
            
        idx = np.where(agn_types == agn_type)[0]
        if len(idx) == 0:
            continue
            
        slope = ot_params[agn_type]['A']
        intercept = ot_params[agn_type]['B']
        
        corrected[idx] = intercept + slope * predicted_inv[idx]
        if verbose:
            print(f"  Corrected {len(idx)} {agn_type} predictions")
            
    return corrected

def optimal_transport_correction(predicted_inv, observed_inv, agn_types):
    print("Computing Optimal Transport bias correction...")
    ot_params = fit_optimal_transport(predicted_inv, observed_inv, agn_types, verbose=True)
    corrected = apply_optimal_transport(predicted_inv, agn_types, ot_params, verbose=False)
    return corrected, ot_params

def apply_generalization_correction(predicted_inv, agn_types, ot_params):
    return apply_optimal_transport(predicted_inv, agn_types, ot_params, verbose=True)
