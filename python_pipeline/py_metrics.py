import numpy as np
from scipy.stats import pearsonr

def compute_bias(observed, predicted):
    """Bias: mean of (observed - predicted)"""
    return np.nanmean(observed - predicted)

def compute_nmad(observed, predicted, normalized=False):
    """
    sigma_NMAD: Normalized Median Absolute Deviation
    sigma_NMAD = 1.4826 * median(|delta - median(delta)|)
    """
    if normalized:
        delta = (observed - predicted) / (1.0 + observed)
    else:
        delta = observed - predicted
    
    # Filter out NaNs
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan
        
    med_delta = np.median(delta)
    mad = np.median(np.abs(delta - med_delta))
    return 1.4826 * mad

def compute_pearson(observed, predicted):
    """Pearson correlation coefficient"""
    mask = np.isfinite(observed) & np.isfinite(predicted)
    obs_clean = observed[mask]
    pred_clean = predicted[mask]
    if len(obs_clean) < 2:
        return np.nan
    r_val, _ = pearsonr(obs_clean, pred_clean)
    return r_val

def compute_rmse(observed, predicted):
    """Root Mean Square Error"""
    diff = observed - predicted
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    return np.sqrt(np.mean(diff ** 2))

def compute_sigma(observed, predicted):
    """Standard deviation of residuals"""
    diff = observed - predicted
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    return np.std(diff, ddof=1)

def compute_outlier_pct_fixed(observed_z, predicted_z):
    """
    Catastrophic outlier percentage:
    Percentage of predictions beyond fixed physical threshold of |delta_z_norm| > 0.15 in linear z scale.
    """
    mask = np.isfinite(observed_z) & np.isfinite(predicted_z)
    obs = observed_z[mask]
    pred = predicted_z[mask]
    if len(obs) == 0:
        return 0.0
        
    delta_z_norm = (pred - obs) / (1.0 + obs)
    outliers = np.abs(delta_z_norm) > 0.15
    return np.mean(outliers) * 100.0

def compute_outlier_pct_2sigma(observed_inv, predicted_inv):
    """
    Catastrophic outlier percentage:
    Percentage of predictions beyond 2 * sigma of the residuals in the 1/(z+1) scale (original paper/R baseline formula).
    """
    residuals = predicted_inv - observed_inv
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 2:
        return 0.0
    sigma = np.std(residuals, ddof=1)
    outliers = np.abs(residuals) > 2.0 * sigma
    return np.mean(outliers) * 100.0

def compute_all_metrics(predicted_inv, observed_inv, predicted_z=None, observed_z=None):
    """
    Compute comprehensive metrics report for both 1/(z+1) and linear z scales
    """
    predicted_inv = np.array(predicted_inv, dtype=float)
    observed_inv = np.array(observed_inv, dtype=float)
    
    if predicted_z is None:
        predicted_z = (1.0 / predicted_inv) - 1.0
    else:
        predicted_z = np.array(predicted_z, dtype=float)
        
    if observed_z is None:
        observed_z = (1.0 / observed_inv) - 1.0
    else:
        observed_z = np.array(observed_z, dtype=float)
        
    delta_z = observed_z - predicted_z
    delta_z_norm = delta_z / (1.0 + observed_z)
    
    metrics = {
        # 1/(z+1) scale metrics
        'inv': {
            'R': compute_pearson(observed_inv, predicted_inv),
            'RMSE': compute_rmse(observed_inv, predicted_inv),
            'bias': compute_bias(observed_inv, predicted_inv),
            'NMAD': compute_nmad(observed_inv, predicted_inv),
            'sigma': compute_sigma(observed_inv, predicted_inv)
        },
        # Linear z scale metrics
        'z': {
            'R': compute_pearson(observed_z, predicted_z),
            'RMSE': compute_rmse(observed_z, predicted_z),
            'bias': compute_bias(observed_z, predicted_z),
            'NMAD': compute_nmad(observed_z, predicted_z, normalized=False),
            'sigma': compute_sigma(observed_z, predicted_z),
            'bias_norm': np.nanmean(delta_z_norm),
            'NMAD_norm': compute_nmad(observed_z, predicted_z, normalized=True),
            'sigma_norm': np.std(delta_z_norm[np.isfinite(delta_z_norm)], ddof=1) if np.sum(np.isfinite(delta_z_norm)) > 1 else np.nan
        },
        'outlier_pct': compute_outlier_pct_fixed(observed_z, predicted_z),
        'outlier_pct_2sigma': compute_outlier_pct_2sigma(observed_inv, predicted_inv)
    }
    return metrics

def print_metrics(metrics, experiment_name="Experiment"):
    """Print metrics report in a structured text layout"""
    print(f"\n=== {experiment_name} Metrics ===")
    
    print("\n--- 1/(z+1) Scale ---")
    print(f"  R (Pearson):   {metrics['inv']['R']:.4f}")
    print(f"  RMSE:          {metrics['inv']['RMSE']:.4f}")
    print(f"  Bias:          {metrics['inv']['bias']:.2e}")
    print(f"  sigma_NMAD:    {metrics['inv']['NMAD']:.4f}")
    print(f"  sigma:         {metrics['inv']['sigma']:.4f}")
    
    print("\n--- Linear z Scale ---")
    print(f"  R (Pearson):   {metrics['z']['R']:.4f}")
    print(f"  RMSE (Dz):     {metrics['z']['RMSE']:.4f}")
    print(f"  Bias (Dz):     {metrics['z']['bias']:.4f}")
    print(f"  sigma_NMAD (Dz): {metrics['z']['NMAD']:.4f}")
    print(f"  sigma (Dz):    {metrics['z']['sigma']:.4f}")
    print(f"  Bias (Dznorm): {metrics['z']['bias_norm']:.2e}")
    print(f"  sigma_NMAD (Dznorm): {metrics['z']['NMAD_norm']:.4f}")
    print(f"  sigma (Dznorm): {metrics['z']['sigma_norm']:.4f}")
    
    print(f"\n  Catastrophic outlier % (Fixed |Dz_norm| > 0.15): {metrics['outlier_pct']:.2f}%")
    if 'outlier_pct_2sigma' in metrics:
        print(f"  Catastrophic outlier % (Circular 2-sigma):        {metrics['outlier_pct_2sigma']:.2f}%")
