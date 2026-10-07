import numpy as np
import pandas as pd
import os
import json
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

from py_model_pipeline import run_python_pipeline
from py_bias_correction import optimal_transport_correction, apply_generalization_correction
from py_metrics import print_metrics, compute_all_metrics
from py_plots import generate_comparison_plots, plot_predicted_vs_observed

def main():
    print("======================================================================")
    # Acting as the user presenting the new Python pipeline
    print("  PYTHON GRADIENT BOOSTING & ADVANCED TABULAR MODELS PIPELINE")
    print("  Comparing XGBoost, LightGBM, CatBoost, Extra Trees, HistGB,")
    print("  NGBoost, TabPFN, TabNet, FT-Transformer and Stacking Ensemble")
    print("======================================================================\n")
    
    # 1. Load Data
    data_dir = "data"
    output_dir = "output/python_results"
    os.makedirs(output_dir, exist_ok=True)
    
    train_eligible_path = os.path.join(data_dir, "training_eligible.csv")
    generalization_path = os.path.join(data_dir, "generalization_set.csv")
    predictor_cols_path = os.path.join(data_dir, "predictor_columns.txt")
    
    if not (os.path.exists(train_eligible_path) and os.path.exists(generalization_path)):
        print("Preprocessed data CSV files not found! Please run 01_download_data.R and 02_preprocess.R first.")
        return
        
    training_data = pd.read_csv(train_eligible_path)
    generalization_data = pd.read_csv(generalization_path)
    
    with open(predictor_cols_path, "r") as f:
        predictors = [line.strip() for line in f if line.strip()]
        
    # Exclude LabelNo if it represents target/class details that shouldn't be a predictor,
    # but since R pipeline used it as predictor_cols, we keep it to maintain exact scientific equivalence.
    print(f"Loaded {len(training_data)} training sources and {len(generalization_data)} generalization sources.")
    print(f"O1 Predictors: {', '.join(predictors)}\n")
    
    # Run Python CV Pipeline with 10 iterations of 10fCV (100 total folds)
    n_iterations = 10
    mean_predictions, agg_predictions, metrics_report, weights_matrix, feature_counts, test_models, pipeline_time = run_python_pipeline(
        training_data, predictors, n_iterations=n_iterations, n_jobs=-1
    )
    
    # 3. Print Metrics for all models
    print("\n" + "="*50)
    print("  MODEL COMPARISONS (CROSS-VALIDATION RESULTS)")
    print("="*50)
    
    # Print metrics for each model
    for model_name in metrics_report.keys():
        print_metrics(metrics_report[model_name], experiment_name=model_name.upper())
        
    # 4. Out-of-fold bias correction has already been nested inside the CV loop.
    # Get predictions
    y_true = training_data['InvRedshift'].values
    y_true_z = training_data['Redshift'].values
    agn_types = training_data['AGN_Type'].values
    
    stack_pred_inv = mean_predictions['stack']
    stack_corrected_inv = mean_predictions['stack_corrected']
    
    stack_uncorrected_z = (1.0 / stack_pred_inv) - 1.0
    stack_corrected_z = (1.0 / stack_corrected_inv) - 1.0
    
    # Load Recalculated R Baseline Metrics
    r_baseline_rec = {}
    r_json_path = os.path.join(data_dir, "../output/r_baseline_metrics_recalculated.json")
    if os.path.exists(r_json_path):
        try:
            with open(r_json_path, "r") as f:
                r_baseline_rec = json.load(f)
        except Exception as e:
            pass

    # 5. Compare with R Baseline Metrics
    print("\n" + "="*95)
    print("  PYTHON VS R BASELINE SUMMARY COMPARISON")
    print("="*95)
    print(f"{'Model':<25} | {'R (linear z)':<14} | {'RMSE (Dz)':<10} | {'Outlier % (Fixed)':<20} | {'Outlier % (2sigma)':<20}")
    print("-"*100)
    
    # Display SL-O1 baseline
    if "SL_O1" in r_baseline_rec:
        sl = r_baseline_rec["SL_O1"]
        print(f"{'R Baseline (SL-O1)':<25} | {sl['R_z']:<14.3f} | {sl['RMSE_z']:<10.3f} | {sl['outlier_pct_fixed']:<18.1f}% | {sl['outlier_pct_2sigma']:<18.1f}%")
    else:
        print(f"{'R Baseline (SL-O1)':<25} | {0.742:<14.3f} | {0.457:<10.3f} | {'43.5':<18}% | {'5.3':<18}%")
        
    # Display SL-O1 BC baseline
    if "SL_O1_BC" in r_baseline_rec:
        sl_bc = r_baseline_rec["SL_O1_BC"]
        print(f"{'R Baseline (SL-O1 BC)':<25} | {sl_bc['R_z']:<14.3f} | {sl_bc['RMSE_z']:<10.3f} | {sl_bc['outlier_pct_fixed']:<18.1f}% | {sl_bc['outlier_pct_2sigma']:<18.1f}%")
    
    for m in test_models + ['stack']:
        m_r = metrics_report[m]['z']['R']
        m_rmse = metrics_report[m]['z']['RMSE']
        m_out_fixed = metrics_report[m]['outlier_pct']
        m_out_2s = metrics_report[m]['outlier_pct_2sigma']
        print(f"{m.upper() + ' (Python)':<25} | {m_r:<14.3f} | {m_rmse:<10.3f} | {m_out_fixed:<18.1f}% | {m_out_2s:<18.1f}%")
        
        # Show bias-corrected metrics
        m_corr = m + '_corrected'
        if m_corr in metrics_report:
            m_r_c = metrics_report[m_corr]['z']['R']
            m_rmse_c = metrics_report[m_corr]['z']['RMSE']
            m_out_fixed_c = metrics_report[m_corr]['outlier_pct']
            m_out_2s_c = metrics_report[m_corr]['outlier_pct_2sigma']
            print(f"{(m + '_BC').upper() + ' (Python)':<25} | {m_r_c:<14.3f} | {m_rmse_c:<10.3f} | {m_out_fixed_c:<18.1f}% | {m_out_2s_c:<18.1f}%")
    
    # Calculate speedup
    r_time_mins = 59.9
    py_time_mins = pipeline_time / 60.0
    speedup = r_time_mins / py_time_mins
    print(f"\nExecution Efficiency Summary:")
    print(f"  R Pipeline Execution Time:      {r_time_mins:.2f} minutes")
    print(f"  Python Pipeline Execution Time: {py_time_mins:.2f} minutes")
    print(f"  COMPUTATIONAL SPEEDUP FACTOR:   {speedup:.1f}x faster!")
    
    # 6. Generate Generalization Set Predictions
    print("\nTraining final Stacking model on full training dataset for generalization predictions...")
    
    # Train final models on full train eligible dataset
    X_train_full = training_data[predictors]
    y_train_full = training_data['InvRedshift']
    X_gen = generalization_data[predictors]
    
    # Train base models on full training data and predict on both training and generalization
    from py_model_pipeline import train_and_predict_base_models
    train_preds_dict, gen_preds_dict = train_and_predict_base_models(
        X_train_full, y_train_full, X_gen, iter_seed=42, predict_train=True
    )
    
    # Stacking prediction using average of fold weights
    avg_weights = np.mean(weights_matrix, axis=0) # shape (len(test_models),)
    print(f"Average Stacking Weights: " + ", ".join([f"{test_models[idx]}: {avg_weights[idx]:.3f}" for idx in range(len(test_models))]))
    
    train_stack_pred_inv = np.zeros(len(X_train_full))
    gen_stack_pred_inv = np.zeros(len(X_gen))
    for idx, m in enumerate(test_models):
        if m in gen_preds_dict:
            gen_stack_pred_inv += avg_weights[idx] * gen_preds_dict[m]
        if m in train_preds_dict:
            train_stack_pred_inv += avg_weights[idx] * train_preds_dict[m]
            
    # Fit OT bias correction parameters on full training predictions (in-sample) to avoid leakage
    from py_bias_correction import fit_optimal_transport, apply_optimal_transport
    ot_params = fit_optimal_transport(train_stack_pred_inv, y_train_full.values, training_data['AGN_Type'].values, verbose=True)
    
    # Apply OT bias correction to generalization predictions
    gen_corrected_inv = apply_optimal_transport(
        gen_stack_pred_inv, generalization_data['AGN_Type'].values, ot_params, verbose=True
    )
    
    gen_uncorrected_z = (1.0 / gen_stack_pred_inv) - 1.0
    gen_corrected_z = (1.0 / gen_corrected_inv) - 1.0
    
    # Create Table 3 predictions dataframe
    name_col = next((c for c in ["Source_Name", "Source_name", "NAME"] if c in generalization_data.columns), None)
    if name_col:
        table3_df = pd.DataFrame({
            'Name': generalization_data[name_col],
            'Predicted_z': np.round(gen_uncorrected_z, 2),
            'Bias_Corrected_z': np.round(gen_corrected_z, 2)
        })
    else:
        table3_df = pd.DataFrame({
            'Index': np.arange(len(X_gen)) + 1,
            'Predicted_z': np.round(gen_uncorrected_z, 2),
            'Bias_Corrected_z': np.round(gen_corrected_z, 2)
        })
        
    # Save Table 3 equivalent
    table3_df.to_csv(os.path.join(output_dir, "Table3_python_predictions.csv"), index=False)
    print(f"Saved generalization predictions to '{output_dir}/Table3_python_predictions.csv'")
    
    # 7. Save pipeline results
    python_results = {
        'mean_predictions': mean_predictions,
        'metrics_report': metrics_report,
        'runtimes': {
            'total_pipeline': pipeline_time
        },
        'weights_matrix': weights_matrix,
        'feature_counts': feature_counts,
        'ot_params': ot_params
    }
    
    with open(os.path.join(output_dir, "python_results.pkl"), "wb") as f:
        pickle.dump(python_results, f)
    print(f"Saved results dictionary to '{output_dir}/python_results.pkl'")
    
    # 8. Generate Visualizations
    generate_comparison_plots(metrics_report, {'total_pipeline': pipeline_time}, {}, output_dir="plots/python_plots")
    
    # Plot predicted vs observed scatter plots
    plot_predicted_vs_observed(y_true_z, stack_uncorrected_z, "Stacking_Regressor_Uncorrected", output_dir="plots/python_plots")
    plot_predicted_vs_observed(y_true_z, stack_corrected_z, "Stacking_Regressor_Bias_Corrected", output_dir="plots/python_plots")
    
    print("\n======================================================================")
    print("  PYTHON REPRODUCTION PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    main()
