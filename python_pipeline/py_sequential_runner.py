"""
Sequential Model Runner - Runs each ML model independently through 100x10-fold CV.
Each model completes fully before the next starts, reducing CPU/memory pressure.
Intermediate results are saved after each model, so progress is never lost.
"""
import numpy as np
import pandas as pd
import time
import os
import pickle
import warnings
import sys
warnings.filterwarnings("ignore")

# Add python_pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from py_metrics import compute_all_metrics, print_metrics
from py_bias_correction import fit_optimal_transport, apply_optimal_transport

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

class LGBMEarlyStoppingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=1000, learning_rate=0.03, max_depth=4, num_leaves=15,
                 subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_ = None
        
    def fit(self, X, y):
        import lightgbm as lgb
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y
            
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_arr, test_size=0.15, random_state=self.random_state
        )
        
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1
        )
        
        self.model_.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
        )
        return self
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        return self.model_.predict(X_arr)

INTERMEDIATE_DIR = "output/python_results/intermediate"

# ─── Optuna Tuning Helper ──────────────────────────────────────────────────

def run_optuna_tuning(model_name, training_data, predictors, output_dir):
    """Runs a 20-trial Optuna hyperparameter optimization study on the training data."""
    import optuna
    import json
    
    params_path = os.path.join(output_dir, f"{model_name}_best_params.json")
    if os.path.exists(params_path):
        print(f"  [OK] Tuned parameters for {model_name.upper()} already exist -- loading from cache.")
        with open(params_path, "r") as f:
            return json.load(f)
            
    print(f"\nRunning Optuna hyperparameter tuning for {model_name.upper()}...")
    
    y = training_data['InvRedshift'].values
    X = training_data[predictors].values
    
    # Simple 5-fold CV to evaluate hyperparameters
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        if model_name == 'xgb':
            import xgboost as xgb
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'n_jobs': 1,
                'random_state': 42
            }
            model_class = xgb.XGBRegressor
        elif model_name == 'lgb':
            import lightgbm as lgb
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 7, 63),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'n_jobs': 1,
                'verbose': -1,
                'random_state': 42
            }
            model_class = lgb.LGBMRegressor
        elif model_name == 'cat':
            import catboost as cb
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 3, 8),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'thread_count': 1,
                'verbose': 0,
                'random_seed': 42
            }
            model_class = cb.CatBoostRegressor
        else:
            raise ValueError(f"Unknown model for tuning: {model_name}")
            
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mse = np.mean((y_val - preds) ** 2)
            scores.append(mse)
            
        return np.mean(scores)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"  [OK] Saved best parameters for {model_name.upper()} to {params_path}")
    print(f"  Best trial value (MSE): {study.best_value:.6f}")
    return best_params


# ─── Model Definitions ───────────────────────────────────────────────────────

def get_model_instance(model_name, seed):
    """Returns a fresh model instance for the given model name."""
    global INTERMEDIATE_DIR
    if model_name == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1
        )
    elif model_name == 'xgb_opt':
        import xgboost as xgb
        import json
        params_path = os.path.join(INTERMEDIATE_DIR, "xgb_best_params.json")
        with open(params_path, "r") as f:
            best_params = json.load(f)
        for k in ['max_depth', 'n_estimators']:
            if k in best_params:
                best_params[k] = int(best_params[k])
        best_params['random_state'] = seed
        best_params['n_jobs'] = 1
        return xgb.XGBRegressor(**best_params)
    elif model_name == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
        )
    elif model_name == 'lgb_opt':
        import lightgbm as lgb
        import json
        params_path = os.path.join(INTERMEDIATE_DIR, "lgb_best_params.json")
        with open(params_path, "r") as f:
            best_params = json.load(f)
        for k in ['max_depth', 'n_estimators', 'num_leaves']:
            if k in best_params:
                best_params[k] = int(best_params[k])
        best_params['random_state'] = seed
        best_params['n_jobs'] = 1
        best_params['verbose'] = -1
        return lgb.LGBMRegressor(**best_params)
    elif model_name == 'lgb_es':
        return LGBMEarlyStoppingRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, random_state=seed
        )
    elif model_name == 'cat':
        import catboost as cb
        return cb.CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=4,
            random_seed=seed, thread_count=1, verbose=0
        )
    elif model_name == 'cat_opt':
        import catboost as cb
        import json
        params_path = os.path.join(INTERMEDIATE_DIR, "cat_best_params.json")
        with open(params_path, "r") as f:
            best_params = json.load(f)
        for k in ['depth', 'iterations']:
            if k in best_params:
                best_params[k] = int(best_params[k])
        best_params['random_seed'] = seed
        best_params['thread_count'] = 1
        best_params['verbose'] = 0
        return cb.CatBoostRegressor(**best_params)
    elif model_name == 'et':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=6, random_state=seed, n_jobs=1
        )
    elif model_name == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_depth=4, random_state=seed
        )
    elif model_name == 'ngb':
        import ngboost as ngb_lib
        return ngb_lib.NGBRegressor(
            n_estimators=200, learning_rate=0.03, random_state=seed, verbose=False
        )
    elif model_name == 'tabpfn':
        from tabpfn import TabPFNRegressor
        os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_vhHlzZBwVf2AHO1kFY4QO1d7or-yb5murRwl67_weqo'
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
        return TabPFNRegressor(random_state=seed, ignore_pretraining_limits=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── Feature Selection (same as original) ────────────────────────────────────

def lasso_feature_selection_1se(X, y, cv=10, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso_cv = LassoCV(cv=cv, random_state=random_state, max_iter=5000, n_jobs=1)
    lasso_cv.fit(X_scaled, y)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    se_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(cv)
    min_idx = np.argmin(mean_mse)
    threshold = mean_mse[min_idx] + se_mse[min_idx]
    selected_idx = min_idx
    for idx in range(len(lasso_cv.alphas_)):
        if mean_mse[idx] <= threshold:
            selected_idx = idx
            break
    alpha_1se = lasso_cv.alphas_[selected_idx]
    lasso_final = Lasso(alpha=alpha_1se, random_state=random_state, max_iter=5000)
    lasso_final.fit(X_scaled, y)
    selected_features = [X.columns[i] for i, coef in enumerate(lasso_final.coef_) if coef != 0]
    return selected_features


# ─── Single Model CV Runner ──────────────────────────────────────────────────

def run_single_model_single_iteration(iter_seed, training_data, y, predictors, model_name, n_folds=10):
    """
    Runs a single model through one iteration of 10-fold CV with LASSO feature selection
    and out-of-fold bias correction. Returns OOF predictions and feature counts.
    """
    np.random.seed(iter_seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=iter_seed)
    
    oof_pred = np.full(len(y), np.nan)
    oof_corrected = np.full(len(y), np.nan)
    feature_counts = {p: 0 for p in predictors}
    
    X = training_data[predictors]
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        agn_train = training_data.iloc[train_idx]['AGN_Type'].values
        agn_test = training_data.iloc[test_idx]['AGN_Type'].values
        
        # LASSO feature selection
        try:
            selected_features = lasso_feature_selection_1se(X_train[predictors], y_train, cv=5, random_state=iter_seed)
        except:
            selected_features = list(predictors)
        if len(selected_features) == 0:
            selected_features = list(predictors)
        for f in selected_features:
            feature_counts[f] += 1
        
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # Train model and predict
        try:
            model = get_model_instance(model_name, iter_seed)
            model.fit(X_train_sel, y_train)
            test_pred = model.predict(X_test_sel)
            train_pred = model.predict(X_train_sel)
            
            oof_pred[test_idx] = test_pred
            
            # Bias correction
            try:
                ot_params = fit_optimal_transport(train_pred, y_train.values, agn_train, verbose=False)
                corrected = apply_optimal_transport(test_pred, agn_test, ot_params, verbose=False)
                oof_corrected[test_idx] = corrected
            except:
                oof_corrected[test_idx] = test_pred
        except Exception as e:
            print(f"  [WARN] {model_name} failed on iter {iter_seed}: {e}")
            oof_pred[test_idx] = np.nanmean(y.values)
            oof_corrected[test_idx] = np.nanmean(y.values)
    
    return oof_pred, oof_corrected, feature_counts


def run_model_cv(model_name, training_data, predictors, n_iterations=100, n_jobs=-1, output_dir="output/python_results/intermediate"):
    """
    Runs a single model through the full 100x10-fold CV pipeline.
    Saves results to disk when complete.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{model_name}_cv_results.pkl")
    
    # Check if already completed
    if os.path.exists(result_path):
        print(f"  [OK] {model_name.upper()} already completed -- loading from cache.")
        with open(result_path, "rb") as f:
            return pickle.load(f)
    
    y = training_data['InvRedshift']
    n_samples = len(y)
    
    print(f"\n{'='*60}")
    print(f"  RUNNING: {model_name.upper()} -- {n_iterations} iterations x 10-fold CV")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Run all iterations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_model_single_iteration)(
            i + 1, training_data, y, predictors, model_name
        )
        for i in range(n_iterations)
    )
    
    elapsed = time.time() - start
    
    # Aggregate
    all_pred = np.zeros((n_samples, n_iterations))
    all_corrected = np.zeros((n_samples, n_iterations))
    total_feat_counts = {p: 0 for p in predictors}
    
    for i, (pred, corrected, feat_counts) in enumerate(results):
        all_pred[:, i] = pred
        all_corrected[:, i] = corrected
        for p in predictors:
            total_feat_counts[p] += feat_counts[p]
    
    mean_pred = np.mean(all_pred, axis=1)
    mean_corrected = np.mean(all_corrected, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(mean_pred, y)
    metrics_corrected = compute_all_metrics(mean_corrected, y)
    
    result = {
        'model_name': model_name,
        'all_pred': all_pred,
        'all_corrected': all_corrected,
        'mean_pred': mean_pred,
        'mean_corrected': mean_corrected,
        'metrics': metrics,
        'metrics_corrected': metrics_corrected,
        'feature_counts': total_feat_counts,
        'elapsed': elapsed
    }
    
    # Save to disk
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    
    print(f"  [OK] {model_name.upper()} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print_metrics(metrics, experiment_name=f"{model_name.upper()}")
    print_metrics(metrics_corrected, experiment_name=f"{model_name.upper()} (Bias Corrected)")
    
    return result


# ─── Stacking Combiner ───────────────────────────────────────────────────────

def run_stacking_from_individual_results(model_results, training_data, predictors, n_iterations=100, stacking_models=None, meta_learner='nnls'):
    """
    Combines individual model CV results into a stacking ensemble.
    Uses NNLS or RidgeCV weights fitted on internal CV of the stacking models.
    """
    from scipy.optimize import minimize
    from sklearn.linear_model import RidgeCV
    
    y = training_data['InvRedshift']
    test_models = list(model_results.keys())
    if stacking_models is None:
        stacking_models = ['xgb', 'lgb', 'cat', 'et', 'hgb']
    n_samples = len(y)
    
    print(f"\n{'='*60}")
    print(f"  BUILDING STACKING ENSEMBLE ({meta_learner.upper()}) FROM {len(test_models)} MODELS")
    print(f"{'='*60}")
    
    # For each iteration, compute stacking weights using internal CV
    all_stack_pred = np.zeros((n_samples, n_iterations))
    all_stack_corrected = np.zeros((n_samples, n_iterations))
    all_weights = []
    
    for iteration in range(n_iterations):
        iter_seed = iteration + 1
        np.random.seed(iter_seed)
        kf = KFold(n_splits=10, shuffle=True, random_state=iter_seed)
        
        iter_stack = np.full(n_samples, np.nan)
        iter_stack_corrected = np.full(n_samples, np.nan)
        
        for train_idx, test_idx in kf.split(np.arange(n_samples)):
            meta_X_train = np.column_stack([
                model_results[m]['all_pred'][train_idx, iteration] for m in stacking_models
                if m in model_results
            ])
            active_stacking = [m for m in stacking_models if m in model_results]
            y_train_fold = y.values[train_idx]
            
            if meta_learner == 'ridge':
                meta_model = RidgeCV(cv=5, fit_intercept=True)
                meta_model.fit(meta_X_train, y_train_fold)
                weights = meta_model.coef_
                
                meta_X_test = np.column_stack([
                    model_results[m]['all_pred'][test_idx, iteration] for m in stacking_models
                    if m in model_results
                ])
                fold_stack = meta_model.predict(meta_X_test)
                fold_stack_train = meta_model.predict(meta_X_train)
            else:
                # Fit NNLS weights
                n_m = meta_X_train.shape[1]
                def objective(w):
                    return np.mean((y_train_fold - np.dot(meta_X_train, w)) ** 2)
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                bounds = [(0.0, 1.0)] * n_m
                w0 = np.ones(n_m) / n_m
                res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
                weights = res.x
                
                # Compute stacked predictions on test fold
                fold_stack = np.zeros(len(test_idx))
                fold_stack_train = np.zeros(len(train_idx))
                for mi, m in enumerate(test_models):
                    fold_stack += weights[active_stacking.index(m)] * model_results[m]['all_pred'][test_idx, iteration] if m in active_stacking else 0.0
                    fold_stack_train += weights[active_stacking.index(m)] * model_results[m]['all_pred'][train_idx, iteration] if m in active_stacking else 0.0
            
            # Map weights to full model list
            full_weights = np.zeros(len(test_models))
            for wi, m in enumerate(active_stacking):
                full_weights[test_models.index(m)] = weights[wi]
            all_weights.append(full_weights)
            
            iter_stack[test_idx] = fold_stack
            
            # Bias correction for stacking
            agn_train = training_data.iloc[train_idx]['AGN_Type'].values
            agn_test = training_data.iloc[test_idx]['AGN_Type'].values
            try:
                ot_params = fit_optimal_transport(fold_stack_train, y_train_fold, agn_train, verbose=False)
                corrected = apply_optimal_transport(fold_stack, agn_test, ot_params, verbose=False)
                iter_stack_corrected[test_idx] = corrected
            except:
                iter_stack_corrected[test_idx] = fold_stack
        
        all_stack_pred[:, iteration] = iter_stack
        all_stack_corrected[:, iteration] = iter_stack_corrected
    
    # Average across iterations
    mean_stack = np.mean(all_stack_pred, axis=1)
    mean_stack_corrected = np.mean(all_stack_corrected, axis=1)
    
    metrics_stack = compute_all_metrics(mean_stack, y)
    metrics_stack_corrected = compute_all_metrics(mean_stack_corrected, y)
    
    print_metrics(metrics_stack, experiment_name="STACKING ENSEMBLE")
    print_metrics(metrics_stack_corrected, experiment_name="STACKING ENSEMBLE (Bias Corrected)")
    
    return {
        'mean_stack': mean_stack,
        'mean_stack_corrected': mean_stack_corrected,
        'metrics_stack': metrics_stack,
        'metrics_stack_corrected': metrics_stack_corrected,
        'weights_matrix': np.array(all_weights),
        'all_stack_pred': all_stack_pred,
        'all_stack_corrected': all_stack_corrected
    }


# ─── Main Sequential Pipeline ────────────────────────────────────────────────

def main():
    import json
    import argparse
    from py_plots import generate_comparison_plots, plot_predicted_vs_observed
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--models', type=str, default='hgb,et,lgb,xgb,cat,xgb_opt,lgb_opt,cat_opt,lgb_es,ngb,tabpfn', help='Comma-separated models to run')
    parser.add_argument('--stacking-models', type=str, default='xgb,lgb,cat,et,hgb,xgb_opt,lgb_opt,cat_opt,lgb_es', help='Comma-separated models for stacking')
    parser.add_argument('--stacking-only', action='store_true', help='Only rebuild stacking ensemble from cached results')
    parser.add_argument('--output-dir', type=str, default='output/python_results', help='Output directory')
    parser.add_argument('--meta-learner', type=str, default='nnls', choices=['nnls', 'ridge'], help='Meta-learner algorithm')
    args = parser.parse_args()
    
    n_iterations = args.iterations
    requested_models = args.models.split(',')
    stacking_models = args.stacking_models.split(',')
    
    print("=" * 70)
    print("  SEQUENTIAL PYTHON PIPELINE -- One Model at a Time")
    print(f"  Iterations: {n_iterations} | Stacking Models: {', '.join(stacking_models)}")
    print("=" * 70)
    
    # 1. Load Data
    data_dir = "data"
    output_dir = args.output_dir
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    
    global INTERMEDIATE_DIR
    INTERMEDIATE_DIR = intermediate_dir
    
    training_data = pd.read_csv(os.path.join(data_dir, "training_eligible.csv"))
    generalization_data = pd.read_csv(os.path.join(data_dir, "generalization_set.csv"))
    with open(os.path.join(data_dir, "predictor_columns.txt"), "r") as f:
        predictors = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(training_data)} training, {len(generalization_data)} generalization sources.")
    print(f"Predictors: {', '.join(predictors)}\n")
    
    # Run Optuna tuning before sequentially training models
    if not args.stacking_only:
        for model_name in ['xgb', 'lgb', 'cat']:
            run_optuna_tuning(model_name, training_data, predictors, intermediate_dir)
            
    # 2. Define models to run (sequential order — fast models first)
    available_models = ['xgb', 'lgb', 'cat', 'et', 'hgb', 'xgb_opt', 'lgb_opt', 'cat_opt', 'lgb_es']
    try:
        import ngboost; available_models.append('ngb')
    except ImportError:
        print("  [SKIP] NGBoost not installed")
    try:
        from tabpfn import TabPFNRegressor; available_models.append('tabpfn')
    except ImportError:
        print("  [SKIP] TabPFN not installed")
    
    models_to_run = [m for m in requested_models if m in available_models]
    print(f"Models to run sequentially: {', '.join(m.upper() for m in models_to_run)}")
    
    # 3. Run each model sequentially
    model_results = {}
    total_start = time.time()
    
    if args.stacking_only:
        print("\nStacking-only mode: Loading all results from cache...")
        for model_name in models_to_run:
            result_path = os.path.join(intermediate_dir, f"{model_name}_cv_results.pkl")
            if os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    model_results[model_name] = pickle.load(f)
                    print(f"  Loaded {model_name.upper()} from cache.")
            else:
                print(f"  [WARN] Cache for {model_name.upper()} not found at {result_path}!")
    else:
        for i, model_name in enumerate(models_to_run):
            print(f"\n[{i+1}/{len(models_to_run)}] Processing {model_name.upper()}...")
            result = run_model_cv(
                model_name, training_data, predictors,
                n_iterations=n_iterations, n_jobs=-1, output_dir=intermediate_dir
            )
            model_results[model_name] = result
            
            # Print running total
            elapsed_so_far = time.time() - total_start
            remaining = len(models_to_run) - (i + 1)
            avg_per_model = elapsed_so_far / (i + 1)
            print(f"\n  Time: Total elapsed: {elapsed_so_far/60:.1f} min | Est. remaining: {remaining * avg_per_model/60:.1f} min")
    
    # Filter stacking models to only those that exist in our loaded results
    stacking_models_active = [m for m in stacking_models if m in model_results]
    print(f"Active stacking models: {', '.join(m.upper() for m in stacking_models_active)}")
    
    # 4. Build Stacking Ensemble
    stacking_result = run_stacking_from_individual_results(
        model_results, training_data, predictors, n_iterations=n_iterations,
        stacking_models=stacking_models_active, meta_learner=args.meta_learner
    )
    
    total_elapsed = time.time() - total_start
    
    # 5. Assemble final results (matching original pipeline format)
    y = training_data['InvRedshift']
    y_true_z = training_data['Redshift'].values
    test_models = list(model_results.keys())
    
    mean_predictions = {}
    metrics_report = {}
    total_feature_counts = {p: 0 for p in predictors}
    
    for m in test_models:
        r = model_results[m]
        mean_predictions[m] = r['mean_pred']
        mean_predictions[m + '_corrected'] = r['mean_corrected']
        metrics_report[m] = r['metrics']
        metrics_report[m + '_corrected'] = r['metrics_corrected']
        for p in predictors:
            total_feature_counts[p] += r['feature_counts'][p]
    
    mean_predictions['stack'] = stacking_result['mean_stack']
    mean_predictions['stack_corrected'] = stacking_result['mean_stack_corrected']
    metrics_report['stack'] = stacking_result['metrics_stack']
    metrics_report['stack_corrected'] = stacking_result['metrics_stack_corrected']
    
    # 6. Print comparison table
    print("\n" + "=" * 95)
    print("  PYTHON VS R BASELINE SUMMARY COMPARISON")
    print("=" * 95)
    print(f"{'Model':<25} | {'R (linear z)':<14} | {'RMSE (Dz)':<10} | {'Outlier % (Fixed)':<20} | {'Outlier % (2sigma)':<20}")
    print("-" * 100)
    
    # R baseline
    r_json_path = os.path.join(data_dir, "../output/r_baseline_metrics_recalculated.json")
    if os.path.exists(r_json_path):
        try:
            with open(r_json_path, "r") as f:
                r_baseline = json.load(f)
            if "SL_O1" in r_baseline:
                sl = r_baseline["SL_O1"]
                print(f"{'R Baseline (SL-O1)':<25} | {sl['R_z']:<14.3f} | {sl['RMSE_z']:<10.3f} | {sl['outlier_pct_fixed']:<18.1f}% | {sl['outlier_pct_2sigma']:<18.1f}%")
            if "SL_O1_BC" in r_baseline:
                sl_bc = r_baseline["SL_O1_BC"]
                print(f"{'R Baseline (SL-O1 BC)':<25} | {sl_bc['R_z']:<14.3f} | {sl_bc['RMSE_z']:<10.3f} | {sl_bc['outlier_pct_fixed']:<18.1f}% | {sl_bc['outlier_pct_2sigma']:<18.1f}%")
        except:
            pass
    
    for m in test_models + ['stack']:
        mr = metrics_report[m]
        print(f"{m.upper() + ' (Python)':<25} | {mr['z']['R']:<14.3f} | {mr['z']['RMSE']:<10.3f} | {mr['outlier_pct']:<18.1f}% | {mr['outlier_pct_2sigma']:<18.1f}%")
        mc = metrics_report.get(m + '_corrected')
        if mc:
            print(f"{(m + '_BC').upper() + ' (Python)':<25} | {mc['z']['R']:<14.3f} | {mc['z']['RMSE']:<10.3f} | {mc['outlier_pct']:<18.1f}% | {mc['outlier_pct_2sigma']:<18.1f}%")
    
    print(f"\nTotal Pipeline Time: {total_elapsed/60:.1f} minutes")
    
    # 7. Generalization predictions
    print("\nTraining final models on full training dataset for generalization predictions...")
    
    def local_train_and_predict_base_models(X_train, y_train, X_test, iter_seed, models_to_run=None, predict_train=False):
        train_predictions = {}
        predictions = {}
        for m in models_to_run:
            try:
                model = get_model_instance(m, iter_seed)
                try:
                    selected_features = lasso_feature_selection_1se(X_train, y_train, cv=5, random_state=iter_seed)
                except Exception as e:
                    selected_features = list(X_train.columns)
                if len(selected_features) == 0:
                    selected_features = list(X_train.columns)
                
                model.fit(X_train[selected_features], y_train)
                predictions[m] = model.predict(X_test[selected_features])
                if predict_train:
                    train_predictions[m] = model.predict(X_train[selected_features])
            except Exception as e:
                print(f"  [WARN] {m} failed in final prediction step: {e}")
                predictions[m] = np.zeros(len(X_test))
                if predict_train:
                    train_predictions[m] = np.zeros(len(X_train))
        if predict_train:
            return train_predictions, predictions
        return predictions

    X_train_full = training_data[predictors]
    y_train_full = training_data['InvRedshift']
    X_gen = generalization_data[predictors]
    
    train_preds_dict, gen_preds_dict = local_train_and_predict_base_models(
        X_train_full, y_train_full, X_gen, iter_seed=42, predict_train=True, models_to_run=test_models
    )
    
    avg_weights = np.mean(stacking_result['weights_matrix'], axis=0)
    print(f"Average Stacking Weights: " + ", ".join([f"{test_models[idx]}: {avg_weights[idx]:.3f}" for idx in range(len(test_models))]))
    
    train_stack_pred_inv = np.zeros(len(X_train_full))
    gen_stack_pred_inv = np.zeros(len(X_gen))
    for idx, m in enumerate(test_models):
        if m in gen_preds_dict:
            gen_stack_pred_inv += avg_weights[idx] * gen_preds_dict[m]
        if m in train_preds_dict:
            train_stack_pred_inv += avg_weights[idx] * train_preds_dict[m]
    
    ot_params = fit_optimal_transport(train_stack_pred_inv, y_train_full.values, training_data['AGN_Type'].values, verbose=True)
    gen_corrected_inv = apply_optimal_transport(gen_stack_pred_inv, generalization_data['AGN_Type'].values, ot_params, verbose=True)
    
    gen_uncorrected_z = (1.0 / gen_stack_pred_inv) - 1.0
    gen_corrected_z = (1.0 / gen_corrected_inv) - 1.0
    
    # Save Table 3
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
    
    table3_df.to_csv(os.path.join(output_dir, "Table3_python_predictions.csv"), index=False)
    print(f"Saved: {output_dir}/Table3_python_predictions.csv")
    
    # 8. Save full results
    python_results = {
        'mean_predictions': mean_predictions,
        'metrics_report': metrics_report,
        'runtimes': {'total_pipeline': total_elapsed},
        'weights_matrix': stacking_result['weights_matrix'],
        'feature_counts': total_feature_counts,
        'ot_params': ot_params,
        'all_stack_pred': stacking_result['all_stack_pred'],
        'all_stack_corrected': stacking_result['all_stack_corrected']
    }
    with open(os.path.join(output_dir, "python_results.pkl"), "wb") as f:
        pickle.dump(python_results, f)
    print(f"Saved: {output_dir}/python_results.pkl")
    
    # 9. Plots
    stack_uncorrected_z = (1.0 / mean_predictions['stack']) - 1.0
    stack_corrected_z = (1.0 / mean_predictions['stack_corrected']) - 1.0
    
    generate_comparison_plots(metrics_report, {'total_pipeline': total_elapsed}, {}, output_dir="plots/python_plots")
    plot_predicted_vs_observed(y_true_z, stack_uncorrected_z, "Stacking_Regressor_Uncorrected", output_dir="plots/python_plots")
    plot_predicted_vs_observed(y_true_z, stack_corrected_z, "Stacking_Regressor_Bias_Corrected", output_dir="plots/python_plots")
    
    print("\n" + "=" * 70)
    print("  SEQUENTIAL PYTHON PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
