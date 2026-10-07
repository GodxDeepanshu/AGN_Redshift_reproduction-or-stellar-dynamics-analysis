import numpy as np
import pandas as pd
import time
import os
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_vhHlzZBwVf2AHO1kFY4QO1d7or-yb5murRwl67_weqo'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from joblib import Parallel, delayed

# Standard sklearn models (always available)
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

# Gradient boosting models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Try imports for optional advanced models with fallback flag
try:
    import ngboost as ngb
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    torch.set_num_threads(1)
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig, OptimizerConfig
    from pytorch_tabular.models import FTTransformerConfig
    HAS_FT_TRANSFORMER = True
except ImportError:
    HAS_FT_TRANSFORMER = False

from py_metrics import compute_all_metrics, print_metrics
from py_bias_correction import fit_optimal_transport, apply_optimal_transport

def lasso_feature_selection_1se(X, y, cv=10, random_state=42):
    """
    Implements the 1SE (one standard error) rule for LASSO feature selection,
    matching the R 'glmnet' behavior.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LassoCV to get the MSE path
    lasso_cv = LassoCV(cv=cv, random_state=random_state, max_iter=5000, n_jobs=1)
    lasso_cv.fit(X_scaled, y)
    
    # mean and SE of MSE across folds for each alpha
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    se_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(cv)
    
    min_idx = np.argmin(mean_mse)
    min_mse = mean_mse[min_idx]
    min_se = se_mse[min_idx]
    
    threshold = min_mse + min_se
    
    # Find the largest alpha where mean_mse <= threshold
    selected_idx = min_idx
    for idx in range(len(lasso_cv.alphas_)):
        if mean_mse[idx] <= threshold:
            selected_idx = idx
            break
            
    alpha_1se = lasso_cv.alphas_[selected_idx]
    
    # Fit final Lasso
    lasso_final = Lasso(alpha=alpha_1se, random_state=random_state, max_iter=5000)
    lasso_final.fit(X_scaled, y)
    
    # Get selected feature names
    selected_features = [X.columns[i] for i, coef in enumerate(lasso_final.coef_) if coef != 0]
    return selected_features

def fit_nnls_weights(meta_X, meta_y):
    """
    Solves for non-negative stacking weights that sum to 1 (SuperLearner NNLS meta-learner).
    """
    n_models = meta_X.shape[1]
    
    def objective(weights):
        pred = np.dot(meta_X, weights)
        return np.mean((meta_y - pred) ** 2)
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    w0 = np.ones(n_models) / n_models
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def train_and_predict_base_models(X_train, y_train, X_test, iter_seed, models_to_run=None, predict_train=False):
    """
    Trains specified base models on X_train and predicts on X_test.
    If predict_train is True, also returns in-sample predictions on X_train.
    Returns (train_predictions, test_predictions) if predict_train else test_predictions.
    """
    predictions = {}
    train_predictions = {}
    
    # 1. XGBoost
    if models_to_run is None or 'xgb' in models_to_run:
        model_xgb = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=iter_seed, n_jobs=1
        )
        model_xgb.fit(X_train, y_train)
        predictions['xgb'] = model_xgb.predict(X_test)
        if predict_train:
            train_predictions['xgb'] = model_xgb.predict(X_train)
    
    # 2. LightGBM
    if models_to_run is None or 'lgb' in models_to_run:
        model_lgb = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, random_state=iter_seed, n_jobs=1, verbose=-1
        )
        model_lgb.fit(X_train, y_train)
        predictions['lgb'] = model_lgb.predict(X_test)
        if predict_train:
            train_predictions['lgb'] = model_lgb.predict(X_train)
    
    # 3. CatBoost
    if models_to_run is None or 'cat' in models_to_run:
        model_cat = cb.CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=4,
            random_seed=iter_seed, thread_count=1, verbose=0
        )
        model_cat.fit(X_train, y_train)
        predictions['cat'] = model_cat.predict(X_test)
        if predict_train:
            train_predictions['cat'] = model_cat.predict(X_train)
    
    # 4. Extra Trees
    if models_to_run is None or 'et' in models_to_run:
        model_et = ExtraTreesRegressor(
            n_estimators=500, max_depth=6, random_state=iter_seed, n_jobs=1
        )
        model_et.fit(X_train, y_train)
        predictions['et'] = model_et.predict(X_test)
        if predict_train:
            train_predictions['et'] = model_et.predict(X_train)
    
    # 5. HistGradientBoosting
    if models_to_run is None or 'hgb' in models_to_run:
        model_hgb = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_depth=4, random_state=iter_seed
        )
        model_hgb.fit(X_train, y_train)
        predictions['hgb'] = model_hgb.predict(X_test)
        if predict_train:
            train_predictions['hgb'] = model_hgb.predict(X_train)
    
    # 6. NGBoost (Optional)
    if (models_to_run is None or 'ngb' in models_to_run) and HAS_NGBOOST:
        try:
            model_ngb = ngb.NGBRegressor(
                n_estimators=200, learning_rate=0.03, random_state=iter_seed, verbose=False
            )
            model_ngb.fit(X_train, y_train)
            predictions['ngb'] = model_ngb.predict(X_test)
            if predict_train:
                train_predictions['ngb'] = model_ngb.predict(X_train)
        except Exception as e:
            pass
            
    # 7. TabPFN (Optional)
    if (models_to_run is None or 'tabpfn' in models_to_run) and HAS_TABPFN:
        try:
            model_tabpfn = TabPFNRegressor(random_state=iter_seed, ignore_pretraining_limits=True)
            model_tabpfn.fit(X_train, y_train)
            predictions['tabpfn'] = model_tabpfn.predict(X_test)
            if predict_train:
                train_predictions['tabpfn'] = model_tabpfn.predict(X_train)
        except Exception as e:
            pass
            
    # 8. TabNet (Optional)
    if (models_to_run is None or 'tabnet' in models_to_run) and HAS_TABNET:
        try:
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)
            
            # Enforce seed parameter for deterministic training
            model_tabnet = TabNetRegressor(
                verbose=0,
                device_name='cpu',
                n_d=8, n_a=8,
                n_steps=3,
                n_shared=2,
                seed=iter_seed
            )
            model_tabnet.fit(
                X_tr_sc, y_train.values.reshape(-1, 1),
                max_epochs=10, patience=2,
                batch_size=256, virtual_batch_size=32
            )
            predictions['tabnet'] = model_tabnet.predict(X_te_sc).flatten()
            if predict_train:
                train_predictions['tabnet'] = model_tabnet.predict(X_tr_sc).flatten()
        except Exception as e:
            pass
            
    # 9. FT-Transformer (Optional)
    if (models_to_run is None or 'ft_trans' in models_to_run) and HAS_FT_TRANSFORMER:
        try:
            train_df = X_train.copy()
            train_df['target'] = y_train
            
            data_config = DataConfig(
                target=['target'],
                continuous_cols=list(X_train.columns),
                categorical_cols=[]
            )
            # Enforce seed parameter in trainer and model configs for deterministic training
            trainer_config = TrainerConfig(
                max_epochs=5,
                batch_size=256,
                checkpoints=None,
                progress_bar="none",
                accelerator="cpu",
                devices=1,
                seed=iter_seed
            )
            model_config = FTTransformerConfig(
                task="regression",
                num_attn_blocks=1,
                num_heads=1,
                input_embed_dim=8,
                seed=iter_seed
            )
            optimizer_config = OptimizerConfig()
            
            model_ft = TabularModel(
                data_config=data_config, 
                model_config=model_config, 
                trainer_config=trainer_config,
                optimizer_config=optimizer_config
            )
            model_ft.logger = False
            model_ft.fit(train_df)
            
            predictions['ft_trans'] = model_ft.predict(X_test)['target_prediction'].values
            if predict_train:
                train_predictions['ft_trans'] = model_ft.predict(X_train)['target_prediction'].values
        except Exception as e:
            pass
            
    if predict_train:
        return train_predictions, predictions
    return predictions

def run_single_cv_iteration(iter_seed, X, y, predictors, n_folds=10):
    """
    Runs a single 10-fold CV iteration with nested feature selection, stacking, and out-of-fold bias correction.
    """
    np.random.seed(iter_seed)
    
    # Define folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=iter_seed)
    
    # Determine which models will run based on imports (excluding slow PyTorch models)
    test_models = ['xgb', 'lgb', 'cat', 'et', 'hgb']
    if HAS_NGBOOST: test_models.append('ngb')
    
    # Initialize out-of-fold predictions with np.nan for robustness
    oof_predictions = {}
    for m in test_models:
        oof_predictions[m] = np.full(len(y), np.nan)
        oof_predictions[m + '_corrected'] = np.full(len(y), np.nan)
    oof_predictions['stack'] = np.full(len(y), np.nan)
    oof_predictions['stack_corrected'] = np.full(len(y), np.nan)
    
    feature_counts = {p: 0 for p in predictors}
    fold_weights = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        agn_train = X.iloc[train_idx]['AGN_Type'].values
        agn_test = X.iloc[test_idx]['AGN_Type'].values
        
        # 1. LASSO Feature Selection with seed propagation
        try:
            selected_features = lasso_feature_selection_1se(X_train[predictors], y_train, cv=5, random_state=iter_seed)
        except Exception as e:
            selected_features = list(predictors)
            
        if len(selected_features) == 0:
            selected_features = list(predictors)
            
        for f in selected_features:
            feature_counts[f] += 1
            
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # 2. Nested Stacking: Internal CV to get OOF predictions for training set
        stacking_models = ['xgb', 'lgb', 'cat', 'et', 'hgb']
        internal_kf = KFold(n_splits=3, shuffle=True, random_state=iter_seed)
        internal_oof = {m: np.zeros(len(y_train)) for m in stacking_models}
        
        for int_train, int_val in internal_kf.split(X_train_sel):
            X_int_tr, X_int_val = X_train_sel.iloc[int_train], X_train_sel.iloc[int_val]
            y_int_tr, y_int_val = y_train.iloc[int_train], y_train.iloc[int_val]
            
            # Train internal models and predict on internal validation fold
            int_preds = train_and_predict_base_models(X_int_tr, y_int_tr, X_int_val, iter_seed, models_to_run=stacking_models)
            for m in stacking_models:
                if m in int_preds:
                    internal_oof[m][int_val] = int_preds[m]
                    
        # Fit stacking weights on internal OOF predictions
        active_models = [m for m in stacking_models if np.sum(np.abs(internal_oof[m])) > 0]
        meta_X = np.column_stack([internal_oof[m] for m in active_models])
        
        weights = fit_nnls_weights(meta_X, y_train)
        
        # Map back to all test_models (assigning 0 weight to inactive models)
        full_weights_vector = np.zeros(len(test_models))
        for idx, m in enumerate(active_models):
            orig_idx = test_models.index(m)
            full_weights_vector[orig_idx] = weights[idx]
            
        fold_weights.append(full_weights_vector)
        
        # 3. Fit models on full fold training set and predict on train & test indexes
        train_preds, test_preds = train_and_predict_base_models(X_train_sel, y_train, X_test_sel, iter_seed, models_to_run=test_models, predict_train=True)
        
        # Accumulate out-of-fold predictions and apply nested bias correction to base models
        for m in test_models:
            if m in test_preds:
                oof_predictions[m][test_idx] = test_preds[m]
                try:
                    ot_params_m = fit_optimal_transport(train_preds[m], y_train.values, agn_train, verbose=False)
                    corrected_m = apply_optimal_transport(test_preds[m], agn_test, ot_params_m, verbose=False)
                    oof_predictions[m + '_corrected'][test_idx] = corrected_m
                except Exception as e:
                    oof_predictions[m + '_corrected'][test_idx] = test_preds[m]
                
        # Calculate stacked prediction
        fold_stack = np.zeros(len(test_idx))
        for idx, m in enumerate(test_models):
            if m in test_preds:
                fold_stack += full_weights_vector[idx] * test_preds[m]
        oof_predictions['stack'][test_idx] = fold_stack
        
        # Calculate training stacked predictions to fit stacking OT parameters
        train_stack = np.zeros(len(train_idx))
        for idx, m in enumerate(test_models):
            if m in train_preds:
                train_stack += full_weights_vector[idx] * train_preds[m]
                
        # Fit OT on stacked training predictions and apply to test fold stacking
        try:
            ot_params_stack = fit_optimal_transport(train_stack, y_train.values, agn_train, verbose=False)
            corrected_stack = apply_optimal_transport(fold_stack, agn_test, ot_params_stack, verbose=False)
            oof_predictions['stack_corrected'][test_idx] = corrected_stack
        except Exception as e:
            oof_predictions['stack_corrected'][test_idx] = fold_stack
        
    return oof_predictions, fold_weights, feature_counts

def run_python_pipeline(training_data, predictors, n_iterations=100, n_jobs=-1):
    """
    Executes the full CV pipeline across multiple iterations in parallel.
    """
    X = training_data[predictors]
    y = training_data['InvRedshift']
    
    # Determine test models
    test_models = ['xgb', 'lgb', 'cat', 'et', 'hgb']
    if HAS_NGBOOST: test_models.append('ngb')
    
    print(f"Starting Python modeling pipeline...")
    print(f"  Iterations: {n_iterations} x 10-fold CV")
    print(f"  Predictors: {len(predictors)}")
    print(f"  Dataset size: {len(training_data)} rows")
    print(f"  Base Models: {', '.join(test_models)}")
    
    start_time = time.time()
    
    # Execution of CV iterations
    if n_jobs == 1:
        results = []
        for i in range(n_iterations):
            t_start_iter = time.time()
            print(f"  [Iter {i+1}/{n_iterations}] Running 10-fold CV...", flush=True)
            res = run_single_cv_iteration(i + 1, training_data, y, predictors)
            results.append(res)
            print(f"  [Iter {i+1}/{n_iterations}] Completed in {time.time() - t_start_iter:.1f}s", flush=True)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_cv_iteration)(i + 1, training_data, y, predictors)
            for i in range(n_iterations)
        )
    
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds ({elapsed/60.0:.2f} minutes)!")
    
    # Aggregate predictions (including uncorrected and corrected models)
    n_samples = len(y)
    all_model_keys = []
    for m in test_models:
        all_model_keys.extend([m, m + '_corrected'])
    all_model_keys.extend(['stack', 'stack_corrected'])
    
    agg_predictions = {k: np.zeros((n_samples, n_iterations)) for k in all_model_keys}
    
    all_weights = []
    total_feature_counts = {p: 0 for p in predictors}
    
    for i, (preds, weights, feat_counts) in enumerate(results):
        for k in agg_predictions.keys():
            if k in preds:
                agg_predictions[k][:, i] = preds[k]
        all_weights.extend(weights)
        for p in predictors:
            total_feature_counts[p] += feat_counts[p]
            
    # Calculate final mean predictions (averaging across the 100 iterations)
    mean_predictions = {k: np.mean(agg_predictions[k], axis=1) for k in agg_predictions.keys()}
    
    # Compute metrics for each model
    metrics_report = {}
    for model_name in mean_predictions.keys():
        metrics_report[model_name] = compute_all_metrics(mean_predictions[model_name], y)
        
    return mean_predictions, agg_predictions, metrics_report, np.array(all_weights), total_feature_counts, test_models, elapsed
