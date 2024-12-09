# **Import Required Libraries**
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml # YAML for reading configuration files

# Regressor models for machine learning
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# Model evaluation, parameter tuning, and scoring
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    make_scorer
)
from sklearn.inspection import partial_dependence
import shap

# Advanced machine learning libraries
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from pmdarima import ARIMA, auto_arima
from scipy.stats import randint, uniform

# **Load configuration file**
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# **Get Models Function**
def get_models(has_time=False):
    """
    Returns a dictionary of model name to model instance.
    If has_time is True, an 'ARIMA' entry will also be included for time series.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Extra Trees Regressor': ExtraTreesRegressor(),
        'LGBM Regressor': LGBMRegressor(),
        'XGB Regressor': XGBRegressor(),
        'CatBoost Regressor': CatBoostRegressor(verbose=0)
    }

    if has_time:
        # ARIMA entry - handled differently in train_selected_model
        models['ARIMA'] = 'ARIMA_PLACEHOLDER'
    
    return models

# **Parse Distribution Function**
def parse_distribution(param_config):
    """
    Parses and returns a distribution for RandomizedSearchCV.
    Supports randint, uniform, or direct list of values.
    """
    if isinstance(param_config, list):
        return param_config  # If the parameter is a list, return it as-is
    if isinstance(param_config, dict):
        dist_type = param_config.get("dist")
        if dist_type == "randint":
            low = param_config["low"]
            high = param_config["high"]
            return randint(low, high) # Return randint distribution
        elif dist_type == "uniform":
            low = param_config["low"]
            width = param_config["width"]
            return uniform(low, width)  # Return uniform distribution
        else:
            return None
    else:
        return [param_config]

# **Get Parameter Distributions Function**
def get_param_distributions(model_name):
    """
    Returns hyperparameter distributions for a given model from the configuration file.
    """
    model_params = config.get("model_parameters", {}).get(model_name, None)
    if model_params is None:
        return None

    param_dist = {}
    for param_name, param_config in model_params.items():
        distribution = parse_distribution(param_config)
        if distribution is not None:
            param_dist[param_name] = distribution
    return param_dist if param_dist else None

# **Generate Model Report Function**
def generate_model_report(model, X_train, y_train, feature_names=None):
    """
    Generates cross-validation metrics and feature importances for the given model.
    """
    # Cross-validation for MSE, MAE, MAPE, and R²
    cv_score_mse = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    mse_mean = np.abs(np.mean(cv_score_mse))
    rmse_mean = np.sqrt(mse_mean)

    cv_score_r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=10)
    r2_mean = np.mean(cv_score_r2)

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv_score_mae = cross_val_score(model, X_train, y_train, scoring=mae_scorer, cv=10)
    mae_mean = np.abs(np.mean(cv_score_mae))

    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    cv_score_mape = cross_val_score(model, X_train, y_train, scoring=mape_scorer, cv=10)
    mape_mean = np.abs(np.mean(cv_score_mape))

    # In-sample prediction metrics
    prediction = model.predict(X_train)
    r2_train = r2_score(y_train, prediction)
    mse_train = mean_squared_error(y_train, prediction)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, prediction)
    mape_train = mean_absolute_percentage_error(y_train, prediction)

    metrics = {
        'cv_neg_mse_scores': cv_score_mse,
        'cv_neg_mse_mean': mse_mean,
        'cv_rmse_mean': rmse_mean,
        'cv_mae_mean': mae_mean,
        'cv_mape_mean': mape_mean,
        'cv_r2_scores': cv_score_r2,
        'cv_r2_mean': r2_mean,
        'r2_train': r2_train,
        'mse_train': mse_train,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'mape_train': mape_train
    }

    # Generate feature importance plot (for tree-based models or linear models)
    if feature_names is not None:
        if hasattr(model, 'coef_'):
            coef = pd.Series(model.coef_, index=feature_names).sort_values()
            importance_fig = px.bar(coef, x=coef.values, y=coef.index, orientation='h', title='Model Coefficients')
        elif hasattr(model, 'feature_importances_'):
            coef = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            importance_fig = px.bar(coef, x=coef.values, y=coef.index, orientation='h', title='Feature Importances')
        else:
            importance_fig = None
    else:
        importance_fig = None

    return metrics, importance_fig

# **Generate Additional Plots and Metrics**
def generate_additional_plots_and_metrics(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Generates residuals plots, partial dependence plots, and SHAP values for tree-based models.
    """
    predictions = model.predict(X_test)
    medae = median_absolute_error(y_test, predictions)
    
    # **Predicted vs Actual Scatter Plot**
    pred_vs_actual_fig = px.scatter(
        x=y_test,
        y=predictions,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Predicted vs Actual Values'
    )
    pred_vs_actual_fig.add_shape(
        type='line',
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    
    # **Residuals vs Fitted Plot**
    residuals = y_test - predictions
    residuals_vs_fitted_fig = px.scatter(
        x=predictions,
        y=residuals,
        labels={'x': 'Fitted (Predicted)', 'y': 'Residuals'},
        title='Residuals vs Fitted'
    )
    residuals_vs_fitted_fig.add_hline(y=0, line_dash='dash', line_color='red')
    
    # **Partial Dependence Plot**
    pdp_figs = []
    tree_based_models = (RandomForestRegressor, ExtraTreesRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor)
    
    if isinstance(model, tree_based_models) and feature_names is not None and len(feature_names) > 0:
        feature_for_pdp = [feature_names[0]]
        try:
            pd_results = partial_dependence(
                model, X_train,
                features=feature_for_pdp,
                kind='average'
            )
            avg = pd_results.average[0]
            vals = pd_results.grid_values[0]
            
            pdp_fig = px.line(
                x=vals, 
                y=avg, 
                title=f'Partial Dependence for {feature_for_pdp[0]}',
                labels={'x': feature_for_pdp[0], 'y': 'Partial Dependence'}
            )
            pdp_figs.append(pdp_fig)
        except Exception as e:
            print(f"Warning: Could not generate PDP plot: {str(e)}")
            pdp_figs = []
    
    # SHAP values for tree-based models
    shap_fig = None
    if isinstance(model, tree_based_models) and feature_names is not None and len(feature_names) > 0:
        try:
            explainer = shap.TreeExplainer(model)
            X_sample = X_train.iloc[:100] if X_train.shape[0] > 100 else X_train
            shap_values = explainer.shap_values(X_sample)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
            shap_fig = px.bar(
                shap_importance,
                x=shap_importance.values,
                y=shap_importance.index,
                orientation='h',
                title='SHAP Feature Importance'
            )
        except Exception as e:
            print(f"Warning: Could not generate SHAP plot: {str(e)}")
            shap_fig = None
    
    return {
        'medae': medae,
        'pred_vs_actual_fig': pred_vs_actual_fig,
        'residuals_vs_fitted_fig': residuals_vs_fitted_fig,
        'pdp_figs': pdp_figs,
        'shap_fig': shap_fig
    }

def train_selected_model(model_name, X_train, y_train, X_test, y_test, feature_names,
                         is_time_series=False, original_data=None):
    """
    Train the selected model. 
    If is_time_series is True and model_name is 'ARIMA', handle ARIMA training.
    If is_time_series is True and model_name != 'ARIMA', only 'ARIMA' is supported.
    If not time series, we must have X_train, y_train, X_test, y_test defined.
    """
    if is_time_series:
        if model_name == 'ARIMA':
            df = original_data.copy()
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time').sort_index()
            
            # If duplicates in index, aggregate numeric columns by mean
            if df.index.duplicated().any():
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    raise ValueError("No numeric columns found for aggregation, but duplicates detected.")
                numeric_df = numeric_df.groupby(level=0).mean()
                if 'Item_Outlet_Sales' not in numeric_df.columns:
                    raise ValueError("'Item_Outlet_Sales' not found among numeric columns after aggregation.")
                df = numeric_df

            # Convert to a daily frequency
            df = df.asfreq('D')

            # We'll assume the target is 'Item_Outlet_Sales'
            if 'Item_Outlet_Sales' not in df.columns:
                raise ValueError("'Item_Outlet_Sales' column not found in the dataframe.")

            y_full = df['Item_Outlet_Sales'].dropna()

            # Time-based split 
            split_point = int(len(y_full) * 0.8)
            y_train_arima = y_full.iloc[:split_point]
            y_test_arima = y_full.iloc[split_point:]

            # Step 1: Use auto_arima to find the best (p, d, q) order
            print("Running auto ARIMA to find the best (p, d, q)...")
            arima_model = auto_arima(
                y_train_arima, 
                start_p=0, max_p=5, 
                start_q=0, max_q=5, 
                d=None, 
                seasonal=False, 
                stepwise=True, 
                trace=True, 
                error_action='ignore', 
                suppress_warnings=True, 
                max_order=10, 
                information_criterion='aic'
            )

            best_p, best_d, best_q = arima_model.order
            print(f"Best ARIMA order: p={best_p}, d={best_d}, q={best_q}")

            # Fit ARIMA model
            arima_fitted = ARIMA(order=(best_p, best_d, best_q)).fit(y_train_arima)

            forecast = arima_fitted.predict(n_periods=len(y_test_arima))
            forecast_series = pd.Series(forecast, index=y_test_arima.index)

            # Drop any NaN values if present
            y_test_arima = y_test_arima.dropna()
            forecast_series = forecast_series.dropna()

            # Align by common index
            common_index = y_test_arima.index.intersection(forecast_series.index)
            y_test_arima = y_test_arima.loc[common_index]
            forecast_series = forecast_series.loc[common_index]

            if y_test_arima.empty or forecast_series.empty:
                raise ValueError("No overlapping data between forecast and actual test data after cleaning.")

            if forecast_series.isna().any() or y_test_arima.isna().any():
                raise ValueError("NaN values found in y_test_arima or forecast_series after cleaning.")

            r2_test = r2_score(y_test_arima, forecast_series)
            mse_test = mean_squared_error(y_test_arima, forecast_series)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test_arima, forecast_series)
            mape_test = mean_absolute_percentage_error(y_test_arima, forecast_series)
            medae_test = median_absolute_error(y_test_arima, forecast_series)

            importance_fig = None

            # Residuals
            residuals = y_test_arima - forecast_series
            residual_fig = px.histogram(residuals, nbins=30, title='Residuals Distribution (ARIMA)')

            # Predicted vs Actual
            pred_vs_actual_fig = px.scatter(
                x=y_test_arima,
                y=forecast_series,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Predicted vs Actual Values (ARIMA)'
            )
            pred_vs_actual_fig.add_shape(
                type='line',
                x0=y_test_arima.min(), y0=y_test_arima.min(),
                x1=y_test_arima.max(), y1=y_test_arima.max(),
                line=dict(color='red', dash='dash')
            )

            # Residuals vs Fitted
            residuals_vs_fitted_fig = px.scatter(
                x=forecast_series,
                y=residuals,
                labels={'x': 'Fitted (Predicted)', 'y': 'Residuals'},
                title='Residuals vs Fitted (ARIMA)'
            )
            residuals_vs_fitted_fig.add_hline(y=0, line_dash='dash', line_color='red')

            pdp_figs = None
            shap_fig = None

            metrics = {
                'r2_test': r2_test,
                'mse_test': mse_test,
                'rmse_test': rmse_test,
                'mae_test': mae_test,
                'mape_test': mape_test,
                'medae_test': medae_test,
                'cv_r2_mean': 'NA',
                'cv_rmse_mean': 'NA',
                'cv_neg_mse_scores': [],
                'cv_neg_mse_mean': 'NA',
                'cv_mae_mean': 'NA',
                'cv_mape_mean': 'NA',
                'r2_train': 'NA',
                'mse_train': 'NA',
                'rmse_train': 'NA',
                'mae_train': 'NA',
                'mape_train': 'NA'
            }

            results = {
                'model': arima_fitted,
                'metrics': metrics,
                'r2_score': r2_test,
                'best_params': None,
                'importance_fig': importance_fig,
                'residual_fig': residual_fig,
                'pred_vs_actual_fig': pred_vs_actual_fig,
                'residuals_vs_fitted_fig': residuals_vs_fitted_fig,
                'pdp_figs': pdp_figs,
                'shap_fig': shap_fig,
                # Store the ARIMA test set for predictions
                'y_test_arima': y_test_arima.to_list()  # Store as list for serialization
            }

            return results
        else:
            raise ValueError(f"Time-series mode is active. Only 'ARIMA' model is supported. '{model_name}' was requested.")
    else:
        # Non-time-series scenario
        if X_train is None or y_train is None:
            raise ValueError(f"X_train and y_train must not be None for non-time-series model: {model_name}")

        models = get_models(has_time=False)
        model = models[model_name]

        param_dist = get_param_distributions(model_name)
        if param_dist is not None:
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=10,
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=10,
                n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            best_params = random_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_params = None

        predictions = model.predict(X_test)
        # Ensure no NaNs in predictions
        if np.isnan(predictions).any():
            raise ValueError("NaN values found in model predictions.")

        r2_test = r2_score(y_test, predictions)
        mse_test = mean_squared_error(y_test, predictions)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, predictions)
        mape_test = mean_absolute_percentage_error(y_test, predictions)

        metrics, importance_fig = generate_model_report(model, X_train, y_train, feature_names)
        residuals = y_test - predictions
        residual_fig = px.histogram(residuals, nbins=30, title='Residuals Distribution')

        metrics.update({
            'r2_test': r2_test,
            'mse_test': mse_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'mape_test': mape_test
        })

        additional_results = generate_additional_plots_and_metrics(model, X_train, y_train, X_test, y_test, feature_names)
        metrics['medae_test'] = additional_results['medae']

        results = {
            'model': model,
            'metrics': metrics,
            'r2_score': r2_test,
            'best_params': best_params,
            'importance_fig': importance_fig,
            'residual_fig': residual_fig,
            'pred_vs_actual_fig': additional_results['pred_vs_actual_fig'],
            'residuals_vs_fitted_fig': additional_results['residuals_vs_fitted_fig'],
            'pdp_figs': additional_results['pdp_figs'],
            'shap_fig': additional_results['shap_fig']
        }

        return results
