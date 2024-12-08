import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
from scipy.stats import uniform, randint
import shap

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_models():
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
    return models

def parse_distribution(param_config):
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, dict):
        dist_type = param_config.get("dist")
        if dist_type == "randint":
            low = param_config["low"]
            high = param_config["high"]
            return randint(low, high)
        elif dist_type == "uniform":
            low = param_config["low"]
            width = param_config["width"]
            return uniform(low, width)
        else:
            return None
    else:
        return [param_config]

def get_param_distributions(model_name):
    model_params = config.get("model_parameters", {}).get(model_name, None)
    if model_params is None:
        return None

    param_dist = {}
    for param_name, param_config in model_params.items():
        distribution = parse_distribution(param_config)
        if distribution is not None:
            param_dist[param_name] = distribution
    return param_dist if param_dist else None

def generate_model_report(model, X_train, y_train, feature_names=None):
    from sklearn.metrics import make_scorer

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

def generate_additional_plots_and_metrics(model, X_train, y_train, X_test, y_test, feature_names):
 
    predictions = model.predict(X_test)
    medae = median_absolute_error(y_test, predictions)
    
    # Predicted vs Actual
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
    
    # Residuals vs Fitted
    residuals = y_test - predictions
    residuals_vs_fitted_fig = px.scatter(
        x=predictions,
        y=residuals,
        labels={'x': 'Fitted (Predicted)', 'y': 'Residuals'},
        title='Residuals vs Fitted'
    )
    residuals_vs_fitted_fig.add_hline(y=0, line_dash='dash', line_color='red')
    
    pdp_figs = []
    tree_based_models = (RandomForestRegressor, ExtraTreesRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor)
    
    if isinstance(model, tree_based_models) and len(feature_names) > 0:
        
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
    if isinstance(model, tree_based_models):
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

def train_selected_model(model_name, X_train, y_train, X_test, y_test, feature_names):
    models = get_models()
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
