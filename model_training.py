# model_training.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from visualizations import plot_model_performance, plot_feature_importance

def get_models():
    """Returns a dictionary of regression models to be trained."""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    return models

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    """Trains, evaluates, and plots performance for a dictionary of models."""
    results_list = []
    models = get_models()

    for name, model in models.items():
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        print(f"--- Training {name} ---")
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results_list.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        plot_model_performance(y_test, y_pred, name)

    return pd.DataFrame(results_list)

def tune_random_forest(X_train, y_train, preprocessor):
    """Performs hyperparameter tuning for Random Forest and returns the best model."""
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20],
        'regressor__min_samples_split': [2, 5]
    }
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_
