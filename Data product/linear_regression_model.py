# ============================================
# Bike sharing hourly demand prediction - Refactored for Streamlit
# ============================================

import os
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Global variables to cache the model
_model = None
_rmse = None

def load_and_train():
    global _model, _rmse
    
    if _model is not None:
        return _model, _rmse
        
    # Path to CSV
    # Assuming this file is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "bikehour.csv")
    
    if not os.path.exists(csv_path):
        # Fallback for when running from root
        csv_path = os.path.join("Data product", "bikehour.csv")

    df = pd.read_csv(csv_path)

    feature_cols = [
        "hr", "weekday", "workingday", "holiday", "mnth", "season", "yr",
        "weathersit", "temp", "atemp", "hum", "windspeed",
    ]
    target_col = "cnt"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Train-test split by year (yr=0 is 2011, yr=1 is 2012)
    train_mask = df["yr"] == 0
    test_mask = df["yr"] == 1

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    categorical_features = ["season", "yr", "mnth", "hr", "weekday", "workingday", "holiday", "weathersit"]
    numeric_features = ["temp", "atemp", "hum", "windspeed"]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    # Calculate RMSE for CI
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    
    _model = model
    _rmse = rmse
    
    return model, rmse

def month_to_season(mnth: int) -> int:
    if mnth in (3, 4, 5): return 1
    elif mnth in (6, 7, 8): return 2
    elif mnth in (9, 10, 11): return 3
    else: return 4

def extract_date_features(date_str: str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    yr = 0 if dt.year <= 2011 else 1
    mnth = dt.month
    season = month_to_season(mnth)
    py_weekday = dt.weekday()
    weekday = (py_weekday + 1) % 7
    workingday = 1 if weekday in (1, 2, 3, 4, 5) else 0
    holiday = 0
    return {
        "yr": yr, "mnth": mnth, "season": season,
        "weekday": weekday, "workingday": workingday, "holiday": holiday,
    }

def predict_one_hour(model, date_feats, hour, temp_c, weathersit, hum=0.6, windspeed=0.2):
    temp_norm = temp_c / 41.0
    atemp_norm = temp_norm
    
    data = pd.DataFrame([{
        "hr": hour,
        "weekday": date_feats["weekday"],
        "workingday": date_feats["workingday"],
        "holiday": date_feats["holiday"],
        "mnth": date_feats["mnth"],
        "season": date_feats["season"],
        "yr": date_feats["yr"],
        "weathersit": weathersit,
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum,
        "windspeed": windspeed,
    }])
    
    return float(model.predict(data)[0])

def get_predictions_for_streamlit(date_str: str, weather_cat: int, temp_c: float):
    model, rmse = load_and_train()
    
    date_feats = extract_date_features(date_str)
    hours = []
    preds = []
    
    for hr in range(24):
        pred = predict_one_hour(model, date_feats, hr, temp_c, weather_cat)
        hours.append(hr)
        preds.append(pred)
        
    results = pd.DataFrame({'Hour': hours, 'Predicted_Demand': preds})
    
    z_score = 1.645 # 90% CI
    results['Lower_CI'] = results['Predicted_Demand'] - z_score * rmse
    results['Upper_CI'] = results['Predicted_Demand'] + z_score * rmse
    
    # Clip
    results['Predicted_Demand'] = results['Predicted_Demand'].clip(lower=0)
    results['Lower_CI'] = results['Lower_CI'].clip(lower=0)
    results['Upper_CI'] = results['Upper_CI'].clip(lower=0)
    
    return results
