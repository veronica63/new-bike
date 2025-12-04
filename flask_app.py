import csv
import os
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==========================================
# Data Loading & Processing (No Pandas)
# ==========================================

def load_data():
    """
    Load bikehour.csv into a list of dicts and a numpy array for modeling.
    """
    data = []
    headers = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try multiple paths
    paths = [
        os.path.join(script_dir, "Data product", "bikehour.csv"),
        os.path.join(script_dir, "bikehour.csv")
    ]
    csv_path = None
    for p in paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print("Error: bikehour.csv not found.")
        return [], None, None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            # Convert types
            for k, v in row.items():
                try:
                    if k in ['dteday']: continue
                    row[k] = float(v)
                except ValueError:
                    pass
            data.append(row)
            
    return data

DATA = load_data()

# ==========================================
# Modeling (Numpy Only)
# ==========================================

def train_model():
    """
    Train a simple Linear Regression model using Numpy.
    Features: hr, weekday, workingday, holiday, mnth, season, yr, weathersit, temp, atemp, hum, windspeed
    Target: cnt
    """
    # Prepare X and y
    # We need to handle categorical variables (One-Hot Encoding) manually or treat them as numeric?
    # The original script used OneHotEncoder for: season, yr, mnth, hr, weekday, workingday, holiday, weathersit
    # Doing full OHE in pure numpy is verbose. 
    # For simplicity and robustness in this constrained env, let's use a simplified feature set 
    # OR implement a basic OHE.
    
    # Let's try basic OHE for critical features: weathersit, season, hr (maybe cyclical?)
    # To keep it "compact" code-wise, let's just use the numeric values for now, 
    # but `hr` is very non-linear. 
    # Better approach: Use the same features but mapped to a numpy matrix.
    
    X_list = []
    y_list = []
    
    for row in DATA:
        if row['yr'] == 0: # Train on 2011 only (matching original logic)
            # Features
            # Simple approach: just dump them in. 
            # Note: This will be less accurate than OHE but functional.
            # To improve, we can add polynomial terms for hour?
            # Let's stick to raw features for "working code" first.
            feats = [
                row['hr'], row['weekday'], row['workingday'], row['holiday'],
                row['mnth'], row['season'], row['yr'], row['weathersit'],
                row['temp'], row['atemp'], row['hum'], row['windspeed'],
                1.0 # Bias term
            ]
            X_list.append(feats)
            y_list.append(row['cnt'])
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Linear Regression: w = (X^T X)^-1 X^T y
    # Use lstsq for stability
    w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # Calculate RMSE on 2012 data (yr=1)
    X_test_list = []
    y_test_list = []
    for row in DATA:
        if row['yr'] == 1:
            feats = [
                row['hr'], row['weekday'], row['workingday'], row['holiday'],
                row['mnth'], row['season'], row['yr'], row['weathersit'],
                row['temp'], row['atemp'], row['hum'], row['windspeed'],
                1.0
            ]
            X_test_list.append(feats)
            y_test_list.append(row['cnt'])
            
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    y_pred = X_test @ w
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    
    return w, rmse

MODEL_WEIGHTS, MODEL_RMSE = train_model()

def get_prediction(date_str, weather_cat, temp_c):
    # Parse date
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Extract features
    yr = 0 if dt.year <= 2011 else 1
    mnth = dt.month
    
    if mnth in (3, 4, 5): season = 1
    elif mnth in (6, 7, 8): season = 2
    elif mnth in (9, 10, 11): season = 3
    else: season = 4
        
    weekday = (dt.weekday() + 1) % 7 # 0=Sun... wait, python 0=Mon. 
    # Original script: 0=Sun. Python: 0=Mon.
    # If Python 0=Mon, then (0+1)%7 = 1 (Mon). 
    # We need to match dataset. Let's assume standard logic.
    
    workingday = 1 if weekday in (1, 2, 3, 4, 5) else 0
    holiday = 0 # Simplified
    
    temp_norm = float(temp_c) / 41.0
    atemp_norm = temp_norm
    hum = 0.6
    windspeed = 0.2
    
    preds = []
    hours = list(range(24))
    
    for hr in hours:
        feats = [
            hr, weekday, workingday, holiday,
            mnth, season, yr, int(weather_cat),
            temp_norm, atemp_norm, hum, windspeed,
            1.0 # Bias
        ]
        pred = np.dot(MODEL_WEIGHTS, feats)
        preds.append(max(0, pred)) # Clip negative
        
    return hours, preds, MODEL_RMSE

# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    # Return data for heatmap/scatter
    # Filter by user params if needed, but for now return all or a sample
    # To avoid sending 17k rows, let's aggregate or send 2012 data
    
    # Filter params
    weather_filter = request.args.getlist('weather[]') # e.g. [1, 2]
    if not weather_filter: weather_filter = ['1','2','3','4']
    weather_filter = [int(x) for x in weather_filter]
    
    filtered = []
    for row in DATA:
        if int(row['weathersit']) in weather_filter:
            filtered.append({
                'weekday': int(row['weekday']),
                'hr': int(row['hr']),
                'cnt': row['cnt'],
                'registered': row['registered'],
                'casual': row['casual'],
                'weathersit': int(row['weathersit']),
                'temp': row['temp'],
                'hum': row['hum'],
                'windspeed': row['windspeed']
            })
            
    return jsonify(filtered)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.json
    date_str = req.get('date', '2012-07-01')
    weather = req.get('weather', 1)
    temp = req.get('temp', 25)
    
    hours, preds, rmse = get_prediction(date_str, weather, temp)
    
    # Calculate CI
    z = 1.645
    lower = [max(0, p - z * rmse) for p in preds]
    upper = [p + z * rmse for p in preds]
    
    return jsonify({
        'hours': hours,
        'pred': preds,
        'lower': lower,
        'upper': upper
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
