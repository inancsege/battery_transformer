import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from battery_transformer.preprocess.preprocess import DataPreprocessor

from utils import monitor_idle_gpu_cpu

# MONITORING =============================================================================================

import threading, subprocess, time, psutil

avg_time = 10
avg_power, avg_gpu_util, avg_cpu_util = monitor_idle_gpu_cpu(duration=avg_time)

print(f'\nAverage values over {avg_time} seconds: \nAVG_GPU_POWER = {avg_power}, AVG_GPU_UTIL = {avg_gpu_util}, AVG_CPU_UTIL = {avg_cpu_util}\n')

monitoring = True

def monitor_gpu(log_file = 'gpu_usage_log.csv', interval = 1):

    query_params = [
        "timestamp", "power.draw", "memory.used", "memory.total",
        "utilization.gpu", "utilization.memory", "temperature.gpu",
        "fan.speed", "clocks.sm", "clocks.gr"
    ]
    
    query_str = ",".join(query_params)
    
    with open(log_file, "w") as f:
        f.write("Timestamp,Power (W),Memory Used (MB),Memory Total (MB),GPU Util (%),"
                "Memory Util (%),Temp (C),Fan Speed (%),Clock SM (MHz),Clock Mem (MHz),"
                "CPU Usage (%)\n")
    
    while monitoring:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query_str, "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )

        gpu_data = list(map(float, result.stdout.strip().split(", ")[1:]))
        gpu_data[0] = gpu_data[0] - avg_power
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent() - avg_cpu_util
        
        log_entry = f"{timestamp}," + ",".join(list(map(str, gpu_data))) + f",{cpu_usage}\n"
        
        with open(log_file, "a") as f:
            f.write(log_entry)

        time.sleep(interval)

# DATA PREPROC ===========================================================================================

# Removed torch import as device is not used directly here for XGBoost data
# from utils import load_and_proc_data_xgb # Replaced by DataPreprocessor

import os
directory = "C:/Users/serha/PycharmProjects/Temp/PINN4SOH/data/XJTU_data" # Verify path
file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]

# --- New Preprocessing Steps ---
preprocessor = DataPreprocessor(
    voltage_col='voltage mean', # Adjust to actual voltage column name in CSVs
    current_col='current mean', # Adjust to actual current column name in CSVs
    timestamp_col='timestamp',   # Adjust to actual timestamp column name in CSVs
    window_size_hours=5, 
    sampling_rate_hz=1/60, # Adjust if initial sampling rate is different
    downsample_freq='15T', 
    filter_type='butterworth', # Or 'savgol' or None
    remove_negatives=True,
    normalize=True # Normalization needed as XGBoost benefits from scaled features
)

all_processed_data = []
final_scaler = None # Store the scaler if needed for inverse transform later

for file_path in file_list:
    try:
        processed_df, scaler = preprocessor.process_file(file_path)
        if not processed_df.empty:
            all_processed_data.append(processed_df)
            final_scaler = scaler
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")

if not all_processed_data:
     raise ValueError("No data could be processed. Check file paths, formats, and preprocessing parameters.")
     
combined_df = pd.concat(all_processed_data, ignore_index=True)

# --- Adapt Feature Extraction and Splitting for XGBoost ---
# Note: XGBoost doesn't typically use sequence data like LSTMs/Transformers
# It uses feature vectors directly. We will split the combined data.

# Define features and target based on processed columns
features = [col for col in combined_df.columns if col not in ['timestamp', 'capacity']] # Adjust target name if different
target = 'capacity' 
NUM_FEATURES = len(features)

if target not in combined_df.columns:
    raise ValueError(f"Target column '{target}' not found in preprocessed data.")
if not all(f in combined_df.columns for f in features):
    raise ValueError(f"One or more feature columns not found in preprocessed data. Available: {combined_df.columns.tolist()}")

X = combined_df[features].values
y = combined_df[target].values

# Split data (e.g., 70/15/15 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 0.5 * 0.3 = 0.15

# The old load_and_proc_data_xgb call is replaced
# X_train, X_val, X_test, y_train, y_val, y_test, scaler_data = load_and_proc_data_xgb(...)

# scaler_data is now final_scaler from the preprocessor

# MODELS - TRAINING ======================================================================================

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    tree_method='hist',
    device='cuda',
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42
)

monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_training_XGB.csv', 1), daemon=True)
monitor_thread.start()

# Train the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

monitoring = False

xgb_model.save_model("models/best_XGB.json")

#loaded_model = xgb.XGBRegressor()
#loaded_model.load_model("xgboost_model.json")

# MODELS - TESTING =======================================================================================

def evaluate_model(model, X_test, y_test, plot_model_name, plot_fig = True):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    all_targets = y_test
    all_preds = predictions

    # Compute MAPE (avoid division by zero)
    non_zero_mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[non_zero_mask] - all_preds[non_zero_mask]) / all_targets[non_zero_mask])) * 100

    # Compute Mean Directional Accuracy (MDA)
    direction_actual = np.sign(np.diff(all_targets))
    direction_pred = np.sign(np.diff(all_preds))

    mda = np.mean(direction_actual == direction_pred)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test MDA: {mda}")


        
time.sleep(2)

monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_XGB.csv', 0.001), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(xgb_model, X_test, y_test, 'xgb', plot_fig = True)
print(f'{time.time()-start_time} seconds\n')

monitoring = False