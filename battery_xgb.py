import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import load_and_proc_data_xgb, monitor_idle_gpu_cpu

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
directory = "data/battery/csv"
file_list = csv_files = [directory+'/'+f for f in os.listdir(directory) if f.endswith(".csv")]
for f in file_list:
    print(f)
    
SEQ_LEN = 100
BATCH_SIZE = 32
features = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc']
NUM_FEATURES = len(features)

X_train, X_val, X_test, y_train, y_val, y_test, scaler_data = load_and_proc_data_xgb(file_list,
                                                                                     features = features)

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

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    with open('outputs/error_results_XGB.txt', "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\nTest MAE: {mae:.4f}\nTest R²: {r2:.4f}")

time.sleep(2)

monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_XGB.csv', 0.001), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(xgb_model, X_test, y_test)
print(f'{time.time()-start_time} seconds\n')

monitoring = False