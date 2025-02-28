import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils_CNN import truncated_normal_, CNNModel, SOHDataset, create_sequences

# MONITORING =============================================================================================

import subprocess, threading, time, psutil

def monitor_idle_gpu_cpu(duration=10, interval=1):
    
    power_values = []
    gpu_util_values = []
    cpu_util_values = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        power, gpu_util = map(float, result.stdout.strip().split(", "))
        cpu_util = psutil.cpu_percent(interval=0.1)
        
        power_values.append(power)
        gpu_util_values.append(gpu_util)
        cpu_util_values.append(cpu_util)
        
        time.sleep(interval)
    
    avg_power = sum(power_values) / len(power_values)
    avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values)
    avg_cpu_util = sum(cpu_util_values) / len(cpu_util_values)
    
    return avg_power, avg_gpu_util, avg_cpu_util

avg_time = 1
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

df = pd.read_csv("data/battery/scaledData1_with_soh.csv")

features = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc']
X = df[features].values

y = df["soh (%)"].values if "soh (%)" in df.columns else None

scaler_data = StandardScaler()
X = scaler_data.fit_transform(X)
y = y / 100

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

# MODELS - TRAINING ======================================================================================

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    tree_method='gpu_hist',
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

xgb_model.save_model("models/best_xgb_model.json")

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
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_XGB.csv', 0.01), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(xgb_model, X_test, y_test)
print(f'{time.time()-start_time} seconds\n')

monitoring = False