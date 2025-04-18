import torch
import torch.nn as nn
import torch.optim as optim

from utils import SOHCNN, monitor_idle_gpu_cpu, train_model, evaluate_model

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

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import SOHDataset, create_sequences # Assuming create_sequences and SOHDataset are still in utils
from battery_transformer.preprocess.preprocess import DataPreprocessor # Import the new class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
directory = "C:/Users/serha/PycharmProjects/Temp/PINN4SOH/data/XJTU_data" # Make sure this path is correct
file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]

# --- New Preprocessing Steps ---
# Initialize the preprocessor with desired settings (adjust as needed)
# WARNING: Column names ('voltage', 'current', 'timestamp') must exist in the CSVs. 
# Check CSV headers and adjust DataPreprocessor init or CSVs accordingly.
# Also, ensure the original sampling rate is appropriate for window_size_hours.
preprocessor = DataPreprocessor(
    voltage_col='voltage mean', # Example: Adjust to actual voltage column name
    current_col='current mean', # Example: Adjust to actual current column name
    timestamp_col='timestamp',   # Example: Adjust to actual timestamp column name
    window_size_hours=5, 
    sampling_rate_hz=1/60, # Assuming data is sampled roughly every minute initially
    downsample_freq='15T', 
    filter_type='butterworth', # Or 'savgol' or None
    remove_negatives=True,
    normalize=True # Keep normalization True as it returns the scaler
)

all_processed_data = []
final_scaler = None # To store the scaler from the last processed file (or handle scalers differently)

for file_path in file_list:
    try:
        processed_df, scaler = preprocessor.process_file(file_path)
        if not processed_df.empty:
            all_processed_data.append(processed_df)
            final_scaler = scaler # Overwrite scaler each time, assumes scaling is consistent
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")

if not all_processed_data:
     raise ValueError("No data could be processed. Check file paths, formats, and preprocessing parameters.")
     
# Combine all processed data
combined_df = pd.concat(all_processed_data, ignore_index=True)

# --- Adapt Feature Extraction and Sequencing ---
SEQ_LEN = 10
BATCH_SIZE = 128
# Define features based on the columns *after* preprocessing
# Example: if preprocessing keeps voltage, current, and adds others, list them here.
# The old feature list might be incompatible. Check columns in 'combined_df'.
# Make sure 'capacity' or the target column name is correct.
features = [col for col in combined_df.columns if col not in ['timestamp', 'capacity']] # Example: Exclude timestamp and target
targets = ['capacity'] 
NUM_FEATURES = len(features)

if targets[0] not in combined_df.columns:
    raise ValueError(f"Target column '{targets[0]}' not found in preprocessed data.")
if not all(f in combined_df.columns for f in features):
    raise ValueError(f"One or more feature columns not found in preprocessed data. Available: {combined_df.columns.tolist()}")
    
X_all = combined_df[features].values
y_all = combined_df[targets[0]].values

# Create sequences (ensure create_sequences function is compatible)
X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)

# Split data (e.g., 70/15/15 split)
X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(X_temp_seq, y_temp_seq, test_size=0.5, random_state=42) # 0.5 * 0.3 = 0.15

# Create Datasets and DataLoaders (ensure SOHDataset is compatible)
train_dataset = SOHDataset(X_train_seq, y_train_seq)
val_dataset = SOHDataset(X_val_seq, y_val_seq)
test_dataset = SOHDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle training data
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MODELS - TRAINING ======================================================================================

model = SOHCNN(input_dim=NUM_FEATURES, embed_dim=256).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_training_CNN.csv', 1), daemon=True)
monitor_thread.start()

train_model(model, train_loader, val_loader, criterion, optimizer, "models/best_CNN.pth", device, num_epochs=50)

monitoring = False
time.sleep(2)

# MODELS - TESTING =======================================================================================

monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_CNN.csv', 0.001), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(model, test_loader, "models/best_CNN.pth", 'outputs/error_results_CNN.txt', 'cnn', plot_fig = True, device=device)
print(f'{time.time()-start_time} seconds\n')

monitoring = False