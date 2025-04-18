import torch
import torch.nn as nn
import torch.optim as optim

from utils import SOHTransformer, load_and_proc_data, monitor_idle_gpu_cpu, train_model, evaluate_model

# MONITORING =============================================================================================

import threading, subprocess, time, psutil

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
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + query_str, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            raw_values = result.stdout.strip().split(", ")
            if len(raw_values) < 2: 
                print(f"Warning: Unexpected nvidia-smi output: {result.stdout.strip()}")
                time.sleep(interval)
                continue

            timestamp_str = raw_values[0]
            gpu_data = []
            for val_str in raw_values[1:]:
                if '[N/A]' in val_str:
                    gpu_data.append(0.0)
                else:
                    try:
                        gpu_data.append(float(val_str))
                    except ValueError:
                        gpu_data.append(0.0)
                        print(f"Warning: Replaced non-float value '{val_str}' with 0.0 in nvidia-smi output.")

            if len(gpu_data) > 0:
                if len(raw_values) > 1 and '[N/A]' not in raw_values[1]: 
                     gpu_data[0] = gpu_data[0] - avg_power
                else:
                     if len(gpu_data) > 0:
                           gpu_data[0] = 0.0 
                     else:
                           gpu_data.append(0.0)
            else:
                print("Warning: No GPU data processed after handling N/A values.")
                continue

            current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") 
            cpu_usage = psutil.cpu_percent() - avg_cpu_util
            
            log_entry = f"{current_timestamp}," + ",".join(map(str, gpu_data)) + f",{cpu_usage}\n" 
            
            with open(log_file, "a") as f:
                f.write(log_entry)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}. Output: {e.output}")
            time.sleep(interval * 5)
        except FileNotFoundError:
            print("Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed and nvidia-smi is in the system PATH.")
            monitoring = False
            break 
        except Exception as e:
            print(f"An unexpected error occurred in monitor_gpu: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(interval)

        if not monitoring:
             break
        time.sleep(interval)

# DATA PREPROC ===========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
directory = "data/battery/csv"
file_list = csv_files = [directory+'/'+f for f in os.listdir(directory) if f.endswith(".csv")]
targets = ['available_capacity (Ah)']
for f in file_list:
    print(f)

SEQ_LEN = 100
BATCH_SIZE = 32
features = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc', 'available_capacity (Ah)']
NUM_FEATURES = len(features)

_, _, train_loader, val_loader, test_loader, scaler_data = load_and_proc_data(file_list,
                                                                              features = features,
                                                                              targets=targets,
                                                                              SEQ_LEN = SEQ_LEN,
                                                                              BATCH_SIZE = BATCH_SIZE)

# MODELS - TRAINING ======================================================================================

model = SOHTransformer(input_dim=NUM_FEATURES, embed_dim=256).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_training_TRANSFORMER.csv', 1), daemon=True)
monitor_thread.start()

train_model(model, train_loader, val_loader, criterion, optimizer, "models/best_TRANSFORMER.pth", device, num_epochs=5)

monitoring = False
time.sleep(2)

# MODELS - TESTING =======================================================================================

monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_TRANSFORMER.csv', 0.01), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(model, test_loader, "models/best_TRANSFORMER.pth", 'outputs/error_results_TRANSFORMER.txt', 'transformer', plot_fig = True, device=device)
print(f'{time.time()-start_time} seconds\n')

monitoring = False