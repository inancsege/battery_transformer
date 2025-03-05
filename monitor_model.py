import torch
import numpy as np
import subprocess, threading, time, psutil
from torch.utils.data import DataLoader, TensorDataset

from utils import SOHTransformer, monitor_idle_gpu_cpu, evaluate_model

# Load test dataset
BATCH_SIZE = 32

monitoring = True

model_name = 'quantized'

def monitor_gpu(avg_power, avg_gpu_util, avg_cpu_util, max_memory_used, log_file = 'gpu_usage_log.csv', interval = 1):

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
        gpu_data[1] = gpu_data[1] - max_memory_used
        gpu_data[3] = gpu_data[3] - avg_gpu_util
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent() - avg_cpu_util
        
        log_entry = f"{timestamp}," + ",".join(list(map(str, gpu_data))) + f",{cpu_usage}\n"
        
        with open(log_file, "a") as f:
            f.write(log_entry)

        time.sleep(interval)

avg_time = 10
avg_power, avg_gpu_util, avg_cpu_util, memory_used_max = monitor_idle_gpu_cpu(duration=avg_time)

print(f'\nAverage values over {avg_time} seconds: \nAVG_GPU_POWER = {avg_power}, AVG_GPU_UTIL = {avg_gpu_util}, AVG_CPU_UTIL = {avg_cpu_util}, MAX_MEMORY_UTIL = {memory_used_max}\n')

def load_test_data():
    data = torch.load("test_data/test_data.pt")
    X_test, y_test = data["X"], data["y"]
    test_dataset = TensorDataset(X_test, y_test)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_loader = load_test_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == 'distill':

    model = SOHTransformer(input_dim=5, embed_dim=128).to(device)
    model.load_state_dict(torch.load("models/distill_soh_tec_model.pth", map_location=device))

elif model_name == 'pruned':

    model = SOHTransformer(input_dim=5, embed_dim=256).to(device)
    pruned_state_dict = torch.load("models/pruned_soh_tec_model.pth", map_location=device)
    new_state_dict = {k: v for k, v in pruned_state_dict.items() if "_mask" not in k}
    model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore missing entries

elif model_name == 'quantized':

    model = SOHTransformer(input_dim=5, embed_dim=256).to(device)
    quantized_state_dict = torch.load("models/quantized_soh_tec_model.pth", map_location=device)
    new_state_dict = {k: v for k, v in quantized_state_dict.items() if "_packed_params" not in k and "scale" not in k and "zero_point" not in k}
    model.load_state_dict(new_state_dict, strict=False)

elif model_name == 'best': # Normal version

    model = SOHTransformer(input_dim=5, embed_dim=256).to(device)
    model.load_state_dict(torch.load("models/best_soh_tec_model.pth", map_location=device))

monitor_thread = threading.Thread(target=monitor_gpu, args=(avg_power, avg_gpu_util, avg_cpu_util, memory_used_max, f'outputs/log_testing_TRANSFORMER_{model_name}.csv', 0.01), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(model, test_loader, f"models/{model_name}_soh_tec_model.pth", f'outputs/error_results_TRANSFORMER_{model_name}.txt', device)
print(f'Time for {model_name}: {time.time()-start_time} seconds\n')

monitoring = False
