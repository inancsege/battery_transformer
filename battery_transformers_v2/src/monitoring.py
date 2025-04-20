import subprocess
import threading
import time
import psutil
import os

def monitor_idle_gpu_cpu(duration=10, interval=1):
    """Monitors idle GPU and CPU usage to establish baseline."""
    power_values = []
    gpu_util_values = []
    cpu_util_values = []

    start_time = time.time()
    print(f"Monitoring idle usage for {duration} seconds...")
    while time.time() - start_time < duration:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            power, gpu_util = map(float, result.stdout.strip().split(", "))
            power_values.append(power)
            gpu_util_values.append(gpu_util)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not get GPU stats via nvidia-smi: {e}. Using 0 for idle values.")
            power_values.append(0)
            gpu_util_values.append(0)
            # Break if nvidia-smi not found
            if isinstance(e, FileNotFoundError):
                break

        cpu_util = psutil.cpu_percent(interval=0.1)
        cpu_util_values.append(cpu_util)
        time.sleep(interval)

    avg_power = sum(power_values) / len(power_values) if power_values else 0
    avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0
    avg_cpu_util = sum(cpu_util_values) / len(cpu_util_values) if cpu_util_values else 0

    print("Idle monitoring complete.")
    return avg_power, avg_gpu_util, avg_cpu_util


class GpuMonitor:
    """Class to handle GPU and CPU monitoring in a separate thread."""
    def __init__(self, avg_power, avg_gpu_util, avg_cpu_util):
        self.monitoring = False
        self.thread = None
        self.avg_power = avg_power
        self.avg_gpu_util = avg_gpu_util
        self.avg_cpu_util = avg_cpu_util
        self.log_file = None
        self.interval = 1

    def _monitor_loop(self):
        """The loop executed by the monitoring thread."""
        query_params = [
            "timestamp", "power.draw", "memory.used", "memory.total",
            "utilization.gpu", "utilization.memory", "temperature.gpu",
            "fan.speed", "clocks.sm", "clocks.gr"
        ]
        query_str = ",".join(query_params)

        header = ("Timestamp,Power_Draw_W,Memory_Used_MB,Memory_Total_MB,GPU_Util_Pct,"
                  "Memory_Util_Pct,Temp_C,Fan_Speed_Pct,Clock_SM_MHz,Clock_Mem_MHz,"
                  "Power_Above_Idle_W,GPU_Util_Above_Idle_Pct,CPU_Util_Above_Idle_Pct\n") # Added calculated fields

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, "w") as f:
            f.write(header)

        while self.monitoring:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + query_str, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )

                raw_values = result.stdout.strip().split(", ")
                timestamp_str = raw_values[0]
                gpu_data = []
                for val_str in raw_values[1:]:
                    if '[N/A]' in val_str:
                        gpu_data.append(0.0) # Handle N/A values
                    else:
                        try:
                            gpu_data.append(float(val_str))
                        except ValueError:
                            gpu_data.append(0.0) # Handle potential non-float values

                if len(gpu_data) < 9:
                     print(f"Warning: Incomplete GPU data received: {gpu_data}")
                     time.sleep(self.interval)
                     continue

                # Calculate values above idle
                power_above_idle = gpu_data[0] - self.avg_power if gpu_data[0] is not None else 0.0
                gpu_util_above_idle = gpu_data[3] - self.avg_gpu_util if gpu_data[3] is not None else 0.0
                cpu_usage = psutil.cpu_percent()
                cpu_above_idle = cpu_usage - self.avg_cpu_util

                log_entry = (f"{timestamp_str},{','.join(map(str, gpu_data))},"
                             f"{power_above_idle:.2f},{gpu_util_above_idle:.2f},{cpu_above_idle:.2f}\n")

                with open(self.log_file, "a") as f:
                    f.write(log_entry)

            except subprocess.CalledProcessError as e:
                print(f"Error running nvidia-smi: {e}")
                time.sleep(self.interval * 5) # Wait longer if error
            except FileNotFoundError:
                print("Error: nvidia-smi command not found.")
                self.monitoring = False # Stop if command fails
                break
            except Exception as e:
                print(f"An unexpected error occurred in monitor_gpu: {e}")
                # Continue monitoring unless it's critical

            time.sleep(self.interval)

    def start(self, log_file, interval):
        """Starts the monitoring thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Monitoring is already running.")
            return

        self.log_file = log_file
        self.interval = interval
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        if self.thread is not None and self.thread.is_alive():
            self.monitoring = False
            self.thread.join() # Wait for the thread to finish
            print("Monitoring thread stopped.")
        self.thread = None
