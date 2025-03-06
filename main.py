import subprocess

python_files = [
    "battery_transformer.py",
    "battery_cnn.py",
    "battery_tcn.py",
    "battery_xgb.py",
    "battery_lstm.py"
]

for script in python_files:
    try:
        print(f"Running {script}...")
        result = subprocess.run(["python", script], check=True)
        print(f"Finished {script} with exit code {result.returncode}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break 
