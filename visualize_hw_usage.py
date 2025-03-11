import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths and corresponding model names
file_paths = {
    "XGBoost": "outputs/log_training_XGB.csv",
    "Transformer": "outputs/log_training_TRANSFORMER.csv",
    "TCN": "outputs/log_training_TCN.csv",
    "LSTM": "outputs/log_training_LSTM.csv",
    "CNN": "outputs/log_training_CNN.csv",
}

# Initialize lists to store values
models = []
total_energy = []
max_memory = []

# Read data from each file
for model, file_path in file_paths.items():
    try:
        df = pd.read_csv(file_path)
        
        # Extract total energy consumption and max memory usage
        if "Power (W)" in df.columns and "Memory Used (MB)" in df.columns:
            models.append(model)
            total_energy.append(df["Power (W)"].sum())  # Summing energy consumption
            max_memory.append(df["Memory Used (MB)"].max())  # Getting max memory usage
    except Exception as e:
        print(f"Error processing {model}: {e}")

# Convert lists to numpy arrays for plotting
total_energy = np.array(total_energy)
max_memory = np.array(max_memory)

# Bar width and x positions
x = np.arange(len(models))
bar_width = 0.35

print(total_energy, max_memory)

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot total energy consumption on the left y-axis
ax1.bar(x, total_energy, bar_width, label="Total Energy Consumption (J)", color='tab:blue')
ax1.set_xlabel("ML Methods")
ax1.set_ylabel("Total Energy Consumption (J)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create second y-axis for memory usage
ax2 = ax1.twinx()
ax2.plot(x, max_memory, marker='o', linestyle='-', color='tab:red', label="Max Memory Usage (MB)")
ax2.set_ylabel("Max Memory Usage (MB)", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Set x-ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels(models)

# Title
plt.title("Energy Consumption and Memory Usage Across ML Methods")

# Show plot
plt.tight_layout()
plt.grid()
plt.savefig('outputs/figures/hw_usage_TRAINING_results.png')
