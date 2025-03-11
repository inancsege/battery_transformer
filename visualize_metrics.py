import matplotlib.pyplot as plt
import numpy as np
import os

file_paths = {
    "XGBoost": "outputs/error_results_XGB.txt",
    "Transformer": "outputs/error_results_TRANSFORMER.txt",
    "TCN": "outputs/error_results_TCN.txt",
    "LSTM": "outputs/error_results_LSTM.txt",
    "CNN": "outputs/error_results_CNN.txt",
}

metrics = {
    "RMSE": [],
    "MAE": [],
    "R²": [],
    "PCC": [],
    "MDA": []
}

models = []

for model, file_path in file_paths.items():
    if os.path.exists(file_path):
        models.append(model)
        with open(file_path, "r") as file:
            for line in file:
                if "Test RMSE" in line:
                    metrics["RMSE"].append(float(line.split(":")[1].strip()))
                elif "Test MAE" in line:
                    metrics["MAE"].append(float(line.split(":")[1].strip()))
                elif "Test R²" in line:
                    metrics["R²"].append(float(line.split(":")[1].strip()))
                elif "Test PCC" in line:
                    metrics["PCC"].append(float(line.split(":")[1].strip()))
                elif "Test MDA" in line:
                    metrics["MDA"].append(float(line.split(":")[1].strip()))

for key in metrics:
    metrics[key] = np.array(metrics[key])

bar_width = 0.15
x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    ax.bar(x + i * bar_width, [metrics[m][i] for m in metrics], bar_width, label=model)

ax.set_xlabel("Metrics")
ax.set_ylabel("Values")
ax.set_title("Comparison of Error Metrics Across Models")
ax.set_xticks(x + (bar_width * (len(models) - 1) / 2))
ax.set_xticklabels(metrics.keys())
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/metric_results.png')
