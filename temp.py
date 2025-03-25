import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess, time, psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

def evaluate_model(model, test_loader, model_save_file, output_save_file, plot_model_name='model', plot_fig = True, device=torch.device("cpu"), return_error_results = False, use_gpu = True):
    model.load_state_dict(torch.load(model_save_file))
    model.eval()

    first_test_flag = True

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            if use_gpu:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            if first_test_flag and plot_fig:
                pred_out = outputs[0].cpu().numpy()
                target_out = batch_y[0].cpu().numpy()
                first_test_flag = False

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute error metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # Compute Pearson Correlation Coefficient (PCC)
    pcc, _ = pearsonr(all_targets, all_preds)

    # Compute Mean Directional Accuracy (MDA)
    direction_actual = np.sign(np.diff(all_targets))
    direction_pred = np.sign(np.diff(all_preds))

    mda = np.mean(direction_actual == direction_pred)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test PCC: {pcc}")
    print(f"Test MDA: {mda}")

    with open(output_save_file, "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test MAE: {mae:.4f}\n")
        f.write(f"Test R²: {r2:.4f}\n")
        f.write(f"Test PCC: {pcc}\n")
        f.write(f"Test MDA: {mda}\n")

    if plot_fig:
        plt.figure(figsize=(10, 5))
        plt.plot(pred_out, label="predicted", linestyle="dashed", alpha=0.7)
        plt.plot(target_out, label="target", linestyle="solid", alpha=0.7)
        
        plt.ylabel("SOH (%)")
        plt.xticks([])
        plt.title(f'{plot_model_name} example')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'outputs/figures/{plot_model_name}_example.png')

    if return_error_results:
        return rmse, mae, r2, pcc, mda