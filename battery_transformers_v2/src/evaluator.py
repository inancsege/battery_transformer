import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from .monitoring import GpuMonitor # Import from monitoring module

class Evaluator:
    """Handles model evaluation and metrics calculation."""
    def __init__(self, model, device, monitor=None, log_file='evaluation_log.csv', monitor_interval=0.01):
        self.model = model
        self.device = device
        self.monitor = monitor
        self.log_file = log_file
        self.monitor_interval = monitor_interval

    def _start_monitoring(self):
        if self.monitor:
            self.monitor.start(self.log_file, self.monitor_interval)
            print(f"Started monitoring evaluation to {self.log_file}")

    def _stop_monitoring(self):
        if self.monitor:
            self.monitor.stop()
            time.sleep(2) # Allow time for final log entry
            print("Stopped monitoring evaluation.")

    def _calculate_metrics(self, all_targets, all_preds):
        """Calculates standard regression metrics."""
        # Ensure inputs are numpy arrays and flattened if necessary
        all_targets_np = np.array(all_targets).flatten()
        all_preds_np = np.array(all_preds).flatten()

        if len(all_targets_np) != len(all_preds_np):
            print(f"Warning: Target ({len(all_targets_np)}) and prediction ({len(all_preds_np)}) lengths differ. Metrics might be inaccurate.")
            # Attempt to align if possible (e.g., if one predicts sequence and other is last step)
            # This requires more context - for now, calculate on available aligned data if possible
            min_len = min(len(all_targets_np), len(all_preds_np))
            if min_len == 0:
                return {'RMSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan'), 'PCC': float('nan'), 'MDA': float('nan')}
            all_targets_np = all_targets_np[:min_len]
            all_preds_np = all_preds_np[:min_len]

        if len(all_targets_np) < 2:
            print("Warning: Need at least 2 data points for R2, PCC, MDA.")
            rmse = np.sqrt(mean_squared_error(all_targets_np, all_preds_np)) if len(all_targets_np) > 0 else float('nan')
            mae = mean_absolute_error(all_targets_np, all_preds_np) if len(all_targets_np) > 0 else float('nan')
            return {'RMSE': rmse, 'MAE': mae, 'R2': float('nan'), 'PCC': float('nan'), 'MDA': float('nan')}

        rmse = np.sqrt(mean_squared_error(all_targets_np, all_preds_np))
        mae = mean_absolute_error(all_targets_np, all_preds_np)
        r2 = r2_score(all_targets_np, all_preds_np)

        # Compute Pearson Correlation Coefficient (PCC)
        pcc, _ = pearsonr(all_targets_np, all_preds_np)

        # Compute Mean Directional Accuracy (MDA)
        direction_actual = np.sign(np.diff(all_targets_np))
        direction_pred = np.sign(np.diff(all_preds_np))
        mda = np.mean(direction_actual == direction_pred)

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'PCC': pcc,
            'MDA': mda
        }
        return metrics

    def evaluate(self, test_loader, model_path, results_save_path, plot_fig=True, figure_save_path='output_figure.png', plot_model_name='model'):
        """Evaluates PyTorch models (LSTM, TCN, Transformer, CNN)."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            # Attempt evaluation even if loading fails (e.g., if model is already trained)

        self.model.eval()
        self._start_monitoring()

        all_preds = []
        all_targets = []
        first_batch_plotted = False

        try:
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)

                    # Handle potential squeezing by the model
                    if outputs.ndim == 1 and batch_y.ndim == 2 and batch_y.shape[1] == 1:
                         batch_y_np = batch_y.squeeze(-1).cpu().numpy()
                    elif outputs.ndim == 2 and batch_y.ndim == 1 and outputs.shape[1] == 1:
                         outputs_np = outputs.squeeze(-1).cpu().numpy()
                         batch_y_np = batch_y.cpu().numpy()
                    else:
                         outputs_np = outputs.cpu().numpy()
                         batch_y_np = batch_y.cpu().numpy()

                    all_preds.append(outputs_np)
                    all_targets.append(batch_y_np)

                    if plot_fig and not first_batch_plotted and outputs_np.ndim > 1:
                        # Plot only if multi-step predictions exist for comparison
                        pred_out_example = outputs_np[0]
                        target_out_example = batch_y_np[0]
                        plt.figure(figsize=(10, 5))
                        plt.plot(pred_out_example, label="predicted", linestyle="dashed", alpha=0.7)
                        plt.plot(target_out_example, label="target", linestyle="solid", alpha=0.7)
                        plt.ylabel("SOH (Scaled)")
                        plt.xlabel("Time Step (Sequence)")
                        plt.title(f'{plot_model_name} Example Prediction vs Target')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(figure_save_path)
                        plt.close()
                        print(f"Example figure saved to {figure_save_path}")
                        first_batch_plotted = True
                    elif plot_fig and not first_batch_plotted and outputs_np.ndim == 1:
                         print("Skipping example plot for single-step predictions.")
                         first_batch_plotted = True # Avoid trying again

            # Concatenate results from all batches
            # Handle potential shape mismatches if batches have varying prediction lengths
            try:
                 all_preds_np = np.concatenate(all_preds, axis=0)
                 all_targets_np = np.concatenate(all_targets, axis=0)
            except ValueError as e:
                 print(f"Warning: Could not concatenate batch results due to shape mismatch: {e}. Calculating metrics per batch.")
                 # Fallback: calculate metrics per batch (less ideal)
                 batch_metrics = [self._calculate_metrics(t, p) for t, p in zip(all_targets, all_preds)]
                 # Average metrics (simple average, might be skewed by batch size)
                 metrics = {k: np.nanmean([m[k] for m in batch_metrics]) for k in batch_metrics[0]} if batch_metrics else {}
            else:
                metrics = self._calculate_metrics(all_targets_np, all_preds_np)

            print(f"\nTest RMSE: {metrics.get('RMSE', float('nan')):.4f}")
            print(f"Test MAE: {metrics.get('MAE', float('nan')):.4f}")
            print(f"Test R2: {metrics.get('R2', float('nan')):.4f}")
            print(f"Test PCC: {metrics.get('PCC', float('nan')):.4f}")
            print(f"Test MDA: {metrics.get('MDA', float('nan')):.4f}")

            with open(results_save_path, "w") as f:
                f.write(f"Test RMSE: {metrics.get('RMSE', float('nan')):.4f}\n")
                f.write(f"Test MAE: {metrics.get('MAE', float('nan')):.4f}\n")
                f.write(f"Test R2: {metrics.get('R2', float('nan')):.4f}\n")
                f.write(f"Test PCC: {metrics.get('PCC', float('nan')):.4f}\n")
                f.write(f"Test MDA: {metrics.get('MDA', float('nan')):.4f}\n")
            print(f"Results saved to {results_save_path}")

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
        finally:
            self._stop_monitoring()

    def evaluate_xgb(self, X_test, y_test, results_save_path):
        """Evaluates the XGBoost model."""
        self.model.eval() # No real effect, but for consistency
        self._start_monitoring()

        try:
            predictions = self.model.predict(X_test)
            metrics = self._calculate_metrics(y_test, predictions)

            print(f"\nTest RMSE: {metrics.get('RMSE', float('nan')):.4f}")
            print(f"Test MAE: {metrics.get('MAE', float('nan')):.4f}")
            print(f"Test R2: {metrics.get('R2', float('nan')):.4f}")
            print(f"Test PCC: {metrics.get('PCC', float('nan')):.4f}")
            print(f"Test MDA: {metrics.get('MDA', float('nan')):.4f}")

            with open(results_save_path, "w") as f:
                f.write(f"Test RMSE: {metrics.get('RMSE', float('nan')):.4f}\n")
                f.write(f"Test MAE: {metrics.get('MAE', float('nan')):.4f}\n")
                f.write(f"Test R2: {metrics.get('R2', float('nan')):.4f}\n")
                f.write(f"Test PCC: {metrics.get('PCC', float('nan')):.4f}\n")
                f.write(f"Test MDA: {metrics.get('MDA', float('nan')):.4f}\n")
            print(f"Results saved to {results_save_path}")

        except Exception as e:
            print(f"An error occurred during XGBoost evaluation: {e}")
        finally:
            self._stop_monitoring()
