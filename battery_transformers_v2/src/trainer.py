import torch
import time
import os
from .monitoring import GpuMonitor # Import from monitoring module

class Trainer:
    """Handles the training loop for PyTorch models."""
    def __init__(self, model, criterion, optimizer, device, monitor=None, log_file='training_log.csv', monitor_interval=1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.monitor = monitor
        self.log_file = log_file
        self.monitor_interval = monitor_interval
        self.monitoring_thread = None

    def _start_monitoring(self):
        if self.monitor:
            self.monitor.start(self.log_file, self.monitor_interval)
            print(f"Started monitoring training to {self.log_file}")

    def _stop_monitoring(self):
        if self.monitor:
            self.monitor.stop()
            time.sleep(2) # Allow time for final log entry
            print("Stopped monitoring training.")

    def train_epoch(self, train_loader):
        """Runs a single training epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            # Adjust target shape if model output is squeezed (e.g., single output)
            if outputs.ndim == 1 and batch_y.ndim == 2 and batch_y.shape[1] == 1:
                batch_y = batch_y.squeeze(-1)
            elif outputs.ndim == 2 and batch_y.ndim == 1 and outputs.shape[1] == 1:
                 outputs = outputs.squeeze(-1)

            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if (i + 1) % 50 == 0: # Print progress
                print(f'  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}')

        return epoch_loss / num_batches

    def validate_epoch(self, val_loader):
        """Runs a single validation epoch."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(val_loader)
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                # Adjust target shape similarly to training
                if outputs.ndim == 1 and batch_y.ndim == 2 and batch_y.shape[1] == 1:
                    batch_y = batch_y.squeeze(-1)
                elif outputs.ndim == 2 and batch_y.ndim == 1 and outputs.shape[1] == 1:
                     outputs = outputs.squeeze(-1)

                loss = self.criterion(outputs, batch_y)
                epoch_loss += loss.item()
        return epoch_loss / num_batches

    def train(self, train_loader, val_loader, num_epochs, model_save_path):
        """Runs the full training process."""
        best_val_loss = float("inf")
        self._start_monitoring()

        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate_epoch(val_loader) if val_loader else float('inf')
                end_time = time.time()

                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"  Best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"An error occurred during training: {e}")
        finally:
            self._stop_monitoring()
