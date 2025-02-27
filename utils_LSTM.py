import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ------------------------------
# LSTM Model for SOH Estimation
# ------------------------------
class SOHLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, dropout=0.1):
        super(SOHLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Regression output for SOH estimation
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, seq_len, hidden_dim)
        x = lstm_out[:, -1, :]  # Take the last time step's output
        return self.fc(x)  # Final regression output

# ------------------------------
# SOH Dataset Class
# ------------------------------
class SOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Ensure float32 for LSTM
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------
# Function to Create Sequences
# ------------------------------
def create_sequences(X, y, seq_len):
    sequences, targets = [], []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        targets.append(y[i+seq_len])  # Predict next SOH value

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
