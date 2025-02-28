import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# Custom Truncated Normal Initialization
# ------------------------------
def truncated_normal_(tensor, mean=0.0, std=0.2, a=-2.0, b=2.0):
    """
    Custom truncated normal initialization.
    """
    lower, upper = (a - mean) / std, (b - mean) / std
    tensor.data = torch.distributions.Normal(mean, std).rsample(tensor.shape)
    tensor.data = torch.clip(tensor.data, min=a, max=b)
    return tensor

# ------------------------------
# TCN-Based Feature Extraction
# ------------------------------
class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.residual = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Ensure the same sequence length for residual and output
        min_length = min(x.shape[2], residual.shape[2])
        x = x[:, :, :min_length]
        residual = residual[:, :, :min_length]
        
        return x + residual

class TCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_size=3, num_layers=3):
        super(TCNFeatureExtractor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TCNBlock(input_dim if i == 0 else embed_dim, embed_dim, kernel_size, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.tcn(x)
        x = self.global_pooling(x).squeeze(-1)
        return x

# ------------------------------
# TCN Model for SOH Estimation
# ------------------------------
class TCNModel(nn.Module):
    def __init__(self, input_dim=5, embed_dim=256):
        super(TCNModel, self).__init__()
        self.feature_extractor = TCNFeatureExtractor(input_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # Regression output
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.mlp_head(x)

class SOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).permute(0, 2, 1)
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Function to create sequences
def create_sequences(X, y, seq_len):
    sequences, targets = [], []
    for i in range(0, len(X) - seq_len, seq_len):
        sequences.append(X[i:i+seq_len])
        if y is not None:
            targets.append(y[i+seq_len-1])
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
