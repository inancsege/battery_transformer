import torch
import torch.nn as nn
from .base_model import BaseModel

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation=1, dropout=0.1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Ensure residual connection matches output dimension if needed
        self.residual = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        self.relu_out = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))

        # Adjust spatial dimension if needed for residual connection
        # This padding/cropping depends on how Conv1d affects length
        # For causal padding='causal' this might not be needed
        # For padding=(k-1)*d, output length should match input
        # assert out.shape[2] == res.shape[2]
        # Simple approach: Crop if necessary (might lose info)
        min_len = min(out.shape[2], res.shape[2])
        out = out[:, :, :min_len]
        res = res[:, :, :min_len]

        return self.relu_out(out + res)

class TCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_size=3, num_layers=3, dropout=0.1):
        super(TCNFeatureExtractor, self).__init__()
        layers = []
        channels = [input_dim] + [embed_dim] * num_layers
        for i in range(num_layers):
            dilation_size = 2**i
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1) # Convert to (batch_size, features, seq_len)
        x = self.tcn(x)
        x = self.global_pooling(x).squeeze(-1) # (batch_size, embed_dim)
        return x

class SOHTCN(BaseModel): # Inherit from BaseModel if defined
    def __init__(self, input_dim, embed_dim=256, output_dim=30, tcn_layers=3, tcn_kernel_size=3, tcn_dropout=0.1):
        super(SOHTCN, self).__init__()
        self.feature_extractor = TCNFeatureExtractor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, output_dim)  # Regression output
        )
        self.output_dim = output_dim

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.mlp_head(x)
        # Ensure output shape matches target [batch_size, output_dim]
        if self.output_dim == 1:
             return output.squeeze(-1)
        return output
