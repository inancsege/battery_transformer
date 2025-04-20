import torch
import torch.nn as nn
from .base_model import BaseModel

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_sizes=[4, 3], strides=[2, 2]):
        super(CNNFeatureExtractor, self).__init__()

        # Ensure lists are not empty
        if not kernel_sizes or not strides or len(kernel_sizes) != len(strides):
            raise ValueError("kernel_sizes and strides must be non-empty lists of the same length")

        layers = []
        in_channels = input_dim
        for i in range(len(kernel_sizes)):
            layers.append(nn.Conv1d(in_channels, embed_dim, kernel_sizes[i], stride=strides[i]))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            in_channels = embed_dim # Output channels of conv become input for next

        self.conv_layers = nn.Sequential(*layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, features)
        Output: (batch_size, embed_dim)
        """
        x = x.permute(0, 2, 1) # Convert to (batch_size, features, seq_len)
        x = self.conv_layers(x) # Shape depends on conv layers, e.g., (batch_size, embed_dim, reduced_seq_len)
        x = self.global_pooling(x).squeeze(-1) # (batch_size, embed_dim)
        return x

class SOHCNN(BaseModel): # Inherit from BaseModel if defined
    def __init__(self, input_dim, embed_dim=256, output_dim=100, cnn_kernel_sizes=[4,3], cnn_strides=[2,2]):
        super(SOHCNN, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides
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
