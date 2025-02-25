import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# Custom Truncated Normal Initialization
# ------------------------------
def truncated_normal_(tensor, mean=0.0, std=0.2, a=-2.0, b=2.0):
    """
    Custom truncated normal initialization.
    This method ensures values stay within the range [a, b].
    """
    lower, upper = (a - mean) / std, (b - mean) / std
    tensor.data = torch.distributions.Normal(mean, std).rsample(tensor.shape)
    tensor.data = torch.clip(tensor.data, min=a, max=b)
    return tensor

# ------------------------------
# Input Embedding Module (CNN-Based)
# ------------------------------
class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_sizes=[4, 3], strides=[2, 2]):
        super(InputEmbedding, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm1d(embed_dim)

        self.cls_token = nn.Parameter(truncated_normal_(torch.empty(1, 1, embed_dim)))
        self.pos_embedding = nn.Parameter(truncated_normal_(torch.empty(1, embed_dim + 1, embed_dim)))

    def forward(self, x):
        """
        x: (batch_size, channels, time_steps)
        Output: (batch_size, seq_len+1, embed_dim)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Reshape for transformer (batch_size, seq_len, embed_dim)

        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.shape[1], :]

        return x

# ------------------------------
# Multi-Head Self-Attention
# ------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(attn_output)

# ------------------------------
# Feed-Forward Network (MLP Block)
# ------------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, ffn_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

# ------------------------------
# DropPath (Stochastic Depth)
# ------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ------------------------------
# Transformer Block
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=16, ffn_dim=1024, drop_path_rate=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), mask))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x

# ------------------------------
# Transformer Encoder
# ------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, drop_path_rate)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ------------------------------
# SOH-TEC Model
# ------------------------------
class SOHTEC(nn.Module):
    def __init__(self, input_dim=5, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1):
        super(SOHTEC, self).__init__()

        self.embedding = InputEmbedding(input_dim, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_blocks, num_heads, ffn_dim, drop_path_rate)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # Regression output for SOH estimation
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])  # Using CLS token for SOH prediction

class SOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).permute(0, 2, 1)
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_sequences(X, y, seq_len):
    
    sequences = []
    targets = []
    
    for i in range(0, len(X) - seq_len, seq_len):
        sequences.append(X[i:i+seq_len])
        if y is not None:
            targets.append(y[i+seq_len-1])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)