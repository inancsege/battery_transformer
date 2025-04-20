import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from ..utils import truncated_normal_ # Import from refactored utils

# --- Components from original utils.py --- #

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
            # Ensure mask has the correct shape [batch_size, num_heads, seq_len, seq_len]
            if mask.dim() == 2: # Typical padding mask [batch_size, seq_len]
                mask = mask[:, None, None, :].expand(batch_size, self.num_heads, seq_len, seq_len)
            elif mask.dim() == 3: # Typical causal mask [seq_len, seq_len]
                 mask = mask[None, None, :, :].expand(batch_size, self.num_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(attn_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, ffn_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Ensure random_tensor has broadcastable shape with x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # Workaround for shape error
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class TransformerBlock(nn.Module):
    # Using PyTorch's built-in layer is generally preferred and optimized
    def __init__(self, embed_dim, num_heads=16, ffn_dim=1024, drop_path_rate=0.1, attn_dropout=0.1, activation="relu", batch_first=True):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=drop_path_rate,
            activation=activation,
            batch_first=batch_first
            # attn_dropout is part of dropout in TransformerEncoderLayer
        )

    def forward(self, x, mask=None, src_key_padding_mask=None):
         # src_mask is for attention masking (e.g., causal), src_key_padding_mask is for padding
        return self.encoder_layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)


# --- SOH Transformer Model --- #

class SOHTransformer(BaseModel): # Inherit from BaseModel if defined
    def __init__(self, input_dim, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1, output_dim=30):
        super(SOHTransformer, self).__init__()
        # Simple linear embedding instead of CNN embedding from original utils.py
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Positional encoding (add if needed, e.g., learned or sinusoidal)
        # self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=drop_path_rate, batch_first=True),
            num_layers=num_blocks
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, output_dim) # Output matches target length
        )
        self.output_dim = output_dim

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x) # Shape: (batch_size, seq_len, embed_dim)
        # Add positional encoding here if using
        # x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x) # Shape: (batch_size, seq_len, embed_dim)
        # Use the output of the last time step for prediction
        output = self.mlp_head(x[:, -1, :]) # Shape: (batch_size, output_dim)
        # Ensure output shape matches target [batch_size, output_dim]
        if self.output_dim == 1:
             return output.squeeze(-1)
        return output

# --- SOH Transformer HDMR Model (if significantly different) --- #
# If SOHTransformerHDMR has the exact same architecture but different params,
# just use SOHTransformer with a different config file.
# If the architecture IS different, define it here.

class SOHTransformerHDMR(SOHTransformer): # Example: Inherit if similar
     def __init__(self, input_dim, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1, output_dim=100):
         # Call parent constructor with potentially different defaults or add modifications
         super(SOHTransformerHDMR, self).__init__(
             input_dim=input_dim,
             embed_dim=embed_dim,
             num_blocks=num_blocks,
             num_heads=num_heads,
             ffn_dim=ffn_dim,
             drop_path_rate=drop_path_rate,
             output_dim=output_dim
         )
         # Add any HDMR-specific layers or modifications here
         print("Initialized SOHTransformerHDMR")

     # Override forward if behavior changes
     # def forward(self, x):
     #     # HDMR-specific forward pass
     #     return super().forward(x) # Or custom logic

