import torch.nn as nn

class BaseModel(nn.Module):
    """Optional: Abstract Base Class for PyTorch models."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def get_params(self):
        """Returns model parameters (useful for config)."""
        return {}