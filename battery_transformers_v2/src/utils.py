import torch

def truncated_normal_(tensor, mean=0.0, std=0.2, a=-2.0, b=2.0):
    """
    Custom truncated normal initialization.
    Fills the input Tensor with values drawn from a truncated
    normal distribution N(mean, std) with values outside [a, b] clipped.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")
    # Calculate clipping points in standard normal space
    lower, upper = (a - mean) / std, (b - mean) / std
    # Sample from standard normal distribution
    torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
    # Clip values in standard normal space
    tensor.data.clamp_(min=lower, max=upper)
    # Scale and shift to the target distribution
    tensor.data.mul_(std).add_(mean)
    return tensor

# Add other truly generic utility functions here if any.
# Avoid putting model definitions, data loading, training, etc. here.
