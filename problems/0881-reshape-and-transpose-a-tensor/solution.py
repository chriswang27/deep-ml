import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    x = x.reshape(-1,)
    x = x.reshape(new_shape)
    return x

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-1, -2)