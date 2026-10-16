import torch

def make_tensor():
    return torch.arange(1, 7).reshape(2, 3).to(dtype=torch.float32)