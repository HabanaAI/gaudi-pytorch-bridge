import torch
torch.ops.load_library("libhabana_device.so")
torch.ops.load_library("libhabana_kernels.so")

def test_hpu_device():
    device = torch.device('habana')
    tensor = torch.randn(5)

    tensor.to(device)