import torch
torch.ops.load_library("libhabana_pytorch_plugin.so")

# @torch.jit.script


def test_hpu_device():
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    cpu_tensor = torch.randn(5)

    print(cpu_tensor)
    hpu_tensor = cpu_tensor.to(hpu)
    cpu_tensor2 = hpu_tensor.to(cpu)

    print(cpu_tensor2)
    pass


test_hpu_device()
