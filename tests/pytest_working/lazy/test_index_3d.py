import torch
from test_utils import compare_tensors, cpu, hpu

s0 = 4
s1 = 3
s2 = 3


def index_original(device) -> torch.Tensor:
    x = torch.arange(s0 * s1 * s2, device=device).view(s0, s1, s2)
    # x = torch.arange(s0*s1*s2).view(s0, s1, s2).to(device)
    torch.Tensor([0, 2]).to(device).to(torch.int64)
    # z = torch.Tensor([1, 2]).to(device).to(torch.int64)
    torch.Tensor([1, 2]).to(device).to(torch.int64)
    torch.Tensor([1]).to(device).to(torch.int64)
    bmask1 = torch.tensor([False, True, False]).to(device)
    return x[:, torch.tensor([0, 1]).to("cpu"), bmask1]


def test_index_3d():
    index_hpu = (index_original(hpu).to("cpu"),)
    index_cpu = index_original(cpu)
    compare_tensors(index_hpu, index_cpu, atol=0, rtol=0)
