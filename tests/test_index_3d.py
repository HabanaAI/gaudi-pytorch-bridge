import habana_frameworks.torch.core as htcore
import torch

s0 = 4
s1 = 3
s2 = 3


def index_test(device, i) -> torch.Tensor:
    # x = torch.arange(s0*s1*s2, device=device).view(s0, s1, s2)
    x = torch.arange(s0 * s1 * s2).to(device).view(s0, s1, s2)
    p = torch.Tensor([0, 2]).to(device).to(torch.int64)
    q = torch.Tensor([1, 2]).to(device).to(torch.int64)
    r = torch.Tensor([1]).to(device).to(torch.int64)
    s = torch.tensor([0, 1, 2, 0, 1, 2]).to(device).to(torch.int64)
    ind1 = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[1, 2, 0], [0, 2, 1]]]).to(torch.int)
    ind2 = torch.tensor([[0, 0], [1, 1]]).to(torch.int)
    bmask = torch.tensor([[True, True, False], [True, False, True], [False, True, True]]).to(device)
    bmask1 = torch.tensor([True, False, True, False]).to(device)
    bmask2 = torch.tensor([[False, True, False], [False, False, True], [True, True, False], [False, True, True]]).to(
        device
    )
    bmask3 = torch.tensor([True, False, True]).to(device)
    if i == 0:
        return x[r, q, ind2]
    elif i == 1:
        return x[p, q, :]
    elif i == 2:
        return x[:, q, r]
    elif i == 3:
        return x[p, :, q]
    elif i == 4:
        return x[p, :, :]
    elif i == 5:
        return x[:, q, :]
    elif i == 6:
        return x[:, :, q]
    elif i == 7:
        return x[..., q]
    elif i == 8:
        return x[q, ...]
    elif i == 9:
        return x[:, bmask]
    elif i == 10:
        return x[bmask1]
    elif i == 11:
        return x[bmask2]
    elif i == 12:
        return x[bmask2, s]
    elif i == 13:
        return x[:, torch.tensor([0, 1]).to(device), bmask3]
    elif i == 14:
        return x[:, torch.tensor([0, 1]).to(device), torch.tensor([1]).to(device)]
    elif i == 15:
        print(ind1.shape)
        return x[ind1[0], :, ind1[1]]
    elif i == 16:
        return x[ind1[0]]
    else:
        return x[torch.tensor([1, 3]).to("cpu"), torch.tensor([0, 1]), torch.tensor([1, 2])]


if __name__ == "__main__":
    for i in range(18):
        print("====================================case : ", i, "===================================================")
        device = torch.device("cpu")
        index_res_cpu = index_test(device, i)
        print("device = ", device, ", res shape = ", index_res_cpu.shape)
        device = torch.device("hpu")
        index_res = index_test(device, i)
        print("device = ", device, ", res shape = ", index_res.shape)
        print("HPU - CPU results match = ", torch.equal(index_res.to("cpu"), index_res_cpu))
    torch._assert(torch.equal(index_res.to("cpu"), index_res_cpu), "CPU and HPU results don't match")
