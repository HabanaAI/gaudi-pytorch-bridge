import habana_frameworks.torch.core as htcore
import torch

s0 = 5
s1 = 4
s2 = 3
s3 = 3


def index_test(device, i) -> torch.Tensor:
    # x = torch.arange(s0*s1*s2*s3, device=device).view(s0, s1, s2, s3)
    x = torch.arange(s0 * s1 * s2 * s3).view(s0, s1, s2, s3).to(device)
    a = torch.Tensor([0, 2]).to(device).to(torch.int64)
    b = torch.Tensor([1, 2]).to(device).to(torch.int64)
    c = torch.Tensor([0, 1]).to(device).to(torch.int64)
    if i == 0:
        return x[c, a, :, :]
    elif i == 1:
        return x[:, a, b, :]
    elif i == 2:
        return x[:, a, b]
    elif i == 3:
        return x[:, a, b]
    elif i == 4:
        return x[torch.tensor([1, 2]).to(device), a]
    elif i == 5:
        return x[
            torch.tensor(
                [
                    [True, False, True, False],
                    [False, True, False, True],
                    [True, False, True, False],
                    [False, True, False, True],
                    [True, True, True, True],
                ]
            ).to(device)
        ]
    elif i == 6:
        return x[:, :, a, b]
    elif i == 7:
        return x[:, a, :, c]
    elif i == 8:
        return x[torch.tensor([1, 2]).to("cpu"), :, :, c]
    else:
        return x[a, :, :, c]


if __name__ == "__main__":
    for i in range(10):
        print("====================================case : ", i, "===================================================")
        device = torch.device("cpu")
        index_res_cpu = index_test(device, i)
        print("device = ", device, ", res shape = ", index_res_cpu.shape)
        device = torch.device("hpu")
        index_res = index_test(device, i)
        print("device = ", device, ", res shape = ", index_res.shape)
        print("HPU - CPU results match = ", torch.equal(index_res.to("cpu"), index_res_cpu))
    torch._assert(torch.equal(index_res.to("cpu"), index_res_cpu), "CPU and HPU results don't match")
