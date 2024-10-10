import torch

s0 = 5
s1 = 4
s2 = 3
s3 = 3


def index_original(device) -> torch.Tensor:
    x = torch.arange(s0 * s1 * s2 * s3, device=device).view(s0, s1, s2, s3)
    # x = torch.arange(s0*s1*s2*s3).view(s0, s1, s2, s3).to(device)
    torch.Tensor([0, 2]).to(device).to(torch.int64)
    torch.Tensor([1, 2]).to(device).to(torch.int64)
    c = torch.Tensor([0, 1]).to(device).to(torch.int64)
    print("input tensor device = ", x.device)
    # return x[c, a, :, :]
    # return x[:, a, b, :]
    # return x[:, :, a, b]
    # return x[:, a, :, c]
    return x[torch.tensor([1, 2]).to("cpu"), :, :, c]
    # return x[a, :, :, c]


if __name__ == "__main__":
    device = torch.device("hpu")
    index_res = index_original(device)
    print(
        "device = ",
        device,
        ", res shape = ",
        index_res.shape,
        ", res = ",
        index_res.to("cpu"),
    )
    device = torch.device("cpu")
    index_res_cpu = index_original(device)
    print(
        "device = ",
        device,
        ", res shape = ",
        index_res_cpu.shape,
        ", res = ",
        index_res_cpu,
    )
    print("HPU - CPU results match = ", torch.equal(index_res.to("cpu"), index_res_cpu))
