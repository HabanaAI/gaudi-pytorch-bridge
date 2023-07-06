import torch

s0 = 4
s1 = 3
s2 = 3


def index_original(device) -> torch.Tensor:
    x = torch.arange(s0 * s1 * s2, device=device).view(s0, s1, s2)
    torch.Tensor([0, 2]).to(device).to(torch.int64)
    torch.Tensor([1, 2]).to(device).to(torch.int64)
    # return x[y, z, :]
    # return x[:, z, y]
    # return x[y, :, z]
    # return x[y, :, :]
    # return x[:, z, :]
    # return x[:, [[True, False, True],[False, True, False],[True, False, False]]]
    return x[
        [
            [True, False, True],
            [False, True, False],
            [True, False, False],
            [False, True, True],
        ],
        :,
    ]


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
    index_res = index_original(device)
    print(
        "device = ",
        device,
        ", res shape = ",
        index_res.shape,
        ", res = ",
        index_res.to("cpu"),
    )
