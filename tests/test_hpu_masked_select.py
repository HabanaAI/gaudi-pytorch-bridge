import torch
from test_utils import compare_tensors


def test_hpu_masked_select():
    device_hpu = torch.device("hpu")
    device_cpu = torch.device("cpu")
    X = torch.randn(5, 4)
    Xh = X.to(device_hpu)

    mask = X.ge(0.6)
    maskh = mask.to(device_hpu)

    B = torch.masked_select(X, mask)
    Bh = torch.masked_select(Xh, maskh)

    B_cpu = Bh.to(device_cpu)
    compare_tensors(B_cpu, B, atol=0.001, rtol=1.0e-3)


def test_hpu_index_bool():
    device_hpu = torch.device("hpu")
    device_cpu = torch.device("cpu")
    X = torch.randn(5, 4)

    mask = X.ge(0.5)
    out_cpu = mask[mask != 0]
    maskh = mask.to(device_hpu)

    out = maskh[maskh != 0]
    out_hpu = out.to(device_cpu)
    compare_tensors(out_cpu, out_hpu, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_hpu_masked_select()
    test_hpu_index_bool()
