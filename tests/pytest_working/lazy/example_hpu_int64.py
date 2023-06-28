#!/usr/bin/env python

import pytest
import torch


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_add(device, dtype1, dtype2=None):
    if not dtype2:
        dtype2 = dtype1
    val1 = int(2**20)
    val2 = val1
    input1 = torch.tensor([val1], dtype=dtype1, device=device)
    input2 = torch.tensor([val2], dtype=dtype2, device=device)

    res = input1 + input2
    res = res.cpu()
    print(res)
    print(res.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_argmax(device, dtype):
    a = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 4]], dtype=dtype, device=device)
    b = torch.argmax(a, dim=1)

    b = b.cpu()
    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_index_select(device, dtype):
    a = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
    indices = torch.tensor([1], dtype=dtype, device=device)
    # This calls gather_fwd_i[32/64] TPC kernel
    b = a.index_select(dim=-1, index=indices)
    b = b.cpu()

    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_gather(device, dtype):
    a = torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    indices = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    # This calls gather_fwd_i[32/64] TPC kernel
    b = a.gather(dim=-1, index=indices)
    b = b.cpu()

    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_empty(device, dtype=torch.int64):
    a = torch.empty((2, 2), dtype=dtype, device=device)
    b = torch.empty((2, 2), dtype=dtype, device=device)

    c = a + b
    print(c.cpu())


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_cumsum(device, dtype=torch.int64):
    a = torch.tensor(
        [[0, 1, 2, 3], [3, 4, 5, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        dtype=dtype,
        device=device,
    )
    b = a.cumsum(dim=-1)

    b = b.cpu()
    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
def test_long_to_float(device):
    a = torch.ones(10, dtype=torch.int64, device=device)
    b = a.to(torch.float)

    b = b.cpu()
    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_pad(device, dtype=torch.int64):
    a = torch.ones([4, 2], device=device, dtype=dtype)
    p1d = (1, 1)
    b = torch.nn.functional.pad(a, p1d, mode="constant", value=0)

    b = b.cpu()
    print(b)
    print(b.dtype)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_arange(device, dtype=torch.int64):
    a = torch.empty(6, device=device, dtype=dtype)
    b = torch.arange(start=0, end=6, device=device, dtype=dtype, out=a)

    b = b.cpu()
    print(b)
    print(b.dtype)
    a = a.cpu()
    print(a)
    print(a.dtype)


def test_copy_tensor_on_device():
    a = torch.arange(start=0, end=6, device=torch.device("hpu:0"), dtype=torch.int64)
    a = a + 3
    b = a.clone()

    b = b.to(torch.int32)
    b = b * 2
    b = b.cpu()
    a = a - 1
    a = a.cpu()

    print(b)
    print(a)
