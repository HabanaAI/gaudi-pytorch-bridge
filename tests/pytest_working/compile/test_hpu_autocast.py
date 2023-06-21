# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import os
import torch
import pathlib
import pytest

# Tests must be executed in separate pytest runs, because habana modules
# have to be reloaded before setting custom list of ops

def load_modules(custom_autocast = False):
    if custom_autocast:
        path = str(pathlib.Path(__file__).parent.resolve())
        os.environ["LOWER_LIST"] = path + "/autocast_files/lower_list.txt"
        os.environ["FP32_LIST"] = path + "/autocast_files/fp32_list.txt"
    import habana_frameworks.torch.core

def assert_dtype(tensors, dtype):
    for tensor in tensors:
        assert tensor.dtype == dtype, f"Wrong dtype. Got {tensor.dtype}, expected {dtype}."

def assert_device(tensors, device):
    for tensor in tensors:
        assert tensor.device == device, f"Wrong device. Got {tensor.device}, expected {device}."

def assert_tensors_equal(tensors, tensor_refs):
    for tensor, tensor_ref in zip(tensors, tensor_refs):
        assert torch.equal(tensor, tensor_ref)

def test_autocast():
    load_modules()
    device = "hpu"
    dtype = torch.bfloat16
    a = torch.rand((5, 5))*10
    b = torch.rand((5, 5))*10
    ah = a.to(device)
    ah_bf16 = ah.to(dtype)
    bh = b.to(device)
    bh_bf16 = bh.to(dtype)
    with torch.autocast(device_type=device, dtype=dtype):
        mm = torch.mm(ah, bh)
        ls = torch.log_softmax(mm, 0)
        ls2 = torch.log_softmax(ah, 0)
        add = torch.add(mm, mm)
        add_float = torch.add(ah, bh)

    mm_ref = torch.mm(ah_bf16, bh_bf16)
    ls_ref = torch.log_softmax(mm_ref, 0)
    ls2_ref = torch.log_softmax(ah_bf16, 0)
    add_ref = torch.add(mm_ref, mm_ref)
    add_float_ref = torch.add(ah, bh)

    assert_dtype((mm, ls, ls2, add), dtype)
    assert_dtype((add_float,), torch.float)
    assert_device((mm, ls, ls2, add, add_float, mm_ref, ls_ref, ls2_ref, add_ref, add_float_ref), ah.device)
    assert_tensors_equal((mm, ls, ls2, add, add_float), (mm_ref, ls_ref, ls2_ref, add_ref, add_float_ref))

@pytest.mark.xfail(reason="Wrong dtype. Got torch.float32, expected torch.bfloat16.")
def test_autocast_custom_list():
    load_modules(True)
    device = "hpu"
    dtype = torch.bfloat16
    a = torch.rand((5, 5))*10
    b = torch.rand((5, 5))*10
    ah = a.to(device)
    ah_bf16 = ah.to(dtype)
    bh = b.to(device)
    bh_bf16 = bh.to(dtype)
    with torch.autocast(device_type=device, dtype=dtype):
        add = torch.add(ah, bh)
        mm = torch.mm(ah, bh)
        matmul = torch.matmul(ah, bh)
        matmul2 = torch.matmul(add, mm)

    add_ref = torch.add(ah_bf16, bh_bf16)
    mm_ref = torch.mm(ah_bf16, bh_bf16)
    matmul_ref = torch.matmul(ah, bh)
    matmul2_ref = torch.matmul(add_ref.to(torch.float), mm_ref.to(torch.float))

    assert_dtype((add, mm), dtype)
    assert_dtype((matmul, matmul2), torch.float)
    assert_device((add, mm, matmul, matmul2, add_ref, mm_ref, matmul_ref, matmul2_ref), ah.device)
    assert_tensors_equal((add, mm, matmul, matmul2), (add_ref, mm_ref, matmul_ref, matmul2_ref))
