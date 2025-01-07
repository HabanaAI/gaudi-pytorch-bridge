###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import os
import pathlib

import pytest
import torch
from test_utils import is_pytest_mode_compile

# Tests must be executed in separate pytest runs, because habana modules
# have to be reloaded before setting custom list of ops


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
    device = "hpu"
    dtype = torch.bfloat16
    a = torch.rand((5, 5)) * 10
    b = torch.rand((5, 5)) * 10
    ah = a.to(device)
    ah_bf16 = ah.to(dtype)
    bh = b.to(device)
    bh_bf16 = bh.to(dtype)

    def fn(a, b, a_f32, b_f32):
        mm = torch.mm(a, b)
        ls = torch.log_softmax(mm, 0)
        ls2 = torch.log_softmax(a, 0)
        add = torch.add(mm, mm)
        add_float = torch.add(a_f32, b_f32)
        return mm, ls, ls2, add, add_float

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    with torch.autocast(device_type=device, dtype=dtype):
        mm, ls, ls2, add, add_float = fn(ah, bh, ah, bh)

    mm_ref, ls_ref, ls2_ref, add_ref, add_float_ref = fn(ah_bf16, bh_bf16, ah, bh)

    assert_dtype((mm, ls, ls2, add), dtype)
    assert_dtype((add_float,), torch.float)
    assert_device(
        (mm, ls, ls2, add, add_float, mm_ref, ls_ref, ls2_ref, add_ref, add_float_ref),
        ah.device,
    )
    assert_tensors_equal((mm, ls, ls2, add, add_float), (mm_ref, ls_ref, ls2_ref, add_ref, add_float_ref))


@pytest.mark.parametrize("is_mask", [True, False])
def test_sdpa(is_mask):
    device = "hpu"
    high_dtype = torch.float
    low_dtype = torch.bfloat16

    qkv_shape = (1, 5, 16, 24)
    mask_shape = (1, 1, 16, 16)
    query = torch.ones(qkv_shape, dtype=high_dtype, device=device)
    key = torch.ones(qkv_shape, dtype=high_dtype, device=device)
    value = torch.ones(qkv_shape, dtype=high_dtype, device=device)
    proj = torch.eye(qkv_shape[-1], dtype=high_dtype, device=device)
    if is_mask:
        attn_mask = torch.ones(mask_shape, dtype=high_dtype, device=device)
        attn_mask_low = attn_mask.to(low_dtype)
    else:
        attn_mask = None
        attn_mask_low = None
    dropout_p = 0
    is_causal = False

    def fn(query, proj, key, value, attn_mask):
        q = query @ proj
        k = key @ proj
        v = value @ proj
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        return attn_output

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    with torch.no_grad(), torch.autocast(device_type=device):
        attn_output = fn(query, proj, key, value, attn_mask)

    attn_output_ref = fn(query.to(low_dtype), proj.to(low_dtype), key.to(low_dtype), value.to(low_dtype), attn_mask_low)

    assert_dtype((attn_output, attn_output_ref), low_dtype)
    assert torch.equal(attn_output, attn_output_ref)


# Below test is meant to run manually with envs set:
# PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=pytest_working/autocast_files/lower_list.txt
# PT_HPU_AUTOCAST_FP32_OPS_LIST=pytest_working/autocast_files/fp32_list.txt
@pytest.mark.skip(reason="Can't set custom autocast list in runtime")
def test_autocast_custom_list():
    device = "hpu"
    dtype = torch.bfloat16
    a = torch.rand((5, 5)) * 10
    b = torch.rand((5, 5)) * 10
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
