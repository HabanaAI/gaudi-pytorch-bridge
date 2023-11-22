###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
import pytest

from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs


@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bernoulli(shape, dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input_a, input_b):
        a = torch.bernoulli(input_a)
        b = torch.bernoulli(input_b)
        c = torch.mul(a, b)
        return c

    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    input_a = torch.empty(shape, dtype=dtype).uniform_(0, 1).to("hpu")
    input_b = torch.empty(shape, dtype=dtype).uniform_(0, 1).to("hpu")

    result_1 = compiled_fn(input_a, input_b).cpu()
    result_2 = compiled_fn(input_a, input_b).cpu()
    assert not torch.equal(result_1, result_2)

    results = torch.tensor((0.0, 1.0), dtype=dtype)
    assert torch.equal(result_1.unique(), results)
    assert torch.equal(result_2.unique(), results)

    check_ops_executed_in_jit_ir("habana_bernoulli")


def test_bernoulli_determinism_one_graph():
    def fn(input):
        return torch.bernoulli(input)

    fn = torch.compile(fn, backend="aot_hpu_training_backend")

    input = torch.empty((3, 4, 5), dtype=torch.float).uniform_(0, 1).to("hpu")

    torch.manual_seed(12345)
    result_1 = fn(input).cpu()
    result_2 = fn(input).cpu()

    torch.manual_seed(12345)
    result_1a = fn(input).cpu()
    result_2a = fn(input).cpu()

    torch.manual_seed(54321)
    result_1b = fn(input).cpu()
    result_2b = fn(input).cpu()

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)
    assert not torch.equal(result_1, result_1b)
    assert not torch.equal(result_2, result_2b)


def test_bernoulli_determinism_two_graphs():
    def fn(input):
        return torch.bernoulli(input)

    def fn2(input):
        a = torch.bernoulli(input)
        return a * 2

    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    compiled_fn2 = torch.compile(fn2, backend="aot_hpu_training_backend")

    input = torch.empty((3, 4, 5), dtype=torch.float).uniform_(0, 1).to("hpu")

    torch.manual_seed(12345)
    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()

    torch.manual_seed(12345)
    result_1a = compiled_fn2(input).cpu() / 2
    result_2a = compiled_fn2(input).cpu() / 2

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)


@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_like", [False, True])
def test_rand(shape, dtype, is_like):
    torch._dynamo.reset()
    clear_t_compile_logs()

    if is_like:
        op = torch.rand_like
        input = torch.empty(shape, dtype=dtype, device="hpu")
    else:
        op = torch.rand
        input = shape

    def fn(input):
        return op(input, dtype=dtype, device="hpu")

    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()
    assert not torch.equal(result_1, result_2)

    assert torch.all(result_1 < 1.0) and torch.all(result_1 >= 0.0)
    assert torch.all(result_2 < 1.0) and torch.all(result_2 >= 0.0)

    check_ops_executed_in_jit_ir("habana_rand")


@pytest.mark.parametrize("shape", [(100, 100), (64, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_like", [False, True])
def test_randn(shape, dtype, is_like):
    torch._dynamo.reset()
    clear_t_compile_logs()

    if is_like:
        op = torch.randn_like
        input = torch.empty(shape, dtype=dtype, device="hpu")
    else:
        op = torch.randn
        input = shape

    def fn(shape):
        return op(shape, dtype=dtype, device="hpu")

    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()
    assert not torch.equal(result_1, result_2)

    mean = torch.mean(result_1)
    assert torch.abs(mean) < 0.1

    # Verify if distribution is normal. There should be:
    # ~68% elements within 1 stddev
    # ~95% elements within 2 stddev
    # ~99.7% elements within 3 stddev
    abs = torch.abs(result_1)
    divider = result_1.numel() / 100
    s1 = torch.count_nonzero(abs < 1.0) / divider
    s2 = torch.count_nonzero(abs < 2.0) / divider
    s3 = torch.count_nonzero(abs < 3.0) / divider

    assert torch.all(s1 < 70.0) and torch.all(s1 > 66.0)
    assert torch.all(s2 < 97.0) and torch.all(s2 > 93.0)
    assert torch.all(s3 < 99.9) and torch.all(s3 > 98.0)

    check_ops_executed_in_jit_ir("habana_randn")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_various_ops(dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input_a, input_b, shape_c):
        a = torch.bernoulli(input_a)
        b = torch.rand_like(a)
        c = torch.bernoulli(input_b)
        d = torch.randn_like(c)
        e = torch.rand(shape_c, dtype=dtype, device="hpu")
        f = torch.randn(shape_c, dtype=dtype, device="hpu")
        ab = torch.mul(a, b)
        cd = torch.div(c, d)
        ef = torch.add(e, f)
        result = torch.addmm(ab, cd, ef)
        return result

    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    shape_a = (4, 12)
    shape_b = (4, 8)
    shape_c = (8, 12)
    input_a = torch.empty(shape_a, dtype=dtype).uniform_(0, 1).to("hpu")
    input_b = torch.empty(shape_b, dtype=dtype).uniform_(0, 1).to("hpu")

    torch.manual_seed(9876543)
    result_1 = compiled_fn(input_a, input_b, shape_c).cpu()
    result_2 = compiled_fn(input_a, input_b, shape_c).cpu()
    result_3 = compiled_fn(input_a, input_b, shape_c).cpu()
    assert not torch.equal(result_1, result_2)
    assert not torch.equal(result_1, result_3)

    torch.manual_seed(9876543)
    result_1a = compiled_fn(input_a, input_b, shape_c).cpu()
    result_2a = compiled_fn(input_a, input_b, shape_c).cpu()
    result_3a = compiled_fn(input_a, input_b, shape_c).cpu()
    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)
    assert torch.equal(result_3, result_3a)

    check_ops_executed_in_jit_ir({"habana_bernoulli", "habana_rand", "habana_randn"})
