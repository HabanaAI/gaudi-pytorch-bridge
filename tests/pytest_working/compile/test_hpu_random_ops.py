###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import numpy as np
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc


@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_bernoulli(shape, dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input_a, input_b):
        a = torch.bernoulli(input_a)
        b = torch.bernoulli(input_b)
        c = torch.mul(a, b)
        return c

    compiled_fn = torch.compile(fn, backend="hpu_backend")

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

    fn = torch.compile(fn, backend="hpu_backend")

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

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    compiled_fn2 = torch.compile(fn2, backend="hpu_backend")

    input = torch.empty((3, 4, 5), dtype=torch.float).uniform_(0, 1).to("hpu")

    torch.manual_seed(12345)
    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()

    torch.manual_seed(12345)
    result_1a = compiled_fn2(input).cpu() / 2
    result_2a = compiled_fn2(input).cpu() / 2

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)


@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_poisson(shape, dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input):
        return torch.poisson(input)

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    input = torch.empty(shape, dtype=dtype).uniform_(0).to("hpu")

    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()
    assert not torch.equal(result_1, result_2)

    assert torch.all(result_1 >= 0.0)
    assert torch.all(result_2 >= 0.0)

    check_ops_executed_in_jit_ir("habana_poisson")


def test_poisson_determinism():
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input):
        return torch.poisson(input)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    input = torch.empty((3, 4, 5), dtype=torch.float32).uniform_(0).to("hpu")

    torch.manual_seed(12345)
    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()

    torch.manual_seed(12345)
    result_1a = compiled_fn(input).cpu()
    result_2a = compiled_fn(input).cpu()

    torch.manual_seed(54321)
    result_1b = compiled_fn(input).cpu()
    result_2b = compiled_fn(input).cpu()

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)
    assert not torch.equal(result_1, result_1b)
    assert not torch.equal(result_2, result_2b)

    check_ops_executed_in_jit_ir("habana_poisson")


@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
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

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    result_1 = compiled_fn(input).cpu()
    result_2 = compiled_fn(input).cpu()
    assert not torch.equal(result_1, result_2)

    assert torch.all(result_1 < 1.0) and torch.all(result_1 >= 0.0)
    assert torch.all(result_2 < 1.0) and torch.all(result_2 >= 0.0)

    check_ops_executed_in_jit_ir("habana_rand")


@pytest.mark.parametrize("shape", [(100, 100), (64, 8, 16)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
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

    compiled_fn = torch.compile(fn, backend="hpu_backend")

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


@pytest.mark.parametrize("shape", [(20, 40), (5, 10, 15)], ids=format_tc)
@pytest.mark.parametrize("low, high", [(2, 200), (-50, 20), (None, 10000)], ids=format_tc)
@pytest.mark.parametrize("is_like", [False, True])
@pytest.mark.parametrize("dtype", [torch.int, torch.long], ids=format_tc)
def test_randint(shape, low, high, is_like, dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    args = (low, high) if low else (high,)
    args = (torch.empty(shape, dtype=dtype, device="hpu"),) + args if is_like else args + (shape,)

    if is_like:
        if low:

            def fn(input, low, high, dtype, device):
                return torch.randint_like(input, low, high, dtype=dtype, device=device)

        else:

            def fn(input, high, dtype, device):
                return torch.randint_like(input, high, dtype=dtype, device=device)

    else:
        if low:

            def fn(low, high, size, dtype, device):
                return torch.randint(low, high, size, dtype=dtype, device=device)

        else:

            def fn(high, size, dtype, device):
                return torch.randint(high, size, dtype=dtype, device=device)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    result_1 = compiled_fn(*args, dtype=dtype, device="hpu").cpu()
    result_2 = compiled_fn(*args, dtype=dtype, device="hpu").cpu()

    if not low:
        low = 0
    assert result_1.dtype == dtype
    assert not torch.equal(result_1, result_2)
    assert torch.all(result_1 < high) and torch.all(result_1 >= low)
    assert torch.all(result_2 < high) and torch.all(result_2 >= low)
    check_ops_executed_in_jit_ir("habana_randint")


@pytest.mark.parametrize("shape", [(10,), (8, 10)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("replacement", [True, False])
def test_multinomial(shape, dtype, replacement):
    torch._dynamo.reset()
    clear_t_compile_logs()

    compiled_fn = torch.compile(torch.multinomial, backend="hpu_backend")

    input = torch.rand(shape, dtype=dtype).to("hpu")
    num_samples = 100 if replacement else 5

    result_1 = compiled_fn(input, num_samples, replacement).cpu()
    result_2 = compiled_fn(input, num_samples, replacement).cpu()
    assert result_1.dtype == torch.long
    assert not torch.equal(result_1, result_2)

    assert torch.all(result_1 >= 0) and torch.all(result_1 < shape[-1])
    assert torch.all(result_2 >= 0) and torch.all(result_2 < shape[-1])

    if replacement and len(shape) == 1:
        input = input.cpu()
        original_prob = input / torch.sum(input)
        result_1_prob = torch.bincount(result_1) / num_samples
        result_2_prob = torch.bincount(result_2) / num_samples

        diff_1 = torch.abs(result_1_prob - original_prob)
        diff_2 = torch.abs(result_2_prob - original_prob)
        standard_error = torch.sqrt((original_prob * (1 - original_prob)) / num_samples)

        assert np.alltrue((3 * standard_error > diff_1).numpy())
        assert np.alltrue((3 * standard_error > diff_2).numpy())

    check_ops_executed_in_jit_ir("habana_multinomial")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_various_ops(dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(input_a, input_b, shape_c, multinomial_input):
        a = torch.bernoulli(input_a)
        b = torch.rand_like(a)
        c = torch.bernoulli(input_b)
        d = torch.randn_like(c)
        e = torch.rand(shape_c, dtype=dtype, device="hpu")
        f = torch.randn(shape_c, dtype=dtype, device="hpu")
        g = torch.randint(10, 300, shape_c, dtype=torch.int, device="hpu").to(dtype)
        h = torch.multinomial(multinomial_input, shape_c[-1], replacement=True).to(dtype)
        i = torch.poisson(g)
        ab = torch.mul(a, b)
        cd = torch.div(c, d)
        ef = torch.add(e, f)
        efg = torch.sub(ef, g)
        efgh = torch.add(efg, h)
        efghi = torch.add(efgh, i)
        result = torch.addmm(ab, cd, efghi)
        return result

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    shape_a = (4, 12)
    shape_b = (4, 8)
    shape_c = (8, 12)
    input_a = torch.empty(shape_a, dtype=dtype).uniform_(0, 1).to("hpu")
    input_b = torch.empty(shape_b, dtype=dtype).uniform_(0, 1).to("hpu")
    multinomial_input = torch.rand(shape_c[-1], dtype=dtype).to("hpu")

    args = (input_a, input_b, shape_c, multinomial_input)

    torch.manual_seed(9876543)
    result_1 = compiled_fn(*args).cpu()
    result_2 = compiled_fn(*args).cpu()
    result_3 = compiled_fn(*args).cpu()
    assert not torch.equal(result_1, result_2)
    assert not torch.equal(result_1, result_3)

    torch.manual_seed(9876543)
    result_1a = compiled_fn(*args).cpu()
    result_2a = compiled_fn(*args).cpu()
    result_3a = compiled_fn(*args).cpu()
    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)
    assert torch.equal(result_3, result_3a)

    check_ops_executed_in_jit_ir(
        {
            "habana_bernoulli",
            "habana_rand",
            "habana_randn",
            "habana_randint",
            "habana_multinomial",
            "habana_poisson",
        }
    )


@pytest.mark.parametrize("n", [(5), (8), (17)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.long], ids=format_tc)
def test_randperm(n, dtype):
    torch._dynamo.reset()
    clear_t_compile_logs()

    def fn(shape):
        return torch.randperm(shape, dtype=dtype, device="hpu")

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    torch.manual_seed(1234)
    result_1 = compiled_fn(n).cpu()
    result_2 = compiled_fn(n).cpu()

    torch.manual_seed(1234)
    result_1a = compiled_fn(n).cpu()
    result_2a = compiled_fn(n).cpu()

    torch.manual_seed(12345)
    result_1b = compiled_fn(n).cpu()
    result_2b = compiled_fn(n).cpu()

    assert result_1.dtype == dtype
    assert not torch.equal(result_1, result_1b)
    assert not torch.equal(result_2, result_2b)

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)

    check_ops_executed_in_jit_ir("habana_randperm")
