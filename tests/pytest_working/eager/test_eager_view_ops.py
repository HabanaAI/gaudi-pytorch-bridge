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

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import pytest
import torch
from test_utils import cpu, hpu, place_on_hpu

Verbose = False


def test_relu_contiguous_view():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_contiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[2:]
    hpu_tensor_slice = hpu_tensor[2:]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_contiguous_as_strided():
    cpu_tensor = torch.randn([2, 3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor.as_strided_([5], [1], 1)
    hpu_tensor.as_strided_([5], [1], 1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_contiguous_multilevel_view():
    cpu_tensor = torch.randn([2, 3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[:].view(-1)
    hpu_tensor = hpu_tensor[:].view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_inplace_view():
    cpu_tensor = torch.randn([4]).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    cpu_tensor = cpu_tensor[2::]
    hpu_tensor = hpu_tensor[2::]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_discontiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[::2]
    hpu_tensor_slice = hpu_tensor[::2]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_2d_discontiguous_slice():
    cpu_tensor = torch.Tensor(np.random.randint(-2, 2, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[0::2, 0::2]
    hpu_tensor_slice = hpu_tensor[0::2, 0::2]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_relu_inplace_1d_noncontiguous_view():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    torch.relu_(hpu_tensor[::2])
    torch.relu_(cpu_tensor[::2])

    torch.allclose(hpu_tensor.cpu(), cpu_tensor, atol=0.001, rtol=0.001)


def test_relu_inplace_2d_noncontiguous_view():
    cpu_tensor = torch.randn([4, 4])
    hpu_tensor = cpu_tensor.to("hpu")

    torch.relu_(hpu_tensor[0::2, 0::2])
    torch.relu_(cpu_tensor[0::2, 0::2])

    torch.allclose(hpu_tensor.cpu(), cpu_tensor, atol=0.001, rtol=0.001)


def test_relu_inplace_noncontiguous_view():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[::2]
    hpu_tensor = hpu_tensor[::2]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_aminmax_multi_output_view_row():
    cpu_tensor = torch.randn([2, 5])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_min_tensor = torch.randn([2, 5])
    hpu_min_tensor = cpu_min_tensor.to("hpu")

    cpu_max_tensor = torch.randn([2, 5])
    hpu_max_tensor = cpu_max_tensor.to("hpu")

    cpu_min_tensor[1], cpu_max_tensor[1] = cpu_tensor.aminmax(dim=0, keepdim=True)
    hpu_min_tensor[1], hpu_max_tensor[1] = hpu_tensor.aminmax(dim=0, keepdim=True)

    assert torch.allclose(hpu_min_tensor.cpu(), cpu_min_tensor, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_max_tensor.cpu(), cpu_max_tensor, atol=0.001, rtol=0.001)


def test_aminmax_multi_output_view_col():
    cpu_tensor = torch.randn([2, 5])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_min_tensor = torch.randn([10])
    hpu_min_tensor = cpu_min_tensor.to("hpu")

    cpu_max_tensor = torch.randn([10])
    hpu_max_tensor = cpu_max_tensor.to("hpu")

    cpu_min_tensor[0::2], cpu_max_tensor[0::2] = cpu_tensor.aminmax(dim=0, keepdim=True)
    hpu_min_tensor[0::2], hpu_max_tensor[0::2] = hpu_tensor.aminmax(dim=0, keepdim=True)

    assert torch.allclose(hpu_min_tensor.cpu(), cpu_min_tensor, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_max_tensor.cpu(), cpu_max_tensor, atol=0.001, rtol=0.001)


def test_aminmax_multi_output_view_col2():
    cpu_tensor = torch.randn([2, 5])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_min_tensor = torch.randn([10])
    hpu_min_tensor = cpu_min_tensor.to("hpu")

    cpu_max_tensor = torch.randn([10])
    hpu_max_tensor = cpu_max_tensor.to("hpu")

    torch.aminmax(cpu_tensor, dim=0, out=[cpu_min_tensor[::2], cpu_max_tensor[::2]])
    torch.aminmax(hpu_tensor, dim=0, out=[hpu_min_tensor[::2], hpu_max_tensor[::2]])

    assert torch.allclose(hpu_min_tensor.cpu(), cpu_min_tensor, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_max_tensor.cpu(), cpu_max_tensor, atol=0.001, rtol=0.001)


def test_d2d_noncontiguous_views_src():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    cpu_dst_tensor = torch.randn([2])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    hpu_dst_tensor.copy_(hpu_src_tensor_view)
    cpu_dst_tensor.copy_(cpu_src_tensor_view)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol=0.001, rtol=0.001)


def test_d2d_noncontiguous_views_dst():
    cpu_src_tensor = torch.randn([2])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_dst_tensor = torch.randn([4])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    cpu_dst_tensor_view = cpu_dst_tensor[::2]
    hpu_dst_tensor_view = hpu_dst_tensor[::2]

    hpu_dst_tensor_view.copy_(hpu_src_tensor)
    cpu_dst_tensor_view.copy_(cpu_src_tensor)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol=0.001, rtol=0.001)


def test_d2d_noncontiguous_views_src_dst():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    cpu_dst_tensor = torch.randn([4])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    cpu_dst_tensor_view = cpu_dst_tensor[::2]
    hpu_dst_tensor_view = hpu_dst_tensor[::2]

    hpu_dst_tensor_view.copy_(hpu_src_tensor_view)
    cpu_dst_tensor_view.copy_(cpu_src_tensor_view)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol=0.001, rtol=0.001)


def test_d2h_noncontiguous_views():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    assert torch.allclose(hpu_src_tensor_view.cpu(), cpu_src_tensor_view, atol=0.001, rtol=0.001)


def test_h2d_noncontiguous_views():
    cpu_src_tensor = torch.randn([4])
    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor = cpu_src_tensor_view.to("hpu")

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor_view, atol=0.001, rtol=0.001)


def test_h2d_chlast():
    a = torch.randn([2, 3, 4, 5]).to(memory_format=torch.channels_last)
    ha = a.to("hpu")

    b = torch.relu(a)
    hb = torch.relu(ha)

    assert torch.allclose(hb.cpu(), b, atol=0.001, rtol=0.001)


def test_h2d_dst_noncontiguous_view():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor2 = torch.randn([2])
    hpu_src_tensor_view = hpu_src_tensor[::2]

    hpu_src_tensor_view.copy_(cpu_src_tensor2)

    cpu_src_tensor[::2].copy_(cpu_src_tensor2)

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor, atol=0.001, rtol=0.001)


def test_d2d_dst_view():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor2 = torch.randn([2])
    hpu_src_tensor2 = cpu_src_tensor2.to("hpu")
    hpu_src_tensor_view = hpu_src_tensor[::2]
    cpu_src_tensor_view = cpu_src_tensor[::2]

    hpu_src_tensor_view.copy_(hpu_src_tensor2)

    cpu_src_tensor_view.copy_(cpu_src_tensor2)

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor, atol=0.001, rtol=0.001)


def test_topk_transpose():
    a = torch.randint(0, 10, [2, 2])
    ha = a.to("hpu")

    b = torch.topk(a, k=2)
    c = b[0].t()

    hb = torch.topk(ha, k=2)
    hc = hb[0].t()
    assert torch.allclose(hc.cpu(), c, atol=0.001, rtol=0.001)


def test_view_copy_cache():
    def fn(x):
        x = x.to(torch.float)
        x1 = x.unsqueeze(-1)
        x2 = x1.transpose(1, 2)
        x1 = x1.to(torch.bfloat16)
        x2 = x2.to(torch.bfloat16)
        y = torch.matmul(x1, x2)
        return y

    a = torch.randn([2, 4]).to(torch.bool)
    ha = a.to("hpu")
    res = fn(a)
    hres = fn(ha)

    assert torch.allclose(hres.cpu(), res, atol=0.001, rtol=0.001)


def test_view_cache2():
    def fn(x, offset):
        x = x[offset : offset + 4 : 2]
        m = torch.nn.ReLU()
        x = m(x)
        return x

    a = torch.randn([10])
    ha = a.to("hpu")
    res = fn(a, 1)
    hres = fn(ha, 1)

    assert torch.allclose(hres.cpu(), res, atol=0.001, rtol=0.001)

    a = torch.randn([10])
    ha = a.to("hpu")
    res = fn(a, 2)
    hres = fn(ha, 2)

    assert torch.allclose(hres.cpu(), res, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="RuntimeError: Wrong PT plugin library loaded in the system. Expected was EAGER, got LAZY")
def test_view_layout1():
    def fn(x, dev):
        m = torch.nn.Conv2d(2, 3, 3, stride=2).to(dev)
        x = m(x)
        x = x[:]
        return x

    a = torch.randn(2, 2, 6, 6)
    ha = a.to("hpu")

    # CPU
    res = fn(a, hpu)

    # HPU
    hres = fn(ha, cpu)
    hres_cpu = hres.cpu()

    assert torch.allclose(hres_cpu, res, atol=0.01, rtol=0.01)


# dtype:different, src:view, dst:view
def test_d2d_src_dst_view_different_dtype():
    cpu_src_tensor = torch.randn([4, 6], dtype=torch.bfloat16)
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_dst_tensor = torch.randn([6, 9], dtype=torch.float32)
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    cpu_dst_tensor_view = cpu_dst_tensor[0::3, 0::3]
    hpu_dst_tensor_view = hpu_dst_tensor[0::3, 0::3]

    cpu_dst_tensor_view.copy_(cpu_src_tensor[0::2, 0::2])
    hpu_dst_tensor_view.copy_(hpu_src_tensor[0::2, 0::2])

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol=0.001, rtol=0.001)


def test_copy_inplace_2d_noncontiguous_view():
    cpu_tensor = torch.randn([4, 4], dtype=torch.float)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_ones = torch.ones([4, 4], dtype=torch.int8)
    hpu_ones = cpu_ones.to("hpu")

    cpu_ones[0::2, 0::2] = cpu_tensor[0::2, 0::2].to(torch.int8)
    hpu_ones[0::2, 0::2] = hpu_tensor[0::2, 0::2].to(torch.int8)
    torch.allclose(hpu_ones.cpu(), cpu_ones, atol=0, rtol=0)


def test_view_permutation_contiguous_view_update():
    torch.manual_seed(0)
    a = torch.randn(2, 2, 6, 6)
    ha = a.to("hpu")
    m = torch.nn.Conv2d(2, 3, 3, stride=2, device="hpu")
    hb = m(ha)
    hc = hb[:]
    hc.fill_(0.0)

    hb_cpu = hb.cpu()

    res = torch.zeros_like(hb_cpu)

    assert torch.allclose(hb_cpu, res, atol=0.01, rtol=0.01)


def test_zero_inplace_2d_noncontiguous_view():
    cpu_tensor = torch.randn([2, 3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor.t_()
    hpu_tensor.t_()

    cpu_tensor.zero_()
    hpu_tensor.zero_()

    torch.allclose(hpu_tensor.cpu(), cpu_tensor, atol=0, rtol=0)


def test_tensorlist_view():
    a = torch.randn([2, 4])
    ha = a.to("hpu")
    b = torch.randn([2, 4])
    hb = b.to("hpu")

    c = torch.cat([a[:, :], b[:, ::2]], dim=1)
    hc = torch.cat([ha[:, :], hb[:, ::2]], dim=1)

    hc_cpu = hc.cpu()
    torch.allclose(hc_cpu, c, atol=0, rtol=0)


def test_normal():
    torch.manual_seed(0)
    a = torch.empty([2, 3], dtype=torch.float)
    ha = a.to("hpu")

    av = a[:, ::2].normal_()
    hav = ha[:, ::2].normal_()
    hav_cpu = hav.cpu()

    torch.allclose(hav_cpu, av, atol=0, rtol=0)


def test_random():
    torch.manual_seed(0)
    a = torch.empty([3, 2], dtype=torch.float32)
    ha = a.to("hpu")

    av = a.t().random_()
    hav = ha.t().random_()
    hav_cpu = hav.cpu()

    torch.allclose(hav_cpu, av, atol=0, rtol=0)


def test_geometric():
    a = torch.empty([3, 2], dtype=torch.float32)
    ha = a.to("hpu")

    av = a[::2, :].geometric_(0.1)
    hav = ha[::2, :].geometric_(0.1)
    hav_cpu = hav.cpu()

    torch.allclose(hav_cpu, av, atol=0, rtol=0)


def test_lognormal():
    torch.manual_seed(0)
    ha1 = torch.empty([3, 2], dtype=torch.float32).to("hpu")
    ha1.permute(0, 1).log_normal_()
    torch.manual_seed(0)
    ha2 = torch.empty([3, 2], dtype=torch.float32).to("hpu")
    ha2.permute(0, 1).log_normal_()

    assert torch.allclose(ha1.cpu(), ha2.cpu(), atol=0.001, rtol=0.001)


def test_ge_inplace():
    torch.manual_seed(0)
    a = torch.randn([3, 2])
    ha = a.to("hpu")

    b = torch.randn([2, 3])
    hb = b.to("hpu")

    a.transpose(0, 1).ge_(b)
    ha.transpose(0, 1).ge_(hb)

    ha_cpu = ha.cpu()
    assert torch.allclose(ha_cpu, a, atol=0.001, rtol=0.001)


def test_lt_inplace():
    torch.manual_seed(0)
    a = torch.randn([3, 2])
    ha = a.to("hpu")

    b = torch.randn([2, 3])
    hb = b.to("hpu")

    a.permute(1, 0).lt_(b)
    ha.permute(1, 0).lt_(hb)

    ha_cpu = ha.cpu()
    assert torch.allclose(ha_cpu, a, atol=0.001, rtol=0.001)


def test_eq_view():
    torch.manual_seed(0)
    a = torch.arange(4).to(torch.float)
    res = torch.zeros(4, dtype=torch.bool)
    a_view = a.as_strided((2, 2), (1, 2))
    res_view = res.as_strided((2, 2), (1, 2))

    ha = a.to("hpu")
    hres = res.to("hpu")
    ha_view = ha.as_strided((2, 2), (1, 2))
    hres_view = hres.as_strided((2, 2), (1, 2))

    # first eq op, scalar dtype double
    torch.eq(a_view, 0.0, out=res_view)
    torch.eq(ha_view, 0.0, out=hres_view)
    hres_view_cpu = hres_view.cpu()
    assert torch.equal(hres_view_cpu, res_view)

    # second eq op, scalar dtype long
    torch.eq(a_view, 0, out=res_view)
    torch.eq(ha_view, 0, out=hres_view)
    hres_view_cpu = hres_view.cpu()
    assert torch.equal(hres_view_cpu, res_view)


@pytest.mark.skip(reason="Test is sporadically failing - SW-167590")
def test_sort_out():
    torch.manual_seed(0)
    a = torch.randn([10])
    ha = a.to("hpu")

    b = torch.empty([20])
    hb = b.to("hpu")

    c = torch.empty([10]).to(torch.long)
    hc = c.to("hpu")

    torch.sort(a, out=[b[::2], c])
    torch.sort(ha, out=[hb[::2], hc])

    hb_cpu = hb.cpu()
    assert torch.allclose(hb_cpu, b, atol=0.001, rtol=0.001)


def test_add_out():
    torch.manual_seed(0)

    def fn():
        a = torch.randn([10])
        ha = a.to("hpu")

        b = torch.randn([22])
        hb = b.to("hpu")

        torch.add(a, 1.0, out=b[2::2])
        torch.add(ha, 1.0, out=hb[2::2])

        hb_cpu = hb.cpu()
        assert torch.allclose(hb_cpu, b, atol=0.001, rtol=0.001)

    fn()
    # test cache hit
    fn()


def test_fill():
    shapes = [(2, 3), (4, 6)]
    for shape in shapes:
        input = torch.randn((shape), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        input.t_().fill_(10)
        input_hpu.t_().fill_(10)

        assert torch.equal(input_hpu.cpu(), input)


# This test case shall start failing as soon as you fix the JIRA issue:
# https://jira.habana-labs.com/browse/SW-152023
@pytest.mark.skip(reason="We are to fix SW-152023 to get pass")
def test_view_split_op_int64_default():
    assert bc.get_pt_enable_int64_support() == False
    t = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    t1 = t.to("hpu")
    assert list(t1[1:3].to("cpu")) == list(t[1:3])


def test_view_split_op_int64_enabled():
    with bc.env_setting("PT_ENABLE_INT64_SUPPORT", True):
        assert bc.get_pt_enable_int64_support() == True
        t = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        t1 = t.to("hpu")
        assert list(t1[1:3].to("cpu")) == list(t[1:3])


# This test case shall start failing as soon as you fix the JIRA issue:
# https://jira.habana-labs.com/browse/SW-152023
@pytest.mark.skip(reason="We are to fix SW-152023 to get pass")
def test_view_split_op_int64_disabled():
    with bc.env_setting("PT_ENABLE_INT64_SUPPORT", False):
        assert bc.get_pt_enable_int64_support() == False
        t = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        t1 = t.to("hpu")
        assert list(t1[1:3].to("cpu")) == list(t[1:3])


@pytest.mark.parametrize("inout", ["in", "out", "inplace"])
@pytest.mark.parametrize("ttl", ["tensor", "boolout", "fill", "tlist"])
def test_view(ttl, inout):
    def complex_default(obj):
        return field(default_factory=lambda: copy.copy(obj))

    @dataclass
    class TestData:
        num_views: int
        command: Callable[[Dict[str, torch.Tensor], List[torch.Tensor]], torch.Tensor]

        num_tensors: int = 1
        view_base_shape: List[int] = complex_default([3, 5])
        tensor_shape: List[int] = complex_default([3])
        normalize: bool = True
        make_view: Callable[[torch.Tensor], torch.Tensor] = lambda t: t[1, 0:5:2]
        store_result: str = "t1"

    test_data = {}

    test_data[("tensor", "in")] = TestData(num_views=1, command=lambda ts, vs: vs[1].logit(eps=0.01))

    test_data[("tensor", "out")] = TestData(
        num_views=2,
        command=lambda ts, vs: torch.logit(vs[1], eps=0.01, out=vs[2]),
    )

    test_data[("tensor", "inplace")] = TestData(num_views=1, command=lambda ts, vs: vs[1].logit_(eps=0.01))

    test_data[("boolout", "in")] = TestData(num_views=2, command=lambda ts, vs: vs[1].ge(vs[2]))

    test_data[("boolout", "out")] = TestData(num_views=3, command=lambda ts, vs: torch.lt(vs[1], vs[2], out=vs[3]))

    test_data[("boolout", "inplace")] = TestData(num_views=2, command=lambda ts, vs: vs[1].eq_(vs[2]))

    test_data[("fill", "inplace")] = TestData(num_views=1, command=lambda ts, vs: vs[1].fill_(-5))

    test_data[("tlist", "in")] = TestData(
        num_views=2,
        num_tensors=2,
        command=lambda ts, vs: torch.cat([ts["t2"]] + vs[1:]),
    )

    test_data[("tlist", "inplace")] = TestData(
        num_views=2,
        num_tensors=1,
        command=lambda ts, vs: torch._foreach_atan_([ts["t1"]] + vs[1:]),
        normalize=False,
    )

    testcase = (ttl, inout)
    if testcase in test_data:
        td = test_data[(ttl, inout)]
    else:
        pytest.skip(f"Testcase {testcase} doesn't exist")

    cpu_tensors = {}

    def add_cpu_tensors(first_value, num, shape, label):
        for i in range(num):
            stop_value = first_value + np.prod(shape)
            cpu_tensors[f"{label}{i+1}"] = torch.Tensor(np.arange(first_value, stop_value).reshape(shape))
            first_value = stop_value
        return first_value

    next_first_value = add_cpu_tensors(1, td.num_views, td.view_base_shape, "vb")
    next_first_value = add_cpu_tensors(next_first_value, td.num_tensors, td.tensor_shape, "t")
    if td.normalize:
        for key in cpu_tensors.keys():
            cpu_tensors[key] /= next_first_value - 1

    if Verbose:
        print(f"{cpu_tensors = }")

    hpu_tensors = place_on_hpu(cpu_tensors)

    for tensors in [cpu_tensors, hpu_tensors]:
        views = [td.make_view(tensors[f"vb{i+1}"]) for i in range(td.num_views)]
        views.insert(0, None)
        result = td.command(tensors, views)
        if result is not None:
            tensors[td.store_result] = result

    for key in cpu_tensors.keys():
        result_cpu = cpu_tensors[key]
        result_hpu = hpu_tensors[key]
        if isinstance(result_cpu, list):
            result_cpu = torch.cat(result_cpu[:])
            result_hpu = torch.cat(result_hpu[:])
        if Verbose:
            print(f"{key = }")
            print(f"{result_cpu = }")
            print(f"result_hpu = {result_hpu.cpu()}")
        assert torch.allclose(result_hpu.cpu(), result_cpu, atol=0.001, rtol=0.001)


@pytest.mark.parametrize(
    "shift_op",
    [torch.ops.aten.__ilshift__, torch.ops.aten.__lshift__, torch.ops.aten.__irshift__, torch.ops.aten.__rshift__],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_shift(shift_op, transpose):
    a = torch.tensor([[1, 2, 4], [1, 2, 4]], dtype=torch.int32)
    if transpose:
        a = a.transpose(1, 0)
    ha = a.to("hpu")

    a_out = shift_op(a, 1)
    ha_out = shift_op(ha, 1)

    assert torch.allclose(ha_out.cpu(), a_out)


def test_sag_view_section_id_1():
    # 1 different tensor view inputs
    a = torch.rand(5)
    b = torch.rand(5)
    ha = a.to("hpu")
    hb = b.to("hpu")

    out1 = torch.mul(a[:3], b[:3])
    hout1 = torch.mul(ha[:3], hb[:3])
    assert torch.equal(hout1.cpu(), out1)

    # 2 same tensor views inputs
    c = torch.rand(10)
    hc = c.to("hpu")

    # view1 is first 5 elements and view2 is last 5 elements of same tensor
    out2 = torch.mul(c[:5], c[5:])
    hout2 = torch.mul(hc[:5], hc[5:])
    assert torch.equal(hout2.cpu(), out2)


def test_sag_view_section_id_2():
    # 1 same tensor views inputs
    a = torch.rand(10, dtype=torch.bfloat16)
    ha = a.to("hpu")

    # view1 is first 5 elements and view2 is last 5 elements of same tensor
    out1 = torch.mul(a[:5], a[5:])
    hout1 = torch.mul(ha[:5], ha[5:])
    assert torch.equal(hout1.cpu(), out1)

    # 2 different tensor view inputs
    b = torch.rand(5, dtype=torch.bfloat16)
    c = torch.rand(5, dtype=torch.bfloat16)
    hb = b.to("hpu")
    hc = c.to("hpu")

    out2 = torch.mul(b[:3], c[:3])
    hout2 = torch.mul(hb[:3], hc[:3])
    assert torch.equal(hout2.cpu(), out2)


@pytest.mark.parametrize(
    "logical_op", [torch.ops.aten.logical_and_, torch.ops.aten.logical_or_, torch.ops.aten.logical_xor_]
)
def test_inplace_slice_logical_op(logical_op):
    dtype = torch.float32
    seed = 4776
    shape = 1024
    repeats = 3
    slice_shape = shape * repeats
    slice_param = slice(None, None, repeats)

    torch.manual_seed(seed)
    cpu_input = torch.rand(slice_shape, dtype=dtype)
    cpu_other = torch.rand(slice_shape, dtype=dtype)
    cpu_input_slice = cpu_input[slice_param]
    cpu_other_slice = cpu_other[slice_param]
    logical_op(cpu_input_slice, cpu_other_slice)

    torch.manual_seed(seed)
    hpu_input = torch.rand(slice_shape, dtype=dtype).to("hpu")
    hpu_other = torch.rand(slice_shape, dtype=dtype).to("hpu")
    hpu_input_slice = hpu_input[slice_param]
    hpu_other_slice = hpu_other[slice_param]
    logical_op(hpu_input_slice, hpu_other_slice)

    assert torch.equal(hpu_input_slice.cpu(), cpu_input_slice)


def test_inplace_binary_strided_insert():
    cpu_self = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    cpu_other = torch.tensor([[2, 2], [2, 2]])
    cpu_self = cpu_self.as_strided([2, 2], [1, 2])

    hpu_self = cpu_self.to("hpu")
    hpu_other = cpu_other.to("hpu")
    hpu_self = hpu_self.as_strided([2, 2], [1, 2])

    cpu_self.add_(cpu_other)
    hpu_self.add_(hpu_other)

    assert torch.equal(hpu_self.cpu(), cpu_self)


def test_lt_out_with_view():
    torch.manual_seed(0)
    params = [((4, 4), (1, 4)), ((2, 8), (1, 2))]
    for shape, strides in params:
        a = torch.arange(16).to(torch.float)
        a_view = a.as_strided(shape, strides)

        ha = a.to("hpu")
        ha_view = ha.as_strided(shape, strides)

        res = torch.zeros(shape, dtype=torch.bfloat16)
        hres = res.to("hpu")

        torch.lt(a_view, 8, out=res)
        torch.lt(ha_view, 8, out=hres)
        assert torch.equal(hres.cpu(), res)


def test_add_with_slice():
    x = torch.rand((1, 4))
    y = torch.rand((1, 4))

    x_hpu = x.to("hpu")
    y_hpu = y.to("hpu")

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y_hpu[:, 0] = (x_hpu[:, 0] + x_hpu[:, 2]) / 2

    assert torch.allclose(y_hpu.cpu(), y, atol=0.001, rtol=0.001)
