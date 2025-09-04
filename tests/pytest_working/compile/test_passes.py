###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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


import torch
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer


def test_pass_fuse_view_chains():
    # It doesn't matter all that much which operations are performed
    # as long as there are view chains inside a graph
    # that need to be eagerized

    def view_chain(inp):
        a = inp[1:]
        b = a.transpose(0, 1)
        d = torch.as_strided(b, (2, 2), (1, 2), 2)
        return d

    def fn(inp):
        a = inp * 2
        b = inp * a
        chain_1 = view_chain(b)
        chain_2 = view_chain(b)
        chain_1_cpu = chain_1.to("cpu")
        e = chain_1 + 5
        return (e, chain_1_cpu, chain_2)

    with FxGraphAnalyzer() as fga:
        inp_hpu = torch.randn(4, 3, device="hpu")
        inp_cpu = inp_hpu.to("cpu")
        fnc_hpu = torch.compile(
            fn,
            dynamic=False,
            backend="hpu_backend",
            options={"use_eager_fallback": True},
        )
        fnc_cpu = torch.compile(fn, dynamic=False, backend="inductor")
        results_hpu = fnc_hpu(inp_hpu)
        results_cpu = fnc_cpu(inp_cpu)
    ops_summary = fga.get_ops_summary()
    assert ops_summary[0]["torch.ops.hpu.batch_as_strided"].eager_count == 1
    for r_hpu, r_cpu in zip(results_hpu, results_cpu, strict=False):
        torch.allclose(r_hpu.to("cpu"), r_cpu)


def test_pass_scalar_reorder_jitfork():
    input_shape = (4, 4)
    scalar = 2
    op_lists = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.sub.Tensor,
    ]

    def raw_function(op, input_tensor, scalar):
        return op(scalar, input_tensor)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend")

    for op in op_lists:
        input_cpu = torch.randn(input_shape, requires_grad=False)
        result = raw_function(op, input_cpu, scalar)
        input_hpu = input_cpu.to("hpu")
        h_result = compiled_fn(op, input_hpu, scalar)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_as_strided_batching():
    def func(x0, x1):
        x = x0 + 1
        y = x + x1
        as_strided = torch.as_strided(y, size=(2, 2), stride=(2, 2))
        z = torch.abs(as_strided)
        return torch.as_strided(z, size=(2, 2), stride=(1, 1)), as_strided

    compiled_func = torch.compile(func, backend="hpu_backend")

    with FxGraphAnalyzer() as fga:
        t1_cpu = torch.ones((8, 8))
        t2_cpu = t1_cpu.clone()
        t1_hpu, t2_hpu = t1_cpu.to("hpu"), t2_cpu.to("hpu")
        cpu_res = func(t1_cpu, t2_cpu)
        hpu_res = tuple([r.to("cpu") for r in compiled_func(t1_hpu, t2_hpu)])
        for c, h in zip(cpu_res, hpu_res, strict=False):
            torch.allclose(c, h)
        ops_summary = fga.get_ops_summary()
        assert ops_summary[0]["torch.ops.hpu.batch_as_strided"].eager_count == 1


def test_as_strided_not_possible_to_merge():
    @torch.compile(backend="hpu_backend")
    def func(x1, x0):
        op0 = torch.abs(x0)
        strided0 = torch.as_strided(op0, size=(2, 2), stride=(1, 1))
        op1 = x1 * 5.0
        strided1 = torch.as_strided(op1, size=(2, 2), stride=(1, 1))
        op2 = torch.add(strided0, strided1)
        strided2 = torch.as_strided(op2, size=(2, 2), stride=(1, 1))
        return strided1, strided2

    t1 = torch.rand(4, 4, device="hpu")
    t2 = torch.rand(4, 4, device="hpu")
    _, _ = func(t1, t2)
