###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import habana_frameworks.torch.hpu as hthpu
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import is_gaudi1

device = "hpu"
backend_compiler = "hpu_backend"


def hpu_partiton_breaker(x):
    x = x.to("cpu")
    x = torch.sigmoid(x)
    x = x.to(device)
    return x


def test_basic():
    def fn(a, b, c):
        x = torch.matmul(a, b)
        x = hpu_partiton_breaker(x)
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, c)
        return x.sum()

    a_ = torch.randn(4, 4, requires_grad=False).to(device)
    b_ = torch.randn(4, 4, requires_grad=False).to(device)
    c_ = torch.randn(4, 4, requires_grad=False).to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        y = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [True, False], "should reuse input"


def test_not_last_use():
    def fn(a, b, c):
        x = torch.matmul(a, b)
        part_in = hpu_partiton_breaker(x)
        x = torch.softmax(part_in, dim=-1)
        x = torch.matmul(x, c)
        # part_in should not be reusable since it's returned to outside
        return x.sum(), part_in

    a_ = torch.randn(4, 4, requires_grad=False).to(device)
    b_ = torch.randn(4, 4, requires_grad=False).to(device)
    c_ = torch.randn(4, 4, requires_grad=False).to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        _ = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [False, False], "should not reuse input"


def test_view():
    def fn(a, b, c):
        x = torch.matmul(a, b)
        x = hpu_partiton_breaker(x)
        # x is not reusable since its viewed y will be used after partition
        y = x.view([-1])
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, c)
        return x.sum(), y

    a_ = torch.randn(4, 4, requires_grad=False).to(device)
    b_ = torch.randn(4, 4, requires_grad=False).to(device)
    c_ = torch.randn(4, 4, requires_grad=False).to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        sum, y = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [
            False,
            False,
        ], "should not reuse the input since it has a view user"


def test_mutation_partition():
    def fn(a, b, c):
        x = torch.matmul(a, b)
        x = hpu_partiton_breaker(x)
        y = x.add_(a)
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, c)
        # y will be partition output, and it shares same memory with x
        # bridge can set it reusable even though it's mutated by partition.
        # then synapse gc will make the final reuse decision inside
        return x.sum(), y

    a = torch.randn(4, 4, requires_grad=False)
    b = torch.randn(4, 4, requires_grad=False)
    c = torch.randn(4, 4, requires_grad=False)
    a_, b_, c_ = a.to(device), b.to(device), c.to(device)
    a_ref, b_ref, c_ref = a.to(device), b.to(device), c.to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        sum, y = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "in_to_out_dups" in target_meta
        assert target_meta["in_to_out_dups"] == {
            0: 0,
        }
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [
            True,
            False,
            False,
        ], "should reuse the input even it' mutated by partition"

    sum_ref, y_ref = fn(a_ref, b_ref, c_ref)
    assert torch.allclose(sum.to("cpu"), sum_ref.to("cpu"), atol=1e-3), "output mismatch"
    assert torch.allclose(y.to("cpu"), y_ref.to("cpu"), atol=1e-3), "output mismatch"


def test_produced_by_mutation_partition():
    def fn(a, b, c):
        x = a.add_(b)
        y = hpu_partiton_breaker(x)
        # part2 inputs [x, y, c]
        # x is not reusable because it shares mem with a
        # y is reusable
        # c is not rusable
        z = x + y
        z = torch.softmax(z, dim=-1)
        z = torch.matmul(z, c)
        return z.sum()

    a = torch.randn(4, 4, requires_grad=False)
    b = torch.randn(4, 4, requires_grad=False)
    c = torch.randn(4, 4, requires_grad=False)
    a_, b_, c_ = a.to(device), b.to(device), c.to(device)
    a_ref, b_ref, c_ref = a.to(device), b.to(device), c.to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        sum = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [
            False,
            True,
            False,
        ], "x is not reusable since it alias graph input"

    sum_ref = fn(a_ref, b_ref, c_ref)
    assert torch.allclose(sum.to("cpu"), sum_ref.to("cpu"), atol=1e-3), "output mismatch"


def test_produced_by_mutation_partition2():
    def fn(a, b, c):
        b_ = torch.sigmoid(b)
        x = a.add_(b_)
        y = hpu_partiton_breaker(x)
        # part2 inputs [x, y, b_, c]
        # x is not reusable because it shares mem with a
        # b_ is reusable
        # y is reusable
        # c is not rusable
        z = x + y
        z = z * b_
        z = torch.softmax(z, dim=-1)
        z = torch.matmul(z, c)
        return z.sum()

    a = torch.randn(4, 4, requires_grad=False)
    b = torch.randn(4, 4, requires_grad=False)
    c = torch.randn(4, 4, requires_grad=False)
    a_, b_, c_ = a.to(device), b.to(device), c.to(device)
    a_ref, b_ref, c_ref = a.to(device), b.to(device), c.to(device)

    compiled_fn = torch.compile(fn, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback():
        sum = compiled_fn(a_, b_, c_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
        assert "is_reusables" in target_meta
        assert target_meta["is_reusables"] == [
            False,
            True,
            True,
            False,
        ], "reusable info not as expected"

    sum_ref = fn(a_ref, b_ref, c_ref)
    assert torch.allclose(sum.to("cpu"), sum_ref.to("cpu"), atol=1e-3), "output mismatch"


@pytest.mark.skipif(is_gaudi1(), reason="random failure on G1")
def test_ws_reduced_e2e():
    batch_size = 1024
    channel_num = 1024
    shape = (batch_size, channel_num)

    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)
            self.fc2 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)
            self.fc3 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            x = torch.sigmoid(x)
            x = hpu_partiton_breaker(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    model = TestModule().to("hpu")
    a_ = torch.randn(shape, dtype=torch.bfloat16, requires_grad=False).to(device)

    compiled_fn = torch.compile(model, backend=backend_compiler)
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback(), torch.no_grad():
        y = compiled_fn(a_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
    assert "is_reusables" in target_meta
    assert target_meta["is_reusables"] == [
        True,
        False,
        False,
        False,
    ], "should reuse input"

    hthpu.synchronize()  # wait for recipe finish to get the exact workspace size
    mem_summary = hthpu.memory._extended_memory_summary_dict()
    ws_size = mem_summary["last_workspace"]  # get the real workspace size

    # run without reuse and compare the WS
    compiled_fn_no_reuse = torch.compile(model, backend=backend_compiler, options={"enable_synapse_input_reuse": False})
    with FxGraphAnalyzer(reset_dynamo=False) as fga, use_eager_fallback(), torch.no_grad():
        _ = compiled_fn_no_reuse(a_)
        part_num = fga.get_partition_num()
        assert part_num == 2
        part_infos = fga.get_partition_infos()
        assert len(part_infos) == part_num
        target_part_info = part_infos[0] if part_infos[0].num_nodes > part_infos[1].num_nodes else part_infos[1]
        target_meta = target_part_info.meta
    assert "is_reusables" not in target_meta

    hthpu.synchronize()
    mem_summary = hthpu.memory._extended_memory_summary_dict()
    orig_ws_size = mem_summary["last_workspace"]
    assert ws_size < orig_ws_size, "reuse doesn't reduce ws size"

    # compare results with eager mode
    with torch.no_grad():
        ref = model(a_)

    y_cpu = y.to("cpu")
    ref_cpu = ref.to("cpu")
    assert torch.allclose(y_cpu, ref_cpu, atol=1e-3), "output mismatch"


@pytest.mark.skipif(is_gaudi1(), reason="random failure on G1")
def test_not_cache_hit_e2e():
    batch_size = 1024
    channel_num = 1024
    shape = (batch_size, channel_num)

    class TestModule1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)
            self.fc2 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)
            self.fc3 = torch.nn.Linear(channel_num, channel_num, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            x = torch.sigmoid(x)
            x = hpu_partiton_breaker(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    class TestModule2(TestModule1):
        def forward(self, x):
            x = torch.sigmoid(x)
            part_in = hpu_partiton_breaker(x)
            x = self.fc1(part_in)
            x = self.fc2(x)
            x = self.fc3(x)
            return x, part_in

    model1 = TestModule1().to("hpu")
    a_ = torch.randn(shape, dtype=torch.bfloat16, requires_grad=False).to(device)

    # run the compiled fn1, the partition graph will be cached
    compiled_fn1 = torch.compile(model1, backend=backend_compiler)
    with use_eager_fallback(), torch.no_grad():
        _ = compiled_fn1(a_)

    hthpu.synchronize()
    mem_summary = hthpu.memory._extended_memory_summary_dict()
    ws_size1 = mem_summary["last_workspace"]

    # run the compiled fn2, the partition graph is same, but the reusable info
    # should be different, so we should not cache hit. Check the resuls, since
    # if cache hit wrongly, the results will be wrong.
    model2 = TestModule2().to("hpu")
    compiled_fn2 = torch.compile(model2, backend=backend_compiler)
    with use_eager_fallback(), torch.no_grad():
        _, part_in = compiled_fn2(a_)

    hthpu.synchronize()
    mem_summary = hthpu.memory._extended_memory_summary_dict()
    ws_size2 = mem_summary["last_workspace"]

    assert ws_size1 < ws_size2, "module 2 ws should be bigger since it won't be reusable"

    with torch.no_grad():
        _, ref = model2(a_)
    # compare results with eager mode
    assert torch.allclose(part_in.to("cpu"), ref.to("cpu"), atol=1e-3), "output mismatch"
