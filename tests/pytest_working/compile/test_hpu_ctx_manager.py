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

import unittest
from collections.abc import Generator
from contextlib import contextmanager

import habana_frameworks.torch as htorch
import pytest
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo.testing import same

# def setup_distributed(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     os.environ["RANK"] = str(rank)

#     # initialize the process group
#     import habana_frameworks.torch.distributed.hccl
#     torch.distributed.init_process_group(backend='hccl', rank=rank, world_size=world_size)


class CtxManagerTests(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_context_manager1(self):
        def fn(x):
            s = torch.hpu.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            current_stream = torch.hpu.current_stream()
            s.wait_stream(current_stream)
            with torch.hpu.stream(s):
                x = torch.relu(x)
            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_across_graph_break(self):
        def fn(x):
            s = torch.hpu.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            tcs = torch.hpu.stream(s)
            current_stream = torch.hpu.current_stream()
            s.wait_stream(current_stream)

            with tcs:
                x = torch.relu(x)

            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_context_manager2(self):
        def fn(x, s):
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            current_stream = torch.hpu.current_stream()
            s.wait_stream(current_stream)

            with torch.hpu.stream(s):
                x = torch.relu(x)

            current_stream.wait_stream(s)
            with torch.hpu.stream(current_stream):
                x = torch.relu(x)

            s2 = torch.hpu.Stream()
            s2.wait_stream(current_stream)
            with torch.hpu.stream(s2):
                x = torch.relu(x)

            current_stream.wait_stream(s2)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        s = torch.hpu.Stream()
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)
            # print(x.to("cpu"))
            new_stream = htorch.hpu.Stream()
            # print("Stream type: ", type(new_stream))
            # print("Stream is instance of ", isinstance(new_stream, _StreamBase))
            with htorch.hpu.stream(new_stream):
                x = torch.sin(x)
                x = torch.add(x, 3)

            cur_stream = htorch.hpu.current_stream()
            cur_stream.wait_stream(new_stream)

            x = torch.add(x, 4)
            is_idle = cur_stream.query()
            cur_stream.synchronize()

            with htorch.hpu.stream(new_stream):
                x = torch.add(x, 5)
            new_stream.synchronize()

            is_equal = cur_stream == new_stream

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        # opt_fn = torch._dynamo.optimize(cnts)(fn)
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        print("Number of graphs: ", cnts.frame_count, " ops:", cnts.op_count)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 20)

    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_compared_with_constant(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.hpu.current_stream()
            if cur_stream is not None:
                return x + 1
            return x - 1

        def fn2(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.hpu.current_stream()
            if cur_stream != "const_str":
                return x + 1
            return x - 1

        x = torch.randn((2, 2), device="hpu")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        res = opt_fn(x)
        res2 = opt_fn2(x)
        self.assertEqual(ref, res)
        self.assertEqual(ref, res2)

    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_stream_compared_with_stream(self):
        def fn(x, s0, s1):
            if s0 == s1:
                return x + 1
            else:
                return x - 1

        s0 = torch.hpu.Stream()
        s1 = torch.hpu.Stream()
        x = torch.randn(2, 2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref0, res0)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        # We have a re-compilation because of chaning inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref1, res1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref1, res1)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        # We have a re-compilation because of chaning inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref0, res0)

    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_event_method_create_stream_outside_of_compile(self):
        def fn(x, cur_stream, new_stream):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            x = torch.add(x, 3)
            event = cur_stream.record_event()
            is_idle = event.query()

            new_stream.wait_event(event)
            with torch.hpu.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.hpu.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            # use new event to sync
            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        cur_stream = torch.hpu.current_stream()
        new_stream = torch.hpu.Stream()
        ref = fn(x, cur_stream, new_stream)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, cur_stream, new_stream)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 19)

    @unittest.skipIf(not torch.hpu.is_available(), "requires hpu")
    def test_hpu_event_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.hpu.current_stream()
            new_stream = torch.hpu.Stream()

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            is_idle = event.query()

            new_stream.wait_event(event)
            with torch.hpu.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.hpu.Event()
            new_event.record(new_stream)

            x = torch.add(x, 5)
            new_event.wait(cur_stream)

            # use new event to sync
            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="hpu")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        # opt_fn = torch._dynamo.optimize("hpu_backend", cnts, nopython=True)(fn)
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 19)


skip_if_no_hpu = pytest.mark.skipif(not torch.hpu.is_available(), reason="hpu required")


@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.hpu.device` for either CPU or HPU device."""
    if device.type != "hpu":
        yield
        return
    with torch.hpu.device(device):
        yield


class TestUseDevice:
    def test_use_device_cpu(self):
        with use_device(torch.device("cpu")):
            pass

    @skip_if_no_hpu
    def test_use_device_hpu(self):
        with use_device(torch.device("hpu")):
            pass
