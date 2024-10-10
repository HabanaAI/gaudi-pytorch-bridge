###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import os
import unittest

import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from packaging.version import Version, parse
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm, same
from torch._streambase import _StreamBase
from torch.nn import functional as F

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

    @unittest.skipIf(
        Version(parse(torch.__version__).base_version) < Version("2.4.0"), "Need patch pytorch/pull/123487"
    )  # version < 2.4 need patch https://github.com/pytorch/pytorch/pull/123487
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


import time
from contextlib import contextmanager
from typing import Generator, List, Union, cast

import habana_frameworks.torch as htorch

if Version(parse(torch.__version__).base_version) < Version("2.4.0"):
    from torch.distributed.pipeline.sync.stream import CPUStream, record_stream

    class CPUStreamType:
        pass

    AbstractStream = Union[torch.hpu.Stream, CPUStreamType]

    def is_hpu(stream) -> bool:
        """Returns ``True`` if the given stream is a valid HPU stream."""
        return stream is not CPUStream

    def as_hpu(stream: AbstractStream) -> torch.hpu.Stream:
        """Casts the given stream as :class:`torch.hpu.Stream`."""
        return cast(torch.hpu.Stream, stream)

    def get_device(stream: AbstractStream) -> torch.device:
        """Gets the device from CPU or HPU stream."""
        if is_hpu(stream):
            return as_hpu(stream).device
        return torch.device("cpu")

    def new_stream(device: torch.device) -> AbstractStream:
        """Creates a new stream for either CPU or HPU device."""
        if device.type != "hpu":
            return CPUStream
        return torch.hpu.Stream(device)

    def current_stream(device: torch.device) -> AbstractStream:
        """:func:`torch.hpu.current_stream` for either CPU or HPU device."""
        if device.type != "hpu":
            return CPUStream
        return torch.hpu.current_stream(device)

    def default_stream(device: torch.device) -> AbstractStream:
        """:func:`torch.hpu.default_stream` for either CPU or HPU device."""
        if device.type != "hpu":
            return CPUStream
        return torch.hpu.default_stream(device)

    @contextmanager
    def use_stream(stream: AbstractStream) -> Generator[None, None, None]:
        """:func:`torch.hpu.stream` for either CPU or HPU stream."""
        if not is_hpu(stream):
            yield
            return

        with torch.hpu.stream(as_hpu(stream)):
            yield

    def wait_stream(source: AbstractStream, target: AbstractStream) -> None:
        """:meth:`torch.hpu.Stream.wait_stream` for either CPU or HPU stream. It
        makes the source stream wait until the target stream completes work queued.
        """
        if is_hpu(target):
            if is_hpu(source):
                # A HPU stream waits another HPU stream.
                as_hpu(source).wait_stream(as_hpu(target))
            else:
                # CPU waits a HPU stream.
                as_hpu(target).synchronize()

        # If the target is CPU, synchronization is not required.

    def _sleep(cycles):
        time.sleep(cycles / 1000000)

    @pytest.fixture(scope="session")
    def hpu_sleep():
        # Warm-up HPU.
        torch.empty(1, device="hpu")

        # From test/test_hpu.py in PyTorch.
        start = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        start.record()
        _sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)

        def hpu_sleep(seconds):
            _sleep(int(seconds * cycles_per_ms * 1000))

        return hpu_sleep

    class TestNewStream:
        def test_new_stream_cpu(self):
            stream = new_stream(torch.device("cpu"))
            assert stream is CPUStream

        @skip_if_no_hpu
        def test_new_stream_hpu(self):
            stream = new_stream(torch.device("hpu"))
            assert isinstance(stream, torch.hpu.Stream)
            assert stream != torch.hpu.default_stream()

    class TestCurrentStream:
        def test_current_stream_cpu(self):
            stream = current_stream(torch.device("cpu"))
            assert stream is CPUStream

        @skip_if_no_hpu
        def test_current_stream_hpu(self):
            stream = current_stream(torch.device("hpu"))
            assert isinstance(stream, torch.hpu.Stream)
            assert stream == torch.hpu.current_stream()

    class TestDefaultStream:
        def test_default_stream_cpu(self):
            stream = default_stream(torch.device("cpu"))
            assert stream is CPUStream

        @skip_if_no_hpu
        def test_default_stream_hpu(self):
            stream = default_stream(torch.device("hpu"))
            assert isinstance(stream, torch.hpu.Stream)
            assert stream == torch.hpu.default_stream()

    class TestUseStream:
        def test_use_stream_cpu(self):
            with use_stream(CPUStream):
                pass

        @skip_if_no_hpu
        def test_use_stream_hpu(self):
            stream = new_stream(torch.device("hpu"))
            with use_stream(stream):
                assert current_stream(torch.device("hpu")) == stream

    class TestGetDevice:
        def test_get_device_cpu(self):
            assert get_device(CPUStream).type == "cpu"

        @skip_if_no_hpu
        def test_get_device_hpu(self):
            stream = current_stream(torch.device("hpu"))
            assert get_device(stream).type == "hpu"

    class TestWaitStream:
        def _test_wait_stream(self, source, target, hpu_sleep=None):
            with use_stream(target):
                if is_hpu(target):
                    hpu_sleep(0.5)
                x = torch.ones(100, 100, device=get_device(target))

            wait_stream(source, target)

            with use_stream(source):
                assert x.sum().item() == 10000

        def test_wait_stream_cpu_cpu(self):
            source = CPUStream
            target = CPUStream
            self._test_wait_stream(source, target)

        @skip_if_no_hpu
        def test_wait_stream_cpu_hpu(self, hpu_sleep):
            source = CPUStream
            target = new_stream(torch.device("hpu"))
            self._test_wait_stream(source, target, hpu_sleep)

        @skip_if_no_hpu
        def test_wait_stream_hpu_cpu(self, hpu_sleep):
            source = new_stream(torch.device("hpu"))
            target = CPUStream
            self._test_wait_stream(source, target, hpu_sleep)

        @skip_if_no_hpu
        def test_wait_stream_hpu_hpu(self, hpu_sleep):
            source = current_stream(torch.device("hpu"))
            target = new_stream(torch.device("hpu"))
            self._test_wait_stream(source, target, hpu_sleep)

    class TestRecordStream:
        def test_record_stream_cpu(self):
            # It should silently ignore CPU tensors.
            x = torch.rand(1, device=torch.device("cpu"))
            record_stream(x, CPUStream)

        @skip_if_no_hpu
        def test_record_stream_hpu(self, hpu_sleep):
            # This test detects unexpected block reallocation. For reliable test,
            # the stream to allocate tensors is isolated. The allocator will not
            # reuse free blocks which were allocated from another stream.
            stream_alloc = new_stream(torch.device("hpu"))
            with torch.hpu.stream(stream_alloc):
                x = torch.rand(1, device=torch.device("hpu"))

            stream = new_stream(torch.device("hpu"))
            record_stream(x, stream)
            with use_stream(stream):
                hpu_sleep(0.5)

            # 'x' is deleted at Python's perspective. But the block of 'x' is still
            # required for 'stream'. 'y' shouldn't be allocated to the block.
            data_ptr = x.data_ptr()
            del x
            stream_alloc.synchronize()
            with torch.hpu.stream(stream_alloc):
                y = torch.rand(1, device=torch.device("hpu"))
            assert y.data_ptr() != data_ptr

            # Pause Python until 'stream' finishes tasks queued. Now the block of
            # 'x' is free to be reallocated.
            # wait_stream(CPUStream, stream)
            # with torch.hpu.stream(stream_alloc):
            #     z = torch.rand(1, device=torch.device("hpu"))
            # assert z.data_ptr() == data_ptr

        @skip_if_no_hpu
        def test_record_stream_shifted_view(self, hpu_sleep):
            # Issue: https://github.com/pytorch/pytorch/issues/27366
            stream_alloc = new_stream(torch.device("hpu"))
            with torch.hpu.stream(stream_alloc):
                x = torch.rand(2, device=torch.device("hpu"))

            y = x[1:]
            assert y.data_ptr() > x.data_ptr()

            stream = new_stream(torch.device("hpu"))
            with use_stream(stream):
                hpu_sleep(1)
            record_stream(y, stream)

            data_ptr = x.data_ptr()
            del x, y

            stream_alloc.synchronize()
            with torch.hpu.stream(stream_alloc):
                z = torch.rand(2, device=torch.device("hpu"))
            stream_alloc.synchronize()
            assert z.data_ptr() != data_ptr


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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
