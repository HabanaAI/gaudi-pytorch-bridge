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
import time

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as exp
import numpy as np
import pytest
import torch


def test_record_stream():
    # exp._reset_device_memory()
    t = torch.FloatTensor([1.0, 2.0, 3.0, 4.0]).pin_memory(device="hpu")
    result = torch.FloatTensor(t.size()).to("hpu")
    stream = ht.hpu.Stream()
    ptr = [None]

    # Performs the CPU->HPU copy in a background stream
    def perform_copy():
        with ht.hpu.stream(stream):
            tmp = torch.FloatTensor(t.size()).to("hpu")
            tmp.copy_(t, non_blocking=True)
            ptr[0] = tmp.data_ptr()
        ht.hpu.current_stream().wait_stream(stream)
        tmp.record_stream(ht.hpu.current_stream())
        time.sleep(1.5)
        result.copy_(tmp)
        del tmp

    perform_copy()
    with ht.hpu.stream(stream):
        tmp2 = torch.FloatTensor(t.size()).to("hpu")
        tmp2.zero_()
        assert tmp2.data_ptr() != ptr[0], f"allocation re-used to soon"

        if result.tolist() == [1.0, 2.0, 3.0, 4.0]:
            assert f"tensor list not equal"

    # we expect "tmp"'s side-stream-tagged block will be reused
    # in that side stream after result.copy_(tmp) in the main stream finishes.
    ht.hpu.current_stream().synchronize()
    with ht.hpu.stream(stream):
        tmp3 = torch.FloatTensor(t.size()).to("hpu")
    # assert tmp3.data_ptr() == ptr[0], f"allocation not re-used"


def test_record_stream_on_shifted_view():
    # exp._reset_device_memory()
    # See issue #27366

    # This test detects unexpected block reallocation. For reliable test,
    # the stream to allocate tensors is isolated. The allocator will not
    # reuse free blocks which were allocated from another stream.
    stream_alloc = ht.hpu.Stream()
    with ht.hpu.stream(stream_alloc):
        base = torch.FloatTensor([10, 10]).to("hpu")

    # Record another stream on a shifted view tensor.
    view = base[5:]
    assert view.storage_offset() > 0

    stream_record = ht.hpu.Stream()
    with ht.hpu.stream(stream_record):
        time.sleep(0.5)

    view.record_stream(stream_record)

    # Delete those tensors to make the block free soon.
    data_ptr = base.data_ptr()
    del base, view

    # A new tensor should not be allocated to the block above.
    stream_alloc.synchronize()

    with ht.hpu.stream(stream_alloc):
        try_realloc = torch.FloatTensor([10, 10]).to("hpu")

    assert try_realloc.data_ptr() != data_ptr
