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

import habana_frameworks.torch as htorch
import pytest
import torch
from torch.testing._internal.common_utils import TestCase


class TestHPU(TestCase):
    def test_generic_stream_event(self):
        stream = torch.Stream("hpu")
        self.assertEqual(stream.device_index, torch.hpu.current_device())
        hpu_stream = torch.hpu.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertEqual(stream.stream_id, hpu_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.hpu.current_stream().stream_id)
        event1 = torch.Event("hpu", enable_timing=True)
        event2 = torch.Event("hpu", enable_timing=True)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.hpu.stream(hpu_stream):
            a_hpu = a.to("hpu", non_blocking=True)
            b_hpu = b.to("hpu", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.hpu.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_hpu = a_hpu + b_hpu
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_hpu.cpu(), a + b)
        self.assertTrue(event1.elapsed_time(event2) > 0)
