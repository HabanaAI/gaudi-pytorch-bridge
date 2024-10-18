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
