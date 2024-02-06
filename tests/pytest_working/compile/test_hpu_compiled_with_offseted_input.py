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
import os
import torch


@torch.compile(backend="hpu_backend")
def hpu_fn(x):
    return x + x


@torch.compile(backend="inductor")
def cpu_fn(x):
    return x + x


def test_compiled_with_view_input():
    t = torch.rand([2, 3], device="hpu")
    input_tensor_hpu = t.transpose(0, 1)
    input_tensor_cpu = input_tensor_hpu.to("cpu")

    cpu_results = []
    hpu_results = []

    for i in range(3):
        cpu_res = cpu_fn(input_tensor_cpu[i])
        hpu_res = hpu_fn(input_tensor_hpu[i])

        cpu_results.append(cpu_res)
        hpu_results.append(hpu_res)

    for cpu, hpu in zip(cpu_results, hpu_results):
        assert torch.allclose(cpu, hpu.cpu(), atol=0.001, rtol=0.001)
