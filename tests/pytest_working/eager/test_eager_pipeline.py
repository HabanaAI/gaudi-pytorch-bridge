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
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest


def test_pipeline_ops_out():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")
    cpu_out = torch.zeros(cpu_tensor.shape)
    hpu_out = torch.zeros(hpu_tensor.shape).to("hpu")

    cpu_tensor1 = torch.Tensor(np.arange(-10.0, 10.0, 0.5))
    hpu_tensor1 = cpu_tensor1.to("hpu")
    cpu_out1 = torch.zeros(cpu_tensor1.shape)
    hpu_out1 = torch.zeros(hpu_tensor1.shape).to("hpu")

    cpu_tensor2 = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor2 = cpu_tensor2.to("hpu")
    cpu_out2 = torch.zeros(cpu_tensor2.shape)
    hpu_out2 = torch.zeros(hpu_tensor2.shape).to("hpu")

    cpu_tensor3 = torch.Tensor(np.arange(-10.0, 10.0, 0.5))
    hpu_tensor3 = cpu_tensor3.to("hpu")
    cpu_out3 = torch.zeros(cpu_tensor3.shape)
    hpu_out3 = torch.zeros(hpu_tensor3.shape).to("hpu")

    torch.sin(hpu_tensor, out=hpu_out)
    torch.sin(cpu_tensor, out=cpu_out)

    torch.pow(hpu_tensor1, 2.0, out=hpu_out1)
    torch.pow(cpu_tensor1, 2.0, out=cpu_out1)

    torch.sin(hpu_tensor2, out=hpu_out2)
    torch.sin(cpu_tensor2, out=cpu_out2)

    torch.pow(hpu_tensor3, 2.0, out=hpu_out1)
    torch.pow(cpu_tensor3, 2.0, out=cpu_out1)

    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out

    result_hpu1 = hpu_out1.to("cpu")
    result_cpu1 = cpu_out1

    result_hpu2 = hpu_out2.to("cpu")
    result_cpu2 = cpu_out2

    result_hpu3 = hpu_out3.to("cpu")
    result_cpu3 = cpu_out3

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)
    assert torch.allclose(result_hpu1, result_cpu1, atol=0.001, rtol=0.001)
    assert torch.allclose(result_hpu2, result_cpu2, atol=0.001, rtol=0.001)
    assert torch.allclose(result_hpu3, result_cpu3, atol=0.001, rtol=0.001)
