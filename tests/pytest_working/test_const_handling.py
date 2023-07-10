###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

import os
import torch
import torch.nn as nn
import numpy
import pytest
from habana_frameworks.torch.utils.library_loader import load_habana_module

from contextlib import contextmanager

def test_same_graph_with_diff_const():
    load_habana_module()
    # Define the input tensor
    input_tensor = torch.randn(1, 3, 32, 32)  # Assuming input size of (batch_size, channels, height, width)

    # Define the weight and bias tensors for the first convolutional layer
    weight1 = torch.randn(6, 3, 5, 5)  # Assuming 6 filters, 3 input channels, and kernel size of 5x5
    bias1 = torch.randn(6)  # One bias term for each filter

    # Define the weight and bias tensors for the second convolutional layer
    weight2 = torch.randn(6, 3, 5, 5)  # Assuming 6 filters, 3 input channels, and kernel size of 5x5
    bias2 = torch.randn(6)  # One bias term for each filter

    # Define the first convolutional layer
    conv1 = nn.Conv2d(3, 6, kernel_size=5)
    conv1.weight.data = weight1
    conv1.bias.data = bias1

    # Define the second convolutional layer
    conv2 = nn.Conv2d(3, 6, kernel_size=5)
    conv2.weight.data = weight2
    conv2.bias.data = bias2

    # Apply the convolutional layers to the input tensor
    with torch.no_grad():
        output1 = conv1(input_tensor)
        output2 = conv2(input_tensor)

    #Run test on HPU
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    input_tensor_hpu = input_tensor.to(hpu)
    conv1_hpu = conv1.to(hpu)
    conv2_hpu = conv2.to(hpu)
    from habana_frameworks.torch.core.quantization import _mark_params_as_const
    _mark_params_as_const(conv1_hpu)
    _mark_params_as_const(conv2_hpu)

    import habana_frameworks.torch.core as htcore
    htcore.hpu_initialize()
    with torch.no_grad():
        output1_hpu = conv1_hpu(input_tensor_hpu)
        htcore.mark_step()

    output1_hpu_cpu = output1_hpu.to(cpu)
    numpy.testing.assert_allclose(
        output1_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)

    with torch.no_grad():
        output2_hpu = conv2_hpu(input_tensor_hpu)
        htcore.mark_step()
    output2_hpu_cpu = output2_hpu.to(cpu)
    numpy.testing.assert_allclose(
        output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.001, rtol=0.001)
