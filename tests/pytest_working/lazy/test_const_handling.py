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

from contextlib import contextmanager

# Fixture to set the environment variable
@pytest.fixture
def set_env_variable():
    variable_name_weight_packing = "ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING"
    original_value_weight_packing = os.environ.get(variable_name_weight_packing)

    variable_name_constant_folding = "ENABLE_CONSTANT_FOLDING"
    original_value_constant_folding= os.environ.get(variable_name_constant_folding)

    variable_name_experimental_flags = "ENABLE_EXPERIMENTAL_FLAGS"
    original_value_experimental_flags= os.environ.get(variable_name_experimental_flags)

    # Set the environment variable to the desired value
    os.environ[variable_name_weight_packing] = "1"
    os.environ[variable_name_constant_folding] = "1"
    os.environ[variable_name_experimental_flags] = "1"

    # Yield to provide the value for the test
    yield "1"

    # Teardown: Restore the original value after the test
    if original_value_weight_packing is not None:
        os.environ[variable_name_weight_packing] = original_value_weight_packing
    else:
        del os.environ[variable_name_weight_packing]

    if original_value_constant_folding is not None:
        os.environ[variable_name_constant_folding] = original_value_constant_folding
    else:
        del os.environ[variable_name_constant_folding]

    if original_value_experimental_flags is not None:
        os.environ[variable_name_experimental_flags] = original_value_experimental_flags
    else:
        del os.environ[variable_name_experimental_flags]

def test_same_graph_with_diff_const(set_env_variable):
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

    import habana_frameworks.torch.core as htcore
    htcore.hpu_set_env()

    #Run test on HPU
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    input_tensor_hpu = input_tensor.to(hpu)
    conv1_hpu = conv1.to(hpu)
    conv2_hpu = conv2.to(hpu)
    from habana_frameworks.torch.core.quantization import _mark_params_as_const, _check_params_as_const
    _mark_params_as_const(conv1_hpu)
    _mark_params_as_const(conv2_hpu)
    _check_params_as_const(conv1_hpu)
    _check_params_as_const(conv2_hpu)

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

    htcore.hpu_reset_env()


def test_same_const_across_recipes():
    # Define input tensors
    input_tensor1 = torch.randn(1, 3, 64, 64)
    input_tensor2 = torch.randn(1, 3, 32, 32)

    # Define kernel size, stride, and padding for the convolutional layers
    kernel_size = 3
    stride = 1
    padding = 1

    # Set the weights and bias for both convolutional layers
    conv_layer1 = torch.nn.Conv2d(3, 6, kernel_size, stride, padding)
    # Initialize the weights of the convolutional layer with random values
    nn.init.xavier_normal_(conv_layer1.weight)
    # Initialize the biases of the convolutional layer with random non-zero values
    nn.init.normal_(conv_layer1.bias)

    # Set the convolutional layers in evaluation mode and disable autograd
    conv_layer1.eval()
    with torch.no_grad():
        # Perform convolution on both nodes
        output1 = conv_layer1(input_tensor1)
        output2 = conv_layer1(input_tensor2)

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    input_tensor1_hpu = input_tensor1.to(hpu)
    input_tensor2_hpu = input_tensor2.to(hpu)
    conv_layer1_hpu = conv_layer1.to(hpu)
    from habana_frameworks.torch.core.quantization import _mark_params_as_const
    _mark_params_as_const(conv_layer1_hpu)

    import habana_frameworks.torch.core as htcore
    htcore.hpu_set_env()

    with torch.no_grad():
        output1_hpu = conv_layer1_hpu(input_tensor1_hpu)
    output1_hpu_cpu = output1_hpu.to(cpu)
    htcore.mark_step()

    with torch.no_grad():
        output2_hpu = conv_layer1_hpu(input_tensor2_hpu)
    output2_hpu_cpu = output2_hpu.to(cpu)

    with torch.no_grad():
        output1_repeat_hpu = conv_layer1_hpu(input_tensor1_hpu)
    output1_repeat_hpu_cpu = output1_repeat_hpu.to(cpu)

    numpy.testing.assert_allclose(
        output1_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)
    numpy.testing.assert_allclose(
        output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.001, rtol=0.001)
    numpy.testing.assert_allclose(
       output1_repeat_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)

    htcore.hpu_reset_env()
