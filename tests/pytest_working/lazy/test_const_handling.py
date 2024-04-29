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
from contextlib import contextmanager

import numpy
import pytest
import torch
import torch.nn as nn


# Fixture to set the environment variable
@pytest.fixture
def set_env_variable(request, arg=False):
    variable_name_weight_packing = "ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING"
    original_value_weight_packing = os.environ.get(variable_name_weight_packing)

    variable_name_constant_folding = "ENABLE_CONSTANT_FOLDING"
    original_value_constant_folding = os.environ.get(variable_name_constant_folding)

    variable_name_experimental_flags = "ENABLE_EXPERIMENTAL_FLAGS"
    original_value_experimental_flags = os.environ.get(variable_name_experimental_flags)

    # Set the environment variable to the desired value
    os.environ[variable_name_weight_packing] = "1"
    os.environ[variable_name_constant_folding] = "1"
    os.environ[variable_name_experimental_flags] = "1"

    arg = request.param
    if arg:
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = "/tmp/cache,false,8192"
        print("Enabled serialization of recipe on disk")

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

    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = ""


@pytest.mark.parametrize("set_env_variable", [False], indirect=True)
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

    htcore.hpu_set_inference_env()

    # Run test on HPU
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    input_tensor_hpu = input_tensor.to(hpu)
    conv1_hpu = conv1.to(hpu)
    conv2_hpu = conv2.to(hpu)

    htcore.hpu_inference_initialize(conv1_hpu)
    htcore.hpu_inference_initialize(conv2_hpu)

    from habana_frameworks.torch.core.quantization import _check_params_as_const

    _check_params_as_const(conv1_hpu)
    _check_params_as_const(conv2_hpu)

    with torch.no_grad():
        output1_hpu = conv1_hpu(input_tensor_hpu)
        htcore.mark_step()

    output1_hpu_cpu = output1_hpu.to(cpu)
    numpy.testing.assert_allclose(output1_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)

    with torch.no_grad():
        output2_hpu = conv2_hpu(input_tensor_hpu)
        htcore.mark_step()
    output2_hpu_cpu = output2_hpu.to(cpu)
    numpy.testing.assert_allclose(output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.001, rtol=0.001)

    htcore.hpu_teardown_inference_env()


@pytest.mark.parametrize("set_env_variable", [False], indirect=True)
def test_same_const_across_recipes(set_env_variable):
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

    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()

    input_tensor1_hpu = input_tensor1.to(hpu)
    input_tensor2_hpu = input_tensor2.to(hpu)
    conv_layer1_hpu = conv_layer1.to(hpu)

    htcore.hpu_inference_initialize(conv_layer1_hpu)

    from habana_frameworks.torch.core.quantization import _check_params_as_const

    _check_params_as_const(conv_layer1_hpu)

    with torch.no_grad():
        output1_hpu = conv_layer1_hpu(input_tensor1_hpu)
    output1_hpu_cpu = output1_hpu.to(cpu)
    htcore.mark_step()

    with torch.no_grad():
        output2_hpu = conv_layer1_hpu(input_tensor2_hpu)
    output2_hpu_cpu = output2_hpu.to(cpu)
    htcore.mark_step()

    with torch.no_grad():
        output1_repeat_hpu = conv_layer1_hpu(input_tensor1_hpu)
    output1_repeat_hpu_cpu = output1_repeat_hpu.to(cpu)
    htcore.mark_step()

    numpy.testing.assert_allclose(output1_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)
    numpy.testing.assert_allclose(output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.001, rtol=0.001)
    numpy.testing.assert_allclose(
        output1_repeat_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001
    )

    htcore.hpu_teardown_inference_env()


@pytest.mark.parametrize("set_env_variable", [False], indirect=True)
def test_user_access_to_modified_tensor(set_env_variable):
    # Define input tensors
    input_tensor = torch.randn(1, 3, 32, 32)

    # Define kernel size, stride, and padding for the convolutional layers
    kernel_size = 3
    stride = 1
    padding = 1

    # Create the original convolutional layer
    conv_layer = torch.nn.Conv2d(3, 6, kernel_size, stride, padding)

    weight_copy = conv_layer.weight.clone()

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()

    input_tensor_hpu = input_tensor.to(hpu)
    conv_layer_hpu = conv_layer.to(hpu)

    htcore.hpu_inference_initialize(conv_layer_hpu)

    from habana_frameworks.torch.core.quantization import _check_params_as_const

    _check_params_as_const(conv_layer_hpu)

    with torch.no_grad():
        output_hpu = conv_layer_hpu(input_tensor_hpu)

    output_hpu_cpu = output_hpu.to(cpu)
    htcore.mark_step()
    weight_hpu_cpu = conv_layer_hpu.weight.to(cpu)
    numpy.testing.assert_allclose(weight_hpu_cpu.detach().numpy(), weight_copy.detach().numpy(), atol=0.001, rtol=0.001)
    htcore.hpu_teardown_inference_env()


# Define the parameterized fixture using pytest.mark.parametrize
@pytest.mark.parametrize("set_env_variable", [True], indirect=True)
def test_zero_sized_tensor(set_env_variable):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            # Define two parameters 'a' and 'b'
            self.a = nn.Parameter(torch.randn(1, requires_grad=True))
            self.b = nn.Parameter(torch.randn(1, requires_grad=True))

            # Define a linear layer with input size 1 and output size 1
            self.linear_layer = nn.Linear(1, 1)

        def forward(self, x):
            # Perform the multiplication of 'a' and 'b' in the forward pass
            result = self.a * self.b
            result = result * x

            # Pass the result through the linear layer
            output = self.linear_layer(result)

            return output

    # Create an instance of the Model
    model = Model()

    # Input tensor
    input_tensor = torch.randn(1, 1)

    # Forward pass
    output = model(input_tensor)

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()

    input_tensor_hpu = input_tensor.to(hpu)
    model_hpu = model.to(hpu)

    htcore.hpu_inference_initialize(model_hpu)

    with torch.no_grad():
        output_hpu = model_hpu(input_tensor_hpu)

    output_hpu_cpu = output_hpu.to(cpu)
    htcore.mark_step()
    numpy.testing.assert_allclose(output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001)

    with torch.no_grad():
        output_repeat_hpu = model_hpu(input_tensor_hpu)

    output_repeat_hpu_cpu = output_repeat_hpu.to(cpu)
    numpy.testing.assert_allclose(
        output_repeat_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001
    )
    htcore.hpu_teardown_inference_env()


@pytest.mark.parametrize("set_env_variable", [False], indirect=True)
def test_same_param_two_models(set_env_variable):
    random_weights = torch.rand(32, 3, 3, 3)
    # First convolutional layer with 16 filters and a different bias
    conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    conv1.weight = nn.Parameter(random_weights)
    conv1.bias = nn.Parameter(torch.rand(32))

    import random

    random.seed(65986)
    # Second convolutional layer with 32 filters and a different bias
    conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
    conv2.weight = conv1.weight
    conv2.bias = nn.Parameter(torch.rand(32))

    input = torch.rand(1, 3, 64, 64)

    output1 = conv1(input)
    output2 = conv2(input)

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()

    conv1_hpu = conv1.to(hpu)
    conv2_hpu = conv2.to(hpu)
    from habana_frameworks.torch.core.quantization import _check_params_as_const

    htcore.hpu_inference_initialize(conv1_hpu)
    _check_params_as_const(conv1_hpu)
    htcore.hpu_inference_initialize(conv2_hpu)
    _check_params_as_const(conv2_hpu)

    input_hpu = input.to(hpu)

    with torch.no_grad():
        output1_hpu = conv1_hpu(input_hpu)

    output1_hpu_cpu = output1_hpu.to(cpu)
    numpy.testing.assert_allclose(output1_hpu_cpu.detach().numpy(), output1.detach().numpy(), atol=0.001, rtol=0.001)

    with torch.no_grad():
        output2_hpu = conv2_hpu(input_hpu)

    output2_hpu_cpu = output2_hpu.to(cpu)
    numpy.testing.assert_allclose(output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.001, rtol=0.001)
    htcore.hpu_teardown_inference_env()
