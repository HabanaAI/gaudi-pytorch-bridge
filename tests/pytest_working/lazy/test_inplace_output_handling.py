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

import habana_frameworks.torch.core as htcore
import numpy
import pytest
import torch
import torch.nn as nn

# Run test on HPU
hpu = torch.device("hpu")
cpu = torch.device("cpu")


def test_inplace():
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.linear1 = nn.Linear(30, 20)
            self.eval()

        def forward(self, x):
            x = self.relu(x)
            htcore.mark_step()
            x = self.relu(x)
            x = self.conv(x)  # Assuming the input is 1D, unsqueeze to make it 2D
            x = self.linear1(x.view(x.size(0), -1))  # Flatten the output before linear layer
            x = self.relu(x)
            return x

    # Dummy input for testing
    input = torch.randn(1, 32, 1, 10)
    input2 = torch.randn(1, 32, 1, 10)

    model = CustomModel()
    model.eval()
    htcore.hpu_set_inference_env()

    input_hpu = input.to(hpu)
    input2_hpu = input2.to(hpu)
    model_hpu = model.to(hpu)
    model_hpu.eval()

    htcore.hpu_inference_initialize(model_hpu)

    # Apply the convolutional layers to the input tensor
    with torch.no_grad():
        output_hpu = model_hpu(input_hpu)
        output_hpu_cpu = output_hpu.to(cpu)

    with torch.no_grad():
        output2_hpu = model_hpu(input2_hpu)
        output2_hpu_cpu = output2_hpu.to(cpu)

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    hpugraph_module = wrap_in_hpu_graph(model_hpu, disable_tensor_cache=True)

    print("Wrap in HPUGraph completed, now will run the graphs")
    with torch.no_grad():
        output_hpugraph = hpugraph_module(input_hpu)

    output_hpugraph_cpu = output_hpugraph.to(cpu)
    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output_hpugraph_cpu.detach().numpy(), atol=0.001, rtol=0.001
    )

    with torch.no_grad():
        output2_hpugraph = hpugraph_module(input2_hpu)
    output2_hpugraph_cpu = output2_hpugraph.to(cpu)
    numpy.testing.assert_allclose(
        output2_hpu_cpu.detach().numpy(), output2_hpugraph_cpu.detach().numpy(), atol=0.001, rtol=0.001
    )

    htcore.hpu_teardown_inference_env()


def fmt_float(value, c):
    return "{:.2f}{}".format(value, c)  # Formats float to 2 decimal pla


def printStat():
    import habana_frameworks.torch as ht

    ht.core.mark_step()
    ht.hpu.synchronize()
    mem_stats = ht.hpu.memory.memory_stats()
    max_used = fmt_float(mem_stats["MaxInUse"] / 1024.0 / 1024.0 / 1024.0, "G")
    perc_used = fmt_float(100 * mem_stats["MaxInUse"] / mem_stats["Limit"], "%")
    used = fmt_float(mem_stats["InUse"] / 1024.0 / 1024.0 / 1024.0, "G")
    stats = f" max_hpu_mem:{max_used}, used:{used} ({perc_used})"
    separator = "-" * len(stats)

    print(separator)
    print(stats)
    print(separator)


def test_kvcache_inplace():
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
            self.eval()

        def forward(self, x, y):
            y.index_copy_(0, torch.tensor([1]), y[0:1])
            y = self.relu(y)
            z = x + y
            z_n = self.pool(z.unsqueeze(0)).squeeze(0)
            return z_n

    # Dummy input for testing
    # Initialize y with a size corresponding to 5GB (5 * 1024**3 bytes)
    y_size_bytes = 5 * 1024**2
    y_size_elements = y_size_bytes // 4  # Assuming y is of dtype=torch.float32 (4 bytes per element)

    input = torch.randn(y_size_elements)
    input2 = torch.randn(y_size_elements)
    kvcache = torch.randn(y_size_elements)
    kvcache2 = torch.randn(y_size_elements)

    htcore.hpu_set_inference_env()

    model = CustomModel()
    model.eval()

    input_hpu = input.to(hpu)
    input2_hpu = input2.to(hpu)
    kvcache_hpu = kvcache.to(hpu)
    kvcache2_hpu = kvcache2.to(hpu)
    model_hpu = model.to(hpu)
    model_hpu.eval()

    # Apply the convolutional layers to the input tensor
    with torch.no_grad():
        output_hpu = model_hpu(input_hpu, kvcache_hpu)
        output_hpu_cpu = output_hpu.to(cpu)

    with torch.no_grad():
        output2_hpu = model_hpu(input2_hpu, kvcache2_hpu)
        output2_hpu_cpu = output2_hpu.to(cpu)

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    hpugraph_module = wrap_in_hpu_graph(model_hpu, disable_tensor_cache=True)

    print("Wrap in HPUGraph completed, now will run the graphs")
    print("======BEFORE FIRST FORWARD=====")
    printStat()
    with torch.no_grad():
        output_hpugraph = hpugraph_module(input_hpu, kvcache_hpu)
    output_hpugraph_cpu = output_hpugraph.to(cpu)
    print("======AFTER FIRST FORWARD=====")
    printStat()

    hpugraph_module.clear_inputs()
    del kvcache_hpu
    del input_hpu
    print("======AFTER FIRST FORWARD - MOVE KV CACHE=====")
    printStat()

    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output_hpugraph_cpu.detach().numpy(), atol=0.001, rtol=0.001
    )

    with torch.no_grad():
        output2_hpugraph = hpugraph_module(input2_hpu, kvcache2_hpu)
    output2_hpugraph_cpu = output2_hpugraph.to(cpu)
    print("======AFTER SECOND FORWARD=====")
    printStat()

    hpugraph_module.clear_inputs()
    del kvcache2_hpu
    del input2_hpu
    print("======AFTER SECOND FORWARD - MOVE KV CACHE=====")
    printStat()
    numpy.testing.assert_allclose(
        output2_hpu_cpu.detach().numpy(), output2_hpugraph_cpu.detach().numpy(), atol=0.001, rtol=0.001
    )

    htcore.hpu_teardown_inference_env()


def test_input_reuse():
    class InplaceOperationNet(nn.Module):
        def __init__(self):
            super(InplaceOperationNet, self).__init__()
            self.relu = nn.ReLU(inplace=True)
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30)
            self.fc3 = nn.Linear(10, 30)

        def forward(self, x1):
            # x1: input to be modified in-place
            # In-place operation on x1
            x1 = self.relu(x1)
            htcore.mark_step()

            # First layer with modified x1
            out1 = self.fc1(x1)
            htcore.mark_step()

            # Second layer with output of first layer
            out2 = self.fc2(out1)
            htcore.mark_step()

            # Third layer using original x1
            out3 = self.fc3(x1)

            # Combine the outputs
            combined = out2 + out3

            return combined

    model = InplaceOperationNet()
    model.eval()

    x1 = torch.randn(1, 10)  # Random input for x1

    htcore.hpu_set_inference_env()
    x1_hpu = x1.to(hpu)
    model_hpu = model.to(hpu)

    with torch.no_grad():
        output_hpu = model_hpu(x1_hpu)
    output_hpu_cpu = output_hpu.to(cpu)

    htcore.hpu_inference_initialize(model_hpu)
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    hpugraph_module = wrap_in_hpu_graph(model_hpu, disable_tensor_cache=True)

    with torch.no_grad():
        output_hpugraph = hpugraph_module(x1_hpu)
    output_hpugraph_cpu = output_hpugraph.to(cpu)

    with torch.no_grad():
        output1_hpugraph = hpugraph_module(x1_hpu)
    output1_hpugraph_cpu = output1_hpugraph.to(cpu)

    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output1_hpugraph_cpu.detach().numpy(), atol=0.001, rtol=0.001
    )
    htcore.hpu_teardown_inference_env()
