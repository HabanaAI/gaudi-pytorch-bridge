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


import pytest
import torch


@pytest.mark.skip(reason="KeyError: 'torch_dynamo_backends'")
@pytest.mark.parametrize("use_eager_conv", [True, False])
def test_data_layout_prop(use_eager_conv):
    conv_op = torch.nn.Conv2d(16, 33, 3, stride=2)
    if not use_eager_conv:
        conv_op = torch.compile(conv_op, backend="hpu_backend")

    def raw_function(x):
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2).to(device="hpu")
        x = torch.relu(x)
        x = maxpool(x)
        return x

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.randn(20, 16, 50, 100).to(device="hpu")
    conv_op = conv_op.to(device="hpu")

    conv_out = conv_op(tensor)

    conv_out_cpu = conv_out.to(device="cpu")
    conv_out_depermuted = conv_out_cpu.to(device="hpu")

    out_tensor = compiled_function(conv_out)
    out_tensor_depermuted = compiled_function(conv_out_depermuted)

    assert torch.allclose(out_tensor, out_tensor_depermuted)
