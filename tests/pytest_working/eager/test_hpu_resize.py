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

import numpy as np
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize(
    "shape_in, shape_out", [((2, 3), (4, 6)), ((4, 6), (2, 3)), ((2, 3, 4, 5), (3, 4, 5, 6))], ids=format_tc
)
@pytest.mark.parametrize("blocking_flag", [True, False])
def test_resize_inplace(shape_in, shape_out, blocking_flag):
    num_elements = np.multiply.reduce(shape_in)
    cpu_tensor = torch.Tensor(np.reshape(np.arange(num_elements, dtype=np.int32), shape_in)).type(torch.int32)
    hpu_tensor = cpu_tensor.to("hpu", non_blocking=blocking_flag)
    result_cpu = cpu_tensor.resize_(shape_out).numpy().flatten()[:num_elements]
    result_hpu = hpu_tensor.resize_(shape_out).to("cpu").numpy().flatten()[:num_elements]

    assert np.array_equal(result_hpu, result_cpu)


def test_empty_resize():
    hpu_tensor = torch.empty([], device="hpu")
    hpu_tensor.resize_(10)
    cpu_tensor = hpu_tensor.to("cpu")
    assert np.equal(cpu_tensor.size()[0], 10)


@pytest.mark.parametrize("src_shape, dst_shape", [((2, 4), (5,)), ((2, 3), (2, 2, 2))], ids=format_tc)
def test_output_resize(src_shape, dst_shape):
    src_num_elements = np.multiply.reduce(src_shape)
    dst_num_elements = np.multiply.reduce(dst_shape)
    num_elements = min(src_num_elements, dst_num_elements)

    cpu_tensor = torch.rand(src_shape)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor.resize_(dst_shape)
    torch.ops.aten._resize_output_(hpu_tensor, dst_shape, hpu_tensor.device)

    cpu_result = cpu_tensor.numpy().flatten()[:num_elements]
    hpu_result = hpu_tensor.to("cpu").numpy().flatten()[:num_elements]

    assert hpu_tensor.size() == dst_shape
    assert np.array_equal(hpu_result, cpu_result)
