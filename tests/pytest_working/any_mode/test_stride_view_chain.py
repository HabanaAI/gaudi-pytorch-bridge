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


def gn(x, y):
    add = torch.add(x, y)
    unsqueeze = torch.ops.aten.unsqueeze_(add, 0)
    transpose = torch.ops.aten.as_strided_(unsqueeze, (3, 2, 1, 1), (1, 3, 3, 6), 0)
    squeeze = torch.ops.aten.squeeze_.dims(transpose, [-1])
    return squeeze


torch.manual_seed(1234)


def test_stride_view_chain():
    # CPU
    cpu_input2 = torch.randn((2, 1, 3), dtype=torch.float32)
    cpu_input1 = torch.randn((2, 1, 3), dtype=torch.float32)  # (128, 1024)

    cpu_result = gn(cpu_input1, cpu_input2)
    # HPU
    input_hpu1 = cpu_input1.to("hpu")
    input_hpu2 = cpu_input2.to("hpu")
    hpu_result = gn(input_hpu1, input_hpu2)
    assert torch.equal(cpu_result, hpu_result.cpu())


if __name__ == "__main__":
    test_stride_view_chain()
