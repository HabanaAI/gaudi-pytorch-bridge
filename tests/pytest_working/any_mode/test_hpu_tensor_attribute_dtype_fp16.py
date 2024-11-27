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


import habana_frameworks.torch.hpu
import pytest
import torch
from test_utils import format_tc, is_gaudi1


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
def test_hpu():
    tmp_cpu = torch.nn.Linear(12, 12, dtype=torch.float16)
    print(f"{tmp_cpu.weight.dtype=} | {tmp_cpu.weight.requires_grad=}")
    tmp_hpu = torch.nn.Linear(12, 12, dtype=torch.float16, device="hpu")
    print(f"{tmp_hpu.weight.dtype=} | {tmp_hpu.weight.requires_grad=}")
    assert tmp_cpu.weight.requires_grad == tmp_hpu.weight.requires_grad
