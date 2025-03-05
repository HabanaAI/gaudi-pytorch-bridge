###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
# import copy
# import torchvision
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import is_pytest_mode_compile


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_einsum(dtype):
    torch.manual_seed(1234)

    def fn(a, b):
        out = torch.einsum("bhql,blc->bhqc", a, b)
        return out

    # CPU
    cpu_input1 = torch.rand((1, 128, 108, 108), dtype=dtype)
    cpu_input2 = torch.rand((1, 108, 512), dtype=dtype)
    cpu_result = fn(cpu_input1, cpu_input2)

    if is_pytest_mode_compile():
        fn = torch.compile(fn, backend="hpu_backend")

    # HPU
    hpu_input1 = cpu_input1.to("hpu")
    hpu_input2 = cpu_input2.to("hpu")

    with use_eager_fallback():
        hpu_result = fn(hpu_input1, hpu_input2)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(cpu_result, hpu_result.cpu(), atol=tolerance, rtol=tolerance)
