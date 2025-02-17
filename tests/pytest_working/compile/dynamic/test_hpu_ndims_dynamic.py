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

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import is_gaudi3, setup_teardown_env_fixture


@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
class TestHpuNdimsDynamic:
    @staticmethod
    def test_ndims_dynamic(setup_teardown_env_fixture):
        if is_gaudi3():
            pytest.skip("DSD not supported on G3")

        def fn(input, shape):
            view = torch.ops.aten.view(input, shape)
            sum = torch.ops.aten.sum(view, [0, 2, 4, 6])
            return sum

        shapes = [(2, 9, 16, 8 * (2**i)) for i in range(0, 4)]
        view_shapes = [(2, 1, 3, 3, 4, 4, 4, 2 * (2**i)) for i in range(0, 4)]
        cpu_input = [torch.rand(shape, dtype=torch.float32) for shape in shapes]
        hpu_input = [input.to("hpu") for input in cpu_input]

        torch._dynamo.reset()

        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")

        cpu_output = []
        hpu_output = []
        for idx, shape in enumerate(view_shapes):
            cpu_output.append(fn(cpu_input[idx], shape))
        for idx, shape in enumerate(view_shapes):
            hpu_output.append(hpu_wrapped_fn(hpu_input[idx], shape).cpu())

        for i in range(len(cpu_output)):
            torch.allclose(cpu_output[i], hpu_output[i])
