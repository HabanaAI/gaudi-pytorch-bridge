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
from test_utils import compare_tensors, format_tc, hpu


@pytest.mark.parametrize("shape", [[10, 20]], ids=format_tc)
@pytest.mark.parametrize("is_inplace", [True, False])
def test_hpu_masked_scatter(shape, is_inplace):
    def fn(self, mask, source, is_inplace):
        return self.masked_scatter_(mask, source) if is_inplace else self.masked_scatter(mask, source)

    cpu_tensor = torch.randn(shape)
    mask = torch.randn(shape[1]) < 0
    cpu_source = torch.randn(shape)

    cpu_args = [cpu_tensor, mask, cpu_source, is_inplace]
    hpu_args = [cpu_tensor.to(hpu), mask.to(hpu), cpu_source.to(hpu), is_inplace]
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_result = fn(*cpu_args)
    hpu_result = hpu_wrapped_fn(*hpu_args)

    compare_tensors([hpu_result], [cpu_result], atol=0, rtol=0)
