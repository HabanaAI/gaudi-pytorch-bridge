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
from test_utils import compare_tensors, hpu, is_pytest_mode_compile

test_case_list = [
    # N, C, fill_val,
    (2, 10, 2.2),
    (2, 10, 0.0),
]


@pytest.mark.parametrize("N, C, fill_val", test_case_list)
@pytest.mark.parametrize("is_masked", [True, False])
@pytest.mark.parametrize("is_inplace", [True, False])
@pytest.mark.parametrize("fill_with_scalar", [True, False])
def test_hpu_fill(N, C, fill_val, is_masked, is_inplace, fill_with_scalar):
    def fn(is_masked, is_inplace):
        def fill(self, value):
            return torch.fill(self, value)

        def fill_(self, value):
            return self.fill_(value)

        def masked_fill(self, mask, value):
            return self.masked_fill(mask, value)

        def masked_fill_(self, mask, value):
            return self.masked_fill_(mask, value)

        if is_masked:
            return masked_fill_ if is_inplace else masked_fill
        else:
            return fill_ if is_inplace else fill

    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)
    ref_fn = fn(is_masked, is_inplace)
    hpu_fn = fn(is_masked, is_inplace)
    cpu_args = [cpu_tensor]
    hpu_args = [hpu_tensor]

    if is_masked:
        mask = torch.randn(C) < 0
        cpu_args.append(mask)
        hpu_args.append(mask.to(hpu))

    if fill_with_scalar:
        cpu_args.append(fill_val)
        hpu_args.append(fill_val)
    else:
        fill_tensor = torch.tensor(fill_val)
        cpu_args.append(fill_tensor)
        hpu_args.append(fill_tensor.to(hpu))

    expected_result = ref_fn(*cpu_args)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        hpu_fn = torch.compile(hpu_fn, backend="hpu_backend")

    real_result = hpu_fn(*hpu_args)

    compare_tensors([real_result], [expected_result], atol=0, rtol=0)
