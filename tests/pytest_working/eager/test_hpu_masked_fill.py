###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
from test_utils import format_tc


@pytest.mark.parametrize("self_dtype", [torch.bfloat16, torch.float, torch.int, torch.long], ids=format_tc)
@pytest.mark.parametrize("scalar_type", [torch.float, torch.int], ids=format_tc)
def test_hpu_masked_fill_scalar_mixed_type(self_dtype, scalar_type):
    shape = (2, 3)
    cpu_self = torch.randint(low=-100, high=100, size=shape).to(self_dtype)
    hpu_self = cpu_self.to("hpu")
    cpu_mask = torch.ones(shape, dtype=torch.bool)
    hpu_mask = cpu_mask.to("hpu")
    fill_value = 1 if scalar_type == torch.int else 1.5
    cpu_results = cpu_self.masked_fill(cpu_mask, fill_value)
    hpu_results = hpu_self.masked_fill(hpu_mask, fill_value)
    assert torch.equal(cpu_results, hpu_results.cpu())
