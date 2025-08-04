###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import habana_frameworks.torch.internal.bridge_config as bc
import torch


def test_eager_cache_input_offsets():
    # create a rare hash collision of input offsets
    # 8192 and 8192 -> hash: 175248275422
    # 0 and 477184 -> hash: 175248275422

    offset_1 = 8192 // 2
    offset_2 = 477184 // 2
    block = 32

    def m(x):
        # slice the first part
        x1 = x[0:block]
        # slice the second part
        x2 = x[offset_1 : offset_1 + block]

        y = torch.empty(offset_2 + block, dtype=x.dtype, device=x.device)
        y1 = y[offset_2 : offset_2 + block]
        y2 = y[offset_1 : offset_1 + block]

        y1.copy_(x1)
        y2.copy_(x2)

        return y1 + y2

    input = torch.rand(offset_1 + block, dtype=torch.bfloat16)
    input_hpu = input.to("hpu")

    # Run with eager cache enabled
    with bc.env_setting("PT_HPU_ENABLE_EAGER_CACHE", True):
        assert bc.get_pt_hpu_enable_eager_cache() is True

        result_cpu = m(input)
        result_hpu = m(input_hpu)

        result_hpu = result_hpu.to("cpu")
        assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)
