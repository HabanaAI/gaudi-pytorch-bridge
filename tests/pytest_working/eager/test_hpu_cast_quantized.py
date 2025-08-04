###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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

import torch


def test_float_to_bfloat16_quantized():
    x = torch.randn([5, 3], dtype=torch.float32)
    out = torch.ops.quantization._FloatToBfloat16Quantized(x.to("hpu"))
    out_cpu = torch.ops.quantization._FloatToBfloat16Quantized(x)
    assert torch.allclose(out.to("cpu"), out_cpu)


def test_bfloat16_quantized_to_float():
    x = torch.randn([5, 3], dtype=torch.float16)
    out = torch.ops.quantization._Bfloat16QuantizedToFloat(x.to("hpu"))
    out_cpu = torch.ops.quantization._Bfloat16QuantizedToFloat(x)
    assert torch.allclose(out.to("cpu"), out_cpu)
