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


@pytest.mark.parametrize(
    "dtype",
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_complex(dtype):
    part_dtype, result_dtype = dtype
    device = torch.device("hpu")
    real = torch.tensor([1, 2], dtype=part_dtype)
    imag = torch.tensor([3, 4], dtype=part_dtype)
    z = 2 * torch.complex(real, imag) + 1.0
    real_h = real.to(device)
    imag_h = imag.to(device)
    z_h = torch.complex(real_h, imag_h)
    z_h = 2 * z_h + 1.0
    assert str(z_h.device) == "cpu"
    assert torch.allclose(z, z_h, atol=0.001, rtol=0.001)
    assert z_h.dtype == result_dtype
