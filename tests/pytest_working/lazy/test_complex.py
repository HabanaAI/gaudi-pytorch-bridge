###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
