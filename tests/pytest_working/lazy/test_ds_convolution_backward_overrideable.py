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

import os

import pytest
import torch
from test_utils import setup_teardown_env_fixture

Verbose = False


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
@pytest.mark.parametrize("output_mask_v", range(0, 8))
def test_hpu_lazy_dynamic_shape(output_mask_v, setup_teardown_env_fixture):
    output_mask = [bool(output_mask_v & (1 << i)) for i in range(3)]
    if Verbose:
        print(f"{output_mask_v = }, {output_mask = }")

    for N, C, H, W, C2 in [
        [64, 128, 5, 5, 63],
        [64, 64, 10, 10, 63],
        [64, 32, 20, 20, 63],
    ]:
        grad_output = torch.rand([N, C2, H, W]).to("hpu")
        input = torch.rand([N, C, H, W]).to("hpu")
        weight = torch.rand([C2, C, 1, 1]).to("hpu")

        if Verbose:
            print(f"{grad_output.size() = }")
            print(f"{input.size() = }")
            print(f"{weight.size() = }")

        # convolution_backward_overrideable is not implemented on CPU
        # just check if it works without validating the results
        result = torch.ops.aten.convolution_backward_overrideable(
            grad_output,
            input,
            weight,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=output_mask,
        )

        for r in result:
            r_cpu = r.cpu()
            if Verbose:
                print(f"{r_cpu.size() = }")

        assert os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] == "1"
