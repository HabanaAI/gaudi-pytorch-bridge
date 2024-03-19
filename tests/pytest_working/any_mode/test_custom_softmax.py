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

import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.kernels import CustomSoftmax
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, hpu, is_pytest_mode_compile


def test_custom_softmax():
    input = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [1000, 1000, 0.0, 1004.0], [-9984, -9984, -9984, -1000]],
        dtype=torch.bfloat16,
    )
    ref_output = torch.tensor(
        [
            [0.0320, 0.0869, 0.2373, 0.6445],
            [0.0177, 0.0177, 0.0000, 0.9648],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
    )

    op = CustomSoftmax.apply
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        op = torch.compile(CustomSoftmax.apply, backend="hpu_backend")

    out = op(torch.clone(input).detach().to(hpu), 0)
    out_cpu = out.cpu().to(torch.float32)

    assert np.allclose(out_cpu.numpy(), ref_output.numpy(), atol=1e-03)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("custom_softmax")


if __name__ == "__main__":
    test_custom_softmax()
