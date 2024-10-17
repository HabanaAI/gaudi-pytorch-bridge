###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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


def gn(x, y):
    add = torch.add(x, y)
    unsqueeze = torch.ops.aten.unsqueeze_(add, 0)
    transpose = torch.ops.aten.as_strided_(unsqueeze, (3, 2, 1, 1), (1, 3, 3, 6), 0)
    squeeze = torch.ops.aten.squeeze_.dims(transpose, [-1])
    return squeeze


torch.manual_seed(1234)


def test_stride_view_chain():
    # CPU
    cpu_input2 = torch.randn((2, 1, 3), dtype=torch.float32)
    cpu_input1 = torch.randn((2, 1, 3), dtype=torch.float32)  # (128, 1024)

    cpu_result = gn(cpu_input1, cpu_input2)
    # HPU
    input_hpu1 = cpu_input1.to("hpu")
    input_hpu2 = cpu_input2.to("hpu")
    hpu_result = gn(input_hpu1, input_hpu2)
    assert torch.equal(cpu_result, hpu_result.cpu())


if __name__ == "__main__":
    test_stride_view_chain()
