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

import habana_frameworks.torch._core_C as htcore
import habana_frameworks.torch.hpu as hpu
import pytest
import torch

torch.manual_seed(2)


def test_graph():
    input = [(2, 3, 4, 4), (2, 3, 6, 6), (2, 3, 8, 8), (2, 3, 10, 10), (2, 3, 2, 2)]

    def raw_function(input_tensor):
        out1 = torch.mul(input_tensor, 2)
        out2 = torch.add(input_tensor, out1)
        return out2

    for s in input:
        t = torch.randn(s, requires_grad=False)
        t_hpu = t.to("hpu")
        result = raw_function(t_hpu)
        htcore._mark_step()
        print(result.to("cpu"))


test_graph()
