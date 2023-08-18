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

import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

def test_op_addr():
    input_shapes = [
        (6, 6),
        (8, 8),
        (10, 10),
    ]

    def raw_function(t1, t2, t3):
        out = torch.addr(t1, t2, t3)
        return out

    compiled_fn = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        v = torch.randn(s[1])
        t = torch.randn(s)
        result = raw_function(t, v, v)

        # HPU
        v_h = v.to("hpu")
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h, v_h, v_h);
        assert torch.allclose(h_result.to("cpu"), result, atol = 0.001, rtol = 0.001)
