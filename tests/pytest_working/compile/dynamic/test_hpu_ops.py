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

def test_view_negative_dim():
        inputs = [(4, 7, 7, 8), (4, 10, 10, 8)]
        shapes = [(4, -1, 8), (4, -1, 8)]

        def raw_function(input_tensor, shape):
            t = torch.relu(input_tensor)
            out = t.view(shape)
            return out

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s1 , s2 in zip(inputs, shapes):
            t = torch.randn(s1, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h, s2)
            out_c = raw_function(t, s2)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)
