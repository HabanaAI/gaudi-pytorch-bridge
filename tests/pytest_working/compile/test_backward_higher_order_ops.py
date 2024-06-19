###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
# test ported from pytorch repo:
# https://github.com/pytorch/pytorch/blob/main/test/dynamo/test_backward_higher_order_ops.py

import habana_frameworks.torch
import pytest
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm


def _multiply(x):
    return x * x


def _multiply_invoke(grad):
    return trace_wrapped(grad, fn=_multiply)


class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):

    def test_invoke_in_pt2_compiled_autograd(self):
        graph = None
        device = "hpu"
        backend = "hpu_backend"

        def compiler_fn(gm):
            nonlocal graph
            self.assertEqual(graph, None)
            graph = gm
            return torch.compile(gm, backend=backend, fullgraph=True, dynamic=True, options={"inference": False})

        torch._dynamo.reset()
        x = torch.tensor([0.5, 0.5], device=device, requires_grad=True)
        y = torch.tensor([0.5, 0.5], device=device, requires_grad=True)

        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x + y

        fn = torch._dynamo.optimize(backend)(fn)
        out = fn(x, y)
        grad_out = torch.tensor([2.0, 2.0], device=device)
        with compiled_autograd.enable(compiler_fn):
            out.backward(grad_out)
        actual = normalize_gm(graph.print_readable(False))
        self.assertEqual(x.grad, grad_out * grad_out)
        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, s0 : torch.SymInt, L_inputs_0_ : torch.Tensor):
        getitem = L_inputs_0_

        new_grad = torch.clone(getitem)

        call_hook = getitem * getitem;  getitem = None

        new_grad_1 = torch.clone(call_hook);  call_hook = None
        return (new_grad, new_grad_1)
"""
        # self.assertExpectedInline(actual, expected)

        graph = None
