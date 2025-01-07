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
import torch
from habana_frameworks.torch import _hpex_C


class CustomSoftmax(torch.autograd.Function):
    """Optimized approximate softmax implementation. Limited to bfloat16 only."""

    @staticmethod
    def forward(ctx, inp, flavor):
        """Performs forward pass

        Parameters
        ----------
        inp : Tensor
            Input tensor.

        flavor : int
            Flavor of computation.
            0 (default) - using default exponent function.
            1 - use non-LUT approximation which has better performance for large tensors at cost of reduced accuracy.
        """
        softmax_result = torch.ops.hpu.custom_softmax(inp, flavor)
        ctx.save_for_backward(softmax_result)
        return softmax_result

    @staticmethod
    def backward(ctx, grad_output):
        (softmax_result,) = ctx.saved_tensors
        grad_input = torch._softmax_backward_data(grad_output, softmax_result, softmax_result.dim() - 1, torch.bfloat16)
        return grad_input, None
