###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import logging

import torch
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import (
    OptimizationPassPlacement,
    OptimizerContext,
)
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    register_pass_at_optimization_pass,
)

logger: logging.Logger = logging.getLogger(__name__)


def test_register_custom_pass():
    def pass_reorder_custom_ops(ctx: OptimizerContext) -> bool:
        logger.debug("####### pass_reorder_custom_ops")
        return False

    class CustomOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            out = x * 2.0
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def custom_model(x):
        x = CustomOp.apply(x)
        return x

    x = torch.rand(10, requires_grad=True)

    register_pass_at_optimization_pass(pass_reorder_custom_ops, OptimizationPassPlacement.PRE_PLACEMENT)

    model = torch.compile(custom_model, backend="hpu_backend", fullgraph=True)

    fwd_result = model(x)
    loss = fwd_result.sum()
    loss.backward()
