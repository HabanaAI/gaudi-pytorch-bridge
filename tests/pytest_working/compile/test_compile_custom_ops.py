###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
from habana_frameworks.torch.dynamo._custom_op_registrations import (
    register_post_ops,
    register_prepare_ops,
)
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs

OP_STATE = 0


def test_reorder_custom_ops():
    global OP_STATE
    OP_STATE = 0

    def pre_op(state: torch.Tensor, num: int) -> None:
        global OP_STATE
        OP_STATE += num

    def _fake_pre_op(state, num) -> None:
        return

    def post_op(state: torch.Tensor, num: int) -> None:
        global OP_STATE
        OP_STATE -= num

    def _fake_post_op(state, num) -> None:
        return

    register_prepare_ops(pre_op, _fake_pre_op, "pre_op", "hpu")
    register_post_ops(post_op, _fake_post_op, "post_op", "hpu")

    class CustomFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input: torch.Tensor, pre_num: int, post_num: int) -> torch.Tensor:
            ctx.pre_num = pre_num
            ctx.post_num = post_num
            out = input * 2.0
            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            torch.ops.hpu_prepare_ops.pre_op(grad_output, ctx.pre_num)
            out = grad_output * 2.0
            torch.ops.hpu_post_ops.post_op(out, ctx.post_num)
            return out

    def mymodel(t):
        out = CustomFun().apply(t, 10, 5)
        out = CustomFun().apply(out, 20, 10)
        return out

    t = torch.tensor([-1.0, 2.0, -4.0], dtype=torch.float, device="hpu", requires_grad=True)

    clear_t_compile_logs()
    torch._dynamo.reset()

    # Compile the model with fullgraph=True to ensure that the custom ops are traced in the fx graph
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)

    fwd_result = mycompiledmodel(t)
    loss = fwd_result.sum()
    loss.backward()

    assert OP_STATE == 15
    check_ops_executed_in_jit_ir({"mul"})
