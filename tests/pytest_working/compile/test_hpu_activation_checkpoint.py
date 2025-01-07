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

import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.random_utils import HABANA_CHECKPOINT_OPS
from habana_frameworks.torch.dynamo.compile_backend.shared_layer import hpu_fallback_op_list
from test_dynamo_utils import use_eager_fallback
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs


def bernoulli(x):
    return torch.bernoulli(x) * x


def poisson(x):
    return torch.poisson(x) * x


def rand(x):
    return torch.rand_like(x) * x


def randn(x):
    return torch.randn_like(x) * x


def randint(x):
    return torch.randint_like(x, 2, 5) * x


def multinomial(x):
    return torch.multinomial(x, x.shape[-1], False) * x


def randperm(x):
    return torch.randperm(x.shape[-1], device=x.device) * x


def native_dropout(x):
    a, b = torch.native_dropout(input=x, p=0.7, train=True)
    return a * x + b


def three_ops(x):
    res_rand = torch.rand(x.shape, dtype=x.dtype, device=x.device)
    res_bernoulli = torch.bernoulli(res_rand) * x
    res_randperm = torch.randperm(x.shape[-1], dtype=x.dtype, device=x.device)
    return res_bernoulli * res_randperm


OPS = [bernoulli, poisson, rand, randn, randint, multinomial, randperm, native_dropout]


class TestModel(torch.nn.Module):
    def __init__(self, op, is_checkpoint):
        super().__init__()
        self.op = op
        self.is_checkpoint = is_checkpoint

    def forward(self, input):
        add = input + 10
        rand = torch.utils.checkpoint.checkpoint(self.op, add) if self.is_checkpoint else self.op(add)
        relu = torch.relu(rand)
        return torch.randint_like(relu, 3, 10, dtype=torch.int) + relu


class TestModelAllOps(torch.nn.Module):
    def __init__(self, is_checkpoint):
        super().__init__()
        self.is_checkpoint = is_checkpoint

    def maybe_checkpoint(self, op, input):
        return torch.utils.checkpoint.checkpoint(op, input) if self.is_checkpoint else op(input)

    def forward(self, input):
        res1 = self.maybe_checkpoint(poisson, input) + randn(input)
        res2 = res1 + randint(res1)
        res3 = multinomial(input) + res2
        res4 = self.maybe_checkpoint(native_dropout, res3)
        res5 = self.maybe_checkpoint(three_ops, res4)
        return res5


def run_model_common(model):
    torch.manual_seed(2137)
    input = torch.rand((12, 16)).to("hpu").requires_grad_(True)
    model = torch.compile(model, backend="hpu_backend")
    out = model(input)
    out.sum().backward()
    return out.cpu(), input.grad.cpu()


def run_model_small(op, is_checkpoint):
    return run_model_common(TestModel(op, is_checkpoint))


def run_model_all_ops(is_checkpoint):
    return run_model_common(TestModelAllOps(is_checkpoint))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("disable_compile", [False, True])
def test_checkpoint(op, disable_compile):
    torch._dynamo.reset()
    clear_t_compile_logs()

    op_name = op.__name__
    habana_op_name = f"habana_{op_name}"
    checkpoint_op_name = f"{habana_op_name}_checkpoint"

    forbidden_ops = set()
    checkpoint_ops = set()

    if disable_compile:
        original_op = op
        op = torch.compiler.disable(op)
        no_checkpoint_ops = {"habana_randint"}
        forbidden_ops = {checkpoint_op_name}
    else:
        no_checkpoint_ops = {habana_op_name, "habana_randint"}
        checkpoint_ops = {habana_op_name, checkpoint_op_name}

    out, grad = run_model_small(op, False)
    check_ops_executed_in_jit_ir(no_checkpoint_ops)
    clear_t_compile_logs()

    out_checkpoint, grad_checkpoint = run_model_small(op, True)
    check_ops_executed_in_jit_ir(checkpoint_ops, forbidden_ops=forbidden_ops)

    if disable_compile:
        op = original_op

    assert torch.equal(out, out_checkpoint)
    assert torch.equal(grad, grad_checkpoint)


CHECKPOINT_OPS = [bernoulli, native_dropout, poisson, rand, randperm]


def test_checkpoint_all_ops():
    torch._dynamo.reset()
    clear_t_compile_logs()

    habana_op_names = {f"habana_{op_name.__name__}" for op_name in OPS}
    checkpoint_op_names = {f"habana_{op_name.__name__}_checkpoint" for op_name in CHECKPOINT_OPS}

    out, grad = run_model_all_ops(False)
    check_ops_executed_in_jit_ir(habana_op_names)
    clear_t_compile_logs()

    out_checkpoint, grad_checkpoint = run_model_all_ops(True)
    check_ops_executed_in_jit_ir(checkpoint_op_names)

    assert torch.equal(out, out_checkpoint)
    assert torch.equal(grad, grad_checkpoint)


@pytest.mark.parametrize("eager_op", [poisson, bernoulli])
def test_checkpoint_all_eager_fallback(eager_op):
    torch._dynamo.reset()
    clear_t_compile_logs()

    compile_ops = OPS.copy()
    compile_ops.remove(eager_op)

    checkpoint_ops = CHECKPOINT_OPS.copy()
    checkpoint_ops.remove(eager_op)

    eager_op_name = eager_op.__name__
    aten_op = f"aten.{eager_op_name}.default"

    habana_op_names = {f"habana_{op_name.__name__}" for op_name in compile_ops}
    checkpoint_op_names = {f"habana_{op_name.__name__}_checkpoint" for op_name in checkpoint_ops}

    checkpoint_op_bckp = HABANA_CHECKPOINT_OPS.pop(aten_op)
    hpu_fallback_op_list.add(eager_op_name)

    with use_eager_fallback():
        out, grad = run_model_all_ops(False)
        check_ops_executed_in_jit_ir(habana_op_names, allowed_fallbacks={eager_op_name})
        clear_t_compile_logs()

        out_checkpoint, grad_checkpoint = run_model_all_ops(True)
        check_ops_executed_in_jit_ir(
            checkpoint_op_names, allowed_fallbacks={"run_with_rng_state", "run_and_save_rng_state"}
        )

    HABANA_CHECKPOINT_OPS[aten_op] = checkpoint_op_bckp
    hpu_fallback_op_list.remove(eager_op_name)

    assert torch.equal(out, out_checkpoint)
    assert torch.equal(grad, grad_checkpoint)
