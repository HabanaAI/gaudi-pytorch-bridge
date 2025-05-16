###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from habana_frameworks.torch.dynamo.compile_backend.random_utils import (
    HABANA_RANDOM_OPS,
)
from habana_frameworks.torch.dynamo.compile_backend.shared_layer import (
    hpu_fallback_op_list,
)
from test_dynamo_utils import use_eager_fallback
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    compile_function_if_compile_mode,
)


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


def _fused_dropout(x):
    a, b = torch._fused_dropout(input=x, p=0.4)
    return a * x + b


def exponential(x):
    a = torch.empty_like(x)
    return a.exponential_() * x


def normal(mean, std):
    return torch.normal(mean, std) * mean * std


def three_ops(x):
    res_rand = torch.rand(x.shape, dtype=x.dtype, device=x.device)
    res_bernoulli = torch.bernoulli(res_rand) * x
    res_randperm = torch.randperm(x.shape[-1], dtype=x.dtype, device=x.device)
    return res_bernoulli * res_randperm


OPS = [
    bernoulli,
    poisson,
    rand,
    randn,
    randint,
    multinomial,
    randperm,
    native_dropout,
    exponential,
    _fused_dropout,
]


class Model(torch.nn.Module):
    def __init__(self, op, is_checkpoint):
        super().__init__()
        self.op = op
        self.is_checkpoint = is_checkpoint

    def forward(self, input):
        add = input + 10
        rand = (
            torch.utils.checkpoint.checkpoint(self.op, add, use_reentrant=False) if self.is_checkpoint else self.op(add)
        )
        relu = torch.relu(rand)
        return torch.randint_like(relu, 3, 10, dtype=torch.int) + relu


class ModelTwoInputs(torch.nn.Module):
    def __init__(self, op, is_checkpoint):
        super().__init__()
        self.op = op
        self.is_checkpoint = is_checkpoint

    def forward(self, input_a, input_b):
        add = input_a + 10
        mul = input_b * 1.3
        rand = (
            torch.utils.checkpoint.checkpoint(self.op, add, mul, use_reentrant=False)
            if self.is_checkpoint
            else self.op(add, mul)
        )
        relu = torch.relu(rand)
        return torch.randint_like(relu, 3, 10, dtype=torch.int) + relu


class ModelAllOps(torch.nn.Module):
    def __init__(self, is_checkpoint):
        super().__init__()
        self.is_checkpoint = is_checkpoint

    def maybe_checkpoint(self, op, input):
        return torch.utils.checkpoint.checkpoint(op, input, use_reentrant=False) if self.is_checkpoint else op(input)

    def forward(self, input):
        res1 = self.maybe_checkpoint(poisson, input) + randn(input)
        res2 = res1 + randint(res1)
        res3 = multinomial(input) + res2
        res4 = self.maybe_checkpoint(native_dropout, res3)
        res5 = self.maybe_checkpoint(three_ops, res4)
        res6 = exponential(res5)
        res7 = self.maybe_checkpoint(_fused_dropout, res6)
        return res7


class ModelDropout(torch.nn.Module):
    def __init__(self, is_checkpoint):
        super().__init__()
        self.is_checkpoint = is_checkpoint

    def maybe_checkpoint(self, op, input):
        return torch.utils.checkpoint.checkpoint(op, input, use_reentrant=False) if self.is_checkpoint else op(input)

    def function(self, input):
        input = torch.relu(input)
        input = torch.nn.functional.dropout(input, p=0.5, training=True)
        input = torch.sin(input)
        return input

    def forward(self, input):
        res1 = self.maybe_checkpoint(self.function, input)
        return torch.sin(res1)


def run_model(model, shape=(12, 16)):
    torch.manual_seed(2137)
    input = torch.rand(shape).to("hpu").requires_grad_(True)
    model = compile_function_if_compile_mode(model)
    out = model(input)
    out.sum().backward()
    return out.cpu(), input.grad.cpu()


def run_model_with_deterministic_algorithms(model, shape=(12, 16), deterministic_flag=True):
    torch.use_deterministic_algorithms(deterministic_flag)
    out, grad = run_model(model, shape)
    torch.use_deterministic_algorithms(False)
    return out, grad


def run_model_two_inputs(model, input_a, input_b):
    model = compile_function_if_compile_mode(model)
    out = model(input_a, input_b)
    out.sum().backward()
    return (
        out.cpu(),
        input_a.grad.cpu() if isinstance(input_a, torch.Tensor) else None,
        input_b.grad.cpu() if isinstance(input_b, torch.Tensor) else None,
    )


def run_model_with_deterministic_algorithms_two_inputs(model, input_a, input_b, deterministic_flag=True):
    torch.use_deterministic_algorithms(deterministic_flag)
    out, grad_a, grad_b = run_model_two_inputs(model, input_a, input_b)
    torch.use_deterministic_algorithms(False)
    return out, grad_a, grad_b


def get_habana_op_names(op_names):
    return {f"habana_{op_name.__name__}" for op_name in op_names}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("disable_compile", [False, True])
def test_checkpoint(op, disable_compile):
    torch._dynamo.reset()
    clear_t_compile_logs()

    habana_op_names = get_habana_op_names([op])

    if disable_compile:
        original_op = op
        op = torch.compiler.disable(op)
        checkpoint_ops = {"habana_randint"}
    else:
        checkpoint_ops = habana_op_names.union({"habana_randint"})

    out, grad = run_model_with_deterministic_algorithms(Model(op, False), deterministic_flag=not disable_compile)

    check_ops_executed_in_jit_ir(checkpoint_ops)
    clear_t_compile_logs()

    out_checkpoint, grad_checkpoint = run_model(Model(op, True))
    check_ops_executed_in_jit_ir(checkpoint_ops)

    if disable_compile:
        op = original_op

    compare_tensors(out, out_checkpoint)
    compare_tensors(grad, grad_checkpoint)


def test_checkpoint_dropout():
    torch._dynamo.reset()
    clear_t_compile_logs()

    op = native_dropout
    habana_op_names = get_habana_op_names([op])

    out, grad = run_model_with_deterministic_algorithms(ModelDropout(False), (120, 160))
    check_ops_executed_in_jit_ir(habana_op_names)
    torch._dynamo.reset()
    clear_t_compile_logs()

    out_checkpoint, grad_checkpoint = run_model(ModelDropout(True), (120, 160))
    check_ops_executed_in_jit_ir(habana_op_names)

    compare_tensors(out, out_checkpoint)
    compare_tensors(grad, grad_checkpoint)


@pytest.mark.parametrize("is_mean_tensor, is_std_tensor", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("disable_compile", [False, True])
def test_checkpoint_normal(is_mean_tensor, is_std_tensor, disable_compile):
    torch._dynamo.reset()
    clear_t_compile_logs()
    op = normal
    shape = (12, 16)

    habana_op_names = {"habana_normal"}

    if disable_compile:
        original_op = op
        op = torch.compiler.disable(op)
        checkpoint_ops = {"habana_randint"}
    else:
        checkpoint_ops = habana_op_names.union({"habana_randint"})

    torch.manual_seed(2137)
    mean = torch.rand(shape).to("hpu").requires_grad_(True) if is_mean_tensor else 2.5
    std = torch.rand(shape).to("hpu").requires_grad_(True) if is_std_tensor else 1.5

    out, grad_a, grad_b = run_model_with_deterministic_algorithms_two_inputs(
        ModelTwoInputs(op, False), mean, std, deterministic_flag=not disable_compile
    )

    check_ops_executed_in_jit_ir(checkpoint_ops)
    clear_t_compile_logs()

    torch.manual_seed(2137)
    mean = torch.rand(shape).to("hpu").requires_grad_(True) if is_mean_tensor else 2.5
    std = torch.rand(shape).to("hpu").requires_grad_(True) if is_std_tensor else 1.5

    out_checkpoint, grad_a_checkpoint, grad_b_checkpoint = run_model_two_inputs(ModelTwoInputs(op, True), mean, std)
    check_ops_executed_in_jit_ir(checkpoint_ops)

    if disable_compile:
        op = original_op

    compare_tensors(out, out_checkpoint)
    if is_mean_tensor:
        compare_tensors(grad_a, grad_a_checkpoint)
    if is_std_tensor:
        compare_tensors(grad_b, grad_b_checkpoint)


def test_checkpoint_all_ops():
    torch._dynamo.reset()
    clear_t_compile_logs()

    habana_op_names = get_habana_op_names(OPS)

    out, grad = run_model_with_deterministic_algorithms(ModelAllOps(False))

    check_ops_executed_in_jit_ir(habana_op_names)
    clear_t_compile_logs()

    out_checkpoint, grad_checkpoint = run_model(ModelAllOps(True))
    check_ops_executed_in_jit_ir(habana_op_names)

    compare_tensors(out, out_checkpoint)
    compare_tensors(grad, grad_checkpoint)


CHECKPOINT_OPS = [bernoulli, native_dropout, poisson, rand, randperm]


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

    habana_op_names = get_habana_op_names(compile_ops)

    checkpoint_op_bckp = HABANA_RANDOM_OPS.pop(aten_op)
    hpu_fallback_op_list.add(eager_op_name)

    with use_eager_fallback():
        out, grad = run_model_with_deterministic_algorithms(ModelAllOps(False))
        check_ops_executed_in_jit_ir(habana_op_names, allowed_fallbacks={eager_op_name})
        clear_t_compile_logs()

        out_checkpoint, grad_checkpoint = run_model(ModelAllOps(True))
        check_ops_executed_in_jit_ir(
            habana_op_names, allowed_fallbacks={"run_with_rng_state", "run_and_save_rng_state"}
        )

    HABANA_RANDOM_OPS[aten_op] = checkpoint_op_bckp
    hpu_fallback_op_list.remove(eager_op_name)

    compare_tensors(out, out_checkpoint)
    compare_tensors(grad, grad_checkpoint)
