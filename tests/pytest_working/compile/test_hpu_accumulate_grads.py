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
import pytest
import torch
from test_dynamo_utils import use_eager_fallback
from test_utils import clear_t_compile_logs, compare_tensors, is_torch_at_least
from torch._dynamo import compiled_autograd


def compiler_fn(gm):
    return torch.compile(gm, backend="hpu_backend", fullgraph=True)


def is_op_found_in_ir(ns, op):
    from habana_frameworks.torch.dynamo.compile_backend.passes import logger as graph_logger

    target = f"call_function[target=torch.ops.{ns}.{op}]"

    for log in graph_logger.data:
        if log.startswith("[PT_COMPILE] IR:"):
            for line in log.split("\n"):
                if target in line:
                    return True
    return False


class ExampleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the weight at 1
        self.weight = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def forward(self, x):
        return self.weight * x


@pytest.mark.skip(reason="https://gerrit.habana-labs.com/#/c/399842 removes accumulate_grad")
def test_accumulate_grad():
    with use_eager_fallback():
        clear_t_compile_logs()
        model_cpu = ExampleLinear()
        model_hpu = ExampleLinear().to("hpu")

        optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        optimizer_hpu = torch.optim.SGD(model_hpu.parameters(), lr=0.01)

        def calculate_loss(x, model) -> torch.Tensor:
            y = 2.0 * x
            y_hat = model(x)
            loss = (y - y_hat) ** 2
            return loss.mean()

        # With multiple batches of size 1
        batches_cpu = [torch.tensor([4.0]), torch.tensor([2.0])]
        batches_hpu = [batches_cpu[0].to("hpu"), batches_cpu[1].to("hpu")]

        optimizer_cpu.zero_grad()
        optimizer_hpu.zero_grad()
        for batch_cpu, batch_hpu in zip(batches_cpu, batches_hpu):
            # The loss needs to be scaled, because the mean should be taken across the whole
            # dataset, which requires the loss to be divided by the number of batches.
            loss_cpu = calculate_loss(batch_cpu, model_cpu) / len(batch_cpu)
            loss_cpu.backward()
            loss_hpu = calculate_loss(batch_hpu, model_hpu) / len(batch_hpu)
            with compiled_autograd.enable(compiler_fn):
                loss_hpu.backward()

            compare_tensors(model_hpu.weight.grad, model_cpu.weight.grad, atol=0, rtol=0)
            compare_tensors(model_hpu.weight, model_cpu.weight, atol=0, rtol=0)

        # Updating the model only after all batches
        optimizer_cpu.step()
        optimizer_hpu.step()
        compare_tensors(model_hpu.weight.grad, model_cpu.weight.grad, atol=0, rtol=0)
        compare_tensors(model_hpu.weight, model_cpu.weight, atol=0, rtol=0)

        if is_torch_at_least("2.2.0a0"):
            assert is_op_found_in_ir("hpu", "accumulate_grads_")
