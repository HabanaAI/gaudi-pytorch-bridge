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

from functools import partial
from typing import Union

import pytest
import torch
from test_utils import clear_t_compile_logs
from torch.testing._internal.common_utils import TestCase

skip_test = False
try:
    from torch._higher_order_ops.hints_wrap import hints_wrapper
except ModuleNotFoundError:
    skip_test = True


def check_hints_in_jit_ir(op_name: str, expected_hints: Union[list, dict], op_idx=0):
    """
    op_name: name of op to be checked
    expected_hints: can be a list or a dict
        list: list of hint names to be checked
        dict: both hint names and corresponding values will be checked
    op_idx: in a single JIT IR, the same op may occur more than one times,
        started from 0
    """
    import re

    from habana_frameworks.torch.dynamo.compile_backend.passes import logger as graph_logger

    op_found = False
    pattern = r"::(\w+)(\[|\()"
    skip_fwd_graph = "backward" in op_name

    for log in graph_logger.data:
        if "####Annotated JIT IR graph" in log:
            if skip_fwd_graph:
                # check backward graph
                skip_fwd_graph = False
                continue
            op_occurence = 0
            for line in log.split("\n"):
                found_op_name = None
                m = re.search(pattern, line)
                if m:
                    found_op_name = m.group(1)
                if op_name == found_op_name:
                    op_occurence += 1
                    if op_occurence < op_idx + 1:
                        continue

                    op_found = True
                    if "hints" not in line:
                        continue
                    # format is lilke: [hints="name1:value1;name2:value2;"]
                    start_pos = line.find("[")
                    end_pos = line.find("]")
                    # only keep hint str -> name1:value1;name2:value2;
                    real_hints_str = line[start_pos + 8 : end_pos - 1]
                    real_hints = {}
                    for h in real_hints_str.split(";"):
                        if not h:
                            break
                        key = h.split(":")[0]
                        val = h.split(":")[-1]
                        real_hints[key] = val

                    if isinstance(expected_hints, dict):
                        for n, v in expected_hints.items():
                            if n not in real_hints:
                                assert False, f"hint {n} is not present in JIT IR"
                            if v != real_hints[n]:
                                assert (
                                    False
                                ), f"value for hint {n} doesn't match with expected one, {real_hints[n]} vs. {v}"
                    elif isinstance(expected_hints, list):
                        for n in expected_hints:
                            if n not in real_hints:
                                assert False, f"hint {n} is not present in JIT IR"
                    break
            if op_found:
                break

    assert op_found, f"op {op_name} is not found in generated JIT IR"


class ToyModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)
        return torch.reshape(out, (out.size(0), out.size(1)))


class ToyModelOuterHint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def outer_hint_function(input):
            out = input
            out = self.layers[0](out)
            out = self.layers[1](out)
            out = self.layers[2](out)
            return self.softmax(out)

        out = hints_wrapper(outer_hint_function, (x,), {}, hints={"outer_hint": "True"})

        return torch.reshape(out, (out.size(0), out.size(1)))


class ToyModelNestedHint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block):
            return self.layers[block](input)

        def outer_hint_function(input):
            out = input
            out = hints_wrapper(inner_hint_function, (out, 0), {}, hints={"inner_hint": "1"})
            out = hints_wrapper(inner_hint_function, (out, 1), {}, hints={"inner_hint": "2"})
            out = hints_wrapper(inner_hint_function, (out, 2), {}, hints={"inner_hint": "3"})
            return self.softmax(out)

        out = hints_wrapper(outer_hint_function, (x,), {}, hints={"outer_hint": "True"})

        return torch.reshape(out, (out.size(0), out.size(1)))


class LinearConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2
        return tensor * const1 + const2

    @staticmethod
    def backward(ctx, grad_output):
        # Dummy, mathematically incorrect, implementation of BWD just to show hints.
        return grad_output * ctx.const1 + ctx.const2, None, None


class ToyModelFunctionalLinearWithHints(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        def outer_hint_function(input1, input2):
            out = torch.nn.functional.linear(input1, input2)
            return out

        out = hints_wrapper(outer_hint_function, (x, y), {}, hints={"outer_functional_linear_hint": "true"})

        return torch.reshape(out, (out.size(0), out.size(1)))


class ToyModelAutogradOverrideWithoutHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)

        out = LinearConstant.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0), out.size(1)))


class LinearConstantHinted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2

        def forward_hinted(tensor, const1, const2):
            return tensor * const1 + const2

        return hints_wrapper(forward_hinted, (tensor, const1, const2), {}, hints={"fwd_custom_linear": "True"})

    @staticmethod
    def backward(ctx, grad_output):
        # Dummy, mathematically incorrect, implementation of BWD just to show hints.
        def backward_hinted(grad_output, const1, const2):
            return grad_output * const1 + const2

        return (
            hints_wrapper(
                backward_hinted,
                (grad_output, ctx.const1, ctx.const2),
                {},
                hints={"bwd_custom_linear": "True"},
            ),
            None,
            None,
        )


class ToyModelAutogradOnlyOverrideWithHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)

        out = LinearConstantHinted.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0), out.size(1)))


class ToyModelAutogradOverrideWithHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block):
            return self.layers[block](input)

        def outer_hint_function(input):
            out = input
            out = hints_wrapper(inner_hint_function, (out, 0), {}, hints={"inner_hint": "1"})
            out = hints_wrapper(inner_hint_function, (out, 1), {}, hints={"inner_hint": "2"})
            out = hints_wrapper(inner_hint_function, (out, 2), {}, hints={"inner_hint": "3"})
            return self.softmax(out)

        out = hints_wrapper(outer_hint_function, (x,), {}, hints={"outer_hint": "True"})

        out = LinearConstantHinted.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0), out.size(1)))


class LinearConstantNestedHinted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2

        def forward_mul(tensor, const1):
            return tensor * const1

        def forward_add(tensor, const2):
            return tensor + const2

        def forward_hinted(tensor, const1, const2):
            out = hints_wrapper(forward_mul, (tensor, const1), {}, hints={"part": "mul"})
            out = hints_wrapper(forward_add, (out, const2), {}, hints={"part": "add"})
            return out

        return hints_wrapper(forward_hinted, (tensor, const1, const2), {}, hints={"fwd_custom_linear": "True"})

    @staticmethod
    def backward(ctx, grad_output):
        # Dummy, mathematically incorrect, implementation of BWD just to show hints.
        def backward_mul(tensor, const1):
            return tensor * const1

        def backward_add(tensor, const2):
            return tensor + const2

        def backward_hinted(grad_output, const1, const2):
            out = hints_wrapper(backward_mul, (grad_output, const1), {}, hints={"part": "mul"})
            out = hints_wrapper(backward_add, (out, const2), {}, hints={"part": "add"})
            return out

        return (
            hints_wrapper(
                backward_hinted,
                (grad_output, ctx.const1, ctx.const2),
                {},
                hints={"bwd_custom_linear": "True"},
            ),
            None,
            None,
        )


class ToyModelAutogradOverrideWithNestedHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block):
            return self.layers[block](input)

        def outer_hint_function(input):
            out = input
            out = hints_wrapper(inner_hint_function, (out, 0), {}, hints={"inner_hint": "1"})
            out = hints_wrapper(inner_hint_function, (out, 1), {}, hints={"inner_hint": "2"})
            out = hints_wrapper(inner_hint_function, (out, 2), {}, hints={"inner_hint": "3"})
            return self.softmax(out)

        out = hints_wrapper(outer_hint_function, (x,), {}, hints={"outer_hint": "True"})

        out = LinearConstantNestedHinted.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0), out.size(1)))


class ComplexOperationWithPreservedOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2

        def forward_part1(tensor, const1):
            return tensor * const1

        def forward_part2_serial(tensor):
            # Path A
            a0 = torch.relu(tensor)
            a1 = a0 * 1.5
            a2 = a1 + 2
            a3 = a2 * 2.5
            a4 = a3 + 3
            a5 = a4 * 3.5

            # Path B
            b0 = torch.selu(tensor)
            b1 = b0 + 2.5
            b2 = b1 * 3
            b3 = b2 + 3.5
            b4 = b3 * 4
            b5 = b4 + 4.5

            # Path C
            c0 = torch.celu(tensor)
            c1 = c0 + 2.7
            c2 = c1 * 0.8
            c3 = c2 - 2
            c4 = c3 * 1.1
            c5 = c4 + 2

            return a5 + b5 + c5

        def forward_part2_interleaved(tensor):
            a0 = torch.relu(tensor)
            b0 = torch.selu(tensor)
            c0 = torch.celu(tensor)

            a1 = a0 * 1.5
            b1 = b0 + 2.5
            c1 = c0 + 2.7

            a2 = a1 + 2
            b2 = b1 * 3
            c2 = c1 * 0.8

            a3 = a2 * 2.5
            b3 = b2 + 3.5
            c3 = c2 - 2

            a4 = a3 + 3
            b4 = b3 * 4
            c4 = c3 * 1.1

            a5 = a4 * 3.5
            b5 = b4 + 4.5
            c5 = c4 + 2
            return a5 + b5 + c5

        def forward_part3(tensor, const2):
            return tensor + const2

        def forward_hinted(tensor, const1, const2):
            out = torch.ops.higher_order.hints_wrapper(forward_part1, (tensor, const1), {}, hints={"part": "pre"})
            out = torch.ops.higher_order.hints_wrapper(
                forward_part2_serial, (out,), {}, hints={"part_id": "middle_serial"}
            )
            out = torch.ops.higher_order.hints_wrapper(
                forward_part2_interleaved, (out,), {}, hints={"part_id": "middle_interleaved"}
            )
            out = torch.ops.higher_order.hints_wrapper(forward_part3, (out, const2), {}, hints={"part_id": "post"})
            return out

        return torch.ops.higher_order.hints_wrapper(
            forward_hinted,
            (tensor, const1, const2),
            {},
            hints={"some_complex_op_fwd": "true", "preserve_order": "true"},
        )

    @staticmethod
    def backward(ctx, grad_output):
        def backward_part1(tensor, const1):
            return tensor * const1

        def backward_part2_serial(tensor):
            # Path A
            a0 = torch.relu(tensor)
            a1 = a0 * 1.5
            a2 = a1 + 2
            a3 = a2 * 2.5
            a4 = a3 + 3
            a5 = a4 * 3.5

            # Path B
            b0 = torch.selu(tensor)
            b1 = b0 + 2.5
            b2 = b1 * 3
            b3 = b2 + 3.5
            b4 = b3 * 4
            b5 = b4 + 4.5

            # Path C
            c0 = torch.celu(tensor)
            c1 = c0 + 2.7
            c2 = c1 * 0.8
            c3 = c2 - 2
            c4 = c3 * 1.1
            c5 = c4 + 2

            return a5 + b5 + c5

        def backward_part2_interleaved(tensor):
            a0 = torch.relu(tensor)
            b0 = torch.selu(tensor)
            c0 = torch.celu(tensor)

            a1 = a0 * 1.5
            b1 = b0 + 2.5
            c1 = c0 + 2.7

            a2 = a1 + 2
            b2 = b1 * 3
            c2 = c1 * 0.8

            a3 = a2 * 2.5
            b3 = b2 + 3.5
            c3 = c2 - 2

            a4 = a3 + 3
            b4 = b3 * 4
            c4 = c3 * 1.1

            a5 = a4 * 3.5
            b5 = b4 + 4.5
            c5 = c4 + 2
            return a5 + b5 + c5

        def backward_part3(tensor, const2):
            return tensor + const2

        def backward_hinted(grad_output, const1, const2):
            out = torch.ops.higher_order.hints_wrapper(backward_part1, (grad_output, const1), {}, hints={"part": "pre"})
            out = torch.ops.higher_order.hints_wrapper(
                backward_part2_serial, (out,), {}, hints={"part_id": "middle_serial"}
            )
            out = torch.ops.higher_order.hints_wrapper(
                backward_part2_interleaved, (out,), {}, hints={"part_id": "middle_interleaved"}
            )
            out = torch.ops.higher_order.hints_wrapper(backward_part3, (out, const2), {}, hints={"part_id": "post"})
            return out

        return (
            torch.ops.higher_order.hints_wrapper(
                backward_hinted,
                (grad_output, ctx.const1, ctx.const2),
                {},
                hints={"some_complex_op_bwd": "true", "preserve_order": "true"},
            ),
            None,
            None,
        )


class ToyModelAutogradOverrideWithPreservedOrder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(10, 15), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(15, 20), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.ReLU()),
            ]
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)

        out = ComplexOperationWithPreservedOrder.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0), out.size(1)))


class NestReluWithHints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        def forward_relu(input):
            out = torch.relu(input)
            return out

        return torch.ops.higher_order.hints_wrapper(
            forward_relu, (input,), {}, hints={"nest_relu_autograd_fwd": "true"}
        )

    @staticmethod
    def backward(ctx, input):
        # Dummy, mathematically incorrect, implementation of BWD just to show hints.
        def backward_relu(input):
            out = torch.relu(input)
            return out

        return torch.ops.higher_order.hints_wrapper(
            backward_relu, (input,), {}, hints={"nest_relu_autograd_bwd": "true"}
        )


@pytest.mark.skipif(skip_test, reason="hints_wrapper op is not supported yet")
class ContextHintsTests(TestCase):
    def test_basic(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelBase().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)

    def test_outer_hint(self):
        pytest.skip("torch._higher_order.ops.hints_wrapper doesn't support autograd yet.")
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelOuterHint().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)

        check_hints_in_jit_ir("addmm", {"outer_hint": "True"})
        check_hints_in_jit_ir("_softmax", {"outer_hint": "True"})

    def test_nested_hint(self):
        pytest.skip("torch._higher_order.ops.hints_wrapper doesn't support autograd yet.")
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelNestedHint().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)

        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "1"}, 0)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "2"}, 1)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "3"}, 2)
        check_hints_in_jit_ir("relu", ["outer_hint", "inner_hint"])
        check_hints_in_jit_ir("_softmax", {"outer_hint": "True"})

    def test_nested_autograd_nohint(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithoutHints().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        iteration(x, y)

    def test_only_autograd_hint(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelAutogradOnlyOverrideWithHints().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        iteration(x, y)

        check_hints_in_jit_ir("mul", {"fwd_custom_linear": "True"})
        check_hints_in_jit_ir("add", {"fwd_custom_linear": "True"})
        check_hints_in_jit_ir("addmm", [])
        check_hints_in_jit_ir("_softmax_backward_data", [])

    def test_nested_autograd_hint(self):
        pytest.skip("torch._higher_order.ops.hints_wrapper doesn't support autograd yet.")

        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithHints().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        iteration(x, y)

        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "1"}, 0)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "2"}, 1)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "3"}, 2)

    def test_nested_autograd_nestedhint(self):
        pytest.skip("torch._higher_order.ops.hints_wrapper doesn't support autograd yet.")
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithNestedHints().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        iteration(x, y)

        check_hints_in_jit_ir("mul", {"fwd_custom_linear": "True", "part": "mul"})
        check_hints_in_jit_ir("add", {"fwd_custom_linear": "True", "part": "add"})
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "1"}, 0)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "2"}, 1)
        check_hints_in_jit_ir("addmm", {"outer_hint": "True", "inner_hint": "3"}, 2)
        check_hints_in_jit_ir("_softmax", {"outer_hint": "True"})

    def test_outer_hint_functional_linear(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelFunctionalLinearWithHints().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.rand(5, 10).to("hpu")
        z = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y, z):
            result = model(x, y)
            loss = criterion(result, z)
            return loss

        iteration(x, y, z)

        check_hints_in_jit_ir("linear", {"outer_functional_linear_hint": "true"})

    def test_nested_autograd_preserved_order(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithPreservedOrder().to("hpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend="hpu_backend")

        x = torch.rand(4, 10).to("hpu")
        y = torch.ones(4, dtype=torch.long).to("hpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        iteration(x, y)

        check_hints_in_jit_ir(
            "where", {"some_complex_op_fwd": "true", "preserve_order": "true", "part_id": "middle_serial"}
        )

    def test_multiple_outputs(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        def hinted_func(a, b):
            def func(a, b):
                z1, indice_out = torch.topk(a, 2)
                z2 = z1 + b
                return z2, indice_out

            return hints_wrapper(func, (a, b), {}, hints={"preserve_order": "True", "group_id": 1})

        x = torch.randn(2, 5, dtype=torch.float).to("hpu")
        y = 1

        compiled_fn = torch.compile(hinted_func, backend="hpu_backend")
        res = compiled_fn(x, y)

        check_hints_in_jit_ir("topk", ["preserve_order", "exec_order", "group_id"])
        check_hints_in_jit_ir("topk", {"preserve_order": "True", "group_id": "1"})

        check_hints_in_jit_ir("add", ["preserve_order", "exec_order", "group_id"])
        check_hints_in_jit_ir("add", {"preserve_order": "True", "group_id": "1"})

    def test_zero_inputs(self):
        clear_t_compile_logs()
        torch._dynamo.reset()

        def hinted_func():
            def func():
                z1 = torch.arange(0, 10, dtype=torch.float, device="hpu")
                z2 = z1 + 3
                return z2

            return hints_wrapper(func, (), {}, hints={"preserve_order": "True", "group_id": 99})

        compiled_fn = torch.compile(hinted_func, backend="hpu_backend")
        res = compiled_fn()

        check_hints_in_jit_ir("arange", ["preserve_order", "exec_order", "group_id"])
        check_hints_in_jit_ir("arange", {"preserve_order": "True", "group_id": "99"})

        check_hints_in_jit_ir("add", ["preserve_order", "exec_order", "group_id"])
        check_hints_in_jit_ir("add", {"preserve_order": "True", "group_id": "99"})
