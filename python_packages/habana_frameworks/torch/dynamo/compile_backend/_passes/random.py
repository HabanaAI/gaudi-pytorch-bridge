###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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


import torch
from torch._dynamo.utils import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode

from ..symbolic_execution import SymExprNodeManager


def skip_faketensor_propagation(node):
    if node.op != "call_function":
        return False

    # after pass pass_remove_unnecessary_bmm_view, it will make bmm op consume
    # non-3D input tensors and cause "batch 1must be a 3D tensor" error.
    # So just skip the second fake_propagation for bmm node.
    if node.target.__name__.split(".")[0] == "bmm":
        return True

    # skip inplace ops to avoid broadcast shape mismatch error
    # assuming inplace op's metadata is the same as the out-of-place version's
    if node.target.__name__.split(".")[0].endswith("_") or node.target == torch.ops.hpu.weight_permutation:
        # the add_ node in wrap_random ops function should do this faketensor propagation
        return node.name != "add__from_random"

    # skip hpu::slice_ds/hpu::constant_pad_nd_ds faketensor propagation.
    # these dynamic version's metadata should be the same as the static version's.
    if (
        hasattr(node.target, "namespace")
        and node.target.namespace == "hpu"
        and node.target.__name__.split(".")[0].endswith("_ds")
    ):
        return True


def generate_random_inputs(fake_mode, args):
    # Fakeify meta tensors used for random ops
    converter = fake_mode.fake_tensor_converter
    t = torch.tensor(0, dtype=torch.int, device="meta")
    additional_input1 = converter.from_meta_and_device(fake_mode, t, device="hpu")
    additional_input2 = converter.from_meta_and_device(fake_mode, t, device="hpu")
    example_inputs = (additional_input1, additional_input2) + args
    return example_inputs


def propagate_for_random_ops(graph_module: torch.fx.GraphModule, args):
    class _RandomOpsPropagation(torch.fx.Interpreter):
        def __init__(self, graph_module: torch.fx.GraphModule, fake_mode: FakeTensorMode | None = None):
            super().__init__(graph_module)
            self._mode = fake_mode if fake_mode else FakeTensorMode()

        def run_node(self, node: torch.fx.Node):
            args = kwargs = result = None
            if SymExprNodeManager.node_name in node.name and node.op != "placeholder":
                result = node.meta["val"]
                args, kwargs = self.fetch_args_kwargs_from_env(node)
            elif skip_faketensor_propagation(node):
                # skip the fake tensor propagation for some special cases
                result = node.meta["val"]
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                node.val_args = args
                node.val_kwargs = kwargs
                return result
            else:
                result = super().run_node(node)
                args, kwargs = self.fetch_args_kwargs_from_env(node)

            node.val_args = args
            node.val_kwargs = kwargs
            node.meta["val"] = result

            return result

        def propagate(self, *args):
            fake_args = [self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
            return self.propagate_dont_convert_inputs(*fake_args)

        def propagate_dont_convert_inputs(self, *args):
            with self._mode:
                return super().run(*args)

    fake_mode = detect_fake_mode(args)
    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(enabled=False, device_type="cpu"):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            example_inputs = generate_random_inputs(fake_mode, args)
            _RandomOpsPropagation(graph_module, fake_mode).propagate(*example_inputs)
        else:
            example_inputs = generate_random_inputs(fake_mode, args)
            _RandomOpsPropagation(graph_module, fake_mode).propagate_dont_convert_inputs(*example_inputs)
