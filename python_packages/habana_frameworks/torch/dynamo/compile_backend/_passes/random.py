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


import torch
from torch._ops import OpOverload as TorchOpOverload
from torch._subclasses.fake_tensor import FakeTensorMode

from .._helpers.helpers import propagate_meta
from ..random_utils import (
    backward_random_op_inputs,
    is_backward_checkpoint_op,
    is_random_op,
    is_run_and_save_rng_state,
    random_op_inputs,
)
from ..symbolic_execution import SymExprNodeManager


def wrap_random_ops(sub_module: torch.fx.GraphModule):
    """
    This pass goes through hpu cluster and:
    - replaces run_and_save_rng_state ops with hpu wrappers,
    - replaces run_with_rng_state ops with hpu checkpoint wrappers,
    - replaces random ops with hpu wrappers,
    - creates seed and counter tensor for hpu seed_generator,
    - feeds hpu wrappers with generated seed tensors.
    - propagate metadata from later JIT lowering (different from wrap_random_ops from helpers.py)
    """

    random_ops = [node for node in sub_module.graph.nodes if is_random_op(node)]
    backward_random_ops = [node for node in sub_module.graph.nodes if is_backward_checkpoint_op(node)]

    # run_with_rng_state op is replaced with the actual random op with seed acquired from
    # the run_with_rng_state's first input.
    if len(backward_random_ops) > 0:
        for node in backward_random_ops:
            with sub_module.graph.inserting_before(node):
                random_node = sub_module.graph.call_function(*backward_random_op_inputs(node))
                node.replace_all_uses_with(random_node, propagate_meta=True)
                random_node.meta.update(node.meta)
                random_node.meta["deterministic"] = True
                sub_module.graph.erase_node(node)

        sub_module.recompile()

    nbr_of_random = len(random_ops)
    if nbr_of_random == 0:
        return None

    with sub_module.graph.inserting_before():
        counter_pl = sub_module.graph.placeholder("counter_pl")
        seed_pl = sub_module.graph.placeholder("seed_pl")

    with sub_module.graph.inserting_after(counter_pl):
        seeds = sub_module.graph.call_function(
            torch.ops.hpu.habana_seed_generator, (counter_pl, seed_pl, nbr_of_random), {}
        )
        _ = sub_module.graph.call_function(torch.ops.aten.add_.Tensor, (counter_pl, nbr_of_random), {})

    for i, node in enumerate(random_ops):
        with sub_module.graph.inserting_before(node):
            seed = sub_module.graph.call_function(torch.select, (seeds, 0, i), {})
            random_node = sub_module.graph.call_function(*random_op_inputs(node, seed))

            if is_run_and_save_rng_state(node):
                random_node.meta["deterministic"] = True
                for getitem_node in list(node.users):
                    index = getitem_node.args[1]
                    if index not in [0, 1]:
                        raise AssertionError(f"Expecting {node} to produce only two outputs")

                    new_out = seed if index == 0 else random_node
                    getitem_node.replace_all_uses_with(new_out, propagate_meta=False)
                    sub_module.graph.erase_node(getitem_node)
            else:
                node.replace_all_uses_with(random_node, propagate_meta=True)
                random_node.meta.update(node.meta)

            sub_module.graph.erase_node(node)

    sub_module.recompile()

    device = "hpu"
    additional_random_args = (
        torch.tensor(0, dtype=torch.int, device=device),
        torch.tensor(0, dtype=torch.int, device=device),
    )
    return additional_random_args


def propagate_for_random_ops(
    graph_module: torch.fx.GraphModule, args, additional_inputs: tuple[torch.Tensor, torch.Tensor]
):
    # reset the 'constant' field of faketensor to avoid real computation in later meta propagation
    for ainput in additional_inputs:
        assert isinstance(ainput, torch._subclasses.fake_tensor.FakeTensor)
        ainput.constant = None

    full_args = additional_inputs + args

    class _RandomOpsPropagation(torch.fx.Interpreter):
        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            fake_mode: FakeTensorMode | None = None,
        ):
            super().__init__(graph_module)
            if fake_mode is None:
                fake_mode = FakeTensorMode()
            self._mode = fake_mode

        def run_node(self, node: torch.fx.Node):
            args = kwargs = result = None
            if SymExprNodeManager.node_name in node.name:
                result = node.meta["val"]
                args, kwargs = self.fetch_args_kwargs_from_env(node)
            elif (isinstance(node.target, TorchOpOverload) and node.target._name == "aten::bmm") or (
                hasattr(node.target, "default")
                and isinstance(node.target.default, TorchOpOverload)
                and node.target.default._name == "aten::bmm"
            ):
                # dealing with special cases
                # after pass pass_remove_unnecessary_bmm_view, it will make bmm op consume
                # non-3D input tensors and cause "batch 1must be a 3D tensor" error.
                # So just skip the second fake_propagation for bmm node.
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

    propagate_meta(graph_module, full_args, _RandomOpsPropagation)
