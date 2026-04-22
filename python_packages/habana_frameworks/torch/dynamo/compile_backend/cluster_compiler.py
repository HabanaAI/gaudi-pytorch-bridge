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


import copy

import habana_frameworks.torch.internal.bridge_config as bc
from habana_frameworks.torch.dynamo._fx_to_jit_lowering import FxToJitLowering
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import (
    OptimizerContext,
)
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from habana_frameworks.torch.jit.csrc.jit_fork.python_passes.forked_passes import (
    run_jit_fork_passes,
)

import torch

from ._helpers import (
    fill_propagated_tensor_metadata_to_node,
    get_dynamic_config_value,
    is_module_dynamic,
    jit_node_annotation_propagation,
    jit_node_shape_propagation,
    remove_duplicated_outputs,
    remove_no_effect_inplace_add,
    wrap_random_ops,
)
from ._passes.random import propagate_for_random_ops
from .recipe_compiler import get_callable_recipe

logger = get_compile_backend_logger()


class _ClusterCompiler(torch.fx.Interpreter):
    def __init__(self, graph_module: torch.fx.GraphModule, ctx: OptimizerContext):
        super().__init__(graph_module)
        self.graph_module = graph_module
        self.ctx = ctx
        self.subgraph_cnt = 0
        self._has_random_ops = False

    def fx_to_jit_ir(self, submod, args):
        self._has_random_ops = wrap_random_ops(submod)
        if self._has_random_ops:
            propagate_for_random_ops(submod, args)
        remove_duplicated_outputs(submod)
        remove_no_effect_inplace_add(submod)

        submod.graph.lint()
        submod.recompile()

        fx_to_jit_lowering = FxToJitLowering(submod)
        fx_to_jit_lowering.run(*args)

        # Some tests require this pattern in logs, so for quick repair decided to
        #  print here it. Although this is not a PyTorch-generated obviously.
        logger.debug(
            "####PyTorch-generated JIT IR graph for this HPU graph:####\n%s",
            fx_to_jit_lowering.jit_ir,
        )

        run_jit_fork_passes(fx_to_jit_lowering.jit_ir)
        logger.debug(
            "####PyTorch-generated JIT IR graph after run_jit_fork_passes:####\n%s",
            fx_to_jit_lowering.jit_ir,
        )

        return fx_to_jit_lowering.jit_ir

    def run_node(self, n: torch.fx.Node):
        # This function has been overwritten because we need
        # access to FX nodes, not node.target as done in the base
        # run_node function.
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)
        with self._set_current_node(n):
            if "val" not in n.meta:
                raise AssertionError(f"{n=} {n.target=} {n.meta.keys()=}")
            if n.op == "call_module":
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                if not isinstance(args, tuple):
                    raise AssertionError("Not a tuple instance")
                if not isinstance(kwargs, dict):
                    raise AssertionError("Not a dict instance")
                return getattr(self, n.op)(n, args, kwargs)
            return n.meta["val"]

    def call_module(self, node: torch.fx.Node, args, kwargs):
        target = node.target
        submod = self.graph_module.get_submodule(target)

        # "_tensor_constant" nodes originally have get_attr op
        # but when included within fused they're represented as placeholder
        # due to that within pass_fake_propagation it's metadata propagation is being skipped
        # Below loop is to handle such cases
        for n in submod.graph.nodes:
            if n.name.startswith("_tensor_constant"):
                result = n.meta["val"]
                assert result is not None, (
                    f"Node {n.name} in graph {self.ctx.graph_name} expected to have val metadata assigned at this stage"
                )
                fill_propagated_tensor_metadata_to_node(result, n)

        submod_updated = copy.deepcopy(submod)
        jit_ir = self.fx_to_jit_ir(submod_updated, args)
        jit_node_annotation_propagation(jit_ir, submod_updated)

        is_submod_dynamic = is_module_dynamic(submod)
        refine_dynamic = bc.get_pt_hpu_enable_refine_dynamic_shapes()
        optim_output_sif_ds = bc.get_pt_hpu_optim_dynamic_output_sif()
        if not hpu_backend_config.force_static_compile:
            if refine_dynamic:
                is_submod_dynamic = is_submod_dynamic or get_dynamic_config_value()

            if is_submod_dynamic and optim_output_sif_ds:
                jit_node_shape_propagation(jit_ir, submod_updated)

        is_reusables: list[bool] = submod.meta.get("is_reusables", [])
        syngraph_module = get_callable_recipe(
            jit_ir,
            submod,
            self.ctx.graph_name,
            is_training=self.ctx.is_training,
            is_dynamic=is_submod_dynamic,
            has_random_ops=self._has_random_ops,
            is_reusables=is_reusables,
        )

        self.ctx.graph_module.delete_submodule(target)
        self.ctx.graph_module.add_submodule(target, syngraph_module)

        self.subgraph_cnt += 1

        self._has_random_ops = False

        logger.debug(
            "####PyTorch (JIT fork)-generated JIT IR graph for this HPU graph:####\n%s",
            str(syngraph_module.graph) if isinstance(syngraph_module, torch.fx.GraphModule) else jit_ir,
        )

        return node.meta["val"]

    @property
    def graph_changed(self):
        logger.info("Number of subgraphs created: %s", self.subgraph_cnt)
        return self.subgraph_cnt != 0


def pass_compile_clusters_jit_fork_version(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each generated XPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the XPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """

    logger.info("pass_compile_clusters_jit_fork_version")

    cluster_compiler = _ClusterCompiler(ctx.graph_module, ctx)
    cluster_compiler.run(*ctx.example_inputs)
    return cluster_compiler.graph_changed
