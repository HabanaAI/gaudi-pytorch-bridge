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
from habana_frameworks.torch.dynamo._fx_to_jit_lowering import FxToJitLowering
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizerContext
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

from .recipe_compiler import get_callable_recipe

logger = get_compile_backend_logger()


# below is repeated to avoid cross includes... todo: fix me https://jira.habana-labs.com/browse/SW-199903
def is_module_dynamic(input_module: torch.fx.GraphModule) -> bool:
    """
    This function dynamicity per graph module.
    """

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import py_sym_types
    from torch.fx.passes.shape_prop import TensorMetadata

    is_dynamic = False
    for node in input_module.graph.nodes:
        if node.op == "placeholder":
            meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
            if (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
                meta_val, py_sym_types
            ):
                is_dynamic = True
                break

    logger.debug("Module dynamicity %s", is_dynamic)
    return is_dynamic


class _ClusterCompiler(torch.fx.Interpreter):
    def __init__(self, graph_module: torch.fx.GraphModule, ctx: OptimizerContext):
        logger.debug("xtdomagala tmp log _ClusterCompiler . ctor")
        super().__init__(graph_module)
        self.graph_module = graph_module
        self.ctx = ctx
        self.subgraph_cnt = 0

    def fx_to_jit_ir(self, submod, args):
        fx_to_jit_lowering = FxToJitLowering(submod)
        fx_to_jit_lowering.run(*args)
        # todo implement run_jit_passes https://jira.habana-labs.com/browse/SW-199897
        # run_jit_passes(fx_to_jit_lowering.jit_ir)
        return fx_to_jit_lowering.jit_ir

    def run_node(self, n: torch.fx.Node):
        # This function has been overwritten because we need
        # access to FX nodes, not node.target as done in the base
        # run_node function.

        with self._set_current_node(n):
            if n.op == "call_module":
                assert "val" in n.meta.keys(), f"{n=} {n.target=} {n.meta.keys()=}"
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert isinstance(args, tuple)
                assert isinstance(kwargs, dict)
                return getattr(self, n.op)(n, args, kwargs)
            return n.meta["val"]

    def call_module(self, node: torch.fx.Node, args, kwargs):
        target = node.target
        submod = self.graph_module.get_submodule(target)

        # todo  sync with jan's dumping https://jira.habana-labs.com/browse/SW-200868
        # if config.dump_graph:
        #     jit_logger = get_jit_graph_logger()
        #     jit_logger.info(
        #         f"FX_GRAPH:\nNode name: {self.ctx.graph_name}\n" f"Target: {node.target}\n" f"Code: {submod.code}"
        #     )

        jit_ir = self.fx_to_jit_ir(submod, args)

        is_submod_dynamic = is_module_dynamic(submod)
        syngraph_module = get_callable_recipe(
            jit_ir, submod, is_training=self.ctx.is_training, is_dynamic=is_submod_dynamic
        )
        # todo https://jira.habana-labs.com/browse/SW-201169:
        # in our case compilation:
        # - fails with jit forked lowering
        # - is done later - it doesn't seem to be a problem
        # ir Graph converter should solve the issue

        # todo: integrate with our graph dumping https://jira.habana-labs.com/browse/SW-200868
        # if config.dump_graph:
        #     dump_fx_submodule(syngraph_module)

        # todo https://jira.habana-labs.com/browse/SW-199903
        #  do we have it handled elsewhere?:
        # if not syngraph_module.compilation_successful:
        #     self.ctx.fallback_required = True

        self.ctx.graph_module.delete_submodule(target)
        self.ctx.graph_module.add_submodule(target, syngraph_module)

        self.subgraph_cnt += 1

        return node.meta["val"]

    @property
    def graph_changed(self):
        logger.info("Number of subgraphs created: %s", self.subgraph_cnt)
        return self.subgraph_cnt != 0


# todo https://jira.habana-labs.com/browse/SW-199903
# don't neeed measurement now, later use bridge solution (I assume there is available one)
# @dynamo_timed(phase_name="POST_PARTITIONER")
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
