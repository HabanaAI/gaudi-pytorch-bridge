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
import torch.fx
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement
from habana_frameworks.torch.dynamo.compile_backend.recipe_compiler import HabanaGraphModule
from habana_frameworks.torch.dynamo.utils import auto_map, str_join

from ..logger import get_fx_graph_logger
from .visualization import GraphVisualizer, fx_dumping_enabled, get_node_coloring_strategy

logger = get_fx_graph_logger()


@auto_map
def _val(meta_val):
    from torch.fx.passes.shape_prop import _extract_tensor_metadata

    if not isinstance(meta_val, torch.Tensor):
        return f"{type(meta_val)}"

    meta = _extract_tensor_metadata(meta_val)
    shape, dtype, stride, mem_fmt = (
        list(meta.shape),
        meta.dtype,
        meta.stride,
        meta.memory_format,
    )

    return f"Tensor(shape={shape}, dtype={dtype}, stride={stride}, mem_fmt={mem_fmt})"


def _node_meta_str_(node, dump_inputs_meta):
    meta_msg = ""
    inputs_meta_msg = ""

    if "val" not in node.meta:
        return meta_msg, inputs_meta_msg

    meta_val = node.meta["val"]
    meta_msg += str_join(_val(meta_val))

    if dump_inputs_meta:

        def dump_input_nodes_metadata(input_node):
            local_msg = ""
            if isinstance(input_node, torch.fx.node.Node):
                local_msg += _node_meta_str_(input_node, False)[0]
            elif isinstance(input_node, (list, tuple)):
                local_msg += "["
                separator = ", "
                local_msg += separator.join(
                    (_node_meta_str_(arg, False)[0] if isinstance(arg, torch.fx.node.Node) else f"{type(arg)}")
                    for arg in input_node
                )
                local_msg += "]"
            else:
                local_msg += f"{type(input_node)}"

            return local_msg

        separator = ", "
        inputs_meta_msg += separator.join(dump_input_nodes_metadata(arg) for arg in node.args)
    return meta_msg, inputs_meta_msg


def graph_to_str(
    fx_module: torch.fx.GraphModule,
    graph_name: str,
    jit_repr: str = None,
    print_nodes=False,
    dynamic=None,
    inference=None,
) -> str:
    graph_str_list = [
        f"### {graph_name} ###",
        fx_module.print_readable(False),
    ]

    if jit_repr is not None:
        graph_str_list.extend(["\nJIT:", jit_repr])
    else:
        graph_str_list.extend(["\nIR:", str(fx_module.graph)])

    if print_nodes:
        graph_str_list.append("\nNodes:")
        for node in fx_module.graph.nodes:
            graph_str_list.append(f"Node name: {node.name} op: {node.op}")
            if node.op == "call_function":
                graph_str_list.append(f"  target: {node.target.__name__}")
            if "val" in node.meta.keys():
                meta, inputs_meta = _node_meta_str_(node, True)
                if "output_device" in node.meta:
                    graph_str_list.append(f"  device: {node.meta['output_device']}")
                graph_str_list.append(f"  metadata: {meta}")
                if node.op not in [
                    "placeholder",
                    "output",
                ]:
                    graph_str_list.append(f"  inputs metadata: {inputs_meta}")

    if dynamic is not None:
        graph_str_list.append(f"dynamic:\t{dynamic}")
    if inference is not None:
        graph_str_list.append(f"inference:\t{inference}")

    graph_str_list.append("\n")

    return "\n".join(graph_str_list)


def dump_fx_graph(
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    stage: OptimizationPassPlacement,
    pass_counter: int,
    last_pass: str = None,
):
    if not fx_dumping_enabled():
        return

    if last_pass is None and stage != OptimizationPassPlacement.PRE_PLACEMENT:
        return

    if last_pass is not None:
        graph_log_name = f"{graph_name} <{stage} after `{last_pass}`>"
        graph_file_name = f"{graph_name}-{stage.value}-{stage.name}-{pass_counter}-{last_pass}"
    else:
        graph_log_name = f"{graph_name} <{stage}>"
        graph_file_name = f"{graph_name}-{stage.value}-{stage.name}-{pass_counter}"

    node_coloring = get_node_coloring_strategy(stage=stage, graph_pass=last_pass)

    logger.debug(graph_to_str(graph_module, graph_name=graph_log_name, print_nodes=True))

    GraphVisualizer.dump_fx_graph(graph_file_name, graph_module, node_coloring)


def dump_fx_submodule(submodule: HabanaGraphModule):
    if not fx_dumping_enabled():
        return

    logger.info(
        graph_to_str(
            fx_module=submodule.fx_module,
            graph_name=submodule.name,
            jit_repr=submodule.graph_str_repr_with_source_info,
            dynamic=submodule.is_dynamic,
            inference=submodule.is_inference,
        )
    )
    GraphVisualizer.dump_fx_graph(submodule.name, submodule.fx_module, get_node_coloring_strategy())
