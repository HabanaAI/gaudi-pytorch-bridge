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

import os
from unittest import mock

import functorch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.debug_utils.graph_repro_utils import (
    map_hpu_backend_config_snapshot_to_dict,
    store_fx_graph_as_code,
)
from habana_frameworks.torch.dynamo.debug_utils.logger import log_function_start_end
from habana_frameworks.torch.dynamo.utils import str_to_bool

import torch
from torch._dynamo.utils import detect_fake_mode

from .freezing_passes import freeze
from .internal import (
    optimize_post_partitioner,
    optimize_pre_partitioner,
    optimize_pre_placement,
    partition_module,
)


def _gen_graph_name():
    current_ordinal = _gen_graph_name.ordinal
    graph_name = f"graph_{current_ordinal:04d}_base_fx"
    _gen_graph_name.ordinal = current_ordinal + 1
    return graph_name


_gen_graph_name.ordinal = 0


@log_function_start_end
def hpu_freezing_compiler_inner(
    graph_module: torch.fx.GraphModule,
    dyn_graph_module: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    """
    This function will be called for each input FX graph. This will only process
    inference graphs where we run inference specific passes and parameter freezing.
    """

    assert not is_training and not is_backward

    graph_name = _gen_graph_name()

    # Perform optimizations on a graph before running passes for preparing the partitioner.
    optimize_pre_placement(graph_module, graph_name, example_inputs, is_training, is_backward)

    # Perform optimizations on a graph before the partitioner.
    optimize_pre_partitioner(graph_module, graph_name, example_inputs, is_training, is_backward)

    graph_module, non_param_input_ids = freeze(
        dynamo_gm=dyn_graph_module, aot_autograd_gm=graph_module, example_inputs=example_inputs
    )
    optimized_example_inputs = [example_inputs[i] for i in non_param_input_ids]
    fake_mode = detect_fake_mode(optimized_example_inputs)

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        params_flat = tracing_context.params_flat
        assert fw_metadata is not None and params_flat is not None
        for i in range(len(params_flat)):
            if i not in non_param_input_ids:
                params_flat[i] = None

    with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
        # Partition the module based on propagated device placement data.
        partition_module(graph_module, graph_name, optimized_example_inputs, is_training, is_backward)

        # Perform optimizations on a graph after the partitioner.
        optimize_post_partitioner(graph_module, graph_name, optimized_example_inputs, is_training, is_backward)

        # Return the module in boxed format required by AOT Autograd.
        if not hpu_backend_config.use_boxed_input:
            boxed_function = functorch.compile.make_boxed_func(graph_module.forward)
        else:

            def wrapper(args: list):
                return graph_module.forward(args)

            wrapper._boxed_call = True
            boxed_function = wrapper

        def wrapper(args):
            args_new = [args[i] for i in non_param_input_ids]
            args.clear()
            return boxed_function(args_new)

        wrapper._boxed_call = True

        return wrapper


@log_function_start_end
def hpu_compiler_inner(
    graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], is_training: bool, is_backward: bool
):
    """
    This function will be called for each input FX graph. There will be at least
    three separate graphs for FWD, BWD and optimizer. Each of these phases can
    also generate multiple graphs and calls to this function.
    """
    if not is_training and str_to_bool(os.environ.get("PT_HPU_USE_FUSE_SDPA_PASS", False)):
        # optimize the module before partitioning it
        # we will fuse the attention module here
        from habana_frameworks.torch.dynamo.compile_backend._passes.fuse_attention import (
            hpu_recursive_joint_graph_passes,
        )

        hpu_recursive_joint_graph_passes(graph_module)

    if hpu_backend_config.enable_flex_attention:
        from habana_frameworks.torch.dynamo.compile_backend.flex_attention import (
            hpu_flex_attention_passes,
        )

        hpu_flex_attention_passes(graph_module, example_inputs, is_training, is_backward)

    graph_name = _gen_graph_name()

    if hpu_backend_config.dump_graph_repro:
        store_fx_graph_as_code(
            graph_module,
            example_inputs,
            graph_name,
            options=map_hpu_backend_config_snapshot_to_dict(hpu_backend_config),
        )
    # Perform optimizations on a graph before running passes for preparing the partitioner.
    optimize_pre_placement(graph_module, graph_name, example_inputs, is_training, is_backward)

    # Perform optimizations on a graph before the partitioner.
    optimize_pre_partitioner(graph_module, graph_name, example_inputs, is_training, is_backward)

    # Partition the module based on propagated device placement data.
    partition_module(graph_module, graph_name, example_inputs, is_training, is_backward)

    # Perform optimizations on a graph after the partitioner.
    optimize_post_partitioner(graph_module, graph_name, example_inputs, is_training, is_backward)

    # Return the module in boxed format required by AOT Autograd.
    if not hpu_backend_config.use_boxed_input:
        return functorch.compile.make_boxed_func(graph_module.forward)
    else:

        def wrapper(args: list):
            return graph_module.forward(args)

        wrapper._boxed_call = True
        return wrapper


def hpu_training_compiler_fw(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """
    Just passthrough for forward pass training compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, True, False)


def hpu_training_compiler_bw(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """
    Just passthrough for backward pass training compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, True, True)


def hpu_inference_compiler(
    graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], dyn_graph_module: torch.fx.GraphModule
):
    """
    Just passthrough for forward inference compilation.
    """
    if torch._inductor.config.freezing:
        return hpu_freezing_compiler_inner(graph_module, dyn_graph_module, example_inputs, False, False)
    else:
        return hpu_compiler_inner(graph_module, example_inputs, False, False)
