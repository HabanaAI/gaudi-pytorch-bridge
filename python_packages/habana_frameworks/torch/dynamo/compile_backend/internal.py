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

from typing import List

import torch

from .passes import OptimizationPassPlacement, optimize_graph


def optimize_pre_placement(
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    """
    This function is supposed to run optimizations passes on a graph that
    wasn't yet partitioned.
    """
    optimize_graph(
        OptimizationPassPlacement.PRE_PLACEMENT, graph_module, graph_name, example_inputs, is_training, is_backward
    )


def optimize_pre_partitioner(
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    """
    This function is supposed to run optimizations passes on a graph that
    wasn't yet partitioned.
    """
    optimize_graph(
        OptimizationPassPlacement.PRE_PARTITIONER, graph_module, graph_name, example_inputs, is_training, is_backward
    )


def optimize_post_partitioner(
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    """
    This function is supposed to run optimizations passes on a graph that
    was already partitioned.
    """
    optimize_graph(
        OptimizationPassPlacement.POST_PARTITIONER, graph_module, graph_name, example_inputs, is_training, is_backward
    )


def partition_module(
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
) -> torch.fx.GraphModule:
    """
    This function will run passes responsible for creating HPU partitions.
    """
    optimize_graph(
        OptimizationPassPlacement.PARTITIONER, graph_module, graph_name, example_inputs, is_training, is_backward
    )
