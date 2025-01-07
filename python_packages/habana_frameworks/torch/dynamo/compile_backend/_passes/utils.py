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


from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import torch.fx
from habana_frameworks.torch.dynamo.compile_backend.partitioner import HabanaPartitioner
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

logger = get_compile_backend_logger()


class OptimizationPassPlacement(Enum):
    PRE_PLACEMENT = 1
    PRE_PARTITIONER = 2
    PARTITIONER = 3
    POST_PARTITIONER = 4


@dataclass
class OptimizerContext:
    graph_module: torch.fx.GraphModule
    graph_name: str
    example_inputs: List[torch.Tensor]
    is_training: bool
    is_backward: bool
    is_dynamic: bool
    stage: OptimizationPassPlacement
    current_partitions: List
    current_partitions_non_mergeable: List
    habana_partitioner: HabanaPartitioner = None
    is_submod: bool = False


class ColorGraph:
    def __init__(self):
        self.partition_colors = set()
        self.output_colors = set()
        self.colors_to_remove = set()
        self._last_color = 0
        self._graph = defaultdict(set)

    def assign_new_color(self, is_partition_color=False):
        self._last_color += 1
        if is_partition_color:
            self.partition_colors.add(self._last_color)
        else:
            self.colors_to_remove.add(self._last_color)
        return self._last_color

    def add_node(self, user_color, color):
        if user_color != color:
            self._graph[color].add(user_color)

    def add_output_node(self, color):
        # We have cases in which a node without any users belongs to a
        # partition, and that partition can have other users.
        if color not in self._graph:
            self._graph[color] = set()

    def _merge_non_partition_colors(self):
        for color in self.colors_to_remove:
            replacement_set = self._graph.pop(color, {})
            for v in self._graph.values():
                if color in v:
                    v.remove(color)
                    v.update(replacement_set)

        self.colors_to_remove = set()

    def _get_reverse_graph_dict(self):
        logger.debug("Color graph (initial): \n%s", self)
        self._merge_non_partition_colors()
        logger.debug("Color graph (partitions only): \n%s", self)

        reverse_graph_dict = defaultdict(set)
        for color, user_set in self._graph.items():
            user_frozen_set = frozenset(user_set)
            reverse_graph_dict[user_frozen_set].add(color)

        return reverse_graph_dict

    def get_parallel_blocks(self):
        reverse_graph_dict = self._get_reverse_graph_dict()

        parallel_blocks = []

        key_to_remove = frozenset()

        while reverse_graph_dict:
            assert (
                key_to_remove in reverse_graph_dict
            ), "Empty set not a key in reverse_graph_dict. Graph most likely cyclic."
            current_block = reverse_graph_dict.pop(key_to_remove)
            parallel_blocks.append(current_block)
            next_iter_dependencies = [(key - current_block, val) for key, val in reverse_graph_dict.items()]
            reverse_graph_dict = defaultdict(set)
            for next_iter_consumers, next_iter_producers in next_iter_dependencies:
                reverse_graph_dict[next_iter_consumers].update(next_iter_producers)

        return parallel_blocks

    def __str__(self):
        lines = []
        lines.append(f"Partition colors: {self.partition_colors}")
        lines.extend([f"{k} --> {v}" for k, v in self._graph.items()])
        return "\n".join(lines)


class SchedulePolicy(Enum):
    # default mode, based on original execution order
    strict = 1
