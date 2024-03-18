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

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch


class OptimizationPassPlacement(Enum):
    PRE_PARTITIONER = 1
    PARTITIONER = 2
    POST_PARTITIONER = 3


@dataclass
class OptimizerContext:
    graph_module: torch.fx.GraphModule
    example_inputs: List[torch.Tensor]
    is_training: bool
    is_backward: bool
    is_dynamic: bool
    uses_aot: bool
    stage: OptimizationPassPlacement
    current_partitions: List
