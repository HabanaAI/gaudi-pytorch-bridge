###############################################################################
# Copyright (c) 2025 Intel Corporation
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
from torch.distributed.tensor import Replicate
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy
from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
)

"""
Use register_op_strategy when the sharding of the output cannot be determined solely
from the input shardings, or when you want to explore multiple possible execution strategies
and let the distributed runtime choose the best one. This is common for operations like
matrix multiplication (aten.mm), convolutions, or other more complex operations.
"""


def addmm_like_strategy(
    mm_equation: str, mesh: torch.distributed.tensor.device_mesh.DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # bias and input posisios are swapped to use the addmm strategy for linear
    mat1_strategy, mat2_strategy, *self_strategy = op_schema.args_schema
    if self_strategy == []:
        self_strategy = None
    if self_strategy is not None:
        assert isinstance(*self_strategy, OpStrategy)
    assert isinstance(mat1_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    if self_strategy is not None:
        is_self_placement_replicate = self_strategy[0].strategies[0].output_specs.placements[0].is_replicate()
        self_shape = self_strategy[0].shape
    mm_out_shape = torch.Size(
        [
            mat2_strategy.shape[-1] if i == len(mat1_strategy.shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_strategy.shape)
        ]
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        # construct new strategy by consider the self arg
        assert strtg.input_specs is not None
        mat1_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        out_spec = strtg.output_spec
        is_mat1_placement_replicate_or_shard0 = mat1_spec.placements[0].is_replicate() or mat1_spec.placements[
            0
        ].is_shard(0)
        is_mat2_placement_replicate_or_shard0 = mat2_spec.placements[0].is_replicate() or mat2_spec.placements[
            0
        ].is_shard(0)

        # self arg's spec should follow the output of mm, but need
        # to consider broadcast for the self arg
        self_spec = None
        if self_strategy is not None:
            broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
            self_placements = map_placements_after_broadcast(out_spec.placements, mm_out_shape, broadcast_dims_map)
            self_spec = DTensorSpec(mesh=mesh, placements=self_placements)

        if is_tensor_shardable(mat1_strategy.shape, mat1_spec) and is_tensor_shardable(mat2_strategy.shape, mat2_spec):
            # update input specs with new self spec
            if self_strategy is not None:
                strtg.input_specs = (mat1_spec, mat2_spec, self_spec)
            else:
                strtg.input_specs = (mat1_spec, mat2_spec)
            # associate costs
            redistribute_cost = None
            if self_strategy is not None:
                redistribute_cost = [
                    generate_redistribute_costs(mat1_strategy, mat1_spec),
                    generate_redistribute_costs(mat2_strategy, mat2_spec),
                    generate_redistribute_costs(self_strategy[0], self_spec),  # self is bias
                ]
            else:
                redistribute_cost = [
                    generate_redistribute_costs(mat1_strategy, mat1_spec),
                    generate_redistribute_costs(mat2_strategy, mat2_spec),
                ]
            strtg.redistribute_cost = redistribute_cost
            if (
                is_mat1_placement_replicate_or_shard0
                and is_mat2_placement_replicate_or_shard0
                and self_strategy is not None
                and is_self_placement_replicate
            ):
                filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


@register_op_strategy(torch.ops.hpu.linear.default)
def linear_sharding_strategy(mesh: torch.distributed.tensor.device_mesh.DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    mat1_strategy, mat2_strategy, *self_strategy = op_schema.args_schema
    if self_strategy == []:
        self_strategy = None
    if self_strategy is not None:
        assert isinstance(*self_strategy, OpStrategy)
    assert isinstance(mat1_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    # We support only a restricted number of sharding strategies as of now for linear on HPU
    is_mat1_placement_replicate_or_shard0 = mat1_strategy.strategies[0].output_specs.placements[
        0
    ].is_replicate() or mat1_strategy.strategies[0].output_specs.placements[0].is_shard(0)
    is_mat2_placement_replicate_or_shard0 = mat2_strategy.strategies[0].output_specs.placements[
        0
    ].is_replicate() or mat2_strategy.strategies[0].output_specs.placements[0].is_shard(0)
    if self_strategy is not None:
        is_self_placement_replicate = self_strategy[0].strategies[0].output_specs.placements[0].is_replicate()
    elif len(mat1_strategy.shape) <= 2 and len(mat2_strategy.shape) <= 2:
        is_self_placement_replicate = None
    else:  # matmul with dims > 3
        is_self_placement_replicate = True
    linear_strategy = None
    if (
        len(mat1_strategy.shape) <= 2
        and len(mat2_strategy.shape) <= 2
        and is_mat1_placement_replicate_or_shard0
        and is_mat2_placement_replicate_or_shard0
        and is_self_placement_replicate
    ):
        linear_strategy = addmm_like_strategy("mk,kn->mn", mesh, op_schema)
    elif self_strategy is not None:
        linear_strategy = OpStrategy(
            [
                PlacementStrategy(
                    output_specs=DTensorSpec(mesh, (Replicate(),)),
                    input_specs=(
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                    ),
                    redistribute_cost=[[1]],
                )
            ]
        )
    else:
        linear_strategy = OpStrategy(
            [
                PlacementStrategy(
                    output_specs=DTensorSpec(mesh, (Replicate(),)),
                    input_specs=(
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                    ),
                    redistribute_cost=[[1]],
                )
            ]
        )

    return linear_strategy


@register_op_strategy(torch.ops.hpu.linear_backward.default)
def linear_bwd_sharding_strategy(
    mesh: torch.distributed.tensor.device_mesh.DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # We support only a replicate strategy as of now for linear_backward on HPU
    # For autograd overridden ops, if we save the forward inputs in Ctxt, we can
    # access them as saved_tensors in backward(). But, the same is not applicable
    # for strategies for these input tensors and they won't be accessible. So,
    # how can we create any sharding strategy in such cases?
    if len(op_schema.args_schema) == 3:  # matmul_bwd has 3 arguments and linear_bwd has 4.
        return OpStrategy(
            [
                PlacementStrategy(
                    output_specs=(
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                    ),
                    input_specs=(
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                        DTensorSpec(mesh, (Replicate(),)),
                    ),
                    redistribute_cost=[[1]],
                )
            ]
        )
    linear_bwd_strategy = OpStrategy(
        [
            PlacementStrategy(
                output_specs=(
                    DTensorSpec(mesh, (Replicate(),)),  # grad_out
                    DTensorSpec(mesh, (Replicate(),)),  # input
                    DTensorSpec(mesh, (Replicate(),)),  # weight
                ),
                input_specs=(
                    DTensorSpec(mesh, (Replicate(),)),  # grad_out
                    DTensorSpec(mesh, (Replicate(),)),  # input
                    DTensorSpec(mesh, (Replicate(),)),  # weight
                ),
                redistribute_cost=[[1]],
            )
        ]
    )

    return linear_bwd_strategy


@register_op_strategy(torch.ops.hpu.matmul.default)
def matmul_sharding_strategy(mesh: torch.distributed.tensor.device_mesh.DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # Sharding strategy for matmul is like linear without bias
    return linear_sharding_strategy(mesh, op_schema)


@register_op_strategy(torch.ops.hpu.matmul_bwd.default)
def matmul_bwd_sharding_strategy(
    mesh: torch.distributed.tensor.device_mesh.DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # Sharding strategy for matmul_bwd is like linear without bias
    return linear_bwd_sharding_strategy(mesh, op_schema)
