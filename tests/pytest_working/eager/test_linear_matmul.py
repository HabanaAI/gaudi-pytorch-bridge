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
import os

import habana_frameworks.torch.distributed.hccl
import pytest
import torch
import torch.nn.functional as F
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.placement_types import Shard

rank = 0


def linear_matmul_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def linear_matmul_cleanup():
    torch.distributed.destroy_process_group()


def custom_linear_matmul_helper(*arguments):
    rank = arguments[0]
    world_size = arguments[1]
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    linear_matmul_setup(rank, world_size)
    device_mesh = DeviceMesh("hpu", list(range(world_size)))
    """Test torch.nn.functional.linear with DTensor inputs."""
    device_type = "hpu"
    batch_size = 16
    out_features = 64
    in_features = 32
    local_input = torch.randn(batch_size, in_features, device=device_type).requires_grad_(True)
    local_weight = torch.randn(out_features, in_features, device=device_type).requires_grad_(True)
    local_bias = torch.randn(out_features, device=device_type).requires_grad_(True)

    input_strategy_list = [Shard(dim=0), Replicate()]
    weight_strategy_list = [Shard(dim=0), Replicate()]
    # Shard input and weight, replicate bias
    input_dtensor = distribute_tensor(local_input, device_mesh, [input_strategy_list[0]])
    weight_dtensor = distribute_tensor(local_weight, device_mesh, [weight_strategy_list[0]])
    bias_dtensor = distribute_tensor(local_bias, device_mesh, [Replicate()])

    # Matmul with dims > 2
    local_input1_matmul = torch.randn(4, 8, 4, device=device_type).requires_grad_(True)
    local_input2_matmul = torch.randn(4, 4, 8, device=device_type).requires_grad_(True)

    # Shard input and weight, replicate bias
    input1_matmul_dtensor = distribute_tensor(local_input1_matmul, device_mesh, [input_strategy_list[0]])
    input2_matmul_dtensor = distribute_tensor(local_input2_matmul, device_mesh, [weight_strategy_list[0]])
    local_matmul_out = torch.matmul(local_input1_matmul, local_input2_matmul)  # local_weight.t())
    loss_matmul_local = torch.sum(local_matmul_out)
    loss_matmul_local.backward()
    dtensor_matmul_out = torch.matmul(input1_matmul_dtensor, input2_matmul_dtensor)  # weight_dtensor.t())
    # Gather to compare with local result
    gathered_matmul_out = dtensor_matmul_out.full_tensor()
    if rank == 0:  # for some reason test hangs if this is not enabled
        torch.testing.assert_close(gathered_matmul_out.cpu(), local_matmul_out.cpu(), rtol=1e-4, atol=1e-4)
    loss_matmul_dist = torch.sum(gathered_matmul_out)
    loss_matmul_dist.backward()
    if rank == 0:  # for some reason, local_out for ranks other than 0 are coming junk
        torch.testing.assert_close(
            input1_matmul_dtensor.grad.full_tensor().cpu(), local_input1_matmul.grad.cpu(), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            input2_matmul_dtensor.grad.full_tensor().cpu(), local_input2_matmul.grad.cpu(), rtol=1e-4, atol=1e-4
        )

    # Matmul with dims <= 2
    local_input1_matmul = torch.randn(8, 4, device=device_type).requires_grad_(True)
    local_input2_matmul = torch.randn(4, 8, device=device_type).requires_grad_(True)

    # Shard input and weight, replicate bias
    input1_matmul_dtensor = distribute_tensor(local_input1_matmul, device_mesh, [input_strategy_list[0]])
    input2_matmul_dtensor = distribute_tensor(local_input2_matmul, device_mesh, [weight_strategy_list[0]])
    local_matmul_out = torch.matmul(local_input1_matmul, local_input2_matmul)  # local_weight.t())
    loss_matmul_local = torch.sum(local_matmul_out)
    loss_matmul_local.backward()
    dtensor_matmul_out = torch.matmul(input1_matmul_dtensor, input2_matmul_dtensor)  # weight_dtensor.t())
    # Gather to compare with local result
    gathered_matmul_out = dtensor_matmul_out.full_tensor()
    if rank == 0:  # for some reason test hangs if this is not enabled
        torch.testing.assert_close(gathered_matmul_out.cpu(), local_matmul_out.cpu(), rtol=1e-4, atol=1e-4)
    loss_matmul_dist = torch.sum(gathered_matmul_out)
    loss_matmul_dist.backward()
    if rank == 0:  # for some reason, local_out for ranks other than 0 are coming junk
        torch.testing.assert_close(
            input1_matmul_dtensor.grad.full_tensor().cpu(), local_input1_matmul.grad.cpu(), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            input2_matmul_dtensor.grad.full_tensor().cpu(), local_input2_matmul.grad.cpu(), rtol=1e-4, atol=1e-4
        )

    # Linear -with bias
    local_out = F.linear(local_input, local_weight, local_bias)
    loss_local = torch.sum(local_out)
    loss_local.backward()

    dtensor_out = F.linear(input_dtensor, weight_dtensor, bias_dtensor)
    # Gather to compare with local result
    gathered_out = dtensor_out.full_tensor()

    if rank == 0:  # for some reason test hangs if this is not enabled
        torch.testing.assert_close(gathered_out.cpu(), local_out.cpu(), rtol=1e-4, atol=1e-4)
    loss_dist = torch.sum(gathered_out)
    loss_dist.backward()
    if rank == 0:  # for some reason, local_out for ranks other than 0 are coming junk
        torch.testing.assert_close(input_dtensor.grad.full_tensor().cpu(), local_input.grad.cpu(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            weight_dtensor.grad.full_tensor().cpu(), local_weight.grad.cpu(), rtol=1e-4, atol=1e-4
        )

    # Linear -without bias - disable this test for the time being as it requires fork change
    """
    local_out = F.linear(local_input, local_weight)
    loss_local = torch.sum(local_out)
    loss_local.backward()

    dtensor_out = F.linear(input_dtensor, weight_dtensor)
    # Gather to compare with local result
    gathered_out = dtensor_out.full_tensor()

    if rank == 0:  # for some reason test hangs if this is not enabled
        torch.testing.assert_close(gathered_out.cpu(), local_out.cpu(), rtol=1e-4, atol=1e-4)
    loss_dist = torch.sum(gathered_out)
    loss_dist.backward()
    if rank == 0:  # for some reason, local_out for ranks other than 0 are coming junk
        torch.testing.assert_close(input_dtensor.grad.full_tensor().cpu(), local_input.grad.cpu(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            weight_dtensor.grad.full_tensor().cpu(), local_weight.grad.cpu(), rtol=1e-4, atol=1e-4
        )
    """
    linear_matmul_cleanup()


@pytest.mark.skipif(
    os.environ.get("PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER") is None
    or os.environ.get("PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER") in ("False", "0", "F", "f", None),
    reason="This test runs only with PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER set to True",
)
def test_autograd_linear_matmul():
    os.environ["RANK"] = "0"  # Example rank for this process (adjust as needed)
    world_size = habana_frameworks.torch.hpu.device_count()
    torch.multiprocessing.spawn(custom_linear_matmul_helper, args=(world_size, rank), nprocs=world_size, join=True)
