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


import pytest
import torch
import torch.distributed._functional_collectives as funcol
from test_utils import is_torch_at_least
from torch.distributed._tensor import Replicate, Shard, distribute_tensor, init_device_mesh
from torch.distributed._tensor.experimental import local_map
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

if is_torch_at_least("2.5"):
    from torch.distributed.tensor.debug import CommDebugMode
else:
    from torch.distributed._tensor.debug import CommDebugMode

funcol_py = torch.ops.c10d_functional


def check_devices():
    return torch.cuda.device_count() > 1


def equal_allreduce_forward(device_mesh, X, Y):
    eq = torch.tensor([torch.equal(X, Y)], device=X.device)
    eq_gather = funcol.all_reduce(eq, "sum", device_mesh)
    return torch.all(eq_gather).item()


def equal_allgather_forward(device_mesh, X, Y):
    eq = torch.tensor([torch.equal(X, Y)], device=X.device)
    eq_gather = funcol.all_gather_tensor(eq, 0, device_mesh)
    return torch.all(eq_gather).item()


def mm_all_gather_forward(device_mesh, A, B):
    local_mm_result = torch.mm(A, B)
    return funcol.all_gather_tensor(local_mm_result, 0, device_mesh).wait()


class TestLocalMap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(check_devices(), reason="")
    @with_comms
    def test_local_map_out_placements_allreduce(self):
        if torch.cuda.device_count() < self.world_size:
            return

        device_mesh = init_device_mesh(device_type=self.device_type, mesh_shape=(self.world_size,))
        comm_mode = CommDebugMode()

        # X.equal(Y)
        X = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        Y = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        row_wise = [Shard(0)]
        X_dt = distribute_tensor(X, device_mesh, row_wise)
        Y_dt = distribute_tensor(Y, device_mesh, row_wise)
        local_equal_allgather_forward = local_map(
            equal_allreduce_forward,  # equal_allgather_forward,
            out_placements=None,
        )
        with comm_mode:
            equal_dt = local_equal_allgather_forward(device_mesh, X_dt, Y_dt)  # a bool

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertTrue(not equal_dt)
        self.assertTrue(not (X.equal(Y)))

        # Test 2: directly return out if no argument is DTensor
        # matmul in DDP
        replicate = [Replicate()]
        X = torch.randn(4 // self.world_size, 4, device=self.device_type, requires_grad=False)
        W = torch.randn(4, 4, device=self.device_type, requires_grad=False)
        local_mm_all_gather_forward = local_map(
            mm_all_gather_forward,
            out_placements=row_wise,
            in_placements=(None, row_wise, replicate),
        )
        with comm_mode:
            Y = local_mm_all_gather_forward(device_mesh, X, W)

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_mode.get_comm_counts()[funcol_py.all_gather_into_tensor], 1)
        X_replicate = funcol.all_gather_tensor(X, 0, device_mesh).wait()
        Y_replicate = torch.mm(X_replicate, W)
        self.assertEqual(Y, Y_replicate)  # Y is a torch.Tensor

    @pytest.mark.skipif(check_devices(), reason="")
    @with_comms
    def test_local_map_out_placements_allgather(self):

        # Test 1: wrap out into DTensor w/ `out_placements`
        device_mesh = init_device_mesh(device_type=self.device_type, mesh_shape=(self.world_size,))
        comm_mode = CommDebugMode()

        # X.equal(Y)
        X = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        Y = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        row_wise = [Shard(0)]
        X_dt = distribute_tensor(X, device_mesh, row_wise)
        Y_dt = distribute_tensor(Y, device_mesh, row_wise)
        local_equal_allgather_forward = local_map(
            equal_allgather_forward,
            out_placements=None,
        )
        with comm_mode:
            equal_dt = local_equal_allgather_forward(device_mesh, X_dt, Y_dt)  # a bool

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertTrue(not equal_dt)
        self.assertTrue(not (X.equal(Y)))

        # Test 2: directly return out if no argument is DTensor
        # matmul in DDP
        replicate = [Replicate()]
        X = torch.randn(4 // self.world_size, 4, device=self.device_type, requires_grad=False)
        W = torch.randn(4, 4, device=self.device_type, requires_grad=False)
        local_mm_all_gather_forward = local_map(
            mm_all_gather_forward,
            out_placements=row_wise,
            in_placements=(None, row_wise, replicate),
        )
        with comm_mode:
            Y = local_mm_all_gather_forward(device_mesh, X, W)

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_mode.get_comm_counts()[funcol_py.all_gather_into_tensor], 1)
        X_replicate = funcol.all_gather_tensor(X, 0, device_mesh).wait()
        Y_replicate = torch.mm(X_replicate, W)
        self.assertEqual(Y, Y_replicate)  # Y is a torch.Tensor


if __name__ == "__main__":
    run_tests()
