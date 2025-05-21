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

import habana_frameworks.torch as ht
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setuphccl(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    import habana_frameworks.torch.distributed.hccl  # noqa F401

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


device_hpu = torch.device("hpu")


def send_recieve(rank, world_size, contigous_views):
    device = f"{device_hpu}"
    setuphccl(rank=rank, world_size=world_size)

    def local_send(contigous_views):
        if contigous_views:
            _tensor1 = torch.ones(5, 2, 10).to("hpu")
            _tensor = _tensor1.reshape(10, 10)
        else:
            _tensor1 = torch.ones(12, 12).to("hpu")
            _tensor = _tensor1[:10, 1:11]

        torch.distributed.send(_tensor, 1)

    def local_recv():
        _tensor = torch.zeros(10, 10).to("hpu")
        torch.distributed.recv(_tensor, 0)
        return _tensor

    if rank == 0:
        local_send(contigous_views)
    else:
        _tensor = local_recv()
        _tensor_ref = torch.ones(10, 10)
        assert torch.equal(_tensor.cpu(), _tensor_ref)


@pytest.mark.skipif(ht.hpu.device_count() < 2, reason="Test Not supported for less tham 2 card.")
def test_send_recieve():
    mp.spawn(send_recieve, args=(2, True), nprocs=2, join=True)
    mp.spawn(send_recieve, args=(2, False), nprocs=2, join=True)
    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"
    mp.spawn(send_recieve, args=(2, True), nprocs=2, join=True)
    mp.spawn(send_recieve, args=(2, False), nprocs=2, join=True)


if __name__ == "__main__":
    test_send_recieve()
