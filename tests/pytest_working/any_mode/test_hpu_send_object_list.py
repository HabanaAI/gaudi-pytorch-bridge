###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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


device_hpu = torch.device("hpu")


def send_receive_list(rank, world_size, flag):
    device = f"{device_hpu}"
    setuphccl(rank=rank, world_size=world_size)
    if dist.get_rank() == 0:
        objectsent = ["foo", 12, {1: 2}]
        dist.send_object_list(objectsent, dst=1, device=device)
    else:
        object_ref = ["foo", 12, {1: 2}]
        object_received = [None, None, None]
        dist.recv_object_list(object_received, src=0, device=device)
        assert object_received == object_ref


@pytest.mark.skipif(ht.hpu.device_count() < 2, reason="Test Not supported for less tham 2 card.")
def test_send_object_list():
    mp.spawn(send_receive_list, args=(2, True), nprocs=2, join=True)


if __name__ == "__main__":
    test_send_object_list()
