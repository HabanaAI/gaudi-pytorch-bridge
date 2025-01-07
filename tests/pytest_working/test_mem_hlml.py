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

import mmap
import os
import struct
import time

import habana_frameworks.torch as ht
import torch

SHM_PREFIX_PATH = "/dev/shm"
SHM_SIZE = 3 * 8  # 3 * uint64_t
SHM_VERSION = 1


class SharedObject:
    def __init__(self):
        so_name = ht.hpu.memory._get_hlml_shared_object_name()
        if so_name.startswith("/"):
            so_name = so_name[1:]
        self._path = os.path.join(SHM_PREFIX_PATH, so_name)

    def __enter__(self):
        with open(self._path, "rb") as fd:
            self._mem = mmap.mmap(fd.fileno(), 3 * 8, prot=mmap.PROT_READ)
        return self

    def __exit__(self, _exv, _extp, _extb):
        self._mem.close()

    def read_values(self):
        return struct.unpack("QQQ", self._mem[:])

    def read_timestamp(self):
        return self.read_values()[1]

    def read_memory(self):
        return self.read_values()[2]


def test_hlml_created():
    # provoke initialization
    torch.Tensor([1]).to("hpu")
    with SharedObject() as so:
        version, _timestamp, _value = so.read_values()
        assert version == SHM_VERSION


def test_hlml_timestamp_is_updated():
    # provoke initialization
    torch.Tensor([1]).to("hpu")
    with SharedObject() as so:
        timestamp0 = so.read_timestamp()
        time.sleep(3)
        timestamp1 = so.read_timestamp()
        assert timestamp0 < timestamp1


def test_hlml_memory_is_updated():
    # provoke initialization
    torch.Tensor([1]).to("hpu")

    SIZE = 1024 * 1024 * 2

    with SharedObject() as so:
        memory0 = so.read_memory()
        t0 = torch.zeros(SIZE).to("hpu")
        t1 = torch.zeros(SIZE).to("hpu")
        time.sleep(3)
        memory1 = so.read_memory()
        assert (memory0 + 2 * SIZE) <= memory1
        del t0
        time.sleep(3)
        memory2 = so.read_memory()
        assert memory2 <= (memory1 - SIZE)
        assert (memory0 + SIZE) <= memory2
