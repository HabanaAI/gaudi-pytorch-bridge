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
import multiprocessing

import pytest
import torch
from packaging.version import Version

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int,
]  # or dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short, torch.uint8, torch.int8]


@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_pin_memory(dtype):
    ifm = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]], dtype=dtype)
    ifm = ifm.pin_memory()
    assert ifm.is_pinned()


@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_empty_pin_memory(dtype):
    ifm = torch.empty([2, 4], dtype=dtype, pin_memory=True)
    assert ifm.is_pinned()


@pytest.mark.skipif(Version(torch.__version__) < Version("2.7"), reason="Fixed in 2.7")
@pytest.mark.skip(reason="Fails when previous tests are run before this")
def test_pin_memory_poison():
    def fork_and_check_is_pinned():
        # Create a pipe to communicate between parent and child processes
        parent_conn, child_conn = multiprocessing.Pipe()

        def worker(conn):
            try:
                x = torch.randn(10)
                x.is_pinned()
                dev = torch.accelerator.current_accelerator()
                x = torch.ones(10, device=dev)[0].item()
                conn.send(x)
            except Exception as e:
                conn.send(str(e))
            finally:
                conn.close()

        # Fork a new process
        p = multiprocessing.Process(target=worker, args=(child_conn,))
        p.start()
        # Receive the result from the child process
        result = parent_conn.recv()
        parent_conn.close()
        # Wait for the child process to finish
        p.join()
        if isinstance(result, str) and result.startswith("Error"):
            raise RuntimeError(result)
        return result

    x = torch.randn(10)
    # check that is_pinned won't poison future fork
    x.is_pinned()
    ret = fork_and_check_is_pinned()
    assert ret == 1.0
