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


from __future__ import print_function

import os

import pytest
import torch

device = torch.device("hpu")


def check_data_pointers(_dp1, _dp2):
    print("dp1 :: ", _dp1)
    print("\ndp2 :: ", _dp2)
    if _dp1 == _dp2:
        print("\n***memory reused successfully***\n")
        return True
    else:
        print("\nmemory not reused !!\n")
        return False


# RuntimeError: unsupported Storage type
def shared_memory(device):
    hpu_tensor_1 = torch.randn(3, 3).to(device)
    tensor_a = hpu_tensor_1.share_memory_()
    print("tensor 1 shared = ", tensor_a.is_shared())


def device_tensor_create(device):
    hpu_tensor_1 = torch.randn((3, 3), device=device)
    # hpu_tensor_1.fill_(0)
    print(hpu_tensor_1)
    hpu_tensor_2 = torch.randn((3, 3), device=device)
    print(hpu_tensor_2)


def tensor_create(device, pool_id):
    hpu_tensor_1 = torch.randn(8, 8).to(device)
    dp1 = hpu_tensor_1.data_ptr()
    print(hpu_tensor_1.to("cpu"))
    del hpu_tensor_1

    hpu_tensor_2 = torch.randn(9, 9).to(device)
    dp2 = hpu_tensor_2.data_ptr()
    print(hpu_tensor_2.to("cpu"))

    # memory should not be reused due to bigger dp5 size
    if pool_id != "5" and pool_id != "0":
        assert check_data_pointers(dp1, dp2) is False

    hpu_tensor_3 = torch.randn(8, 8).to(device)
    dp3 = hpu_tensor_3.data_ptr()
    print(hpu_tensor_3.to("cpu"))

    # dp1 memory must be reused
    if pool_id != "5" and pool_id != "0":
        assert check_data_pointers(dp1, dp3) is True

    hpu_tensor_4 = torch.randn(3, 3).to(device)
    dp4 = hpu_tensor_4.data_ptr()
    print(hpu_tensor_4.to("cpu"))
    del hpu_tensor_4

    hpu_tensor_5 = torch.randn(3, 3).to(device)
    dp5 = hpu_tensor_5.data_ptr()
    print(hpu_tensor_5.to("cpu"))

    # dp4 memory must be reused
    if pool_id != "5" and pool_id != "0":
        assert check_data_pointers(dp5, dp4) is True

    hpu_tensor_6 = torch.randn(3, 3).to(device)
    dp6 = hpu_tensor_6.data_ptr()
    print(hpu_tensor_6.to("cpu"))

    # dp6 memory must be a new block
    if pool_id != "5" and pool_id != "0":
        assert check_data_pointers(dp6, dp4) is False


def pool_exhaust(device, pool_id):
    gigabyte = 1000 * 1000 * 1000
    pool_size = os.environ.get("PT_HPU_POOL_SIZE")
    print("pool_size :: ", pool_size)

    if pool_id == "1":
        allocated_size = 1
        index = 0
        hpu_tensor_list = []
        pool_sz = 0
        if pool_size == 0 or pool_size is None:
            pool_sz = 1 * gigabyte
        else:
            pool_sz = pool_sz * gigabyte
        print("test bump pooling with pool size :: ", pool_sz)
        while allocated_size < (pool_sz - (pool_sz % allocated_size)):
            hpu_tensor_A = torch.randn(10000, 10000).to(device)
            hpu_tensor_list.append(hpu_tensor_A)
            tensor_size = hpu_tensor_A.element_size() * hpu_tensor_A.nelement()
            allocated_size = allocated_size + tensor_size
            print("tensor size :: ", tensor_size)
            index = index + 1

        print("allocated_size :: ", allocated_size)
        print("total blocks :: ", len(hpu_tensor_list))
        dp0 = hpu_tensor_list[0].data_ptr()
        del hpu_tensor_list[0]
        hpu_tensor_reuse = torch.randn(10000, 10000).to(device)
        dp1 = hpu_tensor_reuse.data_ptr()

        # dp0 memory must be reused
        assert check_data_pointers(dp0, dp1) is True

    elif pool_id == "2":
        print("test dynamic pooling")
        hpu_tensor_B = torch.randn(10000, 10000).to(device)
        dpB = hpu_tensor_B.data_ptr()
        del hpu_tensor_B
        hpu_tensor_C = torch.randn(10000, 10000).to(device)
        dpC = hpu_tensor_C.data_ptr()

        # dp0 memory must be reused
        assert check_data_pointers(dpB, dpC) is True
    else:
        print("no pooling")


def is_aligned(dataptr):
    if (dataptr % 128) == 0:
        return True
    else:
        return False


def is_contiguous(dp1, size, dp2):
    alignedsize = (size + 128 - 1) // 128 * 128
    print("in size :: ", size)
    print("alignedsize :: ", alignedsize)
    if (dp1 + alignedsize) == dp2:
        return True
    else:
        return False


def check_alignment(device, pool_id):
    hpu_tensor_1 = torch.randn(3, 3).to(device)
    tensor_size_1 = hpu_tensor_1.element_size() * hpu_tensor_1.nelement()
    dp_1 = hpu_tensor_1.data_ptr()
    print("t1 size :: ", tensor_size_1)
    print("t1 :: ", dp_1)
    assert is_aligned(dp_1) is True

    hpu_tensor_2 = torch.randn(7, 7).to(device)
    tensor_size_2 = hpu_tensor_2.element_size() * hpu_tensor_2.nelement()
    dp_2 = hpu_tensor_2.data_ptr()
    print("t2 size :: ", tensor_size_2)
    print("t2 :: ", dp_2)
    assert is_aligned(dp_2) is True
    if pool_id != "5":
        assert is_contiguous(dp_1, tensor_size_1, dp_2) is True
    else:
        small_chunk_size = 2097152
        assert is_contiguous(dp_1, small_chunk_size, dp_2) is True


def pool_coalesce(device, pool_id):

    gigabyte = 1000 * 1000 * 1000
    pool_size = os.environ.get("PT_HPU_POOL_SIZE")
    print("pool_size :: ", pool_size)

    if pool_id == "3":
        allocated_size = 1
        index = 0
        hpu_tensor_list = []
        if pool_size is None:
            pool_sz = 1 * gigabyte
        else:
            pool_sz = pool_sz * gigabyte
        print("pool size :: ", pool_sz)
        while allocated_size < pool_sz:
            # print ("tensor :: ", index)
            hpu_tensor_A = torch.randn(30000, 1000).to(device)
            hpu_tensor_list.append(hpu_tensor_A)
            tensor_size = hpu_tensor_A.element_size() * hpu_tensor_A.nelement()
            allocated_size = allocated_size + tensor_size
            print("tensor size :: ", tensor_size)
            index = index + 1
            if index == 8:
                break

        print("allocated_size :: ", allocated_size)
        print("total blocks :: ", len(hpu_tensor_list))
        print("tensor list")
        for i in range(len(hpu_tensor_list)):
            print("tensor ", index, " :: ", hpu_tensor_list[i].data_ptr())

        del hpu_tensor_list


def pool_coalesce_stringent(device, pool_id):

    gigabyte = 1024 * 1024 * 1024
    pool_size = os.environ.get("PT_HPU_POOL_SIZE")
    print("pool_size :: ", pool_size)

    if pool_id == "5":
        allocated_size = 1
        index = 0
        hpu_tensor_list = []
        if pool_size is None:
            pool_size = 1 * gigabyte
        pool_sz = int(pool_size)
        if pool_size == "0":
            pool_sz = 1 * gigabyte
        else:
            pool_sz = pool_sz * gigabyte
        print("pool size :: ", pool_sz)
        while allocated_size < pool_sz:
            # print ("tensor :: ", index)
            hpu_tensor_A = torch.randn(30000, 1000).to(device)
            hpu_tensor_list.append(hpu_tensor_A)
            tensor_size = hpu_tensor_A.element_size() * hpu_tensor_A.nelement()
            allocated_size = allocated_size + tensor_size
            print("tensor size :: ", tensor_size)
            index = index + 1
            if index == 8:
                break

        print("allocated_size :: ", allocated_size)
        print("total blocks :: ", len(hpu_tensor_list))
        print("tensor list")
        for i in range(len(hpu_tensor_list)):
            print("tensor ", index, " :: ", hpu_tensor_list[i].data_ptr())

        # dp0 = hpu_tensor_list[0].data_ptr()
        # print(hpu_tensor_list[2].data_ptr())
        # print(hpu_tensor_list[3].data_ptr())
        del hpu_tensor_list
        # del(hpu_tensor_list[3])

        torch.randn(30000, 2000).to(device)
        # dp1 = hpu_tensor_reuse.data_ptr()
        # print(hpu_tensor_list[4].data_ptr())
        # del(hpu_tensor_list[4])
        # del(hpu_tensor_list[5])
        torch.randn(20000, 1000).to(device)
        # hpu_tensor_reuse2 = torch.randn(20000, 1000).to(device)
        # dp0 memory must be reused
        # assert(check_data_pointers(dp0, dp1) is True)


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize("setup_teardown_env_fixture", [{"PT_HPU_POOL_STRATEGY": "5"}], indirect=True)
def test_memory_pool(setup_teardown_env_fixture):
    pool_used = os.environ.get("PT_HPU_POOL_STRATEGY")
    if pool_used != "5":
        check_alignment(device, pool_used)
    tensor_create(device, pool_used)
    pool_exhaust(device, pool_used)
    pool_coalesce(device, pool_used)
    pool_coalesce_stringent(device, pool_used)
