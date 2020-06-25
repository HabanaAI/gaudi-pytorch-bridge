from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
import time

def check_data_pointers( _dp1, _dp2):
    print("dp1 :: ", _dp1)
    print("\ndp2 :: ", _dp2)
    if(_dp1 == _dp2):
        print("\n***memory reused successfully***\n")
        return True
    else:
        print("\nmemory not reused !!\n")
        return False

#RuntimeError: unsupported Storage type
def shared_memory(device):
    hpu_tensor_1 = torch.randn(3, 3).to(device)
    tensor_a = hpu_tensor_1.share_memory_()
    print("tensor 1 shared = ",tensor_a.is_shared())

def device_tensor_create(device):
    hpu_tensor_1 = torch.randn((3, 3), device = device)
    #hpu_tensor_1.fill_(0)
    print(hpu_tensor_1)
    hpu_tensor_2 = torch.randn((3, 3), device = device)
    print(hpu_tensor_2)

def tensor_create(device):

    hpu_tensor_1 = torch.randn(8, 8).to(device)
    dp1 = hpu_tensor_1.data_ptr()
    print(hpu_tensor_1.to("cpu"))
    del(hpu_tensor_1)

    hpu_tensor_2 = torch.randn(9, 9).to(device)
    dp2 = hpu_tensor_2.data_ptr()
    print(hpu_tensor_2.to("cpu"))

    #memory should not be reused due to bigger dp5 size
    assert(check_data_pointers(dp1, dp2) == False)

    hpu_tensor_3 = torch.randn(8, 8).to(device)
    dp3 = hpu_tensor_3.data_ptr()
    print(hpu_tensor_3.to("cpu"))

    #dp1 memory must be reused
    assert(check_data_pointers(dp1, dp3) == True)

    hpu_tensor_4 = torch.randn(3, 3).to(device)
    dp4 = hpu_tensor_4.data_ptr()
    print(hpu_tensor_4.to("cpu"))
    del(hpu_tensor_4)

    hpu_tensor_5 = torch.randn(3, 3).to(device)
    dp5 = hpu_tensor_5.data_ptr()
    print(hpu_tensor_5.to("cpu"))

    #dp4 memory must be reused
    assert(check_data_pointers(dp5, dp4) == True)

    hpu_tensor_6 = torch.randn(3, 3).to(device)
    dp6 = hpu_tensor_6.data_ptr()
    print(hpu_tensor_6.to("cpu"))

    #dp6 memory must be a new block
    assert(check_data_pointers(dp6, dp4) == False)

def pool_exhaust(device):
    gigabyte = 1000*1000*1000
    pool_used = os.environ.get('ENV_POOL_STRATEGY')
    print("pool_used :: ", pool_used)
    pool_size = os.environ.get('ENV_POOL_SIZE')
    print("pool_size :: ", pool_size)

    if (pool_used == '1'):
        allocated_size = 1
        index = 0
        hpu_tensor_list = []
        pool_sz = 0
        pool_used = "static"
        if (pool_size == "0"):
            pool_sz = 1 * gigabyte
        else:
            pool_sz = pool_sz * gigabyte
        print("test bump pooling with pool size :: ", pool_sz)
        while(allocated_size < (pool_sz - (pool_sz % allocated_size))):
            hpu_tensor_A = torch.randn(10000, 10000).to(device)
            hpu_tensor_list.append(hpu_tensor_A)
            tensor_size = hpu_tensor_A.element_size() * hpu_tensor_A.nelement()
            allocated_size = allocated_size + tensor_size
            print("tensor size :: ", tensor_size)
            index = index + 1

        print("allocated_size :: ", allocated_size)
        print("total blocks :: ", len(hpu_tensor_list))
        dp0 = hpu_tensor_list[0].data_ptr()
        del(hpu_tensor_list[0])
        hpu_tensor_reuse = torch.randn(10000, 10000).to(device)
        dp1 = hpu_tensor_reuse.data_ptr()

        #dp0 memory must be reused
        assert(check_data_pointers(dp0, dp1) == True)

    elif (pool_used == '2'):
        pool_used = "dynamic"
        print("test dynamic pooling")
        hpu_tensor_B = torch.randn(10000, 10000).to(device)
        dpB = hpu_tensor_B.data_ptr()
        del(hpu_tensor_B)
        hpu_tensor_C = torch.randn(10000, 10000).to(device)
        dpC = hpu_tensor_C.data_ptr()

        #dp0 memory must be reused
        assert(check_data_pointers(dpB, dpC) == True)
    else:
        print("no pooling")

def is_aligned(dataptr):
    if((dataptr % 128) == 0):
        return True
    else:
        return False

def is_contiguous(dp1, size, dp2):
    alignedsize = (size + 128 - 1) // 128 * 128
    print("in size :: ",size)
    print("alignedsize :: ",alignedsize)
    if((dp1 + alignedsize) == dp2):
        return True
    else:
        return False

def check_alignment(device):
    hpu_tensor_1 = torch.randn(3, 3).to(device)
    tensor_size_1 = hpu_tensor_1.element_size() * hpu_tensor_1.nelement()
    dp_1 = hpu_tensor_1.data_ptr()
    print("t1 size :: ",tensor_size_1)
    print("t1 :: ",dp_1)
    assert(is_aligned(dp_1) == True)

    hpu_tensor_2 = torch.randn(7, 7).to(device)
    tensor_size_2 = hpu_tensor_2.element_size() * hpu_tensor_2.nelement()
    dp_2 = hpu_tensor_2.data_ptr()
    print("t2 size :: ",tensor_size_2)
    print("t2 :: ",dp_2)
    assert(is_aligned(dp_2) == True)
    assert(is_contiguous(dp_1, tensor_size_1, dp_2) == True)


def main():

    torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
    device = torch.device("habana")

    #tensor_create(device)
    #pool_exhaust(device)
    check_alignment(device)

if __name__ == '__main__':
    main()
