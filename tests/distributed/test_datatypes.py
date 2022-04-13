#!/usr/bin/env python
import os
import torch
import sys
import torch.distributed as dist
import numpy
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from mpi4py import MPI
from habana_frameworks.torch.utils.library_loader import load_habana_module


def check_res(func,rank,out,exp):
        result = "Failed"
        if isinstance(out, list):
            for i in range(len(out)):
                if torch.all(out[i].to('cpu').eq(exp[i])):
                    result = "Passed"
                else:
                    result = "Failed"
                    break
        else:
            if torch.all(out.to('cpu').eq(exp)):
                result = "Passed"
        print("{2} {3} : Rank {0} : {1} ".format(rank,result,func,exp.dtype))

def alltoall(rank):
    input = [1*rank,2*rank,3*rank,4*rank,5*rank,6*rank,7*rank,8*rank]
    output = [0]*8
    exp = [[0]*8]*8
    for i in range(8):
        inp = [1*i,2*i,3*i,4*i,5*i,6*i,7*i,8*i]
        exp[i] = inp

    exp = numpy.transpose(exp)

    ip_tensor = torch.IntTensor(input).to('hpu')
    op_tensor = torch.IntTensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.IntTensor(exp[rank]))

    ip_tensor = torch.ByteTensor(input).to('hpu')
    op_tensor = torch.ByteTensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.ByteTensor(exp[rank]))

    ip_tensor = torch.LongTensor(input).to('hpu')
    op_tensor = torch.LongTensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.LongTensor(exp[rank]))

    ip_tensor = torch.BFloat16Tensor(input).to('hpu')
    op_tensor = torch.BFloat16Tensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.BFloat16Tensor(exp[rank]))

    ip_tensor = torch.FloatTensor(input).to('hpu')
    op_tensor = torch.FloatTensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.FloatTensor(exp[rank]))

    ip_tensor = torch.DoubleTensor(input).to('hpu')
    op_tensor = torch.DoubleTensor(output).to('hpu')
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ",rank,op_tensor,torch.DoubleTensor(exp[rank]))


def broadcast(rank):
    group_id = dist.group.WORLD
    output = [1,2,3,4,5]
    if rank == 0:
        input = [1,2,3,4,5]
    else:
        input = [0,0,0,0,0]

    h_t_input  = torch.ByteTensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.ByteTensor(output))

    h_t_input  = torch.LongTensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.LongTensor(output))

    h_t_input  = torch.IntTensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.IntTensor(output))

    h_t_input  = torch.BFloat16Tensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.BFloat16Tensor(output))

    h_t_input  = torch.FloatTensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.FloatTensor(output))

    h_t_input  = torch.DoubleTensor(input).to('hpu')
    dist.broadcast(h_t_input,0, group_id)
    check_res("Broadcast ",rank,h_t_input,torch.DoubleTensor(output))


def send_recv(rank):
    input = [1,2,3,4,5]
    output = [0,0,0,1,0]
    if rank == 0:
        IN2 = torch.ByteTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.ByteTensor(input))

    if rank == 1:
        IN = torch.ByteTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.LongTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.LongTensor(input))

    if rank == 1:
        IN = torch.LongTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.FloatTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.FloatTensor(input))

    if rank == 1:
        IN = torch.FloatTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.IntTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.IntTensor(input))

    if rank == 1:
        IN = torch.IntTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.BFloat16Tensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.BFloat16Tensor(input))

    if rank == 1:
        IN = torch.BFloat16Tensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.DoubleTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv",rank,IN2,torch.DoubleTensor(input))

    if rank == 1:
        IN = torch.DoubleTensor(input).to("hpu")
        dist.send(IN, 0)


def all_gather(rank,world_size):
    output = [0,0,0,0,0]
    input = [1*rank, 2*rank, 3*rank, 4* rank, 5* rank]
    exp = [[1*i, 2*i, 3*i, 4* i, 5* i] for i in range(world_size)]

    op_tensor_list = [torch.ByteTensor(output).to('hpu') for i in range(world_size)]
    a = torch.ByteTensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.ByteTensor(exp))

    op_tensor_list = [torch.BFloat16Tensor(output).to('hpu') for i in range(world_size)]
    a = torch.BFloat16Tensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.BFloat16Tensor(exp))

    op_tensor_list = [torch.FloatTensor(output).to('hpu') for i in range(world_size)]
    a = torch.FloatTensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.FloatTensor(exp))

    op_tensor_list = [torch.DoubleTensor(output).to('hpu') for i in range(world_size)]
    a = torch.DoubleTensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.DoubleTensor(exp))

    op_tensor_list = [torch.IntTensor(output).to('hpu') for i in range(world_size)]
    a = torch.IntTensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.IntTensor(exp))

    op_tensor_list = [torch.LongTensor(output).to('hpu') for i in range(world_size)]
    a = torch.LongTensor(input).to('hpu')
    dist.all_gather(op_tensor_list, a)
    check_res("all gather",rank,op_tensor_list,torch.LongTensor(exp))

myhost = os.uname()[1]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = 8
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["ID"] = str(rank)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "1234"
os.environ["RANK"]=str(rank)
os.environ["LOCAL_RANK"]= str(rank)

import habana_frameworks.torch.distributed.hccl
dist.init_process_group("hccl", rank=rank, world_size=world_size)


all_gather(rank,world_size)
broadcast(rank)
send_recv(rank)
alltoall(rank)



