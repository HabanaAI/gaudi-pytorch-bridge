#!/usr/bin/env python
import os

import numpy
import torch
import torch.distributed as dist
from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


def check_res(func, rank, out, exp):
    result = "Failed"
    if isinstance(out, list):
        for i in range(len(out)):
            if torch.all(out[i].to("cpu").eq(exp[i])):
                result = "Passed"
            else:
                result = "Failed"
                break
    else:
        if torch.all(out.to("cpu").eq(exp)):
            result = "Passed"
    print("{2} {3} : Rank {0} : {1} ".format(rank, result, func, exp.dtype))


def alltoallv(rank):

    from array import array

    numprocs = world_size
    MAX_MSG_SIZE = 2
    allocate = lambda n: array("i", [0]) * n
    sendbuf = allocate(MAX_MSG_SIZE * numprocs)
    recvbuf = allocate(MAX_MSG_SIZE * numprocs)

    for i in range(MAX_MSG_SIZE * numprocs):
        sendbuf[i] = 2147483637 + rank
        recvbuf[i] = -1

    array_int = lambda n: array("i", [0] * n)
    s_counts = array_int(numprocs)
    s_displs = array_int(numprocs)
    r_counts = array_int(numprocs)
    r_displs = array_int(numprocs)

    disp = 0
    for i in range(world_size):
        s_counts[i] = r_counts[i] = MAX_MSG_SIZE
        s_displs[i] = r_displs[i] = disp
        disp = disp + MAX_MSG_SIZE

    s_msg = [sendbuf, (s_counts, s_displs)]
    r_msg = [recvbuf, (r_counts, r_displs)]

    comm.Alltoallv(s_msg, r_msg)
    ip_tensor = torch.IntTensor(sendbuf).to("hpu")
    op_tensor = torch.IntTensor(recvbuf).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor, r_counts, s_counts)
    check_res("alltoallv ", rank, op_tensor, torch.IntTensor(recvbuf))

    ip_tensor = torch.LongTensor(sendbuf).to("hpu")
    op_tensor = torch.LongTensor(recvbuf).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor, r_counts, s_counts)
    check_res("alltoallv ", rank, op_tensor, torch.LongTensor(recvbuf))


def alltoall(rank):
    a_size = 3
    input = (rank + 1) * numpy.arange(a_size * world_size, dtype=int)
    output = numpy.empty(world_size * a_size, dtype=int)
    exp = numpy.empty(world_size * a_size, dtype=int)

    comm.Alltoall(input, exp)

    ip_tensor = torch.IntTensor(input).to("hpu")
    op_tensor = torch.IntTensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.IntTensor(exp))

    ip_tensor = torch.ByteTensor(input).to("hpu")
    op_tensor = torch.ByteTensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.ByteTensor(exp))

    ip_tensor = torch.LongTensor(input).to("hpu")
    op_tensor = torch.LongTensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.LongTensor(exp))

    ip_tensor = torch.BFloat16Tensor(input).to("hpu")
    op_tensor = torch.BFloat16Tensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.BFloat16Tensor(exp))

    ip_tensor = torch.FloatTensor(input).to("hpu")
    op_tensor = torch.FloatTensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.FloatTensor(exp))

    ip_tensor = torch.DoubleTensor(input).to("hpu")
    op_tensor = torch.DoubleTensor(output).to("hpu")
    dist.all_to_all_single(op_tensor, ip_tensor)
    check_res("alltoall ", rank, op_tensor, torch.DoubleTensor(exp))


def broadcast(rank):
    group_id = dist.group.WORLD
    output = [1, 2, 3, 4, 5]
    if rank == 0:
        input = [1, 2, 3, 4, 5]
    else:
        input = [0, 0, 0, 0, 0]

    h_t_input = torch.ByteTensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.ByteTensor(output))

    h_t_input = torch.LongTensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.LongTensor(output))

    h_t_input = torch.IntTensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.IntTensor(output))

    h_t_input = torch.BFloat16Tensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.BFloat16Tensor(output))

    h_t_input = torch.FloatTensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.FloatTensor(output))

    h_t_input = torch.DoubleTensor(input).to("hpu")
    dist.broadcast(h_t_input, 0, group_id)
    check_res("Broadcast ", rank, h_t_input, torch.DoubleTensor(output))


def send_recv(rank):
    input = [1, 2, 3, 4, 5]
    output = [0, 0, 0, 1, 0]
    if rank == 0:
        IN2 = torch.ByteTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.ByteTensor(input))

    if rank == 1:
        IN = torch.ByteTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.LongTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.LongTensor(input))

    if rank == 1:
        IN = torch.LongTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.FloatTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.FloatTensor(input))

    if rank == 1:
        IN = torch.FloatTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.IntTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.IntTensor(input))

    if rank == 1:
        IN = torch.IntTensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.BFloat16Tensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.BFloat16Tensor(input))

    if rank == 1:
        IN = torch.BFloat16Tensor(input).to("hpu")
        dist.send(IN, 0)

    if rank == 0:
        IN2 = torch.DoubleTensor(output).to("hpu")
        dist.recv(IN2, 1)
        check_res("send recv", rank, IN2, torch.DoubleTensor(input))

    if rank == 1:
        IN = torch.DoubleTensor(input).to("hpu")
        dist.send(IN, 0)


def all_gather(rank, world_size):
    output = [0, 0, 0, 0, 0]
    input = [1 * rank, 2 * rank, 3 * rank, 4 * rank, 5 * rank]
    exp = [[1 * i, 2 * i, 3 * i, 4 * i, 5 * i] for i in range(world_size)]

    op_tensor_list = [torch.ByteTensor(output).to("hpu") for i in range(world_size)]
    a = torch.ByteTensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.ByteTensor(exp))

    op_tensor_list = [torch.BFloat16Tensor(output).to("hpu") for i in range(world_size)]
    a = torch.BFloat16Tensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.BFloat16Tensor(exp))

    op_tensor_list = [torch.FloatTensor(output).to("hpu") for i in range(world_size)]
    a = torch.FloatTensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.FloatTensor(exp))

    op_tensor_list = [torch.tensor(output, dtype=torch.float8_e5m2).to("hpu") for i in range(world_size)]
    a = torch.tensor(input, dtype=torch.float8_e5m2).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.tensor(exp, dtype=torch.float8_e5m2))

    op_tensor_list = [torch.tensor(output, dtype=torch.float8_e4m3fn).to("hpu") for i in range(world_size)]
    a = torch.tensor(input, dtype=torch.float8_e4m3fn).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.tensor(exp, dtype=torch.float8_e4m3fn))

    op_tensor_list = [torch.DoubleTensor(output).to("hpu") for i in range(world_size)]
    a = torch.DoubleTensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.DoubleTensor(exp))

    op_tensor_list = [torch.IntTensor(output).to("hpu") for i in range(world_size)]
    a = torch.IntTensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.IntTensor(exp))

    op_tensor_list = [torch.LongTensor(output).to("hpu") for i in range(world_size)]
    a = torch.LongTensor(input).to("hpu")
    dist.all_gather(op_tensor_list, a)
    check_res("all gather", rank, op_tensor_list, torch.LongTensor(exp))


def all_gather_into_tensor(rank, world_size):
    output = [0, 0, 0, 0, 0] * world_size
    input = [1 * rank, 2 * rank, 3 * rank, 4 * rank, 5 * rank]
    exp_list = [[1 * i, 2 * i, 3 * i, 4 * i, 5 * i] for i in range(world_size)]
    exp = []
    for e in exp_list:
        exp.extend(e)

    op_tensor_list = torch.BFloat16Tensor(output).to("hpu")
    a = torch.BFloat16Tensor(input).to("hpu")
    dist.all_gather_into_tensor(op_tensor_list, a)
    check_res("all gather into tensor", rank, op_tensor_list, torch.BFloat16Tensor(exp))

    op_tensor_list = torch.FloatTensor(output).to("hpu")
    a = torch.FloatTensor(input).to("hpu")
    dist.all_gather_into_tensor(op_tensor_list, a)
    check_res("all gather into tensor", rank, op_tensor_list, torch.FloatTensor(exp))

    op_tensor_list = torch.tensor(output, dtype=torch.float8_e5m2).to("hpu")
    a = torch.tensor(input, dtype=torch.float8_e5m2).to("hpu")
    dist.all_gather_into_tensor(op_tensor_list, a)
    check_res("all gather into tensor", rank, op_tensor_list, torch.tensor(exp, dtype=torch.float8_e5m2))

    op_tensor_list = torch.tensor(output, dtype=torch.float8_e4m3fn).to("hpu")
    a = torch.tensor(input, dtype=torch.float8_e4m3fn).to("hpu")
    dist.all_gather_into_tensor(op_tensor_list, a)
    check_res("all gather into tensor", rank, op_tensor_list, torch.tensor(exp, dtype=torch.float8_e4m3fn))

    op_tensor_list = torch.CharTensor(output).to("hpu")
    a = torch.CharTensor(input).to("hpu")
    dist.all_gather_into_tensor(op_tensor_list, a)
    check_res("all gather into tensor", rank, op_tensor_list, torch.CharTensor(exp))


myhost = os.uname()[1]


os.environ["WORLD_SIZE"] = str(world_size)
# Bridge is using "HLS_MODULE_ID", but "ID" is still needed for internal synapse logging.
os.environ["ID"] = str(rank)
os.environ["HLS_MODULE_ID"] = str(rank)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(rank)

dist.init_process_group("hccl", rank=rank, world_size=world_size)

all_gather(rank, world_size)
all_gather_into_tensor(rank, world_size)
broadcast(rank)
send_recv(rank)
alltoall(rank)
alltoallv(rank)
