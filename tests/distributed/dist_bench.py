#!/usr/bin/env python
import os
import timeit

import torch
import torch.distributed as dist
from habana_frameworks.torch.utils.library_loader import load_habana_module
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT

load_habana_module()


def _init_global_test():
    group = list(range(0, dist.get_world_size()))
    group_id = dist.group.WORLD
    rank = dist.get_rank()
    return (group, group_id, rank)


def setup_HCL(sim=True, size=2):
    world_size = size
    os.environ["WORLD_SIZE"] = str(world_size)
    pwd_sim = os.getcwd()
    sim_file = pwd_sim + "/sim.json"
    os.environ["HCL_CONFIG_PATH"] = sim_file
    os.environ["JSON"] = sim_file
    config_file = open(sim_file, "w")
    file_cont = '{\n"HCL_PORT": 5332,\n'
    if sim == True:
        file_cont += '"HCL_TYPE": "BACK_2_BACK",\n'
        file_cont += '"DISABLED_PORTS": [0,1,2,3,5,6,7,8,9],\n'
    else:
        file_cont += '"HCL_TYPE": "HLS1",\n'

    file_cont += '"HCL_COUNT": ' + str(world_size) + "\n}"
    config_file.write(file_cont)
    config_file.close()

    os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["HLS_MODULE_ID"] = os.environ["RANK"]


def dummy_bcast():
    group, group_id, rank = _init_global_test()
    IN = torch.tensor(10, dtype=torch.float).to("hpu")
    dist.broadcast(IN, 0, group_id)


def send_recv(size, steps, stsize):
    print("length   time/message (sec)    transfer rate (M byte/sec)")
    for length in range(stsize, size + 1, 1):
        num_bytes = pow(2, length)
        IN = torch.tensor((), dtype=torch.uint8)
        IN = IN.new_ones(num_bytes).to("hpu")
        dummy_bcast()
        start = timeit.default_timer()
        for i in range(1, steps + 1):
            dist.send(IN, 1)
            dist.recv(IN, 1)
        end = timeit.default_timer()
        time = end - start
        print(
            "2^{}          {:.4f}               {:.4f}".format(
                length, time / (steps), ((2.0 * steps * num_bytes) / time) / (1024 * 1024)
            )
        )


def recv_send(size, steps, stsize):
    for length in range(stsize, size + 1, 1):
        num_bytes = pow(2, length)
        IN = torch.tensor((), dtype=torch.uint8)
        IN = IN.new_ones(num_bytes).to("hpu")
        dummy_bcast()
        for i in range(1, steps + 1):
            dist.recv(IN, 0)
            dist.send(IN, 0)


def perf_all_reduce(args):
    group, group_id, rank = _init_global_test()
    if rank == 0:
        print("AllReduce test")
        print("length   time/message (sec)    transfer rate (M byte/sec)")
    steps = int(args.s)
    size = int(args.size)
    strts = int(args.startsize)
    for length in range(strts, size + 1, 1):
        num_bytes = pow(2, length)
        IN = torch.tensor((), dtype=torch.float)
        IN = IN.new_ones(int(num_bytes / 4)).to("hpu")
        start = timeit.default_timer()
        for i in range(1, steps + 1):
            dist.all_reduce(IN, dist.ReduceOp.SUM, group_id)
        end = timeit.default_timer()
        time = end - start
        if rank == 0:
            print(
                "2^{}          {:.4f}               {:.4f}".format(
                    length, time / steps, ((steps * num_bytes) / time) / (1024 * 1024)
                )
            )


def perf_send_receive(args):
    group, group_id, rank = _init_global_test()
    if rank == 0:
        send_recv(int(args.size), int(args.s), int(args.startsize))
        dummy_bcast()
    elif rank == 1:
        recv_send(int(args.size), int(args.s), int(args.startsize))
        dummy_bcast()
    else:
        dummy_bcast()


def perf_broadcast(args):
    group, group_id, rank = _init_global_test()
    if rank == 0:
        print("Broadcast test")
        print("length   time/message (sec)    transfer rate (M byte/sec)")
    steps = int(args.s)
    size = int(args.size)
    strts = int(args.startsize)
    for length in range(strts, size + 1, 1):
        num_bytes = pow(2, length)
        IN = torch.tensor((), dtype=torch.float)
        IN = IN.new_ones(int(num_bytes / 4)).to("hpu")
        start = timeit.default_timer()
        for i in range(1, steps + 1):
            dist.broadcast(IN, 0, group_id)
        end = timeit.default_timer()
        time = end - start
        if rank == 0:
            print(
                "2^{}          {:.4f}               {:.4f}".format(
                    length, time / steps, ((steps * num_bytes) / time) / (1024 * 1024)
                )
            )


def perf_test(args, sim=True):
    setup_HCL(sim, int(args.np))
    rank = int(os.environ["RANK"])
    dist.init_process_group("hcl", rank=rank, world_size=int(args.np))
    if rank == 0:
        print("pyTorch tests")
    if args.test == "sendrecv":
        perf_send_receive(args)
    elif args.test == "allreduce":
        perf_all_reduce(args)
    elif args.test == "broadcast":
        perf_broadcast(args)

    dist.destroy_process_group()


def perf_test_cpp(args, sim=True):
    setup_HCL(sim, int(args.np))
    exec_test = "./hcl_bench "
    exec_test += args.np + " " + args.test + " " + args.size + " " + args.startsize + " " + args.s
    os.system(exec_test)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Bandwidth test")
    parser.add_argument("--test", default="sendrecv", help="sendrecv, allreduce, broadcast")
    parser.add_argument("--size", default="30", help="Length of tests")
    parser.add_argument("--startsize", default="10", help="Length of tests")
    parser.add_argument("--np", default="2", help="Number of process")
    parser.add_argument("--s", default="100", help="Number of times each call is called to calculate average time")
    parser.add_argument("--run", default="both", help="py-> python, cpp->cpp, both->run both ")
    parser.add_argument("--sim", default=False, help="Run on Simulator")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.run == "both":
        perf_test_cpp(args=args, sim=args.sim)
        perf_test(args=args, sim=args.sim)
    elif args.run == "cpp":
        perf_test_cpp(args=args, sim=args.sim)
    elif args.run == "py":
        perf_test(args=args, sim=args.sim)
