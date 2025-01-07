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


import functools
import logging
import os
import time
from importlib import reload

import habana_frameworks.torch as ht
import multiprocess
import torch
import torch._dynamo
import torch._dynamo as dynamo
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import os

    path = os.environ.get("PYTORCH_MODULES_ROOT_PATH", None)
    # path = "/root/repos/pytorch-integration/"
    if path:
        import sys

        sys.path.append(path)
    from topologies.tools import SynapseProfilerApi, TraceType
except ImportError:
    SynapseProfilerApi = None
    TraceType = None


class HabanaDeviceProfile(object):
    def __init__(self, profiler, device_profiler_step) -> None:
        if device_profiler_step:
            print(
                f"given device_profiler_step = {device_profiler_step}, captured trace will be device_profiler_step = {device_profiler_step + 1}"
            )
        self.profiler = profiler
        self.device_profiler_step = device_profiler_step
        self.cnt = 0
        self.stop_called = False
        self.step_called = False
        self.profile_all = False
        self.profile_all = True

    def start(self):
        if self.step_called or self.cnt or self.profile_all:
            print("start called")
            self.profiler.profiler_start(TraceType.TraceDevice, device_id=0)

    def stop(self):
        if not self.stop_called:
            print("stop called")
            self.profiler.profiler_sync(device_id=0)
            self.profiler.profiler_stop(TraceType.TraceDevice, device_id=0)
            self.profiler.profiler_get_trace_json(TraceType.TraceDevice, device_id=0)
            self.stop_called = True

    def step(self):
        if not self.profile_all:
            if self.cnt + 1 == self.device_profiler_step:
                self.step_called = True
                self.start()
            if self.cnt == self.device_profiler_step:
                self.stop()
        self.cnt += 1


def init_weights(m):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)


class SplitWeightLinear(nn.Module):
    def __init__(self, weight_size, dtype, num_shards):
        super().__init__()
        # weight_size = 8192, 1024
        # with Tx, self.weight is = 1024, 8192
        self.weight = nn.Parameter(
            torch.arange(weight_size[1] * weight_size[0], dtype=dtype).reshape(weight_size[1], weight_size[0])
        )
        # bias size = 8192
        self.bias = nn.Parameter(torch.empty(1, weight_size[0], dtype=dtype).fill_(0.01))
        self.num_shards = num_shards

    def forward(self, inp, dummy_tensor_for_hccl_sync):
        total_shard_size = inp.size()[0]
        shard_size = int(total_shard_size // self.num_shards)
        start_offset = 0
        output_start_offset = 0

        output_list = []

        for i in range(self.num_shards):
            curr_shard_size = shard_size if i < self.num_shards - 1 else total_shard_size - start_offset
            matmul_out = torch.matmul(inp[start_offset : start_offset + curr_shard_size, :, :], self.weight)
            torch.distributed.all_reduce(matmul_out)
            add_out = matmul_out + self.bias
            output_list.append(add_out)
            start_offset = start_offset + curr_shard_size

        output = torch.cat(output_list)

        return output


class FullLinear(nn.Module):
    def __init__(self, weight_size, dtype):
        super().__init__()
        # with Tx, self.weight is = 1024, 8192
        self.weight = nn.Parameter(
            torch.arange(weight_size[1] * weight_size[0], dtype=dtype).reshape(weight_size[1], weight_size[0])
        )
        # bias size = 8192
        self.bias = nn.Parameter(torch.empty(1, weight_size[0], dtype=dtype).fill_(0.01))

    def forward(self, input, dummy_tensor_for_hccl_sync):
        work = torch.distributed.all_reduce(dummy_tensor_for_hccl_sync, async_op=True)
        work.wait()

        output = torch.matmul(input, self.weight)
        work = torch.distributed.all_reduce(output, async_op=True)
        # work.wait()
        return output + self.bias


class ToyModel(nn.Module):
    def __init__(self, weight_size, dtype, num_shards, do_one_shard=None):
        super().__init__()
        if do_one_shard:
            self.weight_layer1 = FullLinear(weight_size, dtype)
        else:
            self.weight_layer1 = SplitWeightLinear(weight_size, dtype, num_shards)

    def forward(self, input, dummy_tensor_for_hccl_sync):
        weight_layer1_out = self.weight_layer1(input, dummy_tensor_for_hccl_sync)
        return weight_layer1_out


def print_config(
    BS,
    iterations,
    world_size,
    num_shards,
    model,
    input_size,
    linear_weight_dim2,
    do_one_shard,
    do_hpu_graph,
    do_host_profile,
    do_device_profile,
):
    print("RUN config")
    print("==========")
    print(f"Batch Size = {BS}")
    print(f"Input size = [Batch, Input seq. length, dim2] = [{BS}, {input_size}, {linear_weight_dim2})")
    print(f"Model weight size = [dim2, hidden size] = {model.weight_layer1.weight.size()}")
    print(f"Model bias size = [hidden size] = {model.weight_layer1.bias.size()}")

    print(f"Number of nodes (world_Size) = {world_size}")

    print(f"Number of shards = {num_shards}")

    print(f"Iterations = {iterations}")

    if do_one_shard:
        print(f"Running in 1 shard without SFG")
    else:
        print(f"Running in {num_shards} shards with SFG")

    if do_hpu_graph:
        print(f"HPU graph is enabled")

    if do_host_profile:
        print("Host profile is enabled from script")

    if do_device_profile:
        print("Device profile is enabled from script")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Llama 70B inference SFG layer 1 UT")

    parser.add_argument("--batch-size", default=12, type=int, help="Batch size (default: 370)")
    parser.add_argument("--iterations", default=5, type=int, help="Number of iterations (default: 5)")
    parser.add_argument(
        "--world-size", default=2, type=int, help="Number of nodes, represents TP size in model (default: 4)"
    )
    parser.add_argument("--shards", default=4, type=int, help="Number of shards (default: 4)")
    parser.add_argument("--run-one-shard", dest="do_one_shard", action="store_true", help="Set to True to run 1 shard")
    parser.add_argument("--hidden-dim", default=12, type=int, help="Hidden dimension (default: 8192)")
    parser.add_argument("--input-length", default=16, type=int, help="Input seqence length (default: 2048)")
    parser.add_argument("--weight-dim2", default=8, type=int, help="Weight dim 2 (default: 1024)")
    parser.add_argument(
        "--host-profile", dest="do_host_profile", action="store_true", help="Set to True for host profile collection"
    )
    parser.add_argument(
        "--device-profile",
        dest="do_device_profile",
        action="store_true",
        help="Set to True for device profile collection",
    )
    parser.add_argument("--hpu-graph", dest="do_hpu_graph", action="store_true", help="Set to True to enable HPU graph")

    args = parser.parse_args()
    return args


def run_single_node(rank, *arguments):
    args = arguments[0]
    world_size = args.world_size
    do_host_profile = args.do_host_profile
    do_device_profile = args.do_device_profile
    do_hpu_graph = args.do_hpu_graph
    num_shards = args.shards
    do_one_shard = args.do_one_shard

    os.environ["PT_HPU_COMPILE_USE_RECIPES"] = "1"
    os.environ["PT_HPU_USE_INPLACE_COLLECTIVE"] = "1"
    os.environ["PT_HPU_LAZY_MODE"] = "0"

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    enable_sfg = not (os.environ.get("PT_HPU_ENABLE_SFG") is None or os.environ["PT_HPU_ENABLE_SFG"] != "1")
    enable_lazy_collective = not (
        os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES") is None or os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] != "1"
    )

    if do_one_shard:
        if enable_sfg:
            print("ONE_SHARD_TEST and PT_HPU_ENABLE_SFG=1 shouldn't be set together. Exiting...")
            return
        if enable_lazy_collective:
            print("ONE_SHARD_TEST and PT_HPU_ENABLE_LAZY_COLLECTIVES=1 shouldn't be set together. Exiting...")
            return
    else:
        if not enable_sfg:
            print("Set PT_HPU_ENABLE_SFG=1 before running. Exiting...")
            return
        if not enable_lazy_collective:
            print("Set PT_HPU_ENABLE_LAZY_COLLECTIVES=1 before running. Exiting...")
            return

    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl

    torch._inductor.config._fuse_ddp_communication = False

    torch.manual_seed(12345)
    comm_group = torch.distributed.init_process_group("hccl")

    device = torch.device("hpu")

    # Shape configs
    hidden_dimension = args.hidden_dim
    input_size = args.input_length
    output_size = 2048  # unused
    linear_weight_dim2 = args.weight_dim2

    ref_model = ToyModel([hidden_dimension, linear_weight_dim2], torch.bfloat16, num_shards, True)
    ref_model.to(device)

    model = ToyModel([hidden_dimension, linear_weight_dim2], torch.bfloat16, num_shards, do_one_shard)
    model.to(device)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    dynamo.reset()

    model = torch.compile(model, backend="hpu_backend")

    x = 0
    start = time.time()
    BS = args.batch_size

    ITERATIONS = args.iterations

    rank = os.getenv("RANK")
    if rank == "0":
        print_config(
            BS,
            ITERATIONS,
            world_size,
            num_shards,
            model,
            input_size,
            linear_weight_dim2,
            do_one_shard,
            do_hpu_graph,
            do_host_profile,
            do_device_profile,
        )

    dummy_tensor_for_hccl_sync = torch.randn([128], dtype=torch.bfloat16).to(device)

    inputs = []
    outputs_refs = []
    outputs_sfg = []

    for cnt in range(ITERATIONS):
        inp_linear = torch.randn([BS, input_size, linear_weight_dim2], dtype=torch.bfloat16).to(device)
        inputs.append(inp_linear)
        with torch.no_grad():
            output = ref_model(inp_linear, dummy_tensor_for_hccl_sync)
        outputs_refs.append(output)

    def run_iterations(prof=None):
        for cnt in range(ITERATIONS):
            inp_linear = inputs[cnt]
            with torch.no_grad():
                output = model(inp_linear, dummy_tensor_for_hccl_sync)
            if do_host_profile or do_device_profile:
                prof.step()
            outputs_sfg.append(output)
            print("RANK:", os.getenv("RANK"), " Time:", (time.time() - start))

    if do_device_profile:
        if SynapseProfilerApi is None or TraceType is None:
            assert False, "SynapseProfilerApi or TraceType is None, please set PYTORCH_MODULES_ROOT_PATH correctly"
        prof = HabanaDeviceProfile(SynapseProfilerApi(), 5)
        prof.start()
        run_iterations(prof)
        prof.stop()
        print("Finished device profiling")

    elif do_host_profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("host_trace_4shards"),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as prof:
            run_iterations(prof)
    else:
        run_iterations()

    print(ht.hpu.memory.memory_stats())
    for ref, sfg in zip(outputs_refs, outputs_sfg):
        assert torch.allclose(ref.cpu(), sfg.cpu())


if __name__ == "__main__":
    args = parse_args()
    # run_single_node(0, tuple((0,)))
    start = time.time()
    world_size = args.world_size
    input_args = tuple((world_size,))
    torch.multiprocessing.spawn(run_single_node, args=(args,), nprocs=world_size)
    print("Time taken :", time.time() - start)
