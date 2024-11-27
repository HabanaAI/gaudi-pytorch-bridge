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
from typing import Callable, Optional

import habana_frameworks.torch as ht
import torch
import torch._dynamo
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter

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


model_parallel_group = None


def get_model_parallel_group():
    return model_parallel_group


def get_model_parallel_world_size():
    return torch.distributed.get_world_size(model_parallel_group)


def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)


def init_weights(m):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        m.bias.data.fill_(0.01)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        init_weights(self)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        init_weights(self)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        output_parallel = F.linear(input_, self.weight)
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


class ToyModel(nn.Module):
    def __init__(self, weight_size):
        super().__init__()
        self.MLP = FeedForward(dim=weight_size, hidden_dim=4 * weight_size, multiple_of=8, ffn_dim_multiplier=None)
        self.ffn_norm = torch.nn.RMSNorm(weight_size, eps=1e-5)

    def forward(self, inp, num_shards):
        if num_shards > 1:
            ffn_norm_out = self.ffn_norm(inp)
            total_shard_size = inp.size()[0]
            shard_size = int(total_shard_size // num_shards)
            start_offset = 0

            out_shard_list = []
            for i in range(num_shards):
                curr_shard_size = shard_size if i < num_shards - 1 else total_shard_size - start_offset
                if curr_shard_size > 0:
                    mlp_out_shard = self.MLP(ffn_norm_out[start_offset : start_offset + curr_shard_size, :])
                    out_shard = inp[start_offset : start_offset + curr_shard_size, :] + mlp_out_shard
                    out_shard_list.append(out_shard)
                    start_offset += curr_shard_size

            return torch.cat(out_shard_list)

        else:
            mlp_out = self.MLP(self.ffn_norm(inp))
            return inp + mlp_out


def print_config(
    BS,
    iterations,
    world_size,
    num_shards,
    model,
    input_size,
    hidden_dimension,
    do_one_shard,
    do_hpu_graph,
    do_host_profile,
    do_device_profile,
):
    print("RUN config")
    print("==========")
    print(f"Batch Size = {BS}")
    print(f"Input size = [Batch, Input seq. length, dim2] = [{BS}, {input_size}, {hidden_dimension})")

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

    parser.add_argument("--batch-size", default=128, type=int, help="Batch size (default: 370)")
    parser.add_argument("--iterations", default=5, type=int, help="Number of iterations (default: 5)")
    parser.add_argument(
        "--world-size", default=2, type=int, help="Number of nodes, represents TP size in model (default: 4)"
    )
    parser.add_argument("--shards", default=4, type=int, help="Number of shards (default: 4)")
    parser.add_argument("--run-one-shard", dest="do_one_shard", action="store_true", help="Set to True to run 1 shard")
    parser.add_argument("--hidden-dim", default=32, type=int, help="Hidden dimension (default: 8192)")
    parser.add_argument("--input-length", default=16, type=int, help="Input seqence length (default: 2048)")
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
    os.environ["PT_HPU_LAZY_MODE"] = "0"

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    enable_sfg = not (os.environ.get("PT_HPU_ENABLE_SFG") is None or os.environ["PT_HPU_ENABLE_SFG"] != "1")
    enable_lazy_collective = not (
        os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES") is None or os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] != "1"
    )

    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl

    torch._inductor.config._fuse_ddp_communication = False

    torch.manual_seed(12345)
    global model_parallel_group
    model_parallel_group = torch.distributed.init_process_group("hccl")

    device = torch.device("hpu")

    # Shape configs
    hidden_dimension = args.hidden_dim
    input_size = args.input_length
    output_size = 2048  # unused

    model = ToyModel(hidden_dimension)
    model.bfloat16()
    model.to(device)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    dynamo.reset()

    model = torch.compile(model, backend="hpu_backend")

    x = 0
    start = time.time()
    BS = args.batch_size

    ITERATIONS = args.iterations
    num_shards = args.shards

    rank = os.getenv("RANK")
    if rank == "0":
        print_config(
            BS,
            ITERATIONS,
            world_size,
            num_shards,
            model,
            input_size,
            hidden_dimension,
            do_one_shard,
            do_hpu_graph,
            do_host_profile,
            do_device_profile,
        )

    inputs = []
    outputs_refs = []
    outputs_sfg = []

    for cnt in range(ITERATIONS):
        inp_linear = torch.randn([BS, input_size, hidden_dimension], dtype=torch.bfloat16).to(device)
        out_ref = model(inp_linear, 1)
        inputs.append(inp_linear)
        outputs_refs.append(out_ref)

    def run_iterations(prof=None):
        for cnt in range(ITERATIONS):
            inp_linear = inputs[cnt]
            with torch.no_grad():
                output = model(inp_linear, num_shards)
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
    for cnt in range(ITERATIONS):
        assert torch.allclose(outputs_refs[cnt], outputs_sfg[cnt])


if __name__ == "__main__":
    args = parse_args()
    # run_single_node(0, tuple((0,)))
    start = time.time()
    world_size = args.world_size
    device_count = ht.hpu.device_count()
    if device_count < world_size:
        world_size = device_count
    input_args = tuple((world_size,))
    torch.multiprocessing.spawn(run_single_node, args=(args,), nprocs=world_size)
    print("Time taken :", time.time() - start)
