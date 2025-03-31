###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

from collections.abc import Callable

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

model_parallel_group = None


def get_model_parallel_group():
    return model_parallel_group


def get_model_parallel_world_size():
    return torch.distributed.get_world_size(model_parallel_group)


def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    # if torch.distributed.get_world_size(group=group) == 1:
    #    return input_

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
        return _reduce(grad_output)


def init_weights(m):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        m.bias.data.fill_(0.01)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


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
        super().__init__()

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
        # if self.gather_output:
        #     # All-gather across the partitions.
        #     output = gather_from_model_parallel_region(output_parallel)
        # else:
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
        super().__init__()

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
        # if self.input_is_parallel:
        #     input_parallel = input_
        # else:
        #     input_parallel = scatter_to_model_parallel_region(input_)
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
        ffn_dim_multiplier: float | None,
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
