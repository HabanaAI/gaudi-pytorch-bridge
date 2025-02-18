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

"""
This module contains performance test for parallel graph compilation

How to run:
export PT_HPU_LAZY_MODE=0
PT_HPU_COMPILE_THREAD_POOL_SIZE=4 python pytorch-integration/utilities/performance/perf_compile_threads.py

Output example ("Extra info" hided by default):
--------------------------- Eager: POOL_SIZE=4, REP=5 (sec.) ---------------------------
  Op count |  Median  |   Mean   | Extra info
----------------------------------------------------------------------------------------
        2 |  3.74E-01|  3.83E-01|  4.27E-01|  3.72E-01|  3.71E-01|  3.74E-01|  3.74E-01
        4 |  7.43E-01|  7.44E-01|  7.49E-01|  7.43E-01|  7.43E-01|  7.44E-01|  7.42E-01
        8 |  1.49E+00|  1.49E+00|  1.50E+00|  1.49E+00|  1.49E+00|  1.49E+00|  1.49E+00
       16 |  3.02E+00|  3.02E+00|  3.00E+00|  3.02E+00|  3.03E+00|  3.02E+00|  3.01E+00
       32 |  6.03E+00|  6.02E+00|  6.03E+00|  6.03E+00|  6.03E+00|  6.01E+00|  6.01E+00

Eager and Compile modes measurement presented in the same runtime session.
Each mode call the same functor with diffrent number of operation in it. Number of operations
is presented in "Op count" field.
Each measurement repeated "execution_set_count" times with the same conditions.
To illustrate paralel compilation, several ("PT_HPU_COMPILE_THREAD_POOL_SIZE" count) functor execution
launched with common syncronization after it.
Each functor execution uses diffrent shapes of input tensors (with minimal diffrences
like [10,10] and [11,11])

"""

import os
import time

import habana_frameworks
import perf_utils
import torch

# os.environ["PT_HPU_LAZY_MODE"] = "0"


execution_set_count = 1  # number of consequtive, independent execution
env_var_thread_pool_size = int(os.getenv("PT_HPU_COMPILE_THREAD_POOL_SIZE", 1))  # number of compiler threads
compilation_count = 32


def direct_execution(input1_list, input2_list, functor_count_list):
    """
    Executes dependant set of operations as many times
    as input data arrays length.

    :param input1_list: Input torch.Tensor list correspond to number of threads
    :param input2_list: Input torch.Tensor list correspond to number of threads
    :param n_task_list: List of integers represents number of tasks
    :returns:           None
    """

    assert len(input1_list) == len(input2_list)

    perf_utils.print_header(f"Eager: THREAD_POOL_SIZE={env_var_thread_pool_size}, REP={execution_set_count} (sec.)")
    for functor_count in functor_count_list:
        time_list = []
        functor = perf_utils.create_functor(functor_count)

        for _ in range(execution_set_count):
            start = time.perf_counter()
            for input1, input2 in zip(input1_list, input2_list):
                result = functor(input1, input2)
            habana_frameworks.torch.hpu.synchronize()
            stop = time.perf_counter()

            time_list.append(stop - start)

        perf_utils.print_line(functor_count * perf_utils.functor_op_count, time_list)


def compiled_execution(input1_list, input2_list, functor_count_list):
    """
    Compile and executes dependant set of operations as many times
    as input data arrays length.

    :param input1_list:        Input torch.Tensor list correspond to number of threads
    :param input2_list:        Input torch.Tensor list correspond to number of threads
    :param functor_count_list: List of integers represents number of operations in a functor
    :returns:                  None
    """

    assert len(input1_list) == len(input2_list)

    perf_utils.print_header(
        f"Compile: f_cmpl(input1, input2 POOL_SIZE={env_var_thread_pool_size}, REP={execution_set_count} (sec.)"
    )
    for op_count in functor_count_list:
        time_list = []

        functor = perf_utils.create_functor(op_count)
        for _ in range(execution_set_count):
            functor_compiled = torch.compile(functor, backend="hpu_backend")

            start = time.perf_counter()
            for input1, input2 in zip(input1_list, input2_list):
                result = functor_compiled(input1, input2)
            habana_frameworks.torch.hpu.synchronize()
            stop = time.perf_counter()

            time_list.append(stop - start)

        perf_utils.print_line(op_count * perf_utils.functor_op_count, time_list)


if __name__ == "__main__":
    """
    This test prints statistic about comparison of the Eager
    and Compile execution mode
    """

    # used here just to print "HABANA PT BRIDGE" configuration header
    # to avoid printing it among test output lines
    habana_frameworks.torch.hpu.get_device_name()

    functor_count_list = [1, 2, 4, 8, 16, 32, 64]

    perf_utils.print_header_line(f"test start. {compilation_count} tasks with {env_var_thread_pool_size} threads")
    perf_utils.print_dev_stat()

    input1_list = perf_utils.create_tensor_2D_list(10, 10, compilation_count)
    input2_list = perf_utils.create_tensor_2D_list(10, 10, compilation_count)

    direct_execution(input1_list, input2_list, functor_count_list)

    compiled_execution(input1_list, input2_list, functor_count_list)

    perf_utils.print_header_line("test finished")
