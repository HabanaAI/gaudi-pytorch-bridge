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
import math  # for ceil etc
import os
import sys
import time

from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config

hpu_backend_config.use_eager_fallback = True

# FIXME: remove unused packages
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA, PySDPA, PySDPAHinted
from test_utils import compare_tensors


# below are utility functions #
def print_mem_summary(name, short=True):
    print(name)
    if short:
        MB = 1024 * 1024
        mem_workspace = ht.memory._extended_memory_summary_dict()["workspace"]
        mem_persistent = ht.memory._extended_memory_summary_dict()["persistent"]
        mem_max_in_use = ht.memory._extended_memory_summary_dict()["max_in_use"]

        label = "workspace:"
        value = str(mem_workspace) + " ({:.2f}) MB".format(mem_workspace / MB)
        print(" " + label + " " * (40 - (len(label) + len(value))) + value)

        label = "persistent:"
        value = str(mem_persistent) + " ({:.2f}) MB".format(mem_persistent / MB)
        print(" " + label + " " * (40 - (len(label) + len(value))) + value)

        label = "max_in_use:"
        value = str(mem_max_in_use) + " ({:.2f}) MB".format(mem_max_in_use / MB)
        print(" " + label + " " * (40 - (len(label) + len(value))) + value)

        label = "max_in_use-persistent:"
        mem_max_in_use_minus_persistent = mem_max_in_use - mem_persistent
        value = str(mem_max_in_use_minus_persistent) + " ({:.2f}) MB".format(mem_max_in_use_minus_persistent / MB)
        print(" " + label + " " * (40 - (len(label) + len(value))) + value + "\n")
    else:
        print(ht.memory.memory_summary())
        print(ht.memory._extended_memory_summary())


def pre_iteration():
    if pytest.mode == "lazy":
        htcore.mark_step()


def post_iteration(o_hpu, q_hpu, k_hpu, v_hpu, wait_for=4):
    if pytest.mode == "lazy":
        htcore.mark_step()
    o_hpu.detach().to("cpu")
    q_hpu.grad.detach().to("cpu")
    k_hpu.grad.detach().to("cpu")
    v_hpu.grad.detach().to("cpu")
    time.sleep(wait_for)


# below are configurations

# 4 kernel types: naive_eager (default), eager_with_slice, lazy_cguid, compile_naive, compile_with_slice, compile_with_hints
debug_mode = os.environ.get("PYTHON_SDPA_DEBUG_MODE", "0") == "1"
kernel_type = os.environ.get("PYTHON_SDPA_KERNEL_TYPE", "naive_eager")

# default off
memory_check = os.environ.get("PYTHON_SDPA_MEMORY_CHECK", "0") == "1"

# default on
perf_check = os.environ.get("PYTHON_SDPA_PERF_CHECK", "0") == "1"
total_runs = int(os.environ.get("PYTHON_SDPA_TOTAL_RUNS", "3"))

# true -> use pytorch profiler
# false (default) -> use synapse profiler
use_tensorboard = os.environ.get("PYTHON_SDPA_USE_TENSORBOARD", "0") == "1"

# turn on for CI running
ci_run = os.environ.get("PYTHON_SDPA_CI_RUN", "1") == "1"

# parameters extracted from real model
chosen_parameters_for_benchmark = [
    (
        8,  # batch_size,
        32,  # n_heads,
        2048,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        2048,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        # False,  # use_attn_mask,
        # False,  # use_float_mask,
        True,  # use_bf16
        True,  # is_causal
        # False,  # is_partial
        8,  # preferred_slice_size
    ),
]

chosen_parameters_for_ci = [
    (
        1,  # batch_size,
        2,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        4,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        # False,  # use_attn_mask,
        # False,  # use_float_mask,
        True,  # use_bf16
        True,  # is_causal
        # False,  # is_partial
        8,  # preferred_slice_size
    ),
]

chosen_parameters = chosen_parameters_for_ci if ci_run else chosen_parameters_for_benchmark


@pytest.mark.parametrize(
    "batch_size, n_heads, seq_len_N_t, seq_len_N_s, head_dim_qk, head_dim_v, dropout_p, use_bf16, is_causal, preferred_slice_size",
    chosen_parameters,
)
def test_multiple_sdpa_impls(
    batch_size,
    n_heads,
    seq_len_N_t,
    seq_len_N_s,
    head_dim_qk,
    head_dim_v,
    dropout_p,
    use_bf16,
    is_causal,
    preferred_slice_size,
):
    print("\nmemory_check: ", memory_check)
    print("perf_check: ", perf_check)
    print("total runs: ", total_runs)

    print("\nbatch_size = ", batch_size)
    print("num_heads = ", n_heads)
    print("seq_len_N_s = ", seq_len_N_s)
    print("head dim q k = ", head_dim_qk)
    print("head dim v = ", head_dim_v)

    def run_sdpa_once(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice):
        # atten_mask -> None
        # drop_p -> 0.0
        o_hpu = PySDPA.apply(q_hpu, k_hpu, v_hpu, is_causal, with_slice)
        o_hpu.backward(g_hpu)
        return o_hpu

    def run_sdpa_cguid_once(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice):
        # atten_mask -> None
        # drop_p -> 0.0
        o_hpu = FusedSDPA.apply(q_hpu, k_hpu, v_hpu, None, 0.0, is_causal)
        o_hpu.backward(g_hpu)
        return o_hpu

    def run_sdpa_with_hints_once(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice):
        # atten_mask -> None
        # drop_p -> 0.0
        o_hpu = PySDPAHinted.apply(q_hpu, k_hpu, v_hpu, is_causal, with_slice)
        o_hpu.backward(g_hpu)
        return o_hpu

    # for debug
    if debug_mode or memory_check:
        torch.manual_seed(1234567)

    # check for supported kernel types
    if kernel_type not in [
        "naive_eager",
        "eager_with_slice",
        "lazy_cguid",
        "compile_naive",
        "compile_with_slice",
        "compile_with_hints",
    ]:
        raise RuntimeError(
            "Not supported kernel type, please use one of [naive_eager, eager_with_slice, lazy_cguid, compile_naive, compile_with_slice, compile_with_hints]"
        )

    print("kernel type: ", kernel_type)

    with_slice = False
    if "naive" not in kernel_type:
        with_slice = True

    run_hpu_sdpa = run_sdpa_once
    if "hints" in kernel_type:
        assert pytest.mode == "compile", "Python hinted SDPA kernel is expected to be used with compile mode"
        run_hpu_sdpa = run_sdpa_with_hints_once
        with_slice = True

    if "lazy_cguid" == kernel_type:
        assert pytest.mode == "lazy", "CGUID SDPA kernel is expected to be used with lazy mode"
        run_hpu_sdpa = run_sdpa_cguid_once

    if pytest.mode == "compile" and "compile" in kernel_type:
        run_hpu_sdpa = torch.compile(run_hpu_sdpa, backend="hpu_backend", dynamic=False)

    dtype = torch.float32
    grad_dtype = torch.float32
    rtol = 1e-3
    atol = 1e-3

    if use_bf16:
        dtype = torch.bfloat16
        grad_dtype = torch.bfloat16
        rtol = 1e-3
        atol = 0.08

    q_shape = (batch_size, n_heads, seq_len_N_t, head_dim_qk)
    k_shape = (batch_size, n_heads, seq_len_N_s, head_dim_qk)
    v_shape = (batch_size, n_heads, seq_len_N_s, head_dim_v)
    fwd_out_shape = (batch_size, n_heads, seq_len_N_t, head_dim_v)

    print("q shape = ", q_shape)
    print("k shape = ", k_shape)
    print("v shape = ", v_shape)

    q = torch.randn(q_shape).to(dtype)
    k = torch.randn(k_shape).to(dtype)
    v = torch.randn(v_shape).to(dtype)
    g = torch.ones(fwd_out_shape).to(grad_dtype)

    q_hpu = q.to("hpu").detach().requires_grad_()
    k_hpu = k.to("hpu").detach().requires_grad_()
    v_hpu = v.to("hpu").detach().requires_grad_()
    g_hpu = g.to("hpu")

    if memory_check:
        ht.memory.reset_accumulated_memory_stats()
        ht.memory.reset_peak_memory_stats()

        print_mem_summary("####MEMORY PRE WARMUP")

    pre_iteration()
    o_hpu = run_hpu_sdpa(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice)
    post_iteration(o_hpu, q_hpu, k_hpu, v_hpu)

    if memory_check:
        print_mem_summary("####MEMORY POST WARMUP")

        for i in range(3):
            pre_iteration()
            o_hpu = run_hpu_sdpa(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice)
            post_iteration(o_hpu, q_hpu, k_hpu, v_hpu)
            print_mem_summary("####MEMORY POST RUN #" + str(i + 2))

    if perf_check:
        if not use_tensorboard:
            sys.path.append(os.environ["PYTORCH_MODULES_ROOT_PATH"])
            from topologies.tools import SynapseProfilerApi, TraceType

            profile_api = SynapseProfilerApi()
            trace_type = TraceType.TraceDevice
            profile_dev_id = 0

            profile_api.profiler_start(trace_type, profile_dev_id)

            for i in range(total_runs):
                pre_iteration()
                o_hpu = run_hpu_sdpa(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice)
                post_iteration(o_hpu, q_hpu, k_hpu, v_hpu)

            profile_api.profiler_sync(profile_dev_id)
            profile_api.profiler_stop(trace_type, profile_dev_id)
            profile_api.profiler_get_trace_json(trace_type, profile_dev_id)
        else:
            import torch_tb_profiler

            # use pytorch profiler and tensorboard for viewing
            activities = [torch.profiler.ProfilerActivity.HPU]
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=3, active=3, repeat=1),
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
            ) as profiler:
                for i in range(8):
                    pre_iteration()
                    o_hpu = run_hpu_sdpa(g_hpu, q_hpu, k_hpu, v_hpu, is_causal, with_slice)
                    post_iteration(o_hpu, q_hpu, k_hpu, v_hpu)
                    profiler.step()
