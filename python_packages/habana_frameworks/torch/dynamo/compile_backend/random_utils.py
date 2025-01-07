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

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from packaging.version import Version, parse

HABANA_RANDOM_OPS_LIST = [
    "aten.bernoulli.default",
    "aten.poisson.default",
    "aten.rand.default",
    "aten.randn.default",
    "aten.randint.low",
    "aten.multinomial.default",
    "aten.randperm.default",
    "aten.native_dropout.default",
    "aten.uniform.default",
]

# Supported habana wrappers for random ops to proper handling in torch.compile
HABANA_RANDOM_OPS = {op: getattr(torch.ops.hpu, "habana_" + op.split(".")[1]) for op in HABANA_RANDOM_OPS_LIST}
HABANA_RANDOM_OPS.update(
    {
        "hpu.sdpa_recomp_fwd_dropout.default": torch.ops.hpu.sdpa_recomp_fwd_dropout_seed,
        "hpu.sdpa_fwd_dropout.default": torch.ops.hpu.sdpa_fwd_dropout_seed,
        "hpu.fp8_sdpa_fwd_dropout.default": torch.ops.hpu.fp8_sdpa_fwd_dropout_seed,
        "hpu.fp8_sdpa_recomp_fwd_dropout.default": torch.ops.hpu.fp8_sdpa_recomp_fwd_dropout_seed,
        "aten.uniform.default": torch.ops.hpu.habana_uniform,
    }
)
HABANA_RANDOM_OPS = HABANA_RANDOM_OPS if bc.get_pt_hpu_wrap_random_ops_compile() else {}

# Supported habana checkpoint wrappers for random ops to proper handling in torch.compile activation checkpoint
HABANA_CHECKPOINT_OPS = (
    {op: getattr(torch.ops.hpu, "habana_" + op.split(".")[1] + "_checkpoint") for op in HABANA_RANDOM_OPS_LIST}
    if bc.get_pt_hpu_wrap_random_ops_compile()
    else {}
)


def is_random_op(node):
    return node.op == "call_function" and (
        (str(node.target) in HABANA_RANDOM_OPS) or (str(node.target) == "run_and_save_rng_state")
    )


def is_backward_checkpoint_op(node):
    return node.op == "call_function" and str(node.target) == "run_with_rng_state"


def is_multi_output_op(node):
    return str(node.target) == "run_and_save_rng_state" and str(node.args[0]) == "aten.native_dropout.default"


def random_op_inputs(node, seed):
    if str(node.target) == "run_and_save_rng_state":
        op = HABANA_CHECKPOINT_OPS[str(node.args[0])]
        args = (seed,) + node.args[1:]
    else:
        op = HABANA_RANDOM_OPS[str(node.target)]
        args = (seed,) + node.args
    kwargs = node.kwargs.copy()
    kwargs.pop("generator", None)

    return (op, args, kwargs)


def backward_random_op_inputs(node):
    op = HABANA_RANDOM_OPS[str(node.args[1])]
    args = (node.args[0],) + node.args[2:]
    kwargs = node.kwargs.copy()
    kwargs.pop("generator", None)

    return (op, args, kwargs)


# Below part overwrites FakeTensorMode dispatch of run_and_save_rng_state op.
# In compile mode its first output is a seed tensor which we reuse in the backward pass,
# but in eager fallback it is torch.hpu.get_rng_state() which is a ByteTensor,
# the same as CPU and CUDA implementation does.
# Therefore we need to call the original function in the eager fallback,
# but new one aligned with seed tensor in compile mode.

from torch._prims.rng_prims import run_and_save_rng_state
from torch._subclasses.fake_tensor import FakeTensorMode


def is_hpu(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type == "hpu"

    devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
    return any(dev == "hpu" for dev in devices)


if Version(parse(torch.__version__).base_version) >= Version("2.5"):
    old_fn = run_and_save_rng_state.python_key_table.pop(FakeTensorMode)
else:
    old_fn = run_and_save_rng_state.python_key_mode_table.pop(FakeTensorMode)


@run_and_save_rng_state.py_impl(FakeTensorMode)
def impl_fake_tensor_mode(mode, op, *args, **kwargs):
    if is_hpu(args, kwargs) and str(op) in HABANA_CHECKPOINT_OPS:
        with mode:
            return torch.empty([], dtype=torch.int, device="hpu"), op(*args, **kwargs)
    return old_fn(mode, op, *args, **kwargs)
