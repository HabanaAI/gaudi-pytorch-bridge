###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import ctypes
import os
import warnings

# This is to ensure we don't start torch.inductor codecache process pool
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
from habana_frameworks.torch.utils.internal import is_lazy
from packaging.version import Version

REQUIRED_VERSION_FILE = "required_version.txt"
REQUIRED_VERSION_FILE_PATH = os.path.join(os.path.dirname(__file__), REQUIRED_VERSION_FILE)

with open(REQUIRED_VERSION_FILE_PATH) as req_ver_file:
    compile_time_ver = Version(req_ver_file.read())

run_time_ver = Version(torch.__version__)

assert (
    run_time_ver.major == compile_time_ver.major and run_time_ver.minor == compile_time_ver.minor
), f"Error: Compile-time major/minor PyTorch version {compile_time_ver} differs from run-time {run_time_ver}."

lib_to_load = "libhabana_pytorch{}_plugin.so".format("" if is_lazy() else "2")
ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib", lib_to_load), ctypes.RTLD_GLOBAL)

import habana_frameworks.torch.activity_profiler
import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu
import habana_frameworks.torch.internal.bridge_config as bc


def overwrite_torch_optimizers():
    from os import environ

    should_rewrite_optimizers = environ.get("PT_HPU_REPLACE_ADAM_ADAMW", "0").lower()
    if should_rewrite_optimizers not in ["1", "true", "yes"]:
        return
    import habana_frameworks.torch.hpex.optimizers.MarkstepAdam as MarkstepAdam
    import habana_frameworks.torch.hpex.optimizers.MarkstepAdamW as MarkstepAdamW
    import torch.optim
    import torch.optim.adam as modAdam
    import torch.optim.adamw as modAdamW

    modAdam.Adam.step = MarkstepAdam.Adam.step
    modAdamW.AdamW.step = MarkstepAdamW.AdamW.step
    modAdam.Adam = MarkstepAdam.Adam
    modAdamW.AdamW = MarkstepAdamW.AdamW


overwrite_torch_optimizers()

if bc.get_pt_hpu_gpu_migration():
    try:
        import habana_frameworks.torch.gpu_migration
    except ImportError:
        warnings.warn(
            "ImportError: no module named habana_frameworks.torch.gpu_migration. "
            "Check if GPU Migration Toolkit package is installed. "
        )
