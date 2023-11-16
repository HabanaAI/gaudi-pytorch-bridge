###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
from packaging.version import Version

import torch
from habana_frameworks.torch.utils.internal import is_lazy

REQUIRED_VERSION_FILE = "required_version.txt"
REQUIRED_VERSION_FILE_PATH = os.path.join(
    os.path.dirname(__file__), REQUIRED_VERSION_FILE
)

with open(REQUIRED_VERSION_FILE_PATH) as req_ver_file:
    compile_time_ver = Version(req_ver_file.read())

run_time_ver = Version(Version(torch.__version__).base_version)

assert (
    run_time_ver == compile_time_ver
), f"Error: Compile-time pytorch version {compile_time_ver} differs from run-time {run_time_ver}."

lib_to_load = "libhabana_pytorch{}_plugin.so".format("" if is_lazy() else "2")
ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "lib", lib_to_load), ctypes.RTLD_GLOBAL
)

import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu
import habana_frameworks.torch.activity_profiler

def overwrite_torch_optimizers():
    from os import environ
    should_rewrite_optimizers = environ.get("PT_HPU_REPLACE_ADAM_ADAMW", "0").lower()
    if should_rewrite_optimizers not in ["1", "true", "yes"]:
        return
    import torch.optim
    import torch.optim.adam as modAdam
    import torch.optim.adamw as modAdamW
    import habana_frameworks.torch.hpex.optimizers.MarkstepAdam as MarkstepAdam
    import habana_frameworks.torch.hpex.optimizers.MarkstepAdamW as MarkstepAdamW
    modAdam.Adam.step = MarkstepAdam.Adam.step
    modAdamW.AdamW.step = MarkstepAdamW.AdamW.step
    modAdam.Adam = MarkstepAdam.Adam
    modAdamW.AdamW = MarkstepAdamW.AdamW

overwrite_torch_optimizers()