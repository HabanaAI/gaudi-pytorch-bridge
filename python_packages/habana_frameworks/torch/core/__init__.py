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

import warnings

import habana_frameworks.torch.utils.debug as htdebug
import habana_frameworks.torch.utils.experimental as htexp
import torch
from habana_frameworks.torch import hpu
from habana_frameworks.torch.utils.internal import is_lazy

from .torch_overwrites import overwrite_torch_functions


# expose lazy-only APIs
from .step_closure import add_step_closure, iter_mark_step, mark_step

# expose common APIs
from .quantization import hpu_initialize

# expose habana_frameworks.torch.hpu as torch.hpu
torch._register_device_module('hpu', hpu)

# wrap some torch functionalitis required to work with HPU
overwrite_torch_functions()

# expose still used deprecated APIs
def data_ptr(t) -> int:
    warnings.warn("habana_frameworks.torch.core.data_ptr is deprecated. "
            "Please use habana_frameworks.torch.utils.experimental._data_ptr")
    return htexp._data_ptr(t)

def get_device_count() -> int:
    warnings.warn("habana_frameworks.torch.core.get_device_count is deprecated. "
            "Please use habana_frameworks.torch.hpu.device_count")
    import habana_frameworks.torch.hpu as hpu
    return hpu.device_count()

def enable_weight_permute_pass(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_weight_permute_pass is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_weight_permute_pass")
    htdebug._enable_weight_permute_pass(flag)

def memstat_livealloc(msg) -> None:
    warnings.warn("habana_frameworks.torch.core.memstat_livealloc is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._memstat_livealloc")
    htdebug._memstat_livealloc(msg)

def memstat_devmem_start_collect(msg, show_cs) -> None:
    warnings.warn("habana_frameworks.torch.core.memstat_devmem_start_collect is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._memstat_devmem_start_collect")
    htdebug._memstat_devmem_start_collect(msg, show_cs)

def memstat_devmem_stop_collect(msg) -> None:
    warnings.warn("habana_frameworks.torch.core.memstat_devmem_stop_collect is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._memstat_devmem_stop_collect")
    htdebug._memstat_devmem_stop_collect(msg)

def is_enabled_synapse_layout_handling() -> bool:
    warnings.warn("habana_frameworks.torch.core.is_enabled_synapse_layout_handling is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._is_enabled_synapse_layout_handling")
    return htdebug._is_enabled_synapse_layout_handling()

# enable profiler and weight sharing if required

def _enable_profiler_if_needed():
    import os
    if "HABANA_PROFILE" not in os.environ:
        os.environ["HABANA_PROFILE"] = "profile_api_light"

def _enable_weight_sharing_if_needed():
    from os import getenv
    def check_env_flag(name, default=""):
        return getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]

    if check_env_flag("EXPERIMENTAL_WEIGHT_SHARING","1"):
        from .weight_sharing import enable_weight_sharing
        enable_weight_sharing()

if is_lazy():
    _enable_weight_sharing_if_needed()
else:
    # Initialize torch.compile backend in non-lazy mode.
    import habana_frameworks.torch.dynamo.compile_backend

_enable_profiler_if_needed()
