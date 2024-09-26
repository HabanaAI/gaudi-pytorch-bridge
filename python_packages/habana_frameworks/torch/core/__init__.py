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

# expose common APIs
from .quantization import (
    hpu_inference_initialize,
    hpu_initialize,
    hpu_reset_env,
    hpu_set_env,
    hpu_set_inference_env,
    hpu_teardown_inference_env,
)

# expose lazy-only APIs
from .step_closure import add_step_closure, iter_mark_step, mark_step
from .torch_overwrites import overwrite_native_pt2e_quantization_interface, overwrite_torch_functions

# expose habana_frameworks.torch.hpu as torch.hpu
torch._register_device_module("hpu", hpu)

# wrap some torch functionalitis required to work with HPU
overwrite_torch_functions()

# this is to prevent potential circular imports caused by the function *overwrite_native_pt2e_quantization_interface()*
from functools import wraps

did_overwrite_quantization = False


# A wrapper for bootstrap._find_and_load function to call overwrite_native_quantization_interface() when needed
def on_import(wrapped):
    @wraps(wrapped)
    def inner(*args, **kwargs):
        global did_overwrite_quantization
        # we only need to overwrite once, after importing one of these modules
        if args[0] in ["torch._export", "torch.ao.quantization.quantize_pt2e"] and not did_overwrite_quantization:
            did_overwrite_quantization = True
            overwrite_native_pt2e_quantization_interface()  # wrap pt2e-quant apis required to work on HPU with graph-breaks
        return wrapped(*args, **kwargs)

    return inner


import importlib._bootstrap as bootstrap

setattr(bootstrap, "_find_and_load", on_import(bootstrap._find_and_load))


# enable profiler and weight sharing if required
def _enable_profiler_if_needed():
    import os

    if "HABANA_PROFILE" not in os.environ:
        os.environ["HABANA_PROFILE"] = "profile_api_light"


def _enable_weight_sharing_if_needed():
    from os import getenv

    def check_env_flag(name, default=""):
        return getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]

    if check_env_flag("PT_HPU_WEIGHT_SHARING", "1") and check_env_flag("EXPERIMENTAL_WEIGHT_SHARING", "1"):
        from .weight_sharing import enable_weight_sharing

        enable_weight_sharing()


if is_lazy():
    _enable_weight_sharing_if_needed()
else:
    # Initialize torch.compile backend in non-lazy mode.
    import habana_frameworks.torch.dynamo._custom_op_meta_registrations
    import habana_frameworks.torch.dynamo.compile_backend

_enable_profiler_if_needed()
