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
from .torch_overwrites import (
    overwrite_capture_pre_autograd_graph,
    overwrite_native_pt2e_quantization_interface,
    overwrite_torch_functions,
)

# expose habana_frameworks.torch.hpu as torch.hpu
torch._register_device_module("hpu", hpu)

# wrap some torch functionalitis required to work with HPU
overwrite_torch_functions()
overwrite_capture_pre_autograd_graph()

# this is to prevent potential circular imports caused by the function *overwrite_native_pt2e_quantization_interface()*
from functools import wraps

from habana_frameworks.torch.dynamo.compile_backend.backends import import_compilers, import_hpu_partition


def create_and_apply_on_import_wrapper():
    """
    This function wraps a _find_and_load function from importlib._bootstrap module, which is called when importing modules.
    This is done, so that some additional initialization is performed when importing some specific modules,
    which would potentially cause an error when importing them for the first time, i.e. circular import error when importing some modules before torch.

    Here is a simple example to demonstrate one of the behaviour this wrapper fixes:
    *torch and habana_frameworks.torch haven't been imported yet*
    import functorch -> imports torch -> imports habana_frameworks.torch -> ... -> imports torch._export -> imports functorch (which causes a cirular import error)
    """
    import importlib._bootstrap as bootstrap

    module = bootstrap
    fn_name = "_find_and_load"

    original_fn = getattr(module, fn_name)

    # A function to restore the original behaviour of the given function
    def unwrap():
        setattr(module, fn_name, original_fn)

    def wrap():
        setattr(module, fn_name, wrapper)

    did_handle_overwrites = False
    did_handle_backend = False

    @wraps(original_fn)
    def wrapper(*args, **kwargs):
        nonlocal did_handle_backend, did_handle_overwrites
        # we only need to overwrite once, after importing one of these modules
        if args[0] in ["torch.ao.quantization.quantize_pt2e"] and not did_handle_overwrites:
            did_handle_overwrites = True
            overwrite_native_pt2e_quantization_interface()  # wrap pt2e-quant apis required to work on HPU with graph-breaks
            ret = original_fn(*args, **kwargs)
        elif "dynamo" in args[0] and not did_handle_backend:
            # postpone some of the imports in the dynamo module until it's actually used to avoid import errors
            did_handle_backend = True
            import_compilers()
            import_hpu_partition()
            ret = original_fn(*args, **kwargs)
        else:
            ret = original_fn(*args, **kwargs)
        if did_handle_backend and did_handle_overwrites:
            # if both of the conditions above have been met, this wrapper is not needed anymore, so the function we have wrapped is set back to the original
            unwrap()

        return ret

    wrap()


# Autoloader is disabled by default in lazy mode, so there is no need to apply fix with create_and_apply_on_import_wrapper() function
# which is autoloader-speciffic
if is_lazy():
    overwrite_native_pt2e_quantization_interface()  # wrap pt2e-quant apis required to work on HPU with graph-breaks
    import_compilers()
    import_hpu_partition()
else:
    create_and_apply_on_import_wrapper()


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
