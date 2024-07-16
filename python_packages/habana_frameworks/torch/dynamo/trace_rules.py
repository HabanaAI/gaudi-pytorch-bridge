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

import habana_frameworks.torch as htorch
import torch
from packaging.version import Version, parse
from torch._dynamo.trace_rules import SKIP_DIRS, _module_dir, _recompile_re
from torch._dynamo.variables import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable

htorch_skip_list = [
    htorch.hpu,
]

SKIP_DIRS.extend(filter(None, (_module_dir(m) for m in htorch_skip_list)))
_recompile_re()

"""
Map of torch objects to their tracing rules (Dynamo variables).
* TorchVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* TorchCtxManagerClassVariable: The context manager classes are supported by Dynamo. E.g., torch.no_grad
* SkipFilesVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

"""
# Manual function to variable type mapping
_manual_htorch_name_rule_map = {
    # "torch.profiler.profile": TorchCtxManagerClassVariable,          #Example
    # "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,    #Example
}

# Dynamo implemented context managers
_htorch_ctx_manager_classes = {
    k: TorchCtxManagerClassVariable
    for k in [
        # "torch._C.DisableTorchFunctionSubclass",                     #Example
        # "torch.amp.autocast_mode.autocast",                          #Example
    ]
}

# In graph functions (including constant folding) that are C bindings
_htorch_c_binding_in_graph_functions = {
    k: TorchInGraphFunctionVariable
    for k in [
        # "math.acos",                                                 #Example
        # "math.acosh",                                                #Example
        # "torch._C._create_function_from_graph",                      #Example
    ]
}

# In graph functions (including constant folding) that are not C bindings
_htorch_non_c_binding_in_graph_functions = {
    k: TorchInGraphFunctionVariable
    for k in [
        "habana_frameworks.torch.hpu.current_stream",
        "habana_frameworks.torch.hpu.event",
        "habana_frameworks.torch.hpu.set_stream",
        "habana_frameworks.torch.hpu.stream",
        "habana_frameworks.torch.hpu.is_available",
        "habana_frameworks.torch.hpu.current_device",
        "habana_frameworks.torch.hpu.device_count",
        "habana_frameworks.torch.hpu.set_stream_by_id",
        "habana_frameworks.torch.hpu._utils._get_device_index",
    ]
}

from torch._dynamo.trace_rules import torch_name_rule_map

if Version(parse(torch.__version__).base_version) >= Version("2.3"):
    habana_torch_name_rule_list = [
        _manual_htorch_name_rule_map,
        _htorch_ctx_manager_classes,
        _htorch_c_binding_in_graph_functions,
        _htorch_non_c_binding_in_graph_functions,
    ]

    torch_name_rule_map.extend(habana_torch_name_rule_list)

    from torch._dynamo.trace_rules import _allowed_callable_ids

    functions_to_add = [
        htorch.hpu.stream,
        htorch.hpu.current_stream,
        htorch.hpu._utils._get_device_index,
    ]

    for obj in functions_to_add:
        _allowed_callable_ids.add(id(obj))

else:
    habana_torch_name_rule_map = {
        **_manual_htorch_name_rule_map,
        **_htorch_ctx_manager_classes,
        **_htorch_c_binding_in_graph_functions,
        **_htorch_non_c_binding_in_graph_functions,
    }

    torch_name_rule_map.update(habana_torch_name_rule_map)

    """
    A note on allowed functions:

    Dynamo consults _allowed_function_ids in torch._dynamo.allowed_functions to determine
    if a particular function/module is allowed to appear as a node in its fx output.

    If a function is disallowed, it may either be traced-through, or skipped.

    Trace-through means dynamo will continue to trace the interior code for
    the function/module rather than stopping at its boundary and recording it
    as a node in the fx graph. Whether tracing through or allowing, the functionality
    of the function/module is part of the dynamo graph.  Caveat: if tracing through,
    any interior operation could trigger its own graph-break.

    Skips are determined by (torch/_dynamo/skipfiles.py) - see "a note on
    skipfiles" there.
    """
    from torch._dynamo.allowed_functions import _allowed_function_ids

    functions_to_add = [
        htorch.hpu.streams.Stream,
        htorch.hpu.events.Event,
    ]

    for obj in functions_to_add:
        _allowed_function_ids.add(id(obj))

from torch._dynamo.variables.torch import constant_fold_functions

functions_to_add = [
    htorch.hpu.is_available,
    htorch.hpu.current_device,
    htorch.hpu._utils._get_device_index,
]

if Version(parse(torch.__version__).base_version) >= Version("2.4.0"):
    constant_fold_functions.update(dict.fromkeys(functions_to_add))
else:
    constant_fold_functions.extend(functions_to_add)
