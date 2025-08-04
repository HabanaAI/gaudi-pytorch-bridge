###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import habana_frameworks.torch as htorch

from torch._dynamo.trace_rules import (
    SKIP_DIRS,
    _allowed_callable_ids,
    _module_dir,
    _recompile_re,
    get_torch_obj_rule_map,
    manual_torch_name_rule_map,
    torch_name_rule_map,
)
from torch._dynamo.variables import (
    TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)
from torch._dynamo.variables.torch import constant_fold_functions

manual_torch_name_rule_map.pop("torch.cuda.current_device", None)

htorch_skip_list = [
    htorch.hpu.__init__,
    htorch.hpu._proxy_module,
    htorch.hpu._utils,
    htorch.hpu.events,
    htorch.hpu.memory,
    htorch.hpu.random,
    htorch.hpu.streams,
    htorch.hpu.metrics,
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
_htorch_ctx_manager_classes = dict.fromkeys(
    [
        # "torch._C.DisableTorchFunctionSubclass",                     #Example
        # "torch.amp.autocast_mode.autocast",                          #Example
    ],
    TorchCtxManagerClassVariable,
)

# In graph functions (including constant folding) that are C bindings
_htorch_c_binding_in_graph_functions = dict.fromkeys(
    [
        # "math.acos",                                                 #Example
        # "math.acosh",                                                #Example
        # "torch._C._create_function_from_graph",                      #Example
    ],
    TorchInGraphFunctionVariable,
)

# In graph functions (including constant folding) that are not C bindings
_htorch_non_c_binding_in_graph_functions = dict.fromkeys(
    [
        "habana_frameworks.torch.hpu.current_stream",
        "habana_frameworks.torch.hpu.event",
        "habana_frameworks.torch.hpu.set_stream",
        "habana_frameworks.torch.hpu.stream",
        "habana_frameworks.torch.hpu.is_available",
        "habana_frameworks.torch.hpu.current_device",
        "habana_frameworks.torch.hpu.device_count",
        "habana_frameworks.torch.hpu.set_stream_by_id",
        "habana_frameworks.torch.hpu._utils._get_device_index",
    ],
    TorchInGraphFunctionVariable,
)

habana_torch_name_rule_list = [
    _manual_htorch_name_rule_map,
    _htorch_ctx_manager_classes,
    _htorch_c_binding_in_graph_functions,
    _htorch_non_c_binding_in_graph_functions,
]

torch_name_rule_map.extend(habana_torch_name_rule_list)
get_torch_obj_rule_map.cache_clear()


functions_to_add = [
    htorch.hpu.stream,
    htorch.hpu.current_stream,
    htorch.hpu._utils._get_device_index,
]

for obj in functions_to_add:
    _allowed_callable_ids.add(id(obj))


functions_to_add = [
    htorch.hpu.is_available,
    htorch.hpu.current_device,
    htorch.hpu._utils._get_device_index,
]

constant_fold_functions.update(dict.fromkeys(functions_to_add))
