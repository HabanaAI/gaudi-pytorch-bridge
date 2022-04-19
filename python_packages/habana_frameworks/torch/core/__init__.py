import torch
import warnings
from .step_closure import *
from collections import deque
from functools import wraps
from typing import Union
import habana_frameworks.torch._core_C as htcore

from torch.functional import Tensor
name_stack = deque()

def pre_fwd_hook(module, input):
    new_name = name_stack[-1] + "/" + module.custom_name if name_stack else module.custom_name
    name_stack.append(new_name)
    htcore.set_module_name(new_name)

def gen_grad_hook(name):
    def grad_hook(grad):
        htcore.set_module_name(name)
        return grad
    return grad_hook

def post_fwd_hook(module, input, output):
    module_name = name_stack.pop()
    if (name_stack):
        name = name_stack[-1]
    else:
        name = ""
    grad_name = "gradient/" + module_name
    htcore.set_module_name(name)
    try:
        if isinstance(output, Tensor):
            if output.requires_grad:
                output.register_hook(gen_grad_hook(grad_name))
        else:
            for o in output:
                if isinstance(o, Tensor) and o.requires_grad:
                    o.register_hook(gen_grad_hook(grad_name))
    except:
        pass

add_module_orig = torch.nn.modules.Module.add_module

@wraps(torch.nn.modules.Module.add_module)
def wrap_add_module(self, name, module):
    module.custom_name = name
    module.register_forward_pre_hook(pre_fwd_hook)
    module.register_forward_hook(post_fwd_hook)
    add_module_orig(self, name, module)

torch.nn.modules.Module.add_module = wrap_add_module

module_set_attr_orig = torch.nn.Module.__setattr__
@wraps(torch.nn.Module.__setattr__)
def wrap_set_attr(self, name: str, value: Union[torch.Tensor, 'torch.nn.Module']) -> None:
    if isinstance(value, torch.nn.Module):
        value.custom_name = name
        value.register_forward_pre_hook(pre_fwd_hook)
        value.register_forward_hook(post_fwd_hook)
    module_set_attr_orig(self, name, value)

torch.nn.Module.__setattr__ = wrap_set_attr

def compute_stream() -> int:
    warnings.warn("habana_frameworks.torch.core.compute_stream is deprecated. "
            "Please use habana_frameworks.torch.utils.experimental._compute_stream")
    import habana_frameworks.torch.utils.experimental as exp
    return exp._compute_stream()

def data_ptr(t) -> int:
    warnings.warn("habana_frameworks.torch.core.data_ptr is deprecated. "
            "Please use habana_frameworks.torch.utils.experimental._data_ptr")
    import habana_frameworks.torch.utils.experimental as exp
    return exp._data_ptr(t)

def get_device_count() -> int:
    warnings.warn("habana_frameworks.torch.core.get_device_count is deprecated. "
            "Please use habana_frameworks.torch.hpu.device_count")
    import habana_frameworks.torch.hpu as hpu
    return hpu.device_count()

# FIXME below will be cleanup in the debug cleanup
def get_fallback_op_count() -> int:
    return htcore.get_fallback_op_count()

def enable_eliminate_common_subexpression(flag) -> None:
    return htcore.enable_eliminate_common_subexpression(flag)

def enable_weight_permute_pass(flag) -> None:
    htcore.enable_weight_permute_pass(flag)

def is_enabled_weight_permute_pass() -> bool:
    return htcore.is_enabled_weight_permute_pass()

def enable_constant_pooling(flag) -> None:
    return htcore.enable_constant_pooling(flag)

def memstat_livealloc(msg) -> None:
    htcore.memstat_livealloc(msg)

def memstat_devmem_start_collect(msg, show_cs) -> None:
    htcore.memstat_devmem_start_collect(msg, show_cs)

def memstat_devmem_stop_collect(msg) -> None:
    htcore.memstat_devmem_stop_collect(msg)

def is_enabled_synapse_layout_handling() -> bool:
    return htcore.is_enabled_synapse_layout_handling()
