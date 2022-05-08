import torch
import warnings
from .step_closure import *
from collections import deque
from functools import wraps
from typing import Union
import habana_frameworks.torch.utils.debug as htdebug
import habana_frameworks.torch.utils.experimental as htexp

from torch.functional import Tensor
name_stack = deque()

def pre_fwd_hook(module, input):
    new_name = name_stack[-1] + "/" + module.custom_name if name_stack else module.custom_name
    name_stack.append(new_name)
    htdebug._set_module_name(new_name)

def gen_grad_hook(name):
    def grad_hook(grad):
        htdebug._set_module_name(name)
        return grad
    return grad_hook

def post_fwd_hook(module, input, output):
    module_name = name_stack.pop()
    if (name_stack):
        name = name_stack[-1]
    else:
        name = ""
    grad_name = "gradient/" + module_name
    htdebug._set_module_name(name)
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
    if isinstance(module, torch.nn.Module):
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
    return htexp._compute_stream()

def data_ptr(t) -> int:
    warnings.warn("habana_frameworks.torch.core.data_ptr is deprecated. "
            "Please use habana_frameworks.torch.utils.experimental._data_ptr")
    return htexp._data_ptr(t)

def get_device_count() -> int:
    warnings.warn("habana_frameworks.torch.core.get_device_count is deprecated. "
            "Please use habana_frameworks.torch.hpu.device_count")
    import habana_frameworks.torch.hpu as hpu
    return hpu.device_count()

def get_fallback_op_count() -> dict:
    warnings.warn("habana_frameworks.torch.core.get_fallback_op_count is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._get_fallback_op_count")
    return htdebug._get_fallback_op_count()

def enable_eliminate_common_subexpression(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_eliminate_common_subexpression is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_eliminate_common_subexpression")
    htdebug._enable_eliminate_common_subexpression(flag)

def enable_weight_permute_pass(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_weight_permute_pass is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_weight_permute_pass")
    htdebug._enable_weight_permute_pass(flag)

def is_enabled_weight_permute_pass() -> bool:
    warnings.warn("habana_frameworks.torch.core.is_enabled_weight_permute_pass is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._is_enabled_weight_permute_pass")
    return htdebug._is_enabled_weight_permute_pass()

def enable_constant_pooling(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_constant_pooling is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_constant_pooling")
    htdebug._enable_constant_pooling(flag)

def set_dynamic_mode() -> None:
    warnings.warn("habana_frameworks.torch.core.set_dynamic_mode is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._set_dynamic_mode")
    htdebug._set_dynamic_mode()

def set_module_name(name) -> None:
    warnings.warn("habana_frameworks.torch.core.set_module_name is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._set_module_name")
    htdebug._set_module_name(name)

def run_saved_model(device_name) -> None:
    warnings.warn("habana_frameworks.torch.core.run_saved_model is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._run_saved_model")
    htdebug._run_saved_model(device_name)

def enable_eliminate_dead_code(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_eliminate_dead_code is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_eliminate_dead_code")
    htdebug._enable_eliminate_dead_code(flag)

def enable_peephole_optimization(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_peephole_optimization is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_peephole_optimization")
    htdebug._enable_peephole_optimization(flag)

def enable_fuse_t_mm_optimization(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_fuse_t_mm_optimization is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_fuse_t_mm_optimization")
    htdebug._enable_fuse_t_mm_optimization(flag)

def enable_fuse_bn_relu_optimization(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_fuse_bn_relu_optimization is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_fuse_bn_relu_optimization")
    htdebug._enable_fuse_bn_relu_optimization(flag)

def enable_permute_pass(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_permute_pass is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_permute_pass")
    htdebug._enable_permute_pass(flag)

def enable_replace_inplace_ops(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_replace_inplace_ops is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_replace_inplace_ops")
    htdebug._enable_replace_inplace_ops(flag)

def enable_replace_views(flag) -> None:
    warnings.warn("habana_frameworks.torch.core.enable_replace_views is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._enable_replace_views")
    htdebug._enable_replace_views(flag)

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

def dump_refined_recipe_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_refined_recipe_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_refined_recipe_stat")
    htdebug._dump_refined_recipe_stat()

def disable_bucket_refinement() -> None:
    warnings.warn("habana_frameworks.torch.core.disable_bucket_refinement is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._disable_bucket_refinement")
    htdebug._disable_bucket_refinement()

def dump_bucket_memory_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_bucket_memory_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_bucket_memory_stat")
    htdebug._dump_bucket_memory_stat()

def dump_history_memory_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_history_memory_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_history_memory_stat")
    htdebug._dump_history_memory_stat()

def dump_recipe_memory_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_recipe_memory_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_recipe_memory_stat")
    htdebug._dump_recipe_memory_stat()

def dump_synapse_recipe_memory_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_synapse_recipe_memory_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_synapse_recipe_memory_stat")
    htdebug._dump_synapse_recipe_memory_stat()

def dump_dynamic_shape_memory_stat() -> None:
    warnings.warn("habana_frameworks.torch.core.dump_dynamic_shape_memory_stat is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._dump_dynamic_shape_memory_stat")
    htdebug._dump_dynamic_shape_memory_stat()

def is_enabled_synapse_layout_handling() -> bool:
    warnings.warn("habana_frameworks.torch.core.is_enabled_synapse_layout_handling is deprecated. "
            "Please use habana_frameworks.torch.utils.debug._is_enabled_synapse_layout_handling")
    return htdebug._is_enabled_synapse_layout_handling()
