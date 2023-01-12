import torch
import warnings
from os import environ
from .step_closure import *
from collections import deque
from functools import wraps
from typing import Union
import datetime
import habana_frameworks.torch.utils.debug as htdebug
import habana_frameworks.torch.utils.experimental as htexp
import habana_frameworks.torch.hpu.random as rand_hpu
from habana_frameworks.torch.utils import _experimental_C
from torch.fx import symbolic_trace

from torch.functional import Tensor
name_stack = deque()

# expose habana_frameworks.torch.hpu as torch.hpu
from habana_frameworks.torch import hpu
torch._register_device_module('hpu', hpu)

def _record_quant_param(name, min, max) -> None:
    if hpu.is_available():
        _experimental_C.record_quant_param(name, min, max)

def read_min_max_overwrite():
    range_path = environ.get('PT_INFERENCE_RANGE_FILE')
    if range_path:
        with open(range_path) as file:
            for line in file:
                line = line[line.find('/')+1:len(line)]
                line = line.split()
                _record_quant_param(line[0], float(line[1]), float(line[2]))

def handle_quant_stats(model=None):
    if model is not None:
        min_calibration_data = dict()
        max_calibration_data = dict()
        placeholder_dict = dict()
        gm : torch.fx.GraphModule = symbolic_trace(model)
        for node in gm.graph.nodes:
            for i in range(len(node.all_input_nodes)):
                if node.all_input_nodes[i].op == "placeholder":
                   name = '.'.join([node.all_input_nodes[i].target, node.target, "placeholder",str(i)])
                   placeholder_dict[node.all_input_nodes[i].target] = name
        for name, param in model.state_dict().items():
            if name.endswith('.min_val'):
                min_calibration_data[name.replace(".min_val","")] = param.item()
            if name.endswith('.max_val'):
                max_calibration_data[name.replace(".max_val","")] = param.item()
        for name, param in placeholder_dict.items():
            if name in min_calibration_data.keys():
               min_calibration_data[param] = min_calibration_data[name]
               min_calibration_data.pop(name)
            if name in max_calibration_data.keys():
               max_calibration_data[param] = max_calibration_data[name]
               max_calibration_data.pop(name)
        for name, param in min_calibration_data.items():
            try:
                _record_quant_param(name, min_calibration_data[name], max_calibration_data[name])
            except:
                pass
        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, torch.nn.Module) and not names_hook_already_registered(submodule):
               try:
                   submodule.custom_name = submodule_name
                   submodule.register_forward_pre_hook(pre_fwd_hook)
                   submodule.register_forward_hook(post_fwd_hook)
                   submodule.names_hook = True
               except (RuntimeError):
                   pass

def hpu_initialize(model=None, optimizer=None, args=None):
    if model is not None:
        read_min_max_overwrite()
        handle_quant_stats(model)

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

manual_seed_orig = torch.manual_seed

@wraps(torch.manual_seed)
def wrap_manual_seed(seed):
    rand_hpu.manual_seed(seed)
    return manual_seed_orig(seed)

torch.manual_seed = wrap_manual_seed

add_module_orig = torch.nn.modules.Module.add_module

def names_hook_already_registered(module):
    if hasattr(module, 'names_hook') and module.names_hook == True:
        return True
    return False

@wraps(torch.nn.modules.Module.add_module)
def wrap_add_module(self, name, module):
    if isinstance(module, torch.nn.Module) and not names_hook_already_registered(module):
        try:
            module.custom_name = name
            module.register_forward_pre_hook(pre_fwd_hook)
            module.register_forward_hook(post_fwd_hook)
            module.names_hook = True
        except (RuntimeError):
            pass
    add_module_orig(self, name, module)

torch.nn.modules.Module.add_module = wrap_add_module

from torch.distributed.constants import default_pg_timeout

ranks_cache = {}
new_group_orig = torch.distributed.new_group
init_process_group_orig = torch.distributed.init_process_group

@wraps(torch.distributed.new_group)
def wrap_new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None):
    global ranks_cache
    cache_enable = environ.get('PT_ENABLE_COMM_GROUP_CACHE', "False")
    if cache_enable.lower() == "true":
        global ranks_cache
        if ranks == None:
            actual_world_size = torch.distributed.distributed_c10d.get_world_size()
            ranks_tuple = tuple(list(range(0, actual_world_size)))
        else:
            ranks_tuple = tuple(sorted(tuple(ranks)))
        if ranks_tuple in ranks_cache:
            return ranks_cache[ranks_tuple]
        else:
            ranks_cache[ranks_tuple] = new_group_orig(ranks, timeout, backend, pg_options)
            return ranks_cache[ranks_tuple]
    else:
        return new_group_orig(ranks, timeout, backend, pg_options)

@wraps(torch.distributed.init_process_group)
def wrap_init_process_group(backend, init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None):
    global ranks_cache
    cache_enable = environ.get('PT_ENABLE_COMM_GROUP_CACHE', "False")
    if cache_enable.lower() == "true":
        if len(ranks_cache) == 0:
            init_process_group_orig(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)
        actual_world_size = torch.distributed.distributed_c10d.get_world_size()
        ranks_tuple = tuple(list(range(0, actual_world_size)))
        if ranks_tuple in ranks_cache:
            return ranks_cache[ranks_tuple]
        ranks_cache[ranks_tuple] = torch.distributed.distributed_c10d._get_default_group()
        return ranks_cache[ranks_tuple]
    else:
        return init_process_group_orig(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)


torch.distributed.new_group = wrap_new_group
torch.distributed.init_process_group = wrap_init_process_group


module_set_attr_orig = torch.nn.Module.__setattr__
@wraps(torch.nn.Module.__setattr__)
def wrap_set_attr(self, name: str, value: Union[torch.Tensor, 'torch.nn.Module']) -> None:
    if isinstance(value, torch.nn.Module) and not names_hook_already_registered(value):
        try:
            value.custom_name = name
            value.register_forward_pre_hook(pre_fwd_hook)
            value.register_forward_hook(post_fwd_hook)
            value.names_hook = True
        except (RuntimeError):
            pass
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

def enable_profiler_if_needed():
    import os
    if "HABANA_PROFILE" not in os.environ:
        os.environ["HABANA_PROFILE"] = "profile_api_light"

def enable_weight_sharing_if_needed():
    from os import getenv
    def check_env_flag(name, default=""):
        return getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]

    if check_env_flag("EXPERIMENTAL_WEIGHT_SHARING","1"):
        from .weight_sharing import enable_weight_sharing
        enable_weight_sharing()

enable_profiler_if_needed()
enable_weight_sharing_if_needed()