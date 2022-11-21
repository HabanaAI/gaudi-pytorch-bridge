import torch
from habana_frameworks.torch.utils import _debug_C

def _get_fallback_op_count() -> dict:
    return _debug_C.get_fallback_op_count()

def _set_dynamic_mode() -> None:
    _debug_C.set_dynamic_mode()

def _set_module_name(name="") -> None:
    _debug_C.set_module_name(name)

def _enable_eliminate_common_subexpression(flag) -> None:
    _debug_C.enable_eliminate_common_subexpression(flag)

def _enable_eliminate_dead_code(flag) -> None:
    _debug_C.enable_eliminate_dead_code(flag)

def _enable_constant_pooling(flag) -> None:
    _debug_C.enable_constant_pooling(flag)

def _enable_peephole_optimization(flag) -> None:
    _debug_C.enable_peephole_optimization(flag)

def _enable_fuse_t_mm_optimization(flag) -> None:
    _debug_C.enable_fuse_t_mm_optimization(flag)

def _enable_fuse_bn_relu_optimization(flag) -> None:
    _debug_C.enable_fuse_bn_relu_optimization(flag)

def _enable_permute_pass(flag) -> None:
    _debug_C.enable_permute_pass(flag)

def _enable_replace_inplace_ops(flag) -> None:
    _debug_C.enable_replace_inplace_ops(flag)

def _enable_replace_views(flag) -> None:
    _debug_C.enable_replace_views(flag)

def _enable_weight_permute_pass(flag) -> None:
    _debug_C.enable_weight_permute_pass(flag)

def _is_enabled_weight_permute_pass() -> bool:
    return _debug_C.is_enabled_weight_permute_pass()

def _memstat_livealloc(msg="") -> None:
    _debug_C.memstat_livealloc(msg)

def _memstat_devmem_start_collect(msg="", show_cs=True) -> None:
    _debug_C.memstat_devmem_start_collect(msg, show_cs)

def _memstat_devmem_stop_collect(msg="") -> None:
    _debug_C.memstat_devmem_stop_collect(msg)

def _dump_refined_recipe_stat() -> None:
    _debug_C.dump_refined_recipe_stat()

def _disable_bucket_refinement() -> None:
    _debug_C.disable_bucket_refinement()

def _dump_bucket_memory_stat() -> None:
    _debug_C.dump_bucket_memory_stat()

def _dump_history_memory_stat() -> None:
    _debug_C.dump_history_memory_stat()

def _dump_recipe_memory_stat() -> None:
    _debug_C.dump_recipe_memory_stat()

def _dump_synapse_recipe_memory_stat() -> None:
    _debug_C.dump_synapse_recipe_memory_stat()

def _dump_dynamic_shape_memory_stat() -> None:
    _debug_C.dump_dynamic_shape_memory_stat()

def load_ds_checkpoint(path) -> None:
    _debug_C.load_ds_checkpoint(path)

def save_ds_checkpoint(path) -> None:
    _debug_C.save_ds_checkpoint(path)

def _is_enabled_synapse_layout_handling() -> bool:
    return _debug_C.is_enabled_synapse_layout_handling()

def clear_dynamic_bucket_recipe_info() -> None:
    return _debug_C.clear_dynamic_bucket_recipe_info()

def _is_enabled_lazy_collectives() -> bool:
    return _debug_C.is_enabled_lazy_collectives()

def _hb_print(msg) -> None:
    return _debug_C.hb_print(msg)

def _mem_log(msg) -> bool:
    return _debug_C.mem_log(msg)
