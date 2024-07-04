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
from habana_frameworks.torch.utils import _debug_C
from habana_frameworks.torch.utils.debug.logger import Logger
from habana_frameworks.torch.utils.internal import is_lazy


def _get_fallback_op_count() -> dict:
    return _debug_C.get_fallback_op_count()


def _get_shape_agnostic_unsupported_ops() -> set:
    return _debug_C.get_shape_agnostic_unsupported_ops()


def _get_jit_cache_size() -> set:
    return _debug_C.get_jit_cache_size()


def _clear_jit_cache() -> set:
    return _debug_C.clear_jit_cache()


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


def _enable_bn_param_recalculation(flag) -> None:
    _debug_C.enable_bn_param_recalculation(flag)


def _enable_fuse_conv_bn_optimization(flag) -> None:
    _debug_C.enable_fuse_conv_bn_optimization(flag)


def _enable_permute_pass(flag) -> None:
    _debug_C.enable_permute_pass(flag)


def _enable_replace_inplace_ops(flag) -> None:
    _debug_C.enable_replace_inplace_ops(flag)


def _enable_replace_views(flag) -> None:
    _debug_C.enable_replace_views(flag)


def _is_enabled_weight_permute_pass() -> bool:
    return False


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
    return True


def clear_dynamic_bucket_recipe_info() -> None:
    return _debug_C.clear_dynamic_bucket_recipe_info()


def _is_enabled_lazy_collectives() -> bool:
    return _debug_C.is_enabled_lazy_collectives()


def _hb_print(msg) -> None:
    return _debug_C.hb_print(msg)


def _mem_log(msg) -> bool:
    return _debug_C.mem_log(msg)


def _hg_print(msg) -> None:
    _debug_C.hg_print(msg)


if is_lazy():
    from habana_frameworks.torch.utils import _debug_lazy_C

    def _bridge_cleanup():
        _debug_lazy_C.bridge_cleanup()

    def get_tensor_info(tensor):
        if "hpu" not in str(tensor.device):
            return None
        return _debug_lazy_C.get_tensor_info(tensor)

else:
    from habana_frameworks.torch.utils import _debug_eager_C

    def _bridge_cleanup():
        _debug_eager_C.bridge_cleanup()

    def get_tensor_info(tensor):
        if "hpu" not in str(tensor.device):
            return None

        storage = tensor.storage()
        return (storage.data_ptr(), storage.nbytes())


def _dump_memory_reporter() -> None:
    _debug_C.dump_memory_reporter()


def _towl_configure(flag: bool, config=""):
    _debug_C.towl_configure(flag, config)


def _towl_print(text: str):
    _debug_C.towl_print(text)
