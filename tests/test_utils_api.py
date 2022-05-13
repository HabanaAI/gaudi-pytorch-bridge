from habana_frameworks.torch.utils import debug as htdebug
htdebug._set_dynamic_mode()
htdebug._set_module_name("test_name")
htdebug._enable_eliminate_common_subexpression(False)
htdebug._enable_eliminate_dead_code(False)
htdebug._enable_constant_pooling(False)
htdebug._enable_peephole_optimization(False)
htdebug._enable_fuse_t_mm_optimization(False)
htdebug._enable_fuse_bn_relu_optimization(False)
htdebug._enable_permute_pass(False)
htdebug._enable_replace_inplace_ops(False)
htdebug._enable_replace_views(False)
htdebug._enable_weight_permute_pass(False)
print(htdebug._is_enabled_weight_permute_pass())
htdebug._enable_weight_permute_pass(True)
print(htdebug._is_enabled_weight_permute_pass())
#htdebug._run_saved_model()

htdebug._memstat_livealloc("test memory live alloc")
htdebug._memstat_devmem_start_collect("test memory start collect")
htdebug._memstat_devmem_start_collect("test memory start collect", False)
htdebug._memstat_devmem_stop_collect("test memory stop collect")

htdebug._dump_refined_recipe_stat()
htdebug._disable_bucket_refinement()
htdebug._dump_bucket_memory_stat()
htdebug._dump_history_memory_stat()
htdebug._dump_recipe_memory_stat()
htdebug._dump_synapse_recipe_memory_stat()
htdebug._dump_dynamic_shape_memory_stat()

print(htdebug._is_enabled_synapse_layout_handling())

from habana_frameworks.torch.utils import profiler as htprofiler
htprofiler._setup_profiler()
htprofiler._start_profiler()
htprofiler._stop_profiler()

import torch
import habana_frameworks.torch.utils.experimental as htexp
print("device_type", htexp._get_device_type())
print("compute_stream", htexp._compute_stream())
x = torch.randn(10, device='hpu')
print("data_ptr", htexp._data_ptr(x))
device_type = htexp._get_device_type()
if (device_type == htexp.synDeviceType.synDeviceGaudi):
    print("gaudi")