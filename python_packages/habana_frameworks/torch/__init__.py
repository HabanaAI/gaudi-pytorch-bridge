import torch
import ctypes
import os

plugin_ver = None
if os.getenv("PT_HPU_LAZY_MODE", "0") != "0":
    plugin_ver = ""
else:
    if torch.__version__.startswith("2.0"):
        plugin_ver = "2"
    else:
        plugin_ver = ""

lib_to_load = "libhabana_pytorch{}_plugin.so".format(plugin_ver)
ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib", lib_to_load), ctypes.RTLD_GLOBAL)

import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu
import habana_frameworks.torch.activity_profiler
