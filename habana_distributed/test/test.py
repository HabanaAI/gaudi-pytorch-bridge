# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
import torch
import sys, traceback
import faulthandler
import torch.distributed as dist
import torch.distributed as c10d
faulthandler.enable(all_threads=True)

torch.ops.load_library("/usr/lib/habanalabs/libhabana_pytorch_plugin.so")
sys.path.insert(0, "/usr/lib/habanalabs")

a = torch.tensor([1.0, 2.0]).to('hpu')

try:
    import habana_torch_hcl
    dist.init_process_group("hcl")
    dist.all_reduce(a)
    print(a.to("cpu"))
except:
    print("Exception")
    traceback.print_exc()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)

#Command to run the test
#HCL_CONFIG_PATH=hls1.json python -um torch.distributed.launch --nproc_per_node=8 --use_env test.py
