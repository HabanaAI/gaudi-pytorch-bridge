import torch
import sys, traceback
import faulthandler
import torch.distributed as dist
import torch.distributed as c10d
faulthandler.enable(all_threads=True)
import os

os.environ['ID'] = os.getenv('RANK')

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

a = torch.tensor([1.0, 2.0]).to('hpu')

try:
    import habana_frameworks.torch.core
    comm_group1 = dist.init_process_group("hcl")
    dist.all_reduce(a, group=comm_group1)
    print(a.to("cpu"))
except:
    print("Exception")
    traceback.print_exc()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)


comm_group2 = dist.new_group(ranks=[0,1])

if int(os.getenv('RANK')) in [0,1]:
    dist.all_reduce(a, group=comm_group2)
    print(a.to('cpu'))

comm_group3 = dist.new_group(ranks=[2,3])

if int(os.getenv('RANK')) in [2,3]:
    a = torch.tensor([5.0, 10.0]).to('hpu')
    dist.all_reduce(a, group=comm_group3)
    print(a.to('cpu'))

#Command to run the test
#HCL_CONFIG_PATH=hls1.json python -um torch.distributed.launch --nproc_per_node=8 --use_env test.py

