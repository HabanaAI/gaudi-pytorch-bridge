import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import reset_seed, compare_tensors
import hb_torch

@torch.jit.script
def test_log_softmax(x):
    return F.log_softmax(x, 1)

hpu = torch.device("habana")
cpu = torch.device("cpu")
in_t = torch.randn(8, 10)

#m = torch.jit.trace(test_log_softmax, in_t)
#print(m.graph_for(in_t))
#print("Eager Mode..")

with torch.jit.optimized_execution(True):
    #print("--------------------")
    #print ("CPU IR Graph optimized")
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    #print(test_log_softmax.graph_for(in_t))
    model_trace = torch.jit.trace(test_log_softmax, in_t)
    torch.jit.save(model_trace, "cpu_trace.pt")
    model = test_log_softmax(in_t)
    cpu_result = model
    #print("Result CPU: " + str(model))
    #print("--------------------")

try:  
    hb_torch.enable()
    #print("--------------------")
    #print("Moving Tensors to HPU")
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_t = in_t.to(hpu)
    #print("--------------------")
    #print ("HPU IR Graph optimized")
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("habana"))
    print(model_trace_hpu.graph_for(hpu_t))
    out = model_trace_hpu(hpu_t)
    hpu_result = out.to(cpu)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)
    #print("Result HPU: " + str(hpu_result))
    #print("--------------------")
except RuntimeError as err:
    print ("Exiting after printing Fused Graph post fusion pass")
    print("OS error: {0}".format(err))
