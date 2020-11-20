import torch
from torch.testing import FileCheck
import pytest
import torch.nn as nn
import torch.nn.functional as F
from test_utils import reset_seed, compare_tensors
import hb_torch
import os


def slice_func(x):
    return x[0:5:2, 0:2]


test_case_list = [(5, 5)]


@pytest.mark.parametrize("D1, D2", test_case_list)
def test_slice_backward(D1, D2):
    hpu = torch.device("habana")
    cpu = torch.device("cpu")
    in_t = torch.randn(5, 5, requires_grad=True)
    hpu_t = in_t.detach().to(hpu)
    hpu_t.requires_grad = True

    with torch.jit.optimized_execution(True):
        os.environ["HABANA_GRAPH_WHITELIST_FILE"] = os.path.join(
            os.environ["PYTORCH_MODULES_ROOT_PATH"],
            "topologies/configs/BERT_whitelist_ops.txt",
        )
        cpu_result = slice_func(in_t)
        grad_out = torch.randn(3, 2, requires_grad=False)
        #print(grad_out)
        cpu_result.backward(grad_out)
        cpu_grad = in_t.grad.detach()
        #print("CPU grad value")
        #print(cpu_grad)

        hb_torch.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        model_trace_hpu = torch.jit.trace(slice_func, hpu_t, check_trace=False)
        model_trace_hpu_graph = model_trace_hpu.graph_for(hpu_t)
        #print("Fused graph on HPU: ")
        #print(model_trace_hpu_graph)
        hpu_out = model_trace_hpu(hpu_t)
        #print(hpu_out.to(cpu))
        hpu_out.backward(grad_out.detach().to(hpu))
        hpu_grad = hpu_t.grad.detach()

        compare_tensors(hpu_grad, cpu_grad, atol=0.001, rtol=1.0e-3)

    os.environ.pop("HABANA_GRAPH_WHITELIST_FILE")


if __name__ == "__main__":
    test_slice_backward(*test_case_list[0])
