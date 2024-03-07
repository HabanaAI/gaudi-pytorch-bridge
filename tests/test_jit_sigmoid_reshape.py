import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
from test_utils import compare_tensors
from torch.testing import FileCheck

test_case_list = [
    # D1, D2, D3, D4
    (2, 3, 2, 3),
]


class SigmoidReshape(nn.Module):
    def __init__(self):
        super(SigmoidReshape, self).__init__()

    def forward(self, x):
        y = x.reshape(1, x.numel())
        return torch.sigmoid(y)


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
@pytest.mark.parametrize("D1, D2, D3, D4", test_case_list)
def test_sigmoid_reshape(D1, D2, D3, D4):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    in_t = torch.randn(D1, D2, D3, D4)
    hpu_t = in_t.to(hpu)

    # Verify eager mode
    eager_model = SigmoidReshape()
    cpu_eager_result = eager_model(in_t)
    hpu_eager_result = eager_model(hpu_t).to(cpu)
    compare_tensors(hpu_eager_result, cpu_eager_result, atol=0.001, rtol=1.0e-3)

    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model_trace = torch.jit.trace(SigmoidReshape(), (in_t))
        cpu_result = model_trace(in_t)

        htcore.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        model_trace_hpu = torch.jit.trace(SigmoidReshape(), (hpu_t), check_trace=False)
        model_trace_hpu_graph = model_trace_hpu.graph_for(hpu_t)
        FileCheck().check_count("= prim::HabanaFusedOp_0", 2, exactly=True).run(str(model_trace_hpu_graph))
        hpu_result = model_trace_hpu(hpu_t).to(cpu)
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_sigmoid_reshape(*test_case_list[0])
