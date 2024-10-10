import os

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn.functional as F
from test_utils import compare_tensors, env_var_in_scope, hpu

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


def embedding_func(x, y):
    a = F.embedding(x, y, padding_idx=0)
    b = torch.transpose(a, 0, 1)
    return torch.transpose(b, 0, 1)


test_case_list = [((0, 512), (30522, 768)), ((0, 512), (512, 768)), ((0, 1), (2, 768))]


@pytest.mark.parametrize("indices, weight", test_case_list)
@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
def test_embedding(indices, weight):
    with env_var_in_scope(
        {
            "PT_HPU_GRAPH_FUSION_OPS_FILE": os.path.join(
                os.environ["MODEL_GARDEN_PYTORCH_PATH"], "nlp/bert/BERT_Fusion_Ops.txt"
            )
        }
    ):
        x = torch.stack(
            (
                torch.arange(indices[0], indices[1]),
                torch.arange(indices[0], indices[1]),
                torch.arange(indices[0], indices[1]),
                torch.arange(indices[0], indices[1]),
            )
        )
        y = torch.randn(weight, requires_grad=True)
        hpu_x = x.to(torch.int).detach().to(hpu)
        hpu_y = y.detach().to(hpu)
        hpu_y.requires_grad = True

        with torch.jit.optimized_execution(True):
            # print(x,y)
            cpu_result = embedding_func(x, y)
            grad_out = torch.randn(cpu_result.size(), requires_grad=False)
            # print(cpu_result)
            cpu_result.backward(grad_out)
            cpu_grad = y.grad.detach()
            # print("CPU grad value")
            # print(cpu_grad)

            htcore.enable()
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            model_trace_hpu = torch.jit.trace(embedding_func, (hpu_x, hpu_y), check_trace=False)
            model_trace_hpu.graph_for(hpu_x, hpu_y)
            # print("Fused graph on HPU: ")
            # print(model_trace_hpu_graph)
            hpu_out = model_trace_hpu(hpu_x, hpu_y)
            # print(hpu_out.to(cpu))
            hpu_out.backward(grad_out.detach().to(hpu))
            hpu_grad = hpu_y.grad.detach()

            compare_tensors(hpu_out, cpu_result, atol=0.001, rtol=1.0e-3)
            compare_tensors(hpu_grad, cpu_grad, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_embedding(*test_case_list[0])
