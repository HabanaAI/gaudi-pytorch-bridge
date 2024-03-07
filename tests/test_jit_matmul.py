import os

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")

import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import compare_tensors, env_var_in_scope, hpu


def matmul_func(x, y):
    z = torch.matmul(x, y)
    return torch.transpose(z, 1, 2)


test_case_list = [
    ((12, 384, 1024), (1024, 4096)),
    ((12, 16, 384, 64), (12, 16, 64, 384)),
]


@pytest.mark.parametrize("D1, D2", test_case_list)
@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
def test_matmul(D1, D2):
    with env_var_in_scope(
        {
            "PT_HPU_GRAPH_FUSION_OPS_FILE": os.path.join(
                os.environ["MODEL_GARDEN_PYTORCH_PATH"], "nlp/bert/BERT_Fusion_Ops.txt"
            )
        }
    ):
        mat1 = torch.randn(D1, requires_grad=True)
        mat2 = torch.randn(D2, requires_grad=True)
        hpu_mat1 = mat1.detach().to(hpu)
        hpu_mat2 = mat2.detach().to(hpu)
        hpu_mat1.requires_grad = True
        hpu_mat2.requires_grad = True

        with torch.jit.optimized_execution(True):
            cpu_result = matmul_func(mat1, mat2)
            grad_out = torch.randn(cpu_result.size(), requires_grad=False)
            cpu_result.backward(grad_out)
            cpu_mat1_grad = mat1.grad.detach()
            cpu_mat2_grad = mat2.grad.detach()
            # print(cpu_mat1_grad.dim(), cpu_mat2_grad.dim())

            htcore.enable()
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            model_trace_hpu = torch.jit.trace(matmul_func, (hpu_mat1, hpu_mat2), check_trace=False)
            model_trace_hpu.graph_for(hpu_mat1, hpu_mat2)
            # print(model_trace_hpu_graph)
            hpu_out = model_trace_hpu(hpu_mat1, hpu_mat2)
            hpu_out.backward(grad_out.detach().to(hpu))
            hpu_mat1_grad = hpu_mat1.grad.detach()
            hpu_mat2_grad = hpu_mat2.grad.detach()

            compare_tensors(hpu_out, cpu_result, atol=0.01, rtol=1.0e-2)
            compare_tensors(hpu_mat1_grad, cpu_mat1_grad, atol=0.01, rtol=1.0e-2)
            compare_tensors(hpu_mat2_grad, cpu_mat2_grad, atol=0.01, rtol=1.0e-2)


if __name__ == "__main__":
    test_matmul(*test_case_list[0])
