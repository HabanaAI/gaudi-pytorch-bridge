import torch
import torch.nn.functional as F
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

test_case_list = [
   #  N,   C,
   (500,  10,),
]

@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_nllloss(N, C):
    # TODO: extend that test to all features of nll_loss kernel
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    kernel = F.nll_loss
    in_tensor = torch.randn(N, C)
    target = torch.randint(low=0, high=C-1, size=(N,))

    hpu_result = kernel(in_tensor.to(hpu), target.to(hpu))
    cpu_result = kernel(in_tensor.to(cpu), target.to(cpu))

    # print("input", in_tensor)
    # print("target", target)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result.to(cpu))
    np.testing.assert_allclose(hpu_result.to(cpu).detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_nllloss(*test_case_list[0])
