import pytest
import torch
import numpy as np
from copy import deepcopy
torch.ops.load_library("libhabana_pytorch_plugin.so")

hpu = torch.device('habana')
cpu = torch.device('cpu')

def evaluate_fwd_kernel(kernel, tensor_list, kernel_params={}, check_results=True):
    '''Run given kernel with tensor_list as arguments on HPU and
    then CPU. Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function.
    Always return lists of outputs'''

    def _kernel_copy_to_device(kernel, device):
        if hasattr(kernel, 'to'):
            kernel_copy = deepcopy(kernel)
            return kernel_copy.to(device)
        else:
            return kernel

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_tensor_list = [t.to(hpu) for t in tensor_list]
    hpu_result = _run_kernel_on_device(hpu, _kernel_copy_to_device(kernel, hpu), hpu_tensor_list, kernel_params)

    cpu_result = _run_kernel_on_device(cpu, kernel, tensor_list, kernel_params)

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)

    return hpu_result, cpu_result

def evaluate_fwd_bwd_kernel(kernel, tensor_list_fwd, tensor_list_bwd, kernel_params={}, check_results_fwd=True, check_results_bwd=True):
    '''Run given kernel fwd and bwd pass on HPU and then on CPU.
    Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function'''

    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_tensor_list_fwd = [t.to(hpu) for t in tensor_list_fwd]
    hpu_result_fwd = _run_kernel_on_device(hpu, kernel, hpu_tensor_list_fwd, kernel_params)
    hpu_tensor_list_bwd = [t.to(hpu) for t in tensor_list_bwd]
    # TODO: add suport for multiple gradients
    hpu_result_bwd = _run_kernel_on_device(hpu, hpu_result_fwd[0].grad_fn, hpu_tensor_list_bwd)

    cpu_result_fwd = _run_kernel_on_device(cpu, kernel, tensor_list_fwd)
    cpu_result_bwd = _run_kernel_on_device(cpu, cpu_result_fwd[0].grad_fn, tensor_list_bwd)

    if check_results_fwd:
        compare_tensors(hpu_result_fwd, cpu_result_fwd, atol=0.001, rtol=1.e-3)

    if check_results_bwd:
        compare_tensors(hpu_result_bwd, cpu_result_bwd, atol=0.001, rtol=1.e-3)

    return (hpu_result_fwd, hpu_result_bwd), (cpu_result_fwd, cpu_result_bwd)

def _assert_tensors_on_device(tensor_list, device):
    for t in tensor_list:
        assert t.device.type == device.type

def _run_kernel_on_device(device, kernel, tensor_list, kernel_params={}):
    _assert_tensors_on_device(tensor_list, device)
    assert isinstance(kernel_params, dict)

    result = kernel(*tensor_list, **kernel_params)

    if isinstance(result, tuple):
        return list(result)
    elif isinstance(result, torch.Tensor):
        # You can't return list(result), because it will fail on 0-d tensors
        result_list = []
        result_list.append(result)
        return result_list
    else:
        raise TypeError("Can not convert outputs")

def compare_tensors(hpu_tensors, cpu_tensors, atol, rtol):
    if not isinstance(hpu_tensors, list):
        hpu_tensors = list(hpu_tensors)
    if not isinstance(cpu_tensors, list):
        cpu_tensors = list(cpu_tensors)

    assert len(hpu_tensors) == len(cpu_tensors)
    for i in range(len(hpu_tensors)):
        np.testing.assert_allclose(hpu_tensors[i].to(cpu).detach().numpy(), cpu_tensors[i].detach().numpy(), atol=atol, rtol=rtol)

@pytest.fixture(autouse=True)
def reset_seed(seed=[0xC001A1]):
    print("Using seed: ", seed[0])
    torch.manual_seed(seed[0])
    np.random.seed(seed[0])
    seed[0] += 1 # I want to change data between runs not only shapes

    # TODO: for future use
    # random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True