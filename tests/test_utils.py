import pytest
import torch
import numpy as np
from copy import deepcopy
torch.ops.load_library("libhabana_pytorch_plugin.so")

hpu = torch.device('habana')
cpu = torch.device('cpu')


def evaluate_fwd_kernel(kernel, kernel_params, check_results=True):
    '''Run given kernel with tensor_list as arguments on HPU and
    then CPU. Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function.
    Always return lists of outputs'''

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result = _run_kernel_on_device(device=hpu,
                                       kernel=_kernel_copy_to_device(kernel, hpu),
                                       kernel_params=kernel_params)

    cpu_result = _run_kernel_on_device(device=cpu, kernel=kernel, kernel_params=kernel_params)

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)

    return hpu_result, cpu_result


def evaluate_fwd_bwd_kernel(kernel, kernel_params_fwd, tensor_list_bwd, check_results_fwd=True, check_results_bwd=True):
    '''Run given kernel fwd and bwd pass on HPU and then on CPU.
    Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function'''
    # TODO: figure out how can we define kernel_params_bwd and use it instead of tensor_list_bwd

    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result_fwd = _run_kernel_on_device(device=hpu,
                                           kernel=_kernel_copy_to_device(kernel, hpu),
                                           kernel_params=kernel_params_fwd)
    hpu_tensor_list_bwd = [t.to(hpu) for t in tensor_list_bwd]
    # TODO: add suport for multiple gradients
    hpu_result_bwd = _run_kernel_on_device(
        device=hpu,
        kernel=hpu_result_fwd[0].grad_fn,
        tensor_list=hpu_tensor_list_bwd)

    cpu_result_fwd = _run_kernel_on_device(
        device=cpu,
        kernel=kernel,
        kernel_params=kernel_params_fwd)
    cpu_result_bwd = _run_kernel_on_device(
        device=cpu,
        kernel=cpu_result_fwd[0].grad_fn,
        tensor_list=tensor_list_bwd)

    if check_results_fwd:
        compare_tensors(hpu_result_fwd, cpu_result_fwd, atol=0.001, rtol=1.e-3)

    if check_results_bwd:
        compare_tensors(hpu_result_bwd, cpu_result_bwd, atol=0.001, rtol=1.e-3)

    return (hpu_result_fwd, hpu_result_bwd), (cpu_result_fwd, cpu_result_bwd)


def compare_tensors(hpu_tensors, cpu_tensors, atol, rtol, assert_enable=True):
    hpu_tensors = _convert_to_tensor_list(hpu_tensors)
    cpu_tensors = _convert_to_tensor_list(cpu_tensors)

    assert len(hpu_tensors) == len(cpu_tensors)
    for i in range(len(hpu_tensors)):
        if assert_enable:
            np.testing.assert_allclose(hpu_tensors[i].to(cpu).detach().numpy(),
                                       cpu_tensors[i].detach().numpy(), atol=atol, rtol=rtol)
        else:
            print('hpu_result[{}]'.format(i), hpu_tensors[i].to(cpu).detach().numpy())
            print('cpu_result[{}]'.format(i), cpu_tensors[i].detach().numpy())
            return np.allclose(hpu_tensors[i].to(cpu).detach().numpy(),
                               cpu_tensors[i].detach().numpy(), atol=atol, rtol=rtol, equal_nan=True)


@pytest.fixture(autouse=True)
def reset_seed(seed=[0xC001A1]):
    print("Using seed: ", seed[0])
    torch.manual_seed(seed[0])
    np.random.seed(seed[0])
    seed[0] += 1  # I want to change data between runs not only shapes

    # TODO: for future use
    # random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def _assert_tensors_on_device(tensor_list, device):
    for t in tensor_list:
        assert t.device.type == device.type


def _run_kernel_on_device(device, kernel, tensor_list=None, kernel_params=None):
    # print("tensor_list", tensor_list)
    # print("kernel_params", kernel_params)

    if kernel_params and tensor_list:
        raise RuntimeError("Pass tensors using kernel_params")

    if kernel_params:
        assert isinstance(kernel_params, dict)
        for k, v in kernel_params.items():
            if isinstance(v, torch.Tensor):
                kernel_params[k] = v.to(device)

    result = kernel(**kernel_params) if kernel_params else kernel(*tensor_list)

    return _convert_to_tensor_list(result)


def _kernel_copy_to_device(kernel, device):
    if hasattr(kernel, 'to'):
        kernel_copy = deepcopy(kernel)
        return kernel_copy.to(device)
    else:
        return kernel


def _convert_to_tensor_list(tensor_or_tensors):
    if isinstance(tensor_or_tensors, tuple):
        return list(tensor_or_tensors)
    elif isinstance(tensor_or_tensors, list):
        return tensor_or_tensors
    elif isinstance(tensor_or_tensors, torch.Tensor):
        # You can't return list(tensor_or_tensors), because it will fail on 0-d tensors
        result_list = []
        result_list.append(tensor_or_tensors)
        return result_list
    else:
        raise TypeError("Can not convert outputs")
