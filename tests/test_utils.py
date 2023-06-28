import os
import pytest
import torch
import numpy as np
from copy import deepcopy
import habana_frameworks.torch
from collections.abc import Mapping
from typing import Callable, Dict, Optional
import habana_frameworks.torch.hpu as hthpu

hpu = torch.device('hpu')
cpu = torch.device('cpu')


def is_device(device_name):
    return hthpu.get_device_name() == device_name

def is_gaudi3():
    return hthpu.get_device_name() == "GAUDI3"

def is_gaudi1():
    return is_device("GAUDI")


def is_gaudi2():
    return is_device("GAUDI2")


def is_gaudi3():
    return is_device("GAUDI3")


def evaluate_fwd_kernel(kernel, kernel_params, check_results=True, atol=0.001, rtol=1.e-3, copy_kernel=True):
    '''Run given kernel with tensor_list as arguments on HPU and
    then CPU. Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function.
    Always return lists of outputs'''

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result = run_kernel_on_device(device=hpu,
                                      kernel=kernel,
                                      kernel_params=kernel_params, copy_kernel=copy_kernel)

    cpu_result = run_kernel_on_device(device=cpu, kernel=kernel, kernel_params=kernel_params)

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=atol, rtol=rtol)

    return hpu_result, cpu_result


def evaluate_fwd_bwd_kernel(kernel, kernel_params_fwd, tensor_list_bwd, check_results_fwd=True, check_results_bwd=True, atol=0.001, rtol=1.e-3, copy_kernel=True, grad_on_grad_enable=True):
    '''Run given kernel fwd and bwd pass on HPU and then on CPU.
    Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function'''
    # TODO: figure out how can we define kernel_params_bwd and use it instead of tensor_list_bwd

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result_fwd = run_kernel_on_device(device=hpu,
                                          kernel=kernel,
                                          kernel_params=kernel_params_fwd, copy_kernel=copy_kernel)

    if grad_on_grad_enable:
      hpu_result_bwd = run_kernel_on_device(
          device=hpu,
          kernel=hpu_result_fwd[0].grad_fn,
          tensor_list=tensor_list_bwd, copy_kernel=copy_kernel)
    else:
      with torch.no_grad():
        hpu_result_bwd = run_kernel_on_device(
            device=hpu,
            kernel=hpu_result_fwd[0].grad_fn,
            tensor_list=tensor_list_bwd, copy_kernel=copy_kernel)

    cpu_result_fwd = run_kernel_on_device(
        device=cpu,
        kernel=kernel,
        kernel_params=kernel_params_fwd, copy_kernel=copy_kernel)

    if grad_on_grad_enable:
      cpu_result_bwd = run_kernel_on_device(
         device=cpu,
         kernel=cpu_result_fwd[0].grad_fn,
         tensor_list=tensor_list_bwd, copy_kernel=copy_kernel)
    else:
      with torch.no_grad():
        cpu_result_bwd = run_kernel_on_device(
            device=cpu,
            kernel=cpu_result_fwd[0].grad_fn,
            tensor_list=tensor_list_bwd, copy_kernel=copy_kernel)

    if check_results_fwd:
        compare_tensors(hpu_result_fwd, cpu_result_fwd, atol=atol, rtol=rtol)

    if check_results_bwd:
        compare_tensors(hpu_result_bwd, cpu_result_bwd, atol=atol, rtol=rtol)

    return (hpu_result_fwd, hpu_result_bwd), (cpu_result_fwd, cpu_result_bwd)


def evaluate_fwd_inplace_kernel(in_out_tensor, kernel_name, kernel_params, check_results=True, atol=0.001, rtol=1.e-3):
    hpu_result = _run_inplace_kernel_on_device(device=hpu,
                                               in_out_tensor=in_out_tensor,
                                               kernel_name=kernel_name,
                                               kernel_params=kernel_params)

    cpu_result = _run_inplace_kernel_on_device(device=cpu,
                                               in_out_tensor=in_out_tensor,
                                               kernel_name=kernel_name,
                                               kernel_params=kernel_params)

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=atol, rtol=rtol)

    return hpu_result, cpu_result


def compare_tensors(hpu_tensors, cpu_tensors, atol, rtol, assert_enable=True):
    hpu_tensors = _convert_to_tensor_list(hpu_tensors)
    cpu_tensors = _convert_to_tensor_list(cpu_tensors)
    assert len(hpu_tensors) == len(cpu_tensors)
    for i in range(len(hpu_tensors)):
        if cpu_tensors[i] is None and hpu_tensors[i] is None:
            continue

    hpu_tensors = [tensor.to(cpu) if tensor is not None else tensor for tensor in hpu_tensors]

    for i in range(len(hpu_tensors)):
        if cpu_tensors[i] is None and hpu_tensors[i] is None:
            continue
        elif assert_enable:
            hpu_tensors[i] = hpu_tensors[i].float() if hpu_tensors[i].dtype == torch.bfloat16 else hpu_tensors[i]
            cpu_tensors[i] = cpu_tensors[i].float() if cpu_tensors[i].dtype == torch.bfloat16 else cpu_tensors[i]
            np.testing.assert_allclose(hpu_tensors[i].detach().numpy(),
                                cpu_tensors[i].detach().numpy(), atol=atol, rtol=rtol)
        else:
            hpu_tensors[i] = hpu_tensors[i].float() if hpu_tensors[i].dtype == torch.bfloat16 else hpu_tensors[i]
            cpu_tensors[i] = cpu_tensors[i].float() if cpu_tensors[i].dtype == torch.bfloat16 else cpu_tensors[i]
            print('hpu_result[{}]'.format(i), hpu_tensors[i].detach().numpy())
            print('cpu_result[{}]'.format(i), cpu_tensors[i].detach().numpy())
            return np.allclose(hpu_tensors[i].detach().numpy(),
                                cpu_tensors[i].detach().numpy(), atol=atol, rtol=rtol, equal_nan=True)


def generic_setup_teardown_env(temp_test_env: Dict, callback: Optional[Callable] = None):
    assert isinstance(temp_test_env, Mapping)

    for k,v in temp_test_env.items():
        temp_test_env[k] = str(v)

    old_env = dict(os.environ)
    print("Set env: ", temp_test_env)
    os.environ.update(temp_test_env)

    if callback:
        callback()

    yield

    print("Reset env.")
    os.environ.clear()
    os.environ.update(old_env)

# fixutre that can be used for indirect initialization
@pytest.fixture
def setup_teardown_env_fixture(request):
    yield from generic_setup_teardown_env(request.param)

@pytest.fixture(autouse=True)
def reset_seed(seed=0xC001A1):
    print("Using seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # TODO: for future use
    # random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def _assert_tensors_on_device(tensor_list, device):
    for t in tensor_list:
        assert t.device.type == device.type


def run_kernel_on_device(device, kernel, tensor_list=None, kernel_params=None, copy_kernel=True):
    # print("tensor_list", tensor_list)
    # print("kernel_params", kernel_params)
    if copy_kernel:
        kernel = _kernel_copy_to_device(kernel, device)

    if kernel_params and tensor_list:
        raise RuntimeError("Pass tensors using kernel_params")

    if kernel_params:
        # create local version of kernel params dict,
        # else some values are retained across calls
        kernel_params_local = {}
        assert isinstance(kernel_params, dict)
        for k, v in kernel_params.items():
            if isinstance(v, torch.Tensor):
                kernel_params_local[k] = v.to(device)
            elif isinstance(v, tuple) and (len(v) > 0) and isinstance(v[0], torch.Tensor):
                if device == cpu:
                    # HPU does not support dtype=long, therefore use dtype=int
                    # in test-cases and convert it to dtype=long for CPU (CPU
                    # works for dtype=long only)
                    kernel_params_local[k] = tuple(
                        [i.to(device, dtype=torch.long) if i.type() == 'torch.IntTensor' else i.to(device) for i in v])
                else:
                    kernel_params_local[k] = tuple([i.to(device) for i in v])
            elif isinstance(v, list) and (len(v) > 0) and isinstance(v[0], torch.Tensor):
                kernel_params_local[k] = [i.to(device) for i in v]
            else:
                kernel_params_local[k] = kernel_params[k]


    elif tensor_list:
        tensor_list = [tensor.to(device) if tensor != None else tensor for tensor in tensor_list]

    result = kernel(**kernel_params_local) if kernel_params else kernel(*tensor_list)

    return _convert_to_tensor_list(result)


def _run_inplace_kernel_on_device(device, in_out_tensor, kernel_name, tensor_list=None, kernel_params=None):
    assert isinstance(in_out_tensor, torch.Tensor)
    if kernel_params and tensor_list:
        raise RuntimeError("Pass tensors using kernel_params")

    in_out_tensor = in_out_tensor.to(device)
    if kernel_params:
        assert isinstance(kernel_params, dict)
        for k, v in kernel_params.items():
            if isinstance(v, torch.Tensor):
                kernel_params[k] = v.to(device)
    elif tensor_list:
        tensor_list = [tensor.to(device) for tensor in tensor_list]

    if kernel_params:
        result = getattr(in_out_tensor, kernel_name)(**kernel_params)
    elif tensor_list:
        result = getattr(in_out_tensor, kernel_name)(*tensor_list)
    else:
        # unary in place kernels
        result = getattr(in_out_tensor, kernel_name)()

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
