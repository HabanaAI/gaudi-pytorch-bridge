###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


import os
import types
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Dict, Optional

import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.utils.debug as htdebug
import numpy as np
import pytest
import torch
from packaging.version import Version

hpu = torch.device("hpu")
cpu = torch.device("cpu")


def is_torch_at_least(req_ver_str: str):
    return Version(Version(torch.__version__).base_version) >= Version(req_ver_str)


def is_device(device_name):
    return hthpu.get_device_name() == device_name


def is_gaudi1():
    return is_device("GAUDI")


def is_gaudi2():
    return is_device("GAUDI2")


def is_gaudi3():
    return is_device("GAUDI3")


def is_lazy():
    return int(os.environ.get("PT_HPU_LAZY_MODE", 1)) == 1


def evaluate_fwd_kernel(
    kernel,
    kernel_params,
    check_results=True,
    atol=0.001,
    rtol=1.0e-3,
    copy_kernel=True,
):
    """Run given kernel with tensor_list as arguments on HPU and
    then CPU. Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function.
    Always return lists of outputs"""

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result = run_kernel_on_device(
        device=hpu,
        kernel=kernel,
        kernel_params=kernel_params,
        copy_kernel=copy_kernel,
    )

    cpu_result = run_kernel_on_device(device=cpu, kernel=kernel, kernel_params=kernel_params)

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=atol, rtol=rtol)

    return hpu_result, cpu_result


def evaluate_fwd_bwd_kernel(
    kernel,
    kernel_params_fwd,
    tensor_list_bwd,
    check_results_fwd=True,
    check_results_bwd=True,
    atol=0.001,
    rtol=1.0e-3,
    copy_kernel=True,
    grad_on_grad_enable=True,
):
    """Run given kernel fwd and bwd pass on HPU and then on CPU.
    Optionally check results and return them if user wants
    to process them latter e.g. to use custom comparison function"""
    # TODO: figure out how can we define kernel_params_bwd and use it instead of tensor_list_bwd

    # Order of operations matters. I am executing HPU first to fail early
    # in case of missing kernel. Furtheremore if we test inplace operators
    # we are still safe because we already copied tensors to HPU before running
    # CPU kernel.
    hpu_result_fwd = run_kernel_on_device(
        device=hpu,
        kernel=kernel,
        kernel_params=kernel_params_fwd,
        copy_kernel=copy_kernel,
    )

    if grad_on_grad_enable:
        hpu_result_bwd = run_kernel_on_device(
            device=hpu,
            kernel=hpu_result_fwd[0].grad_fn,
            tensor_list=tensor_list_bwd,
            copy_kernel=copy_kernel,
        )
    else:
        with torch.no_grad():
            hpu_result_bwd = run_kernel_on_device(
                device=hpu,
                kernel=hpu_result_fwd[0].grad_fn,
                tensor_list=tensor_list_bwd,
                copy_kernel=copy_kernel,
            )

    cpu_result_fwd = run_kernel_on_device(
        device=cpu,
        kernel=kernel,
        kernel_params=kernel_params_fwd,
        copy_kernel=copy_kernel,
    )

    if grad_on_grad_enable:
        cpu_result_bwd = run_kernel_on_device(
            device=cpu,
            kernel=cpu_result_fwd[0].grad_fn,
            tensor_list=tensor_list_bwd,
            copy_kernel=copy_kernel,
        )
    else:
        with torch.no_grad():
            cpu_result_bwd = run_kernel_on_device(
                device=cpu,
                kernel=cpu_result_fwd[0].grad_fn,
                tensor_list=tensor_list_bwd,
                copy_kernel=copy_kernel,
            )

    if check_results_fwd:
        compare_tensors(hpu_result_fwd, cpu_result_fwd, atol=atol, rtol=rtol)

    if check_results_bwd:
        compare_tensors(hpu_result_bwd, cpu_result_bwd, atol=atol, rtol=rtol)

    return (hpu_result_fwd, hpu_result_bwd), (cpu_result_fwd, cpu_result_bwd)


def evaluate_fwd_inplace_kernel(
    in_out_tensor,
    kernel_name,
    kernel_params,
    check_results=True,
    atol=0.001,
    rtol=1.0e-3,
):
    hpu_result = _run_inplace_kernel_on_device(
        device=hpu,
        in_out_tensor=in_out_tensor,
        kernel_name=kernel_name,
        kernel_params=kernel_params,
    )

    cpu_result = _run_inplace_kernel_on_device(
        device=cpu,
        in_out_tensor=in_out_tensor,
        kernel_name=kernel_name,
        kernel_params=kernel_params,
    )

    if check_results:
        compare_tensors(hpu_result, cpu_result, atol=atol, rtol=rtol)

    return hpu_result, cpu_result


def compare_tensors(hpu_tensors, cpu_tensors, atol, rtol, assert_enable=True):
    hpu_tensors = _convert_to_tensor_list(hpu_tensors)
    cpu_tensors = _convert_to_tensor_list(cpu_tensors)
    assert len(hpu_tensors) == len(cpu_tensors)

    hpu_tensors = [tensor.to(cpu) if tensor is not None else tensor for tensor in hpu_tensors]

    for i in range(len(hpu_tensors)):
        if cpu_tensors[i] is None and hpu_tensors[i] is None:
            continue

        hpu_tensors[i] = (
            hpu_tensors[i].float()
            if hpu_tensors[i].dtype in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
            else hpu_tensors[i]
        )
        cpu_tensors[i] = (
            cpu_tensors[i].float()
            if cpu_tensors[i].dtype in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
            else cpu_tensors[i]
        )
        if assert_enable:
            np.testing.assert_allclose(
                hpu_tensors[i].detach().numpy(),
                cpu_tensors[i].detach().numpy(),
                atol=atol,
                rtol=rtol,
            )
        else:
            print("hpu_result[{}]".format(i), hpu_tensors[i].detach().numpy())
            print("cpu_result[{}]".format(i), cpu_tensors[i].detach().numpy())
            return np.allclose(
                hpu_tensors[i].detach().numpy(),
                cpu_tensors[i].detach().numpy(),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )


@contextmanager
def env_var_in_scope(vars=None):
    def set_flag_in_env(name: str, value):
        assert (
            "PT_HPU_LAZY_MODE" != name
        ), "Setting PT_HPU_LAZY_MODE during test is forbidden. Use python3 -m pytest --mode argument instead"
        if value is None:
            os.environ[name] = ""
        else:
            os.environ[name] = str(value)

    orig_vars = {}
    vars = vars if vars else {}
    for key in vars.keys():
        orig_vars[key] = os.environ.get(key, None)
        set_flag_in_env(key, vars[key])
    try:
        yield
    finally:
        for key in orig_vars.keys():
            # restore environment variable
            if orig_vars[key] is not None:
                os.environ[key] = orig_vars[key]
            else:
                if key in os.environ:
                    del os.environ[key]


def generic_setup_teardown_env(temp_test_env: Dict, callback: Optional[Callable] = None):
    htdebug._bridge_cleanup()
    assert isinstance(temp_test_env, Mapping)

    print("Set env: ", temp_test_env)

    if callback:
        callback()

    with env_var_in_scope(vars=temp_test_env):
        yield

    print("Reset env.")


# fixutre that can be used for indirect initialization
@pytest.fixture
def setup_teardown_env_fixture(request):
    yield from generic_setup_teardown_env(request.param)


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
                        [i.to(device, dtype=torch.long) if i.type() == "torch.IntTensor" else i.to(device) for i in v]
                    )
                else:
                    kernel_params_local[k] = tuple([i.to(device) for i in v])
            elif isinstance(v, list) and (len(v) > 0) and isinstance(v[0], torch.Tensor):
                kernel_params_local[k] = [i.to(device) for i in v]
            else:
                kernel_params_local[k] = kernel_params[k]

    elif tensor_list:
        tensor_list = [tensor.to(device) if tensor is not None else tensor for tensor in tensor_list]

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
    if hasattr(kernel, "to"):
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


def _is_simulator():
    status = False
    if os.path.exists("/sys/class/accel/accel0/device/device_type"):
        import subprocess

        out = subprocess.Popen(
            ["cat", "/sys/class/accel/accel0/device/device_type"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = out.communicate()
        status = "SIM".lower() in str(stdout).lower()
    return status


class TcLimitedFormatter:
    def __init__(self, limit_array=None, limit_str=None):
        self.limit_array = limit_array
        self.limit_str = limit_str
        self.counter = 0

    def format_tc_common(self, val, limit_array=None, limit_str=None):
        """Formats test case parametrization argument. Function intended to use with tc params, to
        print them clearly in pytest --collect-only. If function not used params in printed not by value
        but with appended integer. For example it would be dims0, dims1, etc."""
        if isinstance(val, np.ndarray):
            val = val.tolist()

        if isinstance(val, torch.dtype):  # pylint: disable=no-member
            ret = repr(val)
            return ret.split(sep=".")[1]
        elif isinstance(val, tuple):
            if len(val) == 0:
                ret = 0
            else:
                assert val
                ret = self.format_tc_common(val[0], limit_array)
            for i in range(1, len(val)):
                ret = "{}x{}".format(ret, self.format_tc_common(val[i], limit_array))
            return "[{}]".format(ret)
        elif isinstance(val, list):
            if len(val) == 0:
                return "[]"
            limited = limit_array is not None and len(val) > 2 * limit_array
            ret = f"[{self.format_tc_common(val[0], limit_array)}"
            for i in range(1, len(val)):
                current_value = val[i]
                if limited:
                    if i == limit_array:
                        current_value = "_INNER{}_".format(self.counter)
                        self.counter += 1
                    elif i > limit_array and i < len(val) - limit_array:
                        continue

                ret = "{}x{}".format(ret, self.format_tc_common(current_value, limit_array))
            ret = "{}]".format(ret)
            if limit_str is not None and len(ret) > limit_str:
                ret = ret[0:limit_str] + "___{}".format(self.counter)
                self.counter += 1
            return ret
        elif val is None:
            return "_None_"
        elif isinstance(val, types.MethodDescriptorType):
            return val.__name__
        elif isinstance(val, types.BuiltinMethodType):
            return val.__name__
        else:
            s = str(val)

            pref_to_find = "<class '"
            prefix = s.find(pref_to_find)
            suffix = s.find("'>")
            if prefix >= 0 and suffix >= 0:
                s = s[prefix + len(pref_to_find) : suffix]

            s = s.replace("numpy", "np")

            return s

    def __call__(self, val):
        return self.format_tc_common(val, self.limit_array, self.limit_str)


def format_tc(val):
    """Formats test case parametrization argument. Function intended to use with tc params, to
    print them clearly in pytest --collect-only. If function not used params in printed not by value
    but with appended integer. For example it would be dims0, dims1, etc."""
    return TcLimitedFormatter()(val)


def is_pytest_mode_compile():
    return pytest.mode == "compile"


def is_pytest_mode_eager():
    return pytest.mode == "eager"


def is_pytest_mode_lazy():
    return pytest.mode == "lazy"


def clear_t_compile_logs():
    from habana_frameworks.torch.dynamo.compile_backend.passes import logger as graph_logger
    from habana_frameworks.torch.dynamo.compile_backend.shared_layer import logger as fallback_logger

    graph_logger.set_store_data(True)
    fallback_logger.set_store_data(True)


def compile_function_if_compile_mode(function):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        return torch.compile(function, backend="hpu_backend")
    else:
        return function


def check_ops_executed_in_jit_ir(op_names, verbose=False, allowed_fallbacks=set(), forbidden_ops=set()):
    import re

    from habana_frameworks.torch.dynamo.compile_backend.passes import logger as graph_logger
    from habana_frameworks.torch.dynamo.compile_backend.shared_layer import logger as fallback_logger

    graphs_data = graph_logger.data
    fallback_data = fallback_logger.data

    fallback_ops = set()
    non_fallback_ops = set()
    pattern = r"\[PT_COMPILE\] Node: (\w+) requires fallback: (\w+)"

    for log in fallback_data:
        m = re.match(pattern, log)
        if m:
            op = m.group(1)
            fallback = m.group(2)
            if fallback == "True":
                fallback_ops.add(op)
            elif fallback == "False":
                non_fallback_ops.add(op)

    all_ops = fallback_ops.union(non_fallback_ops)

    # due to special treatment of "_to_copy" in passes
    if "_to_copy" in op_names:
        all_ops.add("_to_copy")

    nodes_in_graphs = set()
    pattern = r"::(\w+)\("

    for log in graphs_data:
        if "####PyTorch-generated JIT IR" in log:
            for line in log.split("\n"):
                m = re.search(pattern, line)
                if m:
                    nodes_in_graphs.add(m.group(1))

    if not isinstance(op_names, set):
        op_names = {op_names}

    if verbose:
        print(f"{op_names = }")
        print(f"{allowed_fallbacks = }")
        print(f"{forbidden_ops = }")
        print(f"{fallback_ops = }")
        print(f"{non_fallback_ops = }")
        print(f"{all_ops = }")
        print(f"{nodes_in_graphs = }")

    op_names.difference_update(nodes_in_graphs)
    found_forbidden = nodes_in_graphs.intersection(forbidden_ops)

    if allowed_fallbacks:
        fallback_ops = fallback_ops - allowed_fallbacks

    if verbose:
        print(f"{op_names = }")
        print(f"{found_forbidden = }")

    graph_logger.set_store_data(False)
    fallback_logger.set_store_data(False)

    # The following loop filters out fallback ops even if they do not match the allowed fallback exactly,
    # it is needed to filter out multiple instances of the same op,
    # e.g. 'select_scatter' and 'select_scatter_1' will be filtered out for 'select_scatter'
    for allowed_fallback in allowed_fallbacks:
        fallback_ops = {elem for elem in fallback_ops if not elem.startswith(allowed_fallback)}

    assert all_ops, "No ops detected"
    assert not fallback_ops, f"These ops fell back to eager: {fallback_ops}"
    assert not op_names, f"Ops {op_names} were not found in the JIT IR graph"
    assert not found_forbidden, f"These forbidden ops were found in the JIT IR graph: {found_forbidden}"


def get_fuser_debug_logs_path():
    return os.path.join(os.environ["HABANA_LOGS"], "fuser_debug_logs")


@pytest.fixture
def clear_fuser_debug_logs():
    import shutil

    shutil.rmtree(get_fuser_debug_logs_path(), ignore_errors=True)
    yield
    shutil.rmtree(get_fuser_debug_logs_path(), ignore_errors=True)


def check_op_in_fuser_fused_ops(op_names):
    from os import listdir, path

    assert path.exists(get_fuser_debug_logs_path())

    import json
    from re import fullmatch

    fused_op_dump_file_names = [
        name
        for name in listdir(get_fuser_debug_logs_path())
        if fullmatch("fusergraph-[0-9]+-fused_kernel.*symbol\\.json", name)
    ]

    assert len(fused_op_dump_file_names) > 0

    for fused_op_dump_file_name in fused_op_dump_file_names:
        with open(path.join(get_fuser_debug_logs_path(), fused_op_dump_file_name)) as fused_op_dump_file:
            fused_op_data = json.load(fused_op_dump_file)

        for node in fused_op_data["nodes"]:
            if node["name"] in op_names:
                return True

    return False


def place_on_hpu(cpu_tensors):
    hpu_tensors = {}
    for key, value in cpu_tensors.items():
        hpu_tensors[key] = value.to("hpu")
    return hpu_tensors


def find_in_hier_list(v, hlist, index=[]):
    try:
        return index + [hlist.index(v)]
    except ValueError:
        for i, e in enumerate(hlist):
            if isinstance(e, Iterable):
                idx = find_in_hier_list(v, e, index + [i])
                if idx:
                    return idx
    return None


def is_dtype_floating_point(dtype):
    return torch.is_floating_point(torch.tensor((), dtype=dtype))


def print_tensors_internal(tensors, index=[]):
    if isinstance(tensors[0], Iterable):
        for i, (tensors_sub) in enumerate(zip(*tensors)):
            print_tensors_internal(tensors_sub, index + [i])
    else:
        len = 22
        s = ""
        for v in tensors:
            s += f"{v:{len}}"
        print(f"{index} {s}")


def print_tensors(labels, tensors):
    for l, t in zip(labels, tensors):
        print(f"{l} : {t.shape}")
    print_tensors_internal([t.tolist() for t in tensors])


def fga_assert_helper(ops_summary, op, count_list):
    assert len(ops_summary) == len(count_list)
    for single_graph_summary, graph_eager_count in zip(ops_summary, count_list):
        if graph_eager_count is None:
            assert op not in single_graph_summary
        else:
            graph_count, eager_count = graph_eager_count
            if graph_count != 0 or eager_count != 0:
                assert op in single_graph_summary
                assert single_graph_summary[op].graph_count == graph_count
                assert single_graph_summary[op].eager_count == eager_count
