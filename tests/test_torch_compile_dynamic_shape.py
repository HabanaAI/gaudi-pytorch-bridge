###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import torch
import torch.nn.functional as F
import numpy as np
import pytest
import torch.nn as nn

from contextlib import contextmanager
os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"

def set_flag_in_env(name: str, value):
    if value is None:
        # Nothing to do here
        return
    elif isinstance(value, str):
        os.environ[name] = value
    elif isinstance(value, bool):
        os.environ[name] = str(int(value))
    elif isinstance(value, int):
        os.environ[name] = str(value)
    else:
        assert False, f"Value '{value}' invalid or not supported"


@contextmanager
def env_var_in_scope(vars={}):
    orig_vars = {}
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



def test_relu_mixed():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore
        def raw_function(x):
            tmp1 = x * 2 - 1
            return torch.relu(tmp1)

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend", dynamic=True)

        tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

        result_nocompile = 2 * raw_function(tensor) + 3

        result_compile_train = compiled_function_training(tensor)
        result_mixed_train = 2 * result_compile_train + 3
        result_mixed_infer = 2 * compiled_function_inference(tensor) + 3
        assert torch.allclose(result_nocompile, result_mixed_train)
        assert torch.allclose(result_mixed_infer, result_mixed_train)


def test_reshape_symlnt():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore
        # TODO - Remove fallback option once symlnt is supported
        torch._dynamo.config.suppress_errors = True

        input_shapes = [
           (3, 6, 4),
           (3, 8, 4),
           (3, 10, 4)
        ]

        input_shapes2 = [
           (3, 24),
           (3, 32),
           (3, 40)
        ]

        def raw_function(t1, x2):
            t = t1.shape
            t1 = torch.relu(t1)
            shape = (t[0], int(t[1] * t[2]))
            t2 = t1.reshape(shape)
            t3 = torch.add(t2, x2)
            return t3

        t1 = torch.randn(input_shapes[0], requires_grad = False)
        t2 = torch.randn(input_shapes2[0], requires_grad = False)
        out_c = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
        result_compile_train = compiled_function_training(t1_h, t2_h)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_view():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore
        # TODO - Remove fallback option once symlnt is supported
        torch._dynamo.config.suppress_errors = True

        input_shapes = [
           (3, 6, 4),
           (3, 8, 4),
           (3, 10, 4)
        ]

        input_shapes2 = [
           (3, 24),
           (3, 32),
           (3, 40)
        ]

        def raw_function(t1, x2):
            t = t1.shape
            t1 = torch.relu(t1)
            shape = (t[0], int(t[1] * t[2]))
            t2 = t1.reshape(shape)
            t3 = torch.add(t2, x2)
            return t3

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s, s2 in zip(input_shapes, input_shapes2):
            t1 = torch.randn(s, requires_grad = False)
            t2 = torch.randn(s2, requires_grad = False)
            out_c = raw_function(t1, t2)
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            result_compile_train = compiled_function_training(t1_h, t2_h)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_simple():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        input_shapes = [
           (3, 6, 4),
           (3, 8, 4),
           (3, 10, 4)
        ]

        def raw_function(t1, t2):
            t3 = torch.mul(t1, t2)
            tmp1 = t3 - 1
            return torch.relu(tmp1)

        for s in input_shapes:
            t1 = torch.randn(s, requires_grad = False)
            t2 = torch.randn(s, requires_grad = False)
            out_c = raw_function(t1, t2)
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
            result_compile_train = compiled_function_training(t1_h, t2_h)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_topk():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        sizes = [5, 10, 15, 18, 16]

        def raw_function(t):
            k = t.shape[0] // 5
            out_hpu = torch.topk(t, k)
            hpu_value0 = out_hpu[0]
            return hpu_value0

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s in sizes:
            t = torch.randn(s)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h)
            out_cpu = raw_function(t)
            assert torch.allclose(result_compile_train.to("cpu"), out_cpu)

def test_dynamic_shape_topk_static_k():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        sizes = [5, 10, 15, 18, 16]
        K = [1, 2, 3, 4, 5]

        def raw_function(t, k):
            out_hpu = torch.topk(t, k)
            hpu_value0 = out_hpu[0]
            return hpu_value0

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
        i = 0
        for s in sizes:
            t = torch.randn(s)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h, K[i])
            out_cpu = raw_function(t, K[i])
            i = i + 1
            assert torch.allclose(result_compile_train.to("cpu"), out_cpu)

def test_dynamic_shape_topk_static_same_k():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        sizes = [5, 10, 15, 18, 16]
        K = [1, 1, 1, 1, 1]

        def raw_function(t, k):
            out_hpu = torch.topk(t, k)
            hpu_value0 = out_hpu[0]
            return hpu_value0

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
        i = 0
        for s in sizes:
            t = torch.randn(s, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h, K[i])
            out_cpu = raw_function(t, K[i])
            i = i + 1
            assert torch.allclose(result_compile_train.to("cpu"), out_cpu)

def test_dynamic_shape_topk_lazy():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "1", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        sizes = [5, 10, 15, 18, 16]
        K = [1, 2, 3, 4, 5]

        def raw_function(t1, k):
            out_hpu = torch.topk(t1, k)
            hpu_value0 = out_hpu[0]
            return hpu_value0
        i = 0
        for s in sizes:
            t1 = torch.randn(s).to("hpu")
            result_compile_train = raw_function(t1, K[i])
            i = i + 1
            out_cpu = raw_function(t, K[i])
            i = i + 1
            assert torch.allclose(result_compile_train.to("cpu"), out_cpu)

def test_dynamic_shape_repeat_static():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        H=4
        input = [[4, 10], [4, 231], [4, 520]]
        sizes = [5, 1, 1]

        def raw_function(input_tensor, sizes):
            out = input_tensor.repeat(sizes)
            return out

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s in input:
            t = torch.randn(s, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h, sizes)
            out_c = raw_function(t, sizes)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_repeat_static_lazy():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "1", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        H=4
        input = [[4, 10], [4, 231], [4, 520]]
        sizes = [5, 1, 1]

        def raw_function(input_tensor, sizes):
            out = input_tensor.repeat(sizes)
            return out

        for s in input:
            t = torch.randn(s, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = raw_function(t_h, sizes)
            out_c = raw_function(t, sizes)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_repeat():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        H=4
        input = [[4, 10], [5, 231], [6, 250]]
        sizes = [5, 1, 1]

        def raw_function(input_tensor, sizes):
            s = input_tensor.shape
            d1 = s[0]+1
            out = input_tensor.repeat([d1, d1, d1])
            return out

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s in input:
            t = torch.randn(s, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = compiled_function_training(t_h, sizes)
            out_c = raw_function(t, sizes)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_repeat_lazy():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "1", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        H=4
        input = [[4, 10], [4, 231], [4, 520]]
        sizes = [5, 1, 1]

        def raw_function(input_tensor, sizes):
            s = input_tensor.shape
            d1 = s[0]+1
            out = input_tensor.repeat([d1, d1, d1])
            return out

        for s in input:
            t = torch.randn(s, requires_grad = False)
            t_h = t.to("hpu")
            result_compile_train = raw_function(t_h, sizes)
            out_c = raw_function(t, sizes)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_control_flow_static():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):

        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        sizes = [5, 10, 15, 18, 16]
        sizes1 = [6, 11, 14, 17, 17]

        def raw_function(t1, t2):
            if t1 < t2:
                out_hpu = torch.add(t1, t2)
            else:
                out_hpu = torch.add(t2, t1)
            return out_hpu

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)
        i = 0
        for s in sizes:
            t1 = torch.tensor(s)
            t2 = torch.tensor(sizes1[i])
            i = i + 1
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            result_compile_train = compiled_function_training(t1_h, t2_h)
            out_c = raw_function(t1, t2)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

def test_dynamic_shape_mult_module_split():
    print("Starting...................")
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore
        print("Starting the test.................")
        input_shapes = [
           (3, 6, 4),
           (3, 8, 4),
           (3, 10, 4)
        ]
        input_shapes2 = [
           (3, 24),
           (3, 32),
           (3, 40)
        ]

        def raw_function(t1, x2):
            t1 = torch.relu(t1)
            t = t1.shape
            shape = (t[0], int(t[1] * t[2]))
            t2 = t1.reshape(shape)
            t3 = torch.add(t2, x2)
            sh = t3.shape
            shape2 = (sh[1], sh[0])
            t5 = t3.reshape(shape2)
            t6 = torch.add(t5, t5)
            return t6

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

        for s , s2 in zip(input_shapes, input_shapes2):
            t1 = torch.randn(s, requires_grad = False)
            t2 = torch.randn(s2, requires_grad = False)
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            result_compile_train = compiled_function_training(t1_h, t2_h)
            out_c = raw_function(t1, t2)
            assert torch.allclose(result_compile_train.to("cpu"), out_c)

if __name__ == '__main__':
    test_relu_mixed()
    test_reshape_symlnt()
    test_dynamic_shape_simple()
    test_dynamic_shape_view()
    test_dynamic_shape_repeat()
    test_dynamic_shape_repeat_lazy()
    test_dynamic_shape_repeat_static()
    test_dynamic_shape_repeat_static_lazy()
    test_dynamic_shape_topk_static_k()
    test_dynamic_shape_topk_static_same_k()
    # test_dynamic_shape_topk()
    test_dynamic_shape_control_flow_static()
    test_dynamic_shape_mult_module_split()

