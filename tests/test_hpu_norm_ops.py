import os
from copy import deepcopy

import numpy
import pytest
import torch
from test_utils import cpu, evaluate_fwd_bwd_kernel, generic_setup_teardown_env, hpu, is_lazy

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    yield from generic_setup_teardown_env({"PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION": "1"})


# N - batch
# H - input height
# W - input width
# C - input channels
batch_norm_test_case_list_2d = [
    # N, H, W, C
    (16, 224, 224, 3),
]

batch_norm_test_case_list_3d = [
    # N, C, D, H, W
    (1, 128, 16, 20, 20),
]

batch_norm_test_case_list_1d = [
    # N, C
    (16, 128),
]

batch_norm_test_case_list_1d_ncl = [
    # N, C, L
    (32, 64, 5),
]

layer_norm_test_case_list = [(2, 5, 10, 10)]

instance_norm3d_test_case_list = [
    # N, C, D, H, W
    (1, 256, 16, 16, 16),
    (1, 320, 8, 8, 8),
]


@pytest.mark.xfail(reason="segv")
@pytest.mark.parametrize("N, H, W, C", layer_norm_test_case_list)
@pytest.mark.parametrize("split_dim", [1, 2, 3])
def test_hpu_native_layer_norm(N, H, W, C, split_dim):
    shape = [N, H, W, C]
    shape_norm = shape[split_dim:]
    kernel = torch.nn.LayerNorm(shape_norm, elementwise_affine=True)
    kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}

    bwd_tensors = [torch.randn(shape)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
        grad_on_grad_enable=False,
    )
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
        grad_on_grad_enable=False,
    )


@pytest.mark.parametrize("N, H, W, C", layer_norm_test_case_list)
@pytest.mark.parametrize("split_dim", [1, 2, 3])
def test_hpu_layer_norm_fwd_bwd(N, H, W, C, split_dim):
    shape = [N, H, W, C]
    shape_norm = shape[split_dim:]
    kernel = torch.nn.LayerNorm(shape_norm)
    kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}

    bwd_tensors = [torch.randn(shape)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, H, W, C", layer_norm_test_case_list)
@pytest.mark.parametrize("split_dim", [1, 2, 3])
def test_hpu_layer_norm_withcache_fwd_bwd(N, H, W, C, split_dim):
    shape = [N, H, W, C]
    shape_norm = shape[split_dim:]
    for _ in range(2):
        kernel = torch.nn.LayerNorm(shape_norm)
        kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}
        bwd_tensors = [torch.randn(shape)]
        evaluate_fwd_bwd_kernel(
            kernel=kernel,
            tensor_list_bwd=bwd_tensors,
            kernel_params_fwd=kernel_params_fwd,
            copy_kernel=True,
        )


@pytest.mark.xfail(reason="KeyError: 'torch_dynamo_backends")
@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_conv_and_batch_norm_2d_fwd_compile_only(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    import habana_frameworks.torch.utils.debug as htdebug

    htdebug._enable_fuse_conv_bn_optimization(True)

    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.conv2 = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, bias=True)
            self.conv2.weight = torch.nn.Parameter(0.2 * torch.ones_like(self.conv2.weight))
            self.conv2.bias = torch.nn.Parameter(0.5 * torch.ones_like(self.conv2.bias))
            self.bn2 = torch.nn.BatchNorm2d(C)
            self.bn2.weight = torch.nn.Parameter(0.12 * torch.ones_like(self.bn2.weight))
            self.bn2.bias = torch.nn.Parameter(0.15 * torch.ones_like(self.bn2.bias))
            self.bn2.running_mean = torch.nn.Parameter(0.01 * torch.ones_like(self.bn2.running_mean))
            self.bn2.running_var = torch.nn.Parameter(0.9 * torch.ones_like(self.bn2.running_var))
            self.train(False)
            self.eval()

        def _forward_impl(self, x):
            y = self.conv2(x)
            z = self.bn2(y)
            return z

        def forward(self, x):
            return self._forward_impl(x)

    model = bn()
    model.eval()
    x = torch.randn(N, C, H, W, dtype=torch.float32, requires_grad=False)
    print("Infer on CPU....................................", flush=True)
    with torch.inference_mode():
        output = model(x)

    model_hpu = model.to(hpu)
    model_hpu.eval()
    x_hpu = x.to(hpu)
    print("Infer on HPU....................................", flush=True)

    def raw_function(tensor):
        model_hpu(tensor)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")
    with torch.inference_mode():
        output_hpu = compiled_function(x_hpu)

    output_hpu_cpu = output_hpu.to(cpu)
    numpy.testing.assert_allclose(output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001)


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_conv_and_batch_norm_2d_fwd_only(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    import habana_frameworks.torch.utils.debug as htdebug

    htdebug._enable_fuse_conv_bn_optimization(True)

    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.conv2 = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False)
            self.conv2.weight = torch.nn.Parameter(0.2 * torch.ones_like(self.conv2.weight))
            # self.conv2.bias = torch.nn.Parameter(0.5 * torch.ones_like(self.conv2.bias))
            self.bn2 = torch.nn.BatchNorm2d(C)
            self.bn2.weight = torch.nn.Parameter(0.12 * torch.ones_like(self.bn2.weight))
            self.bn2.bias = torch.nn.Parameter(0.15 * torch.ones_like(self.bn2.bias))
            self.bn2.running_mean = torch.nn.Parameter(0.01 * torch.ones_like(self.bn2.running_mean))
            self.bn2.running_var = torch.nn.Parameter(0.9 * torch.ones_like(self.bn2.running_var))
            self.train(False)
            self.eval()

        def _forward_impl(self, x):
            y = self.conv2(x)
            z = self.bn2(y)
            return z

        def forward(self, x):
            return self._forward_impl(x)

    model = bn()
    model.eval()
    x = torch.randn(N, C, H, W, dtype=torch.float32, requires_grad=False)
    print("Infer on CPU....................................", flush=True)
    with torch.inference_mode():
        output = model(x)

    model_hpu = model.to(hpu)
    model_hpu.eval()
    x_hpu = x.to(hpu)
    x_hpu = x_hpu.to(torch.bfloat16)
    print("Infer on HPU....................................", flush=True)
    with torch.inference_mode():
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            output_hpu = model_hpu(x_hpu)
    output_hpu = output_hpu.to(torch.float32)
    output_hpu_cpu = output_hpu.to(cpu)
    numpy.testing.assert_allclose(output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_fwd_bwd(N, H, W, C):
    kernel = torch.nn.BatchNorm2d(C)
    kernel_params_fwd = {"input": torch.randn(N, C, H, W, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, H, W)]

    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_fwd_only(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.bn2 = torch.nn.BatchNorm2d(C)
            self.train(False)
            self.eval()

        def _forward_impl(self, x):
            x = self.bn2(x)
            return x

        def forward(self, x):
            return self._forward_impl(x)

    model = bn()
    x = torch.randn(N, C, H, W, dtype=torch.float32, requires_grad=False)

    model.eval()
    output = model(x)
    model.eval()

    model_hpu = model.to(hpu)
    model_hpu.eval()
    x_hpu = x.to(hpu)
    output_hpu = model_hpu(x_hpu)
    model_hpu.eval()
    output_hpu_cpu = output_hpu.to(cpu)

    numpy.testing.assert_allclose(output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("N, C, D, H, W", batch_norm_test_case_list_3d)
def test_hpu_batch_norm_3d_fwd_bwd(N, C, D, H, W):
    kernel = torch.nn.BatchNorm3d(C)
    kernel_params_fwd = {"input": torch.randn(N, C, D, H, W, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, D, H, W)]

    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, C", batch_norm_test_case_list_1d)
def test_hpu_batch_norm_1d_fwd_bwd(N, C):
    kernel = torch.nn.BatchNorm1d(C)
    kernel_params_fwd = {"input": torch.randn(N, C, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C)]

    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, C, L", batch_norm_test_case_list_1d_ncl)
def test_hpu_batch_norm_1d_ncl_fwd_bwd(N, C, L):
    kernel = torch.nn.BatchNorm1d(C)
    kernel_params_fwd = {"input": torch.randn(N, C, L, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, L)]

    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_eval_fwd_bwd(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.bn1 = torch.nn.BatchNorm2d(C)

        def _forward_impl(self, x):
            x = self.bn1(x)
            return x

        def forward(self, x):
            return self._forward_impl(x)

    model = bn()
    model = model.train()
    x = torch.randn((N, C, H, W))
    output = model(x)

    model = model.eval()
    output = model(x)

    model_hpu = model.to(hpu)
    model_hpu = model_hpu.eval()
    x_hpu = x.to(hpu)
    output_hpu = model_hpu(x_hpu)
    output_hpu_cpu = output_hpu.to(cpu)
    numpy.testing.assert_allclose(output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_chlast_fwd_bwd(N, H, W, C):
    kernel = torch.nn.BatchNorm2d(C)
    in_tensor = torch.randn(N, C, H, W, requires_grad=True)
    kernel_params_fwd = {"input": in_tensor.contiguous(memory_format=torch.channels_last)}
    bwd_tensor = torch.randn(N, C, H, W)
    bwd_tensors = [bwd_tensor.contiguous(memory_format=torch.channels_last)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
    )


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_chlast_withcache_fwd_bwd(N, H, W, C):
    for _ in range(2):
        kernel = torch.nn.BatchNorm2d(C)
        in_tensor = torch.randn(N, C, H, W, requires_grad=True)
        kernel_params_fwd = {"input": in_tensor.contiguous(memory_format=torch.channels_last)}
        bwd_tensor = torch.randn(N, C, H, W)
        bwd_tensors = [bwd_tensor.contiguous(memory_format=torch.channels_last)]
        evaluate_fwd_bwd_kernel(
            kernel=kernel,
            tensor_list_bwd=bwd_tensors,
            kernel_params_fwd=kernel_params_fwd,
            copy_kernel=True,
        )


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_eval_withcache_fwd_bwd(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.bn1 = torch.nn.BatchNorm2d(C)

        def _forward_impl(self, x):
            x = self.bn1(x)
            return x

        def forward(self, x):
            return self._forward_impl(x)

    for _ in range(2):
        model = bn()
        model = model.train()
        x = torch.randn((N, C, H, W))
        output = model(x)
        model = model.eval()
        output = model(x)
        model_hpu = model.to(hpu)
        model_hpu = model_hpu.eval()
        x_hpu = x.to(hpu)
        output_hpu = model_hpu(x_hpu)
        output_hpu_cpu = output_hpu.to(cpu)
        numpy.testing.assert_allclose(
            output_hpu_cpu.detach().numpy(),
            output.detach().numpy(),
            atol=0.001,
            rtol=0.001,
        )


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("N, H, W", [(12, 384, 1024)])
@pytest.mark.parametrize("split_dim", [2])
def test_hpu_layer_norm_bert_graphmode(N, H, W, split_dim):
    shape = [N, H, W]
    shape_norm = shape[split_dim:]
    kernel = torch.nn.LayerNorm(shape_norm, elementwise_affine=True)
    kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}

    bwd_tensor1 = torch.randn(shape)
    bwd_tensor2 = None
    bwd_tensor3 = None
    bwd_tensors = [bwd_tensor1, bwd_tensor2, bwd_tensor3]
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
        grad_on_grad_enable=False,
    )
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        copy_kernel=True,
        grad_on_grad_enable=False,
    )


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("N, H", [(1024, 4096), (1024, 4096), (391, 1)])
@pytest.mark.parametrize("lp_norm_op", [torch.norm])
@pytest.mark.parametrize("value", [2.0])
def test_hpu_lp_norm_op_fwd_bwd(N, H, lp_norm_op, value):
    kernel_params_fwd = {"input": torch.randn(N, H, requires_grad=True), "p": value}
    bwd_tensors = [torch.randn(N, H)]
    evaluate_fwd_bwd_kernel(
        kernel=lp_norm_op,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
    )


# Instance Norm bwd is not suported in legacy eager mode
@pytest.mark.skipif(
    os.getenv("PT_HPU_LAZY_MODE") is not None and os.getenv("PT_HPU_LAZY_MODE") == "0",
    reason="Instance Norm bwd is not suported in legacy eager mode",
)
@pytest.mark.parametrize("N, C, D, H, W", instance_norm3d_test_case_list)
def test_hpu_instance_norm_3d_fwd_bwd(N, C, D, H, W):

    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel = torch.nn.InstanceNorm3d(C, affine=True)
    kernel_copy = deepcopy(kernel)

    # cpu forward
    out_cpu = kernel(input_nchw)

    # hpu forward
    input_nchw_hpu = input_nchw.to(hpu).detach()
    input_nchw_hpu.requires_grad = True
    kernel_hpu = kernel_copy.to(hpu)
    out_hpu = kernel_hpu(input_nchw_hpu)

    # cpu bwd
    out_cpu_bwd = torch.ones_like(out_cpu)
    out_cpu.backward(out_cpu_bwd)
    input_nchw_bwd = input_nchw.grad

    # hpu bwd
    out_hpu.backward(out_cpu_bwd.to(hpu))
    tt = input_nchw_hpu.grad.to(cpu)

    numpy.testing.assert_allclose(
        tt.detach().numpy(),
        input_nchw_bwd.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


# Instance Norm bwd is not suported in legacy eager mode
@pytest.mark.skipif(not is_lazy(), reason="Instance Norm bwd is not suported in legacy eager mode")
@pytest.mark.parametrize("N, C, D, H, W", instance_norm3d_test_case_list)
def test_hpu_instance_norm_3d_chlast_fwd_bwd(N, C, D, H, W):

    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel = torch.nn.InstanceNorm3d(C, affine=True)
    kernel_copy = deepcopy(kernel)

    # cpu forward
    out_cpu = kernel(input_nchw)

    # hpu forward
    input_nhwc_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu).detach()
    input_nhwc_hpu.requires_grad = True
    kernel_hpu = kernel_copy.to(hpu)
    out_hpu = kernel_hpu(input_nhwc_hpu)
    tt = out_hpu.to(cpu)

    # cpu bwd
    out_cpu_bwd = torch.ones_like(out_cpu)
    out_cpu.backward(out_cpu_bwd)
    input_nchw_bwd = input_nchw.grad

    # hpu bwd
    out_hpu.backward(out_cpu_bwd.contiguous(memory_format=torch.channels_last_3d).to(hpu))
    tt = input_nhwc_hpu.grad.to(cpu)

    numpy.testing.assert_allclose(
        tt.detach().numpy(),
        input_nchw_bwd.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


if __name__ == "__main__":
    test_hpu_native_layer_norm(*layer_norm_test_case_list[0], 1)
    test_hpu_batch_norm_2d_fwd_bwd(*batch_norm_test_case_list_2d[0])
    test_hpu_batch_norm_1d_fwd_bwd(*batch_norm_test_case_list_1d[0])
    test_hpu_batch_norm_1d_ncl_fwd_bwd(*batch_norm_test_case_list_1d_ncl[0])
