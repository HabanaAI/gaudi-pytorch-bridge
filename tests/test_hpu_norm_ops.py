import torch
import torch.nn.functional as F
import pytest
from test_utils import (
    evaluate_fwd_kernel,
    evaluate_fwd_bwd_kernel,
    reset_seed,
    evaluate_fwd_inplace_kernel,
)
import numpy

# N - batch
# H - input height
# W - input width
# C - input channels
batch_norm_test_case_list_2d = [
    # N, H, W, C
    (16, 224, 224, 3),
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


@pytest.mark.parametrize("N, H, W, C", layer_norm_test_case_list)
@pytest.mark.parametrize("split_dim", [1, 2, 3])
def test_hpu_native_layer_norm(N, H, W, C, split_dim):
    shape = [N, H, W, C]
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


@pytest.mark.parametrize("N, H, W, C", layer_norm_test_case_list)
@pytest.mark.parametrize("split_dim", [1, 2, 3])
def test_hpu_layer_norm_fwd_bwd(N, H, W, C, split_dim):
    shape = [N, H, W, C]
    shape_norm = shape[split_dim:]
    kernel = torch.nn.LayerNorm(shape_norm)
    kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}

    bwd_tensor1 = torch.randn(shape)
    bwd_tensor2 = torch.randn(shape[0:split_dim])
    bwd_tensor3 = torch.randn(shape[0:split_dim])
    bwd_tensors = [bwd_tensor1, bwd_tensor2, bwd_tensor3]
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
    for i in range(2):
        kernel = torch.nn.LayerNorm(shape_norm)
        kernel_params_fwd = {"input": torch.randn(shape, requires_grad=True)}
        bwd_tensor1 = torch.randn(shape)
        bwd_tensor2 = torch.randn(shape[0:split_dim])
        bwd_tensor3 = torch.randn(shape[0:split_dim])
        bwd_tensors = [bwd_tensor1, bwd_tensor2, bwd_tensor3]
        evaluate_fwd_bwd_kernel(
            kernel=kernel,
            tensor_list_bwd=bwd_tensors,
            kernel_params_fwd=kernel_params_fwd,
            copy_kernel=True,
        )


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
    hpu = torch.device("habana")
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
    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.001, rtol=0.001
    )


@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_chlast_fwd_bwd(N, H, W, C):
    kernel = torch.nn.BatchNorm2d(C)
    in_tensor = torch.randn(N, C, H, W, requires_grad=True)
    kernel_params_fwd = {
        "input": in_tensor.contiguous(memory_format=torch.channels_last)
    }
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
    for i in range(2):
        kernel = torch.nn.BatchNorm2d(C)
        in_tensor = torch.randn(N, C, H, W, requires_grad=True)
        kernel_params_fwd = {
            "input": in_tensor.contiguous(memory_format=torch.channels_last)
        }
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
    hpu = torch.device("habana")
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

    for i in range(2):
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


if __name__ == "__main__":
    test_hpu_native_layer_norm(*layer_norm_test_case_list[0], 1)
    test_hpu_batch_norm_2d_fwd_bwd(*batch_norm_test_case_list_2d[0])
    test_hpu_batch_norm_1d_fwd_bwd(*batch_norm_test_case_list_1d[0])
    test_hpu_batch_norm_1d_ncl_fwd_bwd(*batch_norm_test_case_list_1d_ncl[0])

