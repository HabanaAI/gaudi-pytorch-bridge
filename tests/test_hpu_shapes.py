import numpy as np
import pytest
import torch
from test_utils import compare_tensors, cpu, evaluate_fwd_bwd_kernel, evaluate_fwd_kernel, hpu

# N - batch
# H - input height
# W - input width
# C - input channels
mnist_test_case_list = [
    # N, H, W, C
    (64, 3, 3, 50),
]

test_case_list = [
    # N, H, W, C
    (8, 28, 28, 3),
]

index_put_dtype_list = [torch.float, torch.bfloat16, torch.int32, torch.int8]

broadcast_test_case_list = [
    [torch.randn(8, 3, 28, 28), torch.randn(1), torch.randn(1)],
    [torch.randn(1, 4), torch.randn(3, 1), torch.randn(1)],
    [torch.randn(1, 4), torch.randn(3, 1), torch.randn(2, 1, 1)],
]

arange_test_case_list = [
    # start, end, step, dtype
    (0.0, 10.0, 2.0, torch.float),
    (1, 16, 2, torch.int32),
    (1, 16, 2, torch.int32),
    (20, 40, 5, torch.long),
    (0, -10, -2, torch.long),
]

test_case_scatter_add = [
    # N, H, I, S
    (512, 768, 1, 512),
]

test_case_nonzero = [
    # N, H, W, C, value, as_tuple
    # Mix values and as_tupel true
    (2, 3, 2, 4, 0, True),
    # Mix values and as_tupel false
    (2, 3, 2, 4, 0, False),
    # False values and as_tupel true
    (2, 3, 2, 4, 5, True),
    # False values and as_tupel false
    (2, 3, 2, 4, 5, False),
]

gather_test_case_list = [
    # N, C
    (4, 2, torch.gather),
]

gather_data_type_list = [
    torch.float,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    # torch.bfloat16,#RuntimeError: "scatter_gather_tensor_cpu" not implemented for 'BFloat16'
    torch.int64,
    torch.float64,
]


@pytest.mark.xfail
@pytest.mark.parametrize("N, W, C", [(8, 4, 28)])
def test_hpu_view_insert(N, W, C):
    """
    Do scale and transform from xywh to ltrb
    suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    bboxes_in = torch.randn(N, W, C, requires_grad=False)
    scores_in = torch.randn(N, W, C, requires_grad=False)
    dboxes = torch.randn(N, W, C, requires_grad=False)
    dboxes_xywh = torch.randn(N, W, C, requires_grad=False)
    scale_xy = 0.5
    scale_wh = 0.5

    hbboxes_in = bboxes_in.to(hpu)
    hscores_in = scores_in.to(hpu)
    hdboxes = dboxes.to(hpu)
    hdboxes_xywh = dboxes_xywh.to(hpu)

    # CPU
    bboxes_in = bboxes_in.permute(0, 2, 1)
    scores_in = scores_in.permute(0, 2, 1)
    dboxes = dboxes.permute(0, 2, 1)
    dboxes_xywh = dboxes_xywh.permute(0, 2, 1)
    # print(bboxes_in.device, scores_in.device, self.dboxes_xywh.device)

    bboxes_in[:, :, :2] = scale_xy * bboxes_in[:, :, :2]
    bboxes_in[:, :, 2:] = scale_wh * bboxes_in[:, :, 2:]

    bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
    bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * dboxes_xywh[:, :, 2:]

    # Transform format to ltrb
    l, t, r, b = (
        bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2],
        bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3],
        bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2],
        bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3],
    )

    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b

    scores_softmax = torch.nn.functional.softmax(scores_in, dim=-1)

    # HPU
    hbboxes_in = hbboxes_in.permute(0, 2, 1)
    hscores_in = hscores_in.permute(0, 2, 1)
    hdboxes = hdboxes.permute(0, 2, 1)
    hdboxes_xywh = hdboxes_xywh.permute(0, 2, 1)
    # print(bboxes_in.device, scores_in.device, self.dboxes_xywh.device)

    hbboxes_in[:, :, :2] = scale_xy * hbboxes_in[:, :, :2]
    hbboxes_in[:, :, 2:] = scale_wh * hbboxes_in[:, :, 2:]

    hbboxes_in[:, :, :2] = hbboxes_in[:, :, :2] * hdboxes_xywh[:, :, 2:] + hdboxes_xywh[:, :, :2]
    hbboxes_in[:, :, 2:] = hbboxes_in[:, :, 2:].exp() * hdboxes_xywh[:, :, 2:]

    # Transform format to ltrb
    hl, ht, hr, hb = (
        hbboxes_in[:, :, 0] - 0.5 * hbboxes_in[:, :, 2],
        hbboxes_in[:, :, 1] - 0.5 * hbboxes_in[:, :, 3],
        hbboxes_in[:, :, 0] + 0.5 * hbboxes_in[:, :, 2],
        hbboxes_in[:, :, 1] + 0.5 * hbboxes_in[:, :, 3],
    )

    hbboxes_in[:, :, 0] = hl
    hbboxes_in[:, :, 1] = ht
    hbboxes_in[:, :, 2] = hr
    hbboxes_in[:, :, 3] = hb

    hscores_softmax = torch.nn.functional.softmax(hscores_in, dim=-1)

    compare_tensors(hscores_softmax, scores_softmax, atol=0.001, rtol=1.0e-3)
    compare_tensors(hbboxes_in, bboxes_in, atol=0.001, rtol=1.0e-3)


# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_view(N, H, W, C):
    in_tensor = torch.randn(N, C, H, W)

    hpu_result = in_tensor.to(hpu).view(-1, C * H * W)
    cpu_result = in_tensor.to(cpu).view(-1, C * H * W)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_slice_and_select(N, H, W, C):
    in_tensor = torch.randn(N, C, H, W)

    hpu_result = in_tensor.to(hpu)[:, 0, 0:4:2, 0:4]
    cpu_result = in_tensor.to(cpu)[:, 0, 0:4:2, 0:4]
    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)


# All False test case
# Mix of True and False
@pytest.mark.parametrize("N, H, W, C, value, format", test_case_nonzero)
def test_hpu_nonzero(N, H, W, C, value, format):
    dim_list = [N, C, H, W]
    in_tensor = torch.randn(tuple(dim_list)) > value
    hpu_result = torch.nonzero(in_tensor.to(hpu), as_tuple=format)
    cpu_result = torch.nonzero(in_tensor.to(cpu), as_tuple=format)
    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)


# Empty Tensor case
@pytest.mark.parametrize("N, H, W, C, value, format", test_case_nonzero)
def test_hpu_nonzero_empty(N, H, W, C, value, format):
    dim_list = [0, 0, 0, 0]
    in_tensor = torch.empty(tuple(dim_list), dtype=torch.float)
    hpu_result = torch.nonzero(in_tensor.to(hpu), as_tuple=format)
    cpu_result = torch.nonzero(in_tensor.to(cpu), as_tuple=format)
    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)


# Test 1D case
# 1D case all false
@pytest.mark.parametrize("N, H, W, C, value, format", test_case_nonzero)
def test_hpu_nonzero_1D(N, H, W, C, value, format):
    dim_list = [H]
    in_tensor = torch.randn(tuple(dim_list)) > value
    hpu_result = torch.nonzero(in_tensor.to(hpu), as_tuple=format)
    cpu_result = torch.nonzero(in_tensor.to(cpu), as_tuple=format)
    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_unique(N, H, W, C):
    dim_list = [N, C, H, W]
    in_tensor = torch.randint(0, 100, tuple(dim_list))
    hpu_result = torch.unique(in_tensor.to(hpu), False, False, False, None)
    cpu_result = torch.unique(in_tensor, False, False, False, None)
    # Result coming in habana is reverse order than CPU reversing the CPU to match habana
    cpu_flipped = torch.flip(cpu_result, [0])
    compare_tensors(hpu_result, cpu_flipped, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dim", [0, 1, -2, 3])
def test_hpu_index_select(N, H, W, C, dim):
    kernel = torch.index_select

    dim_list = [N, C, H, W]
    kernel_params_fwd = {
        "input": torch.randn(tuple(dim_list), requires_grad=True),
        "dim": dim,
        "index": torch.tensor([0, 2]),
    }

    dim_list[dim] = 2
    bwd_tensors = [torch.randn(tuple(dim_list))]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("acc", [True, False])
def test_hpu_index_put_simple(N, H, W, C, dtype, acc):
    kernel = torch.index_put
    dim_list = [N, C, H, W]
    if dtype == torch.float32:
        in_t = torch.randn(tuple(dim_list), dtype=dtype, requires_grad=True)
        tbool = torch.randint_like(in_t, 0, np.prod(dim_list)) > (np.prod(dim_list) / 2)
        ti = tbool.nonzero().unbind(1)
        tv = torch.randn((ti[0].shape[0]))
    else:
        in_t = torch.randint(16, tuple(dim_list), dtype=dtype)
        tbool = torch.randint_like(in_t, 0, 128) > 8  # keep values small to avoid mismatches due to overflow and acc
        ti = tbool.nonzero().unbind(1)
        tv = torch.randint(16, (ti[0].shape[0],), dtype=dtype)
    kernel_params_fwd = {
        # clone required as tensors that are the result of a differentiable operation are not leaf variables
        "input": in_t.clone(),
        "indices": ti,
        "values": tv,
        "accumulate": acc,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype", index_put_dtype_list)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_bool(N, H, W, C, dtype, acc):
    kernel = torch.index_put
    dim_list = [N, C, H, W]
    if dtype == torch.float32:
        in_t = torch.randn(tuple(dim_list), dtype=dtype, requires_grad=True)
        ti = torch.randint_like(in_t, 0, np.prod(dim_list)) > (np.prod(dim_list) / 2)
        tv = torch.tensor(2.0)
    else:
        in_t = torch.randint(16, tuple(dim_list), dtype=dtype)
        ti = torch.randint_like(in_t, 0, 128) > 8
        tv = torch.tensor(2, dtype=dtype)
    kernel_params_fwd = {
        # clone required as tensors that are the result of a differentiable operation are not leaf variables
        "input": in_t.clone(),
        "indices": [ti],
        "values": tv,
        "accumulate": acc,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_new(N, H, W, C, acc):
    kernel = torch.index_put
    dim_list = [N, C, H, W]
    kernel_params_fwd = {
        "input": torch.randn(tuple(dim_list), requires_grad=True),
        "indices": [torch.tensor([0, 2]), torch.tensor([1, 1])],
        "values": torch.randn(tuple([2, H, W])),
        "accumulate": acc,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype", index_put_dtype_list)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_inplace(N, H, W, C, dtype, acc):
    kernel = torch.index_put_
    dim_list = [N, C, H, W]
    if dtype == torch.float32:
        in_t = torch.randn(tuple(dim_list), dtype=dtype, requires_grad=True)
        tbool = torch.randint_like(in_t, 0, np.prod(dim_list)) > (np.prod(dim_list) / 2)
        ti = tbool.nonzero().unbind(1)
        tv = torch.randn((ti[0].shape[0]))
    else:
        in_t = torch.randint(16, tuple(dim_list), dtype=dtype)
        tbool = torch.randint_like(in_t, 0, 128) > 8
        ti = tbool.nonzero().unbind(1)
        tv = torch.randint(16, (ti[0].shape[0],), dtype=dtype)
    kernel_params_fwd = {
        # clone required as tensors that are the result of a differentiable operation are not leaf variables
        "input": in_t.clone(),
        "indices": ti,
        "values": tv,
        "accumulate": acc,
    }
    bwd_tensors = [torch.randn(tuple(dim_list))]
    if dtype == torch.float32:
        evaluate_fwd_bwd_kernel(
            kernel=kernel,
            tensor_list_bwd=bwd_tensors,
            kernel_params_fwd=kernel_params_fwd,
        )
    else:
        evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype", index_put_dtype_list)
@pytest.mark.parametrize("acc", [True, False])
def test_hpu_index_put_bool_inplace(N, H, W, C, dtype, acc):
    kernel = torch.index_put_
    dim_list = [N, C, H, W]
    if dtype == torch.float32:
        in_t = torch.randn(tuple(dim_list), dtype=dtype, requires_grad=True)
        ti = torch.randint_like(in_t, 0, np.prod(dim_list)) > (np.prod(dim_list) / 2)
        tv = torch.randn(torch.nonzero(ti).shape[0])
    else:
        in_t = torch.randint(16, tuple(dim_list), dtype=dtype)
        ti = torch.randint_like(in_t, 0, 128) > 8
        tv = torch.randint(16, (torch.nonzero(ti).shape[0],), dtype=dtype)
    kernel_params_fwd = {
        "input": in_t.clone(),
        "indices": [ti],
        "values": tv,
        "accumulate": acc,
    }
    bwd_tensors = [torch.randn(tuple(dim_list))]
    if dtype == torch.float32:
        evaluate_fwd_bwd_kernel(
            kernel=kernel,
            tensor_list_bwd=bwd_tensors,
            kernel_params_fwd=kernel_params_fwd,
        )
    else:
        evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dim", [0, 1, -2, 3])
def test_hpu_index_add(N, H, W, C, dim):
    kernel = torch.index_add

    dim_list = [N, C, H, W]
    dim_list_tensor = [N, C, H, W]
    dim_list_tensor[dim] = 2
    kernel_params_fwd = {
        "input": torch.randn(tuple(dim_list), requires_grad=True),
        "dim": dim,
        "index": torch.tensor([0, 2]),
        "source": torch.randn(tuple(dim_list_tensor), requires_grad=True),
        "alpha": 1,
    }

    bwd_tensors = [torch.randn(tuple(dim_list))]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_src(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = torch.scatter(self_t, 0, indices_torch, src)
    thpu_out = torch.scatter(self_hpu, 0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_value(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    value = 1.0  # torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = torch.scatter(self_t, 0, indices_torch, value)
    thpu_out = torch.scatter(self_hpu, 0, indices_torch.to(hpu), value)
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_inplace(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = self_t.scatter_(0, indices_torch, src)
    tcpu_out = tcpu_out + tcpu_out
    thpu_out = self_hpu.scatter_(0, indices_torch.to(hpu), src.to(hpu))
    thpu_out = thpu_out + thpu_out
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_inplace_intermediate(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    in_t = torch.randn(N, H)
    in_hpu = in_t.to(hpu)

    self_t = torch.pow(in_t, 2.0)
    self_hpu = torch.pow(in_hpu, 2.0)
    tcpu_out = self_t.scatter_(0, indices_torch, src)
    tcpu_out = tcpu_out + tcpu_out
    thpu_out = self_hpu.scatter_(0, indices_torch.to(hpu), src.to(hpu))
    thpu_out = thpu_out + thpu_out
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_value_inplace(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    value = 2
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = self_t.scatter_(0, indices_torch, value)
    thpu_out = self_hpu.scatter_(0, indices_torch.to(hpu), value)
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-25327")
@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_add_out(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = torch.scatter_add(self_t, 0, indices_torch, src)
    thpu_out = torch.scatter_add(self_hpu, 0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0.001, rtol=1.0e-3)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-25327")
@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_add_inplace(N, H, I, S):  # noqa
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = self_t.scatter_add_(0, indices_torch, src)
    thpu_out = self_hpu.scatter_add_(0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("test_case_list", broadcast_test_case_list)
def test_hpu_broadcast(test_case_list):
    t1 = test_case_list[0]
    t2 = test_case_list[1]
    t3 = test_case_list[2]

    tcpu_out = torch.broadcast_tensors(t1, t2, t3)
    thpu_out = torch.broadcast_tensors(t1.to(hpu), t2.to(hpu), t3.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.xfail(reason="Int out of range for cast")
@pytest.mark.parametrize("start, end, step, dtype", arange_test_case_list)
@pytest.mark.parametrize("op", [torch.arange])
def test_hpu_arange_op_out(start, end, step, dtype, op):
    kernel_params_fwd = {}
    kernel_params_fwd["start"] = start
    kernel_params_fwd["end"] = end
    kernel_params_fwd["step"] = step
    kernel_params_fwd["dtype"] = dtype
    kernel_params_fwd["out"] = torch.empty(1, dtype=dtype)
    evaluate_fwd_kernel(kernel=op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("test_dtype", [torch.float, torch.long])
def test_hpu_expand(test_dtype):
    tin = torch.arange(0, 3, 1, dtype=test_dtype).view(3, 1)
    tcpu_out = tin.expand(3, 4)
    thpu_out = tin.to(hpu).expand(3, 4)
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize(
    "N, C",
    [
        (32, 8732),
    ],
)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_ssd(N, C, acc):
    dim_list = [N, C]
    label = torch.randint(low=1, high=C, size=tuple(dim_list), requires_grad=False)
    label_hpu = label.to(hpu)
    mask = label > 0
    mask_hpu = label_hpu > 0
    value_tensor = torch.tensor(0.0)
    value_tensor_hpu = value_tensor.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.index_put(
        input=torch.flatten(input_tensor),
        indices=[torch.flatten(mask)],
        values=value_tensor,
        accumulate=acc,
    )
    out_hpu = torch.index_put(
        input=torch.flatten(input_tensor_hpu),
        indices=[torch.flatten(mask_hpu)],
        values=value_tensor_hpu,
        accumulate=acc,
    )
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, C, gather_op", gather_test_case_list)
@pytest.mark.parametrize("dtype", gather_data_type_list)
def test_hpu_gather_op(N, C, gather_op, dtype):
    kernel_params_fwd = {}
    if dtype is torch.bool:
        kernel_params_fwd["input"] = torch.randint(N * C, (N, C)) > N * C / 2
    elif dtype is torch.bfloat16:
        kernel_params_fwd["input"] = torch.randn(N, C, dtype=dtype)
    else:
        kernel_params_fwd["input"] = torch.arange(0, N * C, 1, dtype=dtype).reshape(N, C)
    kernel_params_fwd["dim"] = 0
    kernel_params_fwd["index"] = torch.randint(N, [C, C])
    evaluate_fwd_kernel(kernel=gather_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize(
    "N, C",
    [
        (2, 262194),
    ],
)
@pytest.mark.parametrize("acc", [True])
def test_hpu_index_put_mrcnn(N, C, acc):
    dim_list = [N, C]
    label = torch.randint(low=1, high=C, size=tuple(dim_list), requires_grad=False)
    label.to(hpu)
    mask = label > 0

    for i in range(512):
        mask[0][i] = True
        mask[1][i] = False

    for i in range(512, 262194):
        mask[0][i] = False
        mask[1][i] = False

    print(mask.shape)
    print(torch.nonzero(mask).shape)
    mask_hpu = mask.to(hpu)
    value_tensor = torch.randn(262194 * 2)
    value_tensor_cpu = value_tensor[0:512]
    value_tensor_hpu = value_tensor_cpu.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.index_put(input=input_tensor, indices=[mask], values=value_tensor_cpu, accumulate=acc)
    out_hpu = torch.index_put(
        input=input_tensor_hpu,
        indices=[mask_hpu],
        values=value_tensor_hpu,
        accumulate=acc,
    )
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N, C",
    [
        (2, 161145),
    ],
)
@pytest.mark.parametrize("acc", [True])
def test_hpu_index_put_mrcnn3(N, C, acc):
    dim_list = [N, C, 4]
    label = torch.randint(low=1, high=C, size=tuple([N, C]), requires_grad=False)
    label.to(hpu)
    mask = label > 0

    for i in range(47):
        mask[0][i] = True
        mask[1][i] = False

    for i in range(47, C):
        mask[0][i] = False
        mask[1][i] = False

    print(mask.shape)
    print(torch.nonzero(mask).shape)
    mask_hpu = mask.to(hpu)

    value_tensor = torch.randn(161145 * 2, 4)
    value_tensor_cpu = value_tensor[0:47]
    value_tensor_hpu = value_tensor_cpu.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    print("value_tensor shape '{}'".format(value_tensor_cpu.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.index_put(input=input_tensor, indices=[mask], values=value_tensor_cpu, accumulate=acc)
    out_hpu = torch.index_put(input=input_tensor_hpu, indices=[mask_hpu], values=value_tensor_hpu, accumulate=acc)
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(), out_cpu.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True
    )


@pytest.mark.skip(reason="crash")
@pytest.mark.parametrize(
    "N, C",
    [
        (64, 32768),
    ],
)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_transformer2(N, C, acc):
    dim_list = [N, C]
    label = torch.randn(N, C)
    label_hpu = label.to(hpu)
    mask = label > 0
    mask_hpu = label_hpu > 0
    value_tensor = torch.tensor(1.0)
    value_tensor_hpu = value_tensor.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.index_put(input=input_tensor, indices=[mask], values=value_tensor, accumulate=acc)
    out_hpu = torch.index_put(
        input=input_tensor_hpu,
        indices=[mask_hpu],
        values=value_tensor_hpu,
        accumulate=acc,
    )
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N",
    [
        (16),
    ],
)
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_transformer3(N, acc):
    dim_list = [N]
    label = torch.randn(N)
    label.to(hpu)
    mask = torch.tensor([3, 2])
    mask_hpu = mask.to(hpu)
    value_tensor = torch.tensor(True)
    value_tensor_hpu = value_tensor.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True) > 5
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    print("input type {}".format(input_tensor.dtype))
    print("mask shape {}".format(mask.shape))
    print("mask type {}".format(mask.dtype))
    print("value shape {}".format(value_tensor.shape))
    print("value type {}".format(value_tensor.dtype))
    out_cpu = torch.index_put(input=input_tensor, indices=[mask], values=value_tensor, accumulate=acc)
    out_hpu = torch.index_put(
        input=input_tensor_hpu,
        indices=[mask_hpu],
        values=value_tensor_hpu,
        accumulate=acc,
    )
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


transformer_list = [*range(1, 257, 1)]


@pytest.mark.parametrize("N", transformer_list)
@pytest.mark.parametrize("C", [4])
@pytest.mark.parametrize("acc", [False])
def test_hpu_index_put_transformer1(N, C, acc):
    dim_list = [N, C]
    label = torch.randn(N, C)
    label_hpu = label.to(hpu)
    mask = label > 0
    mask_hpu = label_hpu > 0
    value_tensor = torch.tensor(True)
    value_tensor_hpu = value_tensor.to(hpu)
    input_tensor = torch.randn(tuple(dim_list), requires_grad=True)
    input_tensor = input_tensor > 3
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.index_put(input=input_tensor, indices=[mask], values=value_tensor, accumulate=acc)
    out_hpu = torch.index_put(
        input=input_tensor_hpu,
        indices=[mask_hpu],
        values=value_tensor_hpu,
        accumulate=acc,
    )
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


def test_hpu_index_put_transformer_ip():
    eos_mask = torch.tensor(
        [
            [False, False, False, False, False, False, False, False],
            [True, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, True, False, False, False, False, False],
            [True, True, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, True, False, False, True, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, False, False, True, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [False, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, True, True, False, False, True, True],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
        ]
    )
    eos_mask_hpu = eos_mask.to(hpu)
    beam_size = 4
    cands_to_ignore = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    cands_hpu = cands_to_ignore.to(hpu)

    print("eos_mask shape '{}'".format(eos_mask.shape))
    print("eos_mask type {}".format(eos_mask.dtype))
    print("cands_to_ignore shape {}".format(cands_to_ignore.shape))
    print("cands_to_ignore type {}".format(cands_to_ignore.dtype))
    print("beam_size  {}".format(beam_size))

    eos_mask = eos_mask[:, :beam_size]
    eos_mask_hpu = eos_mask.to(hpu)
    eos_mask[cands_to_ignore] = torch.tensor(0).to(eos_mask)
    eos_mask_hpu[cands_hpu] = torch.tensor(0).to(hpu).to(eos_mask)

    print(eos_mask)
    print(eos_mask_hpu.to(cpu))

    np.testing.assert_allclose(
        eos_mask_hpu.to(cpu).detach().numpy(),
        eos_mask.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


def test_hpu_index_put_transformer_case():
    eos_mask = torch.tensor(
        [
            [False, False, False, False, False, False, False, False],
            [True, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, True, False, False, False, False, False],
            [True, True, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, True, False, False, True, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, False, False, True, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [False, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, True, True, False, False, True, True],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
        ]
    )
    eos_mask_hpu = eos_mask.to(hpu)
    beam_size = 4
    cands_to_ignore = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    cands_hpu = cands_to_ignore.to(hpu)

    print("eos_mask shape '{}'".format(eos_mask.shape))
    print("eos_mask type {}".format(eos_mask.dtype))
    print("cands_to_ignore shape {}".format(cands_to_ignore.shape))
    print("cands_to_ignore type {}".format(cands_to_ignore.dtype))
    print("beam_size  {}".format(beam_size))

    eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
    eos_mask_hpu[:, :beam_size][cands_hpu] = torch.tensor(0).to(hpu).to(eos_mask)
    # eos_mask_temp = eos_mask_hpu[:, :beam_size]
    # out_hpu = torch.index_put(input=eos_mask_temp, indices=[cands_hpu], values=torch.tensor(0).to(hpu).to(eos_mask), accumulate=False)
    # eos_mask_hpu[:, :beam_size] = out_hpu
    print(eos_mask)
    print(eos_mask_hpu.to(cpu))

    # print(eos_mask_temp.to(cpu))
    np.testing.assert_allclose(
        eos_mask_hpu.to(cpu).detach().numpy(),
        eos_mask.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("acc", [True])
def test_hpu_index_put_point(acc):
    x = torch.randint(0, 23, (24, 128, 64), dtype=torch.long)
    y = torch.randint(0, 23, (24, 128, 64), dtype=torch.long)
    x_hpu = x.to(hpu)
    y_hpu = y.to(hpu)
    indices = [x, y]
    indices_hpu = [x_hpu, y_hpu]
    value = torch.rand([24, 128, 64, 128], dtype=torch.float32)
    tensor = torch.rand([24, 512, 128], dtype=torch.float32)
    value_hpu = value.to(hpu)
    tensor_hpu = tensor.to(hpu)
    out_cpu = torch.index_put(input=tensor, indices=indices, values=value, accumulate=acc)
    print(out_cpu.shape)
    out_hpu = torch.index_put(input=tensor_hpu, indices=indices_hpu, values=value_hpu, accumulate=acc)
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().numpy(),
        out_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


def test_hpu_index_put_mmdet():
    x = torch.zeros([3549], dtype=torch.bool)
    x_hpu = x.to(hpu)
    tensor = torch.ones([3549], dtype=torch.long)
    tensor_hpu = tensor.to(hpu)
    tensor[x] = tensor[x] + 1
    tensor_hpu[x_hpu] = tensor_hpu[x_hpu] + 1
    np.testing.assert_allclose(
        tensor_hpu.to(cpu).detach().numpy(),
        tensor.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N, C",
    [
        (16, 100),
    ],
)
@pytest.mark.parametrize("dtype", index_put_dtype_list)
def test_hpu_masked_scatter(N, C, dtype):
    dim_list = [N, C]
    label = torch.randint(low=1, high=N, size=tuple([C]), requires_grad=False)
    label.to(hpu)
    mask = label < C / 2
    print(mask.shape)
    print(torch.nonzero(mask).shape)
    mask_hpu = mask.to(hpu)

    value_tensor = torch.randn(N, torch.nonzero(mask).shape[0]).to(dtype)
    value_tensor_cpu = value_tensor
    value_tensor_hpu = value_tensor_cpu.to(hpu)
    print(value_tensor_cpu.shape)

    input_tensor = torch.randn(tuple(dim_list), requires_grad=True).to(dtype)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = torch.masked_scatter(input=input_tensor, mask=mask, source=value_tensor_cpu)
    out_hpu = torch.masked_scatter(input=input_tensor_hpu, mask=mask_hpu, source=value_tensor_hpu)
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().to(torch.float).numpy(),
        out_cpu.detach().to(torch.float).numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N, C",
    [
        (16, 100),
    ],
)
@pytest.mark.parametrize("dtype", index_put_dtype_list)
def test_hpu_masked_scatter_inplace(N, C, dtype):
    dim_list = [N, C]
    label = torch.randint(low=1, high=N, size=tuple([C]), requires_grad=False)
    label.to(hpu)
    mask = label < C / 2
    print(mask.shape)
    print(torch.nonzero(mask).shape)
    mask_hpu = mask.to(hpu)

    value_tensor = torch.randn(N, torch.nonzero(mask).shape[0]).to(dtype)
    value_tensor_cpu = value_tensor
    value_tensor_hpu = value_tensor_cpu.to(hpu)
    print(value_tensor_cpu.shape)

    input_tensor = torch.randn(tuple(dim_list)).to(dtype)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    out_cpu = input_tensor.masked_scatter_(mask=mask, source=value_tensor_cpu)
    out_hpu = input_tensor_hpu.masked_scatter_(mask=mask_hpu, source=value_tensor_hpu)
    np.testing.assert_allclose(
        out_hpu.to(cpu).detach().to(torch.float32).numpy(),
        out_cpu.detach().to(torch.float32).numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("dtype", index_put_dtype_list)
def test_hpu_index_fill_(dtype):
    index = torch.tensor([0, 1])
    index_hpu = index.to(hpu)
    input_tensor = torch.randn(3, 4, 3).to(dtype)
    print("input_tensor shape '{}'".format(input_tensor.shape))
    input_tensor_hpu = input_tensor.to(hpu)
    dim = torch.randint(input_tensor.dim(), (1,))[0]
    input_tensor.index_fill_(dim, index, -1)
    input_tensor_hpu.index_fill_(dim, index_hpu, -1)
    np.testing.assert_allclose(
        input_tensor_hpu.to(cpu).detach().to(torch.float32).numpy(),
        input_tensor.detach().to(torch.float32).numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


if __name__ == "__main__":
    test_hpu_slice_and_select(*test_case_list[0])
    test_hpu_view(*test_case_list[0])
    test_hpu_index_select(*test_case_list[0], 0)
    test_hpu_index_put_new(*test_case_list[0])
    test_hpu_index_add(*test_case_list[0], 0)
    test_hpu_broadcast(broadcast_test_case_list[2])
    test_hpu_scatter_value_inplace(*test_case_scatter_add[0])
    test_hpu_nonzero(*test_case_nonzero[0])
    test_hpu_nonzero_empty(*test_case_nonzero[0])
    test_hpu_nonzero_1D(*test_case_nonzero[0])
    test_hpu_unique(*test_case_list[0])
