###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import pytest
import torch
import torch.nn
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    is_gaudi1,
    is_pytest_mode_compile,
    print_tensors,
)

Verbose = False

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_embedding(shapes, dtype):
    def fn(input, indices):
        return torch.embedding(input, indices)

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape).to(dtype)
    hpu_input = cpu_input.to("hpu")
    max_index = input_shape[1] - 1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    fn = compile_function_if_compile_mode(fn)

    cpu_output = torch.embedding(cpu_input, cpu_indices)
    hpu_output = fn(hpu_input, hpu_indices)

    if Verbose:
        print(cpu_output)
        print(hpu_output.cpu())

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("embedding")


@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hpu_embedding_bwd(shapes, dtype):
    def fn(input, indices):
        embedding = torch.embedding(input, indices)
        grad = torch.ones_like(embedding)
        embedding.backward(grad)
        return input.grad

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape).to(dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    max_index = input_shape[1] - 1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    hpu_fn = compile_function_if_compile_mode(fn)

    cpu_output = fn(cpu_input, cpu_indices)
    hpu_output = hpu_fn(hpu_input, hpu_indices)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"embedding", "embedding_dense_backward"})


def topo_input_sw_204570():
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 5199, 11413],
        [0, 0, 0, 0, 0, 4907, 2673, 556, 12170, 2175],
        [0, 0, 0, 0, 0, 4100, 11543, 1917, 2231, 11286],
        [0, 0, 0, 0, 0, 0, 1178, 23095, 21025, 5633],
        [0, 0, 0, 0, 0, 0, 0, 0, 5388, 5373],
        [2231, 9334, 4931, 2162, 12740, 4176, 21788, 9431, 795, 6806],
        [0, 0, 0, 0, 0, 0, 0, 0, 1818, 8090],
        [0, 0, 0, 0, 0, 0, 0, 3255, 531, 20874],
        [17380, 25043, 14326, 10622, 9540, 22864, 10661, 17554, 23755, 18503],
        [0, 0, 0, 0, 11460, 5756, 8732, 3693, 2078, 1783],
        [0, 0, 0, 0, 0, 0, 0, 0, 10227, 2934],
        [0, 0, 0, 0, 0, 0, 0, 0, 4419, 1915],
        [0, 0, 0, 0, 0, 0, 0, 9086, 7985, 9706],
        [0, 0, 0, 0, 0, 0, 2016, 7186, 4084, 2211],
        [0, 0, 0, 0, 0, 0, 0, 0, 23950, 17118],
        [9025, 9032, 15875, 9038, 24316, 25402, 25403, 9021, 25353, 21545],
        [1340, 2765, 2695, 2770, 14953, 12636, 22349, 21255, 18681, 9830],
        [0, 0, 0, 0, 0, 0, 0, 0, 5395, 4377],
        [0, 0, 0, 0, 0, 0, 0, 0, 530, 531],
        [0, 0, 0, 0, 0, 0, 0, 6030, 15835, 15792],
        [0, 0, 0, 0, 0, 0, 0, 1905, 14961, 10073],
        [0, 0, 0, 0, 0, 0, 0, 0, 15581, 338],
        [0, 0, 0, 0, 0, 0, 0, 0, 14517, 19642],
        [0, 0, 0, 0, 9745, 5178, 14380, 11901, 19580, 14674],
        [0, 0, 0, 0, 0, 0, 9320, 23502, 19654, 12850],
        [0, 0, 0, 0, 0, 0, 0, 0, 15919, 6474],
        [0, 0, 0, 0, 0, 0, 0, 19588, 10958, 22116],
        [0, 0, 0, 0, 0, 0, 0, 0, 9126, 13597],
        [0, 0, 0, 0, 0, 0, 0, 0, 4141, 3103],
        [0, 0, 0, 0, 0, 0, 232, 6293, 20251, 3086],
        [19867, 2844, 5150, 10618, 8162, 20456, 4042, 23129, 18605, 20444],
        [0, 0, 0, 0, 11570, 3421, 18960, 21026, 23397, 21234],
        [5734, 1816, 1043, 4102, 10764, 15006, 3628, 1809, 8497, 4611],
        [0, 0, 0, 0, 9192, 5120, 23928, 18757, 13534, 11970],
        [0, 0, 0, 0, 0, 3125, 5135, 5136, 5137, 5138],
        [0, 0, 0, 0, 0, 0, 0, 0, 3550, 5597],
        [0, 0, 0, 0, 0, 0, 1886, 314, 2093, 4774],
        [0, 0, 9790, 10097, 6424, 14021, 6422, 19884, 4931, 6289],
        [0, 0, 0, 0, 0, 0, 0, 0, 4861, 12527],
        [4103, 7258, 2128, 1826, 6016, 1109, 11244, 562, 2231, 15621],
        [0, 0, 6817, 16781, 2836, 13758, 1430, 15229, 7849, 2182],
        [0, 0, 0, 0, 0, 0, 14663, 2211, 7084, 1575],
        [0, 0, 0, 0, 0, 0, 0, 0, 326, 22863],
        [0, 0, 0, 0, 0, 0, 0, 0, 3132, 4663],
        [0, 0, 0, 0, 0, 0, 0, 0, 19279, 8105],
        [0, 0, 0, 0, 0, 0, 0, 24409, 562, 1899],
        [5945, 6951, 2295, 232, 1809, 5567, 6765, 10558, 10557, 1892],
        [0, 0, 0, 0, 0, 0, 0, 0, 12938, 2394],
        [0, 0, 0, 0, 0, 0, 0, 2095, 3704, 3065],
        [0, 0, 0, 0, 0, 0, 0, 0, 9943, 1851],
        [0, 0, 5603, 14949, 4672, 2033, 14200, 5546, 4646, 5888],
        [0, 0, 0, 0, 0, 0, 8133, 10458, 17673, 1384],
        [0, 0, 0, 0, 3855, 502, 22429, 16752, 3162, 15244],
        [0, 0, 0, 0, 0, 15135, 14413, 1482, 5462, 3136],
        [0, 0, 0, 0, 0, 0, 0, 0, 18317, 22784],
        [10822, 1601, 7387, 4278, 1879, 22781, 2385, 3894, 267, 8689],
        [23764, 10683, 15115, 2068, 7796, 13798, 7555, 7594, 667, 11727],
        [0, 0, 0, 0, 0, 0, 0, 0, 85, 9614],
        [0, 0, 0, 0, 0, 1818, 15649, 17542, 9566, 5561],
        [0, 0, 0, 0, 0, 2393, 1899, 10924, 11813, 24475],
        [0, 0, 0, 0, 0, 0, 13373, 16508, 12376, 16355],
        [0, 0, 0, 0, 0, 5140, 12956, 23515, 16761, 13016],
        [2705, 9191, 5639, 13267, 7749, 9711, 3854, 1225, 15193, 16218],
        [0, 0, 0, 0, 0, 0, 0, 0, 19784, 16253],
        [0, 0, 0, 0, 0, 0, 0, 0, 6102, 10618],
        [0, 0, 0, 0, 0, 0, 0, 0, 1696, 8918],
        [0, 9778, 2271, 3427, 10207, 6489, 25512, 15479, 25530, 18576],
        [0, 0, 0, 0, 0, 0, 0, 0, 225, 9844],
        [0, 0, 0, 0, 0, 0, 7792, 449, 15847, 20076],
        [0, 0, 0, 0, 0, 0, 0, 0, 13229, 2161],
        [0, 0, 0, 0, 497, 24413, 9446, 13817, 21575, 3033],
        [0, 0, 0, 0, 0, 0, 0, 2181, 4931, 5034],
        [0, 1727, 25288, 12058, 6496, 596, 509, 84, 1899, 16740],
        [0, 0, 0, 0, 0, 0, 0, 0, 1289, 1795],
        [0, 0, 0, 0, 0, 0, 0, 8586, 20130, 4314],
        [0, 0, 0, 0, 0, 0, 0, 0, 4758, 184],
        [0, 0, 0, 0, 0, 0, 17573, 3608, 9711, 228],
        [659, 14575, 25517, 13963, 7008, 24594, 10956, 6885, 1868, 24121],
        [0, 0, 0, 0, 0, 0, 0, 0, 383, 8465],
        [0, 0, 0, 0, 0, 0, 0, 0, 9721, 525],
        [0, 0, 0, 0, 0, 0, 0, 0, 531, 19446],
        [0, 0, 0, 0, 0, 0, 0, 0, 2473, 4728],
        [0, 0, 0, 0, 0, 0, 15774, 3086, 11464, 1835],
        [0, 0, 0, 0, 0, 0, 0, 0, 10703, 21847],
        [0, 0, 7669, 21444, 593, 7463, 1916, 21508, 5401, 6006],
        [0, 0, 0, 0, 0, 0, 0, 0, 323, 475],
        [0, 0, 0, 0, 0, 0, 16248, 1957, 485, 316],
        [0, 0, 0, 13406, 4332, 3090, 8167, 1262, 4226, 19763],
        [0, 0, 0, 0, 0, 0, 0, 0, 8250, 11074],
        [0, 0, 0, 0, 7719, 6404, 7720, 7721, 7722, 5182],
        [0, 0, 0, 0, 0, 0, 0, 0, 9612, 2887],
        [7546, 21338, 7548, 5018, 24127, 18806, 19843, 20516, 21265, 10812],
        [0, 0, 0, 0, 0, 0, 8698, 2128, 10882, 15136],
        [0, 0, 0, 0, 12989, 4942, 1637, 2221, 1960, 20083],
        [9828, 9779, 12845, 24713, 18664, 13396, 18673, 3450, 23015, 2814],
        [0, 0, 0, 0, 0, 0, 0, 2370, 13850, 1337],
        [0, 0, 0, 0, 0, 4159, 509, 17654, 15203, 8686],
        [0, 0, 0, 0, 0, 0, 0, 1851, 509, 10042],
        [0, 0, 0, 0, 0, 0, 0, 15332, 6194, 17043],
        [0, 0, 0, 0, 0, 0, 0, 6558, 184, 4644],
        [0, 0, 0, 0, 0, 0, 0, 1590, 14785, 525],
        [0, 0, 0, 0, 0, 0, 0, 672, 7409, 1821],
        [0, 0, 0, 0, 0, 0, 0, 1855, 9283, 1902],
        [3365, 3367, 5050, 3568, 4497, 316, 8420, 8421, 1305, 4013],
        [0, 0, 1660, 3238, 11210, 11211, 3769, 6288, 11212, 480],
        [0, 0, 0, 0, 0, 0, 2289, 2211, 4225, 13630],
        [0, 0, 0, 0, 0, 0, 5361, 9527, 21256, 5897],
        [0, 0, 0, 0, 0, 0, 0, 22976, 7594, 7595],
        [0, 0, 0, 0, 0, 0, 0, 0, 7252, 2185],
        [14762, 17907, 4094, 8257, 20932, 6771, 15804, 10608, 517, 6278],
        [0, 0, 0, 0, 0, 4403, 3210, 15694, 4524, 10261],
        [0, 12147, 22874, 11865, 17207, 21268, 9835, 3446, 9669, 19811],
        [0, 0, 0, 0, 0, 1851, 509, 2211, 7617, 3071],
        [0, 0, 0, 0, 0, 0, 0, 657, 8071, 15730],
        [0, 0, 0, 3270, 9691, 27, 10587, 10369, 1067, 8016],
        [21462, 19842, 24922, 12273, 21266, 19170, 6201, 6953, 17048, 21749],
        [0, 0, 0, 0, 0, 0, 0, 0, 14708, 5015],
        [3506, 3507, 241, 3508, 3509, 3510, 2846, 2532, 837, 3511],
        [0, 0, 0, 0, 0, 0, 0, 0, 7506, 2040],
        [0, 0, 0, 0, 0, 0, 1979, 5798, 19888, 23012],
        [0, 0, 0, 0, 0, 0, 6581, 7006, 532, 10467],
        [0, 0, 0, 0, 0, 13898, 9467, 6877, 2146, 1720],
        [0, 0, 0, 0, 0, 0, 0, 0, 13895, 13896],
        [0, 0, 0, 0, 0, 0, 7652, 1882, 15105, 8140],
        [23519, 9217, 4566, 14238, 4101, 13271, 21895, 5395, 2714, 3095],
        [0, 0, 0, 0, 0, 0, 6053, 8411, 23936, 25265],
        [0, 0, 0, 13415, 8237, 12624, 11837, 4320, 5585, 5613],
        [0, 0, 0, 0, 0, 0, 0, 11035, 509, 1775],
    ]


def hpu_nn_embedding_test(num_embeddings, embedding_dim, padding_idx, bwd):
    if is_pytest_mode_compile():
        pytest.skip(
            "Output 0 of TracableCreateParameterBackward is a view and its base or another view of its base has been "
            "modified inplace. This view was created inside a custom Function (or because an input was returned as-is) "
            "and the autograd logic to handle view+inplace would override the custom backward associated with "
            "the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by "
            "cloning the output of the custom Function."
        )

    def fn(emb_input, device, init_weight=None):
        emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, device=device)
        if init_weight is None:
            torch.nn.init.xavier_normal_(emb.weight.data)
        else:
            emb.weight.data = init_weight.data.clone()

        if not bwd:
            return emb(emb_input), emb.weight

        emb_fwd = emb(emb_input)
        grad = torch.ones_like(emb_fwd)
        emb_fwd.backward(grad)
        return emb.weight.grad, emb.weight

    def fn_cpu(emb_input, init_weight):
        return fn(emb_input, "cpu", init_weight)

    def fn_hpu(emb_input):
        return fn(emb_input, "hpu")

    fn_hpu = compile_function_if_compile_mode(fn_hpu)

    cpu_input = torch.LongTensor(
        topo_input_sw_204570() if num_embeddings == 25849 else [[0, 1, 2, 4, 5], [4, 3, 2, 9, 0]]
    )
    hpu_input = cpu_input.to("hpu")

    if Verbose:
        print(f"{cpu_input = }")
        print(f"{hpu_input = }")

    hpu_output, hpu_weight = fn_hpu(hpu_input)
    cpu_output, cpu_weight = fn_cpu(cpu_input, hpu_weight.cpu())

    if Verbose:
        print(f"{cpu_output = }")
        print(f"{hpu_output = }")

        print(f"{cpu_weight = }")
        print(f"{hpu_weight = }")

        print_tensors(["grad_hpu", "grad_cpu"], [hpu_output.cpu(), cpu_output], atol=0, rtol=0)

    compare_tensors(hpu_output.cpu(), cpu_output, atol=0.0, rtol=0.0)
    compare_tensors(hpu_weight.cpu(), cpu_weight, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"embedding", "embedding_dense_backward"})


@pytest.mark.parametrize("num_embeddings", [10])
@pytest.mark.parametrize("embedding_dim", [3])
@pytest.mark.parametrize("padding_idx", [0, None])
@pytest.mark.parametrize("bwd", [False, True])
def test_hpu_nn_embedding(num_embeddings, embedding_dim, padding_idx, bwd):
    hpu_nn_embedding_test(num_embeddings, embedding_dim, padding_idx, bwd)


@pytest.mark.parametrize("num_embeddings", [25849])
@pytest.mark.parametrize("embedding_dim", [50])
@pytest.mark.parametrize("padding_idx", [0])
@pytest.mark.parametrize("bwd", [True])
def test_hpu_nn_embedding_topo(num_embeddings, embedding_dim, padding_idx, bwd):
    hpu_nn_embedding_test(num_embeddings, embedding_dim, padding_idx, bwd)
