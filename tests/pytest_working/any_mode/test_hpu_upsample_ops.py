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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
class TestHpuUpsample:
    @staticmethod
    def _common_test(variant, shape, size, scale_factor, align_corners, antialias, mode, dtype):
        if (size != None and scale_factor != None) or (size == None and scale_factor == None):
            pytest.skip("Unsupported test configuration")

        def upsample_fwd_fn(input):
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners, None, antialias)

        def upsample_bwd_fn(input):
            upsample = torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners, None, antialias)
            grad = torch.ones_like(upsample)
            upsample.backward(grad)
            return input.grad

        cpu_input = torch.rand(shape, dtype=dtype)
        hpu_input = cpu_input.to("hpu")
        if variant == "bwd":
            cpu_input.requires_grad = True
            hpu_input.requires_grad = True
            upsample_fn = upsample_bwd_fn
        else:
            upsample_fn = upsample_fwd_fn

        torch._dynamo.reset()
        hpu_wrapped_fn = torch.compile(upsample_fn, backend="hpu_backend") if pytest.mode == "compile" else upsample_fn

        cpu_output = upsample_fn(cpu_input)
        hpu_output = hpu_wrapped_fn(hpu_input).cpu()
        assert torch.allclose(cpu_output, hpu_output, rtol=1e-4)

    @pytest.mark.parametrize("shape_and_size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_bicubic2d(self, shape_and_size, scale_factor, align_corners, antialias, variant, dtype):
        if antialias and (
            (shape_and_size == ((2, 2, 3, 3), (6, 6)) and scale_factor == None)
            or (shape_and_size == ((2, 2, 3, 3), None) and scale_factor == [1, 2])
        ):
            pytest.skip("Unsupported test configuration (aten::_upsample_bicubic2d_aa.out is not yet supported on HPU)")
        if pytest.mode == "compile" and antialias == False:
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, antialias, "bicubic", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_bilinear2d(self, shape_and_size, scale_factor, align_corners, antialias, variant, dtype):
        if antialias and (
            (shape_and_size == ((2, 2, 3, 3), (6, 6)) and scale_factor == None)
            or (shape_and_size == ((2, 2, 3, 3), None) and scale_factor == [1, 2])
        ):
            pytest.skip(
                "Unsupported test configuration (aten::_upsample_bilinear2d_aa.out is not yet supported on HPU)"
            )
        if pytest.mode == "compile" and antialias == False:
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, antialias, "bilinear", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_nearest1d(self, shape_and_size, scale_factor, variant, dtype):
        if pytest.mode == "compile":
            pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_nearest_exact1d(self, shape_and_size, scale_factor, variant, dtype):
        if pytest.mode == "compile":
            pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest-exact", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_nearest2d(self, shape_and_size, scale_factor, variant, dtype):
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 2, 3, 3, 3), None), ((2, 2, 3, 3, 3), (6, 6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2, 3]], ids=format_tc)
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_nearest3d(self, shape_and_size, scale_factor, variant, dtype):
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape_and_size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_linear1d(self, shape_and_size, scale_factor, align_corners, variant, dtype):
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, False, "linear", dtype)

    @pytest.mark.parametrize(
        "shape_and_size",
        [
            ((2, 2, 3, 3, 3), None),
            ((2, 2, 3, 3, 3), (3, 6, 9)),
            ((2, 2, 3, 3, 3), (6, 3, 9)),
            ((2, 2, 3, 3, 3), (9, 6, 3)),
            ((2, 2, 3, 3, 3), (9, 3, 3)),
            ((2, 2, 3, 3, 3), (6, 6, 6)),
            ((2, 2, 3, 3, 3), (6, 3, 6)),
        ],
        ids=format_tc,
    )
    @pytest.mark.parametrize(
        "scale_factor", [None, [1, 2, 3], [2, 2, 2], [2, 1, 2], [2, 1, 3], [3, 2, 1], [3, 1, 1]], ids=format_tc
    )
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_trilinear3d(self, shape_and_size, scale_factor, align_corners, variant, dtype):
        shape, size = shape_and_size

        is_bwd = variant == "bwd"
        illegal_size = size is not None and size[0] != 3
        illegal_scale = scale_factor is not None and scale_factor[0] != 1

        if illegal_size or illegal_scale or is_bwd:
            pytest.skip("SW-188775")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")

        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, False, "trilinear", dtype)
