###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
import pytest
import torch
from test_utils import (
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi3,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
    is_pytest_mode_lazy,
)


@pytest.mark.parametrize("dtype", [torch.float, torch.uint8], ids=format_tc)
@pytest.mark.parametrize(
    "variant",
    [
        "fwd",
        "bwd",
    ],
)
class TestHpuUpsample:
    @staticmethod
    def _common_test(variant, shape, size, scale_factor, align_corners, antialias, mode, dtype, rtol=1e-4):
        if (size is not None and scale_factor is not None) or (size is None and scale_factor is None):
            pytest.xfail("Unsupported test configuration")

        def upsample_fwd_fn(input):
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners, None, antialias)

        def upsample_bwd_fn(input):
            upsample = torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners, None, antialias)
            grad = torch.ones_like(upsample)
            upsample.backward(grad)
            return input.grad

        cpu_input = (
            torch.rand(shape, dtype=dtype)
            if dtype == torch.float
            else torch.randint(size=shape, low=0, high=100, dtype=dtype)
        )
        hpu_input = cpu_input.to("hpu")
        if variant == "bwd":
            cpu_input.requires_grad = True
            hpu_input.requires_grad = True
            upsample_fn = upsample_bwd_fn
        else:
            upsample_fn = upsample_fwd_fn

        hpu_wrapped_fn = compile_function_if_compile_mode(upsample_fn)

        cpu_output = upsample_fn(cpu_input)
        hpu_output = hpu_wrapped_fn(hpu_input).cpu()
        assert torch.allclose(cpu_output, hpu_output, rtol=rtol)

    @pytest.mark.parametrize("shape,size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    def test_upsample_bicubic2d(self, shape, size, scale_factor, align_corners, antialias, variant, dtype):
        if dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if is_gaudi3() and not is_pytest_mode_eager() and not antialias and size is None:
            pytest.skip("SW-215817 Invalid node geometry on Gaudi3")
        TestHpuUpsample._common_test(
            variant,
            shape,
            size,
            scale_factor,
            align_corners,
            antialias,
            "bicubic",
            dtype,
            rtol=1 if dtype == torch.uint8 else 1e-4,
        )

    @pytest.mark.parametrize("shape,size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    def test_upsample_bilinear2d(self, shape, size, scale_factor, align_corners, antialias, variant, dtype):
        if dtype == torch.uint8 and not is_pytest_mode_compile() and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if is_gaudi3() and antialias:
            pytest.skip(reason="SW-215817 Antialiasing is not fully supported on Gaudi3")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(
            variant,
            shape,
            size,
            scale_factor,
            align_corners,
            antialias,
            "bilinear",
            dtype,
            rtol=1 if dtype == torch.uint8 else 1e-4,
        )

    @pytest.mark.parametrize("shape,size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    def test_upsample_nearest1d(self, shape, size, scale_factor, variant, dtype):
        if not is_pytest_mode_compile() and dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if pytest.mode == "compile":
            pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape,size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    def test_upsample_nearest_exact1d(self, shape, size, scale_factor, variant, dtype):
        if is_pytest_mode_lazy() and dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if pytest.mode == "compile":
            pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
        if pytest.mode == "eager" and variant == "bwd":
            pytest.xfail("[SW-205662] Sporadic graph compile fail.")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest-exact", dtype)

    @pytest.mark.parametrize("shape, size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    def test_upsample_nearest2d(self, shape, size, scale_factor, variant, dtype):
        if not is_pytest_mode_compile() and dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape, size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]], ids=format_tc)
    def test_upsample_nearest_exact2d(self, shape, size, scale_factor, variant, dtype):
        if not is_pytest_mode_compile() and variant == "bwd" and dtype == torch.uint8:
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest-exact", dtype)

    @pytest.mark.parametrize("shape,size", [((2, 2, 3, 3, 3), None), ((2, 2, 3, 3, 3), (6, 6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2, 3]], ids=format_tc)
    def test_upsample_nearest3d(self, shape, size, scale_factor, variant, dtype):
        if not is_pytest_mode_compile() and dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest", dtype)

    @pytest.mark.parametrize("shape,size", [((2, 2, 3, 3, 3), None), ((2, 2, 3, 3, 3), (6, 6, 6))], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [1, 2, 3]], ids=format_tc)
    def test_upsample_nearest_exact3d(self, shape, size, scale_factor, variant, dtype):
        if not is_pytest_mode_compile() and dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, None, False, "nearest-exact", dtype)

    @pytest.mark.parametrize("shape,size", [((2, 3, 3), None), ((2, 3, 3), 6)], ids=format_tc)
    @pytest.mark.parametrize("scale_factor", [None, [2]], ids=format_tc)
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_upsample_linear1d(self, shape, size, scale_factor, align_corners, variant, dtype):
        if dtype == torch.uint8 and variant == "fwd":
            pytest.xfail("Unsupported dtype `\"compute_indices_weights_linear\" not implemented for 'Byte'`")
        if dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("RuntimeError: only Tensors of floating point and complex dtype can require gradients")
        if pytest.mode == "compile":
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, False, "linear", dtype)

    @pytest.mark.parametrize(
        "shape,size",
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
    def test_upsample_trilinear3d(self, shape, size, scale_factor, align_corners, variant, dtype):
        if dtype == torch.uint8 and variant == "fwd":
            pytest.xfail("Unsupported dtype `\"compute_indices_weights_trilinear\" not implemented for 'Byte'`")
        if dtype == torch.uint8 and variant == "bwd":
            pytest.xfail("Unsupported dtype: `only Tensors of floating point and complex dtype can require gradients`")

        illegal_size = size is not None and size[0] != 3
        illegal_scale = scale_factor is not None and scale_factor[0] != 1

        if illegal_size or illegal_scale:
            pytest.skip("SW-188775")

        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, False, "trilinear", dtype)
