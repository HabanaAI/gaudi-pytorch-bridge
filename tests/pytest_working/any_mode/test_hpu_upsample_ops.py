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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("antialias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float])
class TestHpuUpsample:
    @staticmethod
    def _common_test(variant, shape, size, scale_factor, align_corners, antialias, mode, dtype):
        if ((size != None and scale_factor != None) or (size == None and scale_factor == None)):
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
        cpu_wrapped_fn = torch.compile(upsample_fn) if pytest.mode == "compile" else upsample_fn
        hpu_wrapped_fn = torch.compile(upsample_fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else upsample_fn

        cpu_output = cpu_wrapped_fn(cpu_input)
        hpu_output = hpu_wrapped_fn(hpu_input).cpu()
        assert torch.allclose(cpu_output, hpu_output, rtol=1e-4)

    @pytest.mark.parametrize("shape_and_size", [((2, 2, 3, 3), None), ((2, 2, 3, 3), (6, 6))])
    @pytest.mark.parametrize("scale_factor", [None, [1, 2]])
    @pytest.mark.parametrize("variant", ["fwd", "bwd"])
    def test_upsample_bicubic2d(self, shape_and_size, scale_factor, align_corners, antialias, variant, dtype):
        if (pytest.mode == "compile" and antialias == False):
            pytest.xfail("[SW-163842] aten._unsafe_index - IndexError: index is out of bounds")
        shape, size = shape_and_size
        TestHpuUpsample._common_test(variant, shape, size, scale_factor, align_corners, antialias, "bicubic", dtype)
