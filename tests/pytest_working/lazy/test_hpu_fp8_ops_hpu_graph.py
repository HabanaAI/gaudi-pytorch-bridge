###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import habana_frameworks.torch.hpu as ht
import pytest
import torch
from fp8_utils import convertExpBiasToScale
from test_utils import (
    compare_tensors,
    is_gaudi1,
    is_gaudi2,
    is_pytest_mode_eager,
)

Verbose = False

# Disable dynamic shapes
ht.disable_dynamic_shape()

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")]


def common_h2d_scales(
    src_dtype, batched_tensors, fuse_cast, scale_values, scale_out_values, is_hw_aligned, dtc, dry_run
):
    ht.enable_inference_mode()

    fp8_dtype = torch.float8_e4m3fn
    # change shape to generate different graphs for hw/non-hw modes
    shape_a = (4, 8) if is_hw_aligned else (6, 8)
    shape_b = (8, 16)
    if batched_tensors:
        shape_a = (2, 1) + shape_a
        shape_b = (2,) + shape_b

    import habana_frameworks.torch.utils.experimental as htexp

    htexp._set_scale_attributes(is_hw_aligned, 10)

    def generate_inputs(scale_a, scale_b, scale_out):
        a = torch.randn(shape_a, dtype=src_dtype)
        b = torch.randn(shape_b, dtype=src_dtype)
        if scale_a == 256.0:
            a /= 4.0
        if scale_b == 256.0:
            b /= 4.0
        ah = a.to("hpu")
        bh = b.to("hpu")
        sa = torch.tensor(scale_a, dtype=src_dtype)
        sb = torch.tensor(scale_b, dtype=src_dtype)
        so = torch.tensor(scale_out, dtype=src_dtype)

        return a, b, ah, bh, sa, sb, so

    def fn_hpu(a, b, sa, sb, sa_inv, sb_inv, scale_out):
        scaled_a, _ = torch.ops.hpu.cast_to_fp8_v2(a, sa, False, False, fp8_dtype)
        scaled_b, _ = torch.ops.hpu.cast_to_fp8_v2(b, sb, False, False, fp8_dtype)
        if fuse_cast:
            return torch.ops.hpu.cast_to_fp8_v2(
                torch.ops.hpu.fp8_gemm_v2(
                    scaled_a, False, scaled_b, False, None, src_dtype, sa_inv, sb_inv, None, False
                ),
                scale_out,
                False,
                False,
                fp8_dtype,
            )[0]
        else:
            return torch.ops.hpu.fp8_gemm_v2(
                scaled_a, False, scaled_b, False, None, src_dtype, sa_inv, sb_inv, None, False
            )

    def fn_cpu(a, b, sa, sb, sa_inv, sb_inv, scale_out):
        scaled_a = (a * sa).to(fp8_dtype).to(src_dtype)
        scaled_b = (b * sb).to(fp8_dtype).to(src_dtype)
        res = torch.matmul(scaled_a, scaled_b) * (sa_inv * sb_inv)
        if fuse_cast:
            res = (res * scale_out).to(fp8_dtype)
        return res

    fn_hpu = ht.hpu.wrap_in_hpu_graph_func(fn_hpu, asynchronous=False, disable_tensor_cache=dtc, dry_run=dry_run)
    i = 0
    for sa_val in scale_values:
        for sb_val in scale_values:
            for so_val in scale_out_values:
                # print(f"=== iteration {i}")
                i = i + 1
                a, b, ah, bh, sa, sb, so = generate_inputs(sa_val, sb_val, so_val)

                # Scales as CPU Tensors are intentional.
                res_hpu = fn_hpu(ah, bh, sa, sb, 1 / sa, 1 / sb, so)
                res_cpu = fn_cpu(a, b, sa, sb, 1 / sa, 1 / sb, so)

                tol = 1e-5 if src_dtype == torch.float else 0.125

                compare_tensors(res_hpu, res_cpu, atol=tol, rtol=tol)
    htexp._set_scale_attributes(False, 0)
    ht.disable_inference_mode()


@pytest.mark.skipif(is_pytest_mode_eager(), reason="Eager mode doesn't support H2D scales.")
@pytest.mark.parametrize("src_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("batched_tensors", [False, True])
@pytest.mark.parametrize("fuse_cast", [False])
@pytest.mark.parametrize("dtc", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_h2d_scales(src_dtype, batched_tensors, fuse_cast, dtc, dry_run):
    torch.manual_seed(42)  # For reproducibility
    bias_values = [3, 7, 11, 15] if is_gaudi2() else [2, 6, 12, 15]
    scale_values = convertExpBiasToScale(bias_values)
    scale_out_values = convertExpBiasToScale((3, 11)) if fuse_cast else (1.0,)

    common_h2d_scales(
        src_dtype,
        batched_tensors,
        fuse_cast,
        scale_values,
        scale_out_values,
        is_hw_aligned=True,
        dtc=dtc,
        dry_run=dry_run,
    )


@pytest.mark.skipif(is_pytest_mode_eager(), reason="Eager mode doesn't support H2D scales.")
@pytest.mark.parametrize("src_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fuse_cast", [False, True])
@pytest.mark.parametrize("dtc", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_non_hw_h2d_scales(src_dtype, fuse_cast, dtc, dry_run):
    scale_values = (2.5, 0.7)
    scale_out_values = (0.4, 3.0) if fuse_cast else (1.0,)

    common_h2d_scales(
        src_dtype, False, fuse_cast, scale_values, scale_out_values, is_hw_aligned=False, dtc=dtc, dry_run=dry_run
    )
