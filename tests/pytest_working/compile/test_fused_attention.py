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

import itertools
import math
import os

import pytest
import torch
import torch._inductor.config
import torch.utils.checkpoint
from habana_frameworks.torch.dynamo.utils import str_to_bool
from test_dynamo_utils import use_eager_fallback
from torch._dynamo.utils import counters

# In this test file we only check if the test cases can fuse the attention pattern and map to autograd overwrite sdpa implementation
# PT_HPU_USE_OVERRIDE_ATEN_SDPA=True and PT_HPU_USE_FUSE_SDPA_PASS=True are required otherwise we will skip the test
# Also need the
check_aten_sdpa_fusion_flag = str_to_bool(os.environ.get("PT_HPU_USE_FUSE_SDPA_PASS", False)) and str_to_bool(
    os.environ.get("PT_HPU_USE_OVERRIDE_ATEN_SDPA", False)
)


class TestHpuFusedAttention:

    def compare_outputs(output1, output2, rtol=1e-05, atol=1e-08):
        """
        Compares two outputs, which can be tensors or tuples of tensors,
        using torch.allclose.
        """
        if isinstance(output1, tuple) and isinstance(output2, tuple):
            if len(output1) != len(output2):
                return False
            return all(
                torch.allclose(t1, t2.cpu(), rtol=rtol, atol=atol) for t1, t2 in zip(output1, output2, strict=False)
            )

        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.allclose(output1, output2.cpu(), rtol=rtol, atol=atol)

        return False

    @staticmethod
    def _clone_inputs_to_hpu(inputs):
        """
        Convert the cpu tensors to hpu tensors
        """

        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone().to("hpu")

        return [clone(x) for x in inputs]

    @staticmethod
    def _check_common(
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
        check_train=False,
        override_check_equal=False,
        dtype=torch.float,
        rtol=1.3e-6,
        use_static_shapes=True,
    ):
        if not check_aten_sdpa_fusion_flag:
            pytest.skip(
                "Unsupported test configuration, to enable this need to use `PT_HPU_USE_OVERRIDE_ATEN_SDPA` and `PT_HPU_USE_FUSE_SDPA_PASS` flag"
            )
        # if os.environ.get("SERIALIZED_PATTERN_PATH", "DEFAULT") == "DEFAULT":
        #     pytest.skip("working on default pytorch pattern, need to use the `SERIALIZED_PATTERN_PATH`")
        from torch._inductor.pattern_matcher import _seen_patterns

        # need to clear the seen patterns required to run multiple test
        # otherwise it will give error
        _seen_patterns.clear()
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device="cpu", dtype=dtype),
                torch.randn(tensor_shape, device="cpu", dtype=dtype),
                torch.randn(tensor_shape, device="cpu", dtype=dtype),
            ]
        else:
            args1 = list(args1)
        args2 = TestHpuFusedAttention._clone_inputs_to_hpu(args1)

        for training in [True] if check_train else [False]:
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            if not use_static_shapes:
                torch._dynamo.mark_dynamic(args2[0], 0)
                torch._dynamo.mark_dynamic(args2[1], 0)
                torch._dynamo.mark_dynamic(args2[2], 0)

            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            counters.clear()
            torch._dynamo.reset()
            dot_prod_attention_hpu = torch.compile(dot_prod_attention, backend="hpu_backend", fullgraph=True)
            result2 = dot_prod_attention_hpu(*(args2 + dropout_arg))

            if has_fuse_pattern:
                assert counters["inductor"]["fuse_attention"] >= 1, "Fused pattern is not applied"

            # some tests configured with very low dropout where we still want to check equality
            if not has_dropout or override_check_equal:
                TestHpuFusedAttention.compare_outputs(result1, result2, atol=atol, rtol=1.3e-6)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2, strict=False):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and (not has_dropout or override_check_equal)
                    ):
                        assert torch.allclose(arg1.grad, arg2.grad.cpu(), atol=atol, rtol=rtol)

    def test_sdpa_rewriter_01(self):
        def dot_prod_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1)).div(math.sqrt(key.shape[-1])).softmax(dim=-1).matmul(value)
            )

        for dtype in [torch.float]:
            atol = 0.001
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            if dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            TestHpuFusedAttention._check_common(dot_prod_attention, dtype=dtype, atol=atol, rtol=rtol)
            TestHpuFusedAttention._check_common(dot_prod_attention, dtype=dtype, atol=atol, rtol=rtol, check_train=True)

    def test_sdpa_rewriter_02(self):
        def dot_prod_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        TestHpuFusedAttention._check_common(dot_prod_attention)
        TestHpuFusedAttention._check_common(dot_prod_attention, check_train=True)

    def test_sdpa_rewriter_03(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training: bool
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        TestHpuFusedAttention._check_common(dot_prod_attention, contains=False, has_dropout=True)
        TestHpuFusedAttention._check_common(dot_prod_attention, contains=False, has_dropout=True, check_train=True)

    def test_sdpa_rewriter_04(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                p=0.2,
                inplace=False,
                training=training,
            ).matmul(value)

        TestHpuFusedAttention._check_common(dot_prod_attention, contains=False, has_dropout=True)
        TestHpuFusedAttention._check_common(dot_prod_attention, contains=False, has_dropout=True, check_train=True)

    def test_sdpa_rewriter_bert_large(self):
        def dot_prod_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = (
                torch.randn((1, 1, 16, 16), dtype=torch.float, device=query.device).tril(diagonal=0)
                * -3.4028234663852886e38
            )
            query = query.permute(0, 2, 1, 3).contiguous()
            key = key.permute(0, 2, 1, 3).contiguous()
            value = value.permute(0, 2, 1, 3).contiguous()
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores.to(torch.float32)
            attention_scores = attention_scores / (8.0)
            attention_scores = attention_scores + attn_mask
            attention_scores = attention_scores.to(torch.bfloat16)
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = torch.nn.functional.dropout(attention_probs, p=0.1, inplace=False)
            y = torch.matmul(attention_probs, value)
            y = y.permute([0, 2, 1, 3]).to(torch.float32)
            return y

        tensor_shape = (32, 16, 32, 32)
        args = [
            torch.randn(tensor_shape, dtype=torch.bfloat16, device="hpu"),
            torch.randn(tensor_shape, dtype=torch.bfloat16, device="hpu"),
            torch.randn(tensor_shape, dtype=torch.bfloat16, device="hpu"),
        ]
        with use_eager_fallback():  # to allow transpose.int to fallback to eager
            TestHpuFusedAttention._check_common(
                dot_prod_attention, args1=args, contains=False, has_dropout=True, check_train=True
            )
