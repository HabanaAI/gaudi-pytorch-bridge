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

# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import os
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from unittest import skipUnless

import habana_frameworks.torch as htorch
import torch
import torch._inductor.config
import torch.utils.checkpoint
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    BlockMask,
    _identity,
    _score_mod_signature,
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16
from torch.testing._internal.common_utils import IS_MACOS
from torch.utils._triton import has_triton

# Use this decorator only when hitting Triton bugs on H100
running_on_a100_only = skipUnless(
    torch.cuda.is_available() and has_triton() and torch.cuda.get_device_capability() == (8, 0),
    "Requires A100 and Triton",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


@contextmanager
def temp_float32_matmul_precision(precision: str):
    """
    Temporarily set the float32 matmul precision and restore it after the context is exited.

    Args:
    precision (str): The precision to set ('highest', 'high', or 'medium').
    """
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def create_attention(score_mod, block_mask, enable_gqa=False, return_lse=False):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
    )


def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod,
        1,
        1,
        query.shape[-2],
        key.shape[-2],
        query.device,
    )
    return block_mask


TEST_ON_CUDA = (
    torch.cuda.is_available() and torch.utils._triton.has_triton() and torch.cuda.get_device_capability() >= (8, 0)
)

TEST_ON_HPU = htorch.hpu.is_available()

if TEST_ON_CUDA:
    test_device = "cuda"
    test_dtypes = (
        [torch.float32, torch.bfloat16, torch.float16] if PLATFORM_SUPPORTS_BF16 else [torch.float16, torch.float32]
    )
    test_dtypes_fast = [torch.float16]
elif TEST_ON_HPU:
    test_device = "hpu"
    test_dtypes = [torch.float32]
else:
    test_device = "cpu"
    torch_config_string = torch.__config__.show()
    LONG_COMPILATION_ON_CPU = False
    if "CLANG" in torch_config_string.upper():
        # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
        # TODO: check reason of long compile time
        LONG_COMPILATION_ON_CPU = True

    import os

    # skip since currently flex attention requires at least `avx2` support on CPU.
    IS_PLATFORM_SUPPORTED = (
        not torch.xpu.is_available()
        and not IS_MACOS
        and torch.cpu._is_avx2_supported()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
    )

    test_dtypes = (
        [torch.float32, torch.bfloat16]
        if torch.backends.mkldnn.is_available() and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    )
    test_dtypes_fast = [torch.float32]


# --------- Useful score mod functions for testing ---------
def _causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return score + (token_q - token_kv)


def _rel_causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


def _generate_alibi_bias(num_heads: int):
    def _alibi_bias(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        token_q: Tensor,
        token_kv: Tensor,
    ) -> Tensor:
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias


def _inverse_causal(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))


def _times_two(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * 2


def _squared(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * score


def _head_offset(dtype: torch.dtype):
    """Captured Buffer"""
    head_offset = torch.rand(H, device="cuda", dtype=dtype)

    def score_mod(score, b, h, m, n):
        return score * head_offset[h]

    return score_mod


def _trig(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return torch.sin(torch.cos(score)) + torch.tan(b)


def _trig2(score, b, h, m, n):
    """Branching joint graph"""
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    z = cos_score * sin_score + torch.tan(b)
    return z


# --------- Useful mask mod functions for testing ---------
def _causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q >= token_kv


def _inverse_causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q <= token_kv


test_score_mods = [
    _identity,
    _times_two,
    _squared,
    _causal,
    _inverse_causal,
    _rel_bias,
    _rel_causal,
    # _generate_alibi_bias(8),
]

test_score_mask_mod_map = {
    _identity: noop_mask,
    _times_two: noop_mask,
    _squared: noop_mask,
    _causal: _causal_mask,
    _inverse_causal: _inverse_causal_mask,
    _rel_bias: noop_mask,
    _rel_causal: _causal_mask,
    # _generate_alibi_bias(8): noop_mask,
}

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 1
H = 4
S = 4
D = 4

test_Hq_Hkv = [
    (4, 2),
    (4, 1),
]

test_Bq_Bkv = [
    (3, 1),
    (4, 1),
    (5, 1),
]

test_block_size = [
    128,
    256,
    (128, 256),
    (256, 128),
]


def query_key_value_clones(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None, device="cpu"
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).to(device=device).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).to(device=device).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).to(device=device).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device
        if self.device == "cpu":
            if LONG_COMPILATION_ON_CPU:
                self.skipTest("skip UT for CPU due to long compilation time found in CI")
            if not IS_PLATFORM_SUPPORTED:
                self.skipTest("skip UT due to not support on those platforms")

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: str | None = None,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        is_paged_attention: bool = False,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
                if is_paged_attention:
                    # paged attention is less accurate since it may reorder
                    # the blocks from block mask
                    fudge_factor = 20.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

    def _check_out_and_grad(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query")
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key")
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value")

    def _check_grads(
        self,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        dtype = q.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query")
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key")
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value")

    def _check_grads(
        self,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        dtype = q.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query")
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key")
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value")

    def _check_only_ref_grads_with_grads(
        self,
        q_ref: torch.Tensor,
        q_1_grad: torch.Tensor,
        q_2_grad: torch.Tensor,
        k_ref: torch.Tensor,
        k_1_grad: torch.Tensor,
        k_2_grad: torch.Tensor,
        v_ref: torch.Tensor,
        v_1_grad: torch.Tensor,
        v_2_grad: torch.Tensor,
    ):
        dtype = q_ref.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(q_ref.grad, q_1_grad, q_2_grad, q_fudge_factor, "Grad_Query")
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(k_ref.grad, k_1_grad, k_2_grad, k_fudge_factor, "Grad_Key")
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(v_ref.grad, v_1_grad, v_2_grad, v_fudge_factor, "Grad_Value")

    def run_test(
        self,
        return_lse,
        block_size,
        traning,
        gqa,
        score_mod: _score_mod_signature,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int | None = None,
        KV_H: int | None = None,
        KV_S: int | None = None,
        V_D: int | None = None,
        block_mask: BlockMask | None = None,
    ):
        if return_lse and traning:
            return

        # skip tests ToDo
        if gqa and traning:
            return

        if KV_B is None:
            KV_B = Q_B
        if KV_H is None:
            KV_H = Q_H
        if KV_S is None:
            KV_S = Q_S
        if V_D is None:
            V_D = Q_D

        if gqa:
            KV_H = Q_H // 2

        test_inference_only = not traning

        torch.manual_seed(3874)
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )

        mask_mod = test_score_mask_mod_map[score_mod]

        if block_mask is None:
            block_mask = create_block_mask(
                mask_mod, Q_B, Q_H, Q_S, KV_S, device="cpu", BLOCK_SIZE=block_size  # self.device,
            )

        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        sdpa_partial_ref = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H), return_lse=return_lse)

        golden_out = sdpa_partial_ref(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial_ref(q_ref, k_ref, v_ref)

        block_mask = block_mask.to(device=self.device)
        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H), return_lse=return_lse)
        compiled_sdpa = torch.compile(sdpa_partial, backend="hpu_backend", fullgraph=True)

        if test_inference_only:
            if not return_lse:
                compiled_out = compiled_sdpa(q, k, v).to("cpu")
                self._check_out(
                    golden_out,
                    ref_out,
                    compiled_out,
                    is_paged_attention=False,
                )
            else:
                compiled_out = compiled_sdpa(q, k, v)
                lse_out = compiled_out[1].to("cpu")
                compiled_out = compiled_out[0].to("cpu")
                self._check_out(
                    golden_out[0],
                    ref_out[0],
                    compiled_out,
                    is_paged_attention=False,
                )
                self._check_out(
                    golden_out[1] * 0.6931471805599453,
                    ref_out[1] * 0.6931471805599453,
                    lse_out,
                    is_paged_attention=False,
                )
        else:
            compiled_out = compiled_sdpa(q, k, v)
            golden_out.sum().backward()
            ref_out.sum().backward()
            compiled_out.sum().backward()

            q_hpu = q.to("cpu")
            q_hpu.grad = q.grad.to("cpu")
            k_hpu = k.to("cpu")
            k_hpu.grad = k.grad.to("cpu")
            v_hpu = v.to("cpu")
            v_hpu.grad = v.grad.to("cpu")
            compiled_out_hpu = compiled_out.to("cpu")
            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out_hpu,
                q_gold,
                q_ref,
                q_hpu,
                k_gold,
                k_ref,
                k_hpu,
                v_gold,
                v_ref,
                v_hpu,
            )

    def preprocess_paged_attention(
        self,
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype = torch.float16,
        page_size: int = 128,
    ) -> tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:
        assert block_mask is not None, "Must provide block_mask"
        Q_B, Q_H, Q_S, _ = q.shape
        KV_B, KV_H, KV_S, QK_D = k.shape
        _, _, _, V_D = v.shape

        # test with different batch size
        max_batch_size = max(Q_B, KV_B) + 3

        n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

        # allocate cache
        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            QK_D,
            device=self.device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device=self.device,
            dtype=dtype,
        )

        # For testing purposes, we randomly initialize the page table, which maps
        # (batch_idx, logical_block_idx) to physical_block_idx. Specifically, PagedAttention
        # maintains a stack empty_pages of unused physical_block_idx. The `batch_reserve`
        # function grabs physical_block_idx from the top of empty_pages until there are enough
        # pages for each batch index (i.e., num pages for batch_idx >= target_seq_len[batch_idx]).
        # For example, at the first batch_reserve call, physical block indices (1,...,KV_S//4)
        # are allocated to batch index 0, and physical block indices
        # (KV_S//4+1, ..., KV_S//4 + KV_S//2) are allocated to batch index 1, etc.
        # Thus, kv tensors of batch index 1 will be scattered in the kv cache, simulating
        # a real use case of paged attention.
        paged_attention = PagedAttention(n_pages, page_size, max_batch_size, device=self.device)
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device=self.device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device=self.device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device=self.device),
        )
        batch_reserve(paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device=self.device))

        # update cache with k and v
        input_pos = torch.arange(KV_S, device=self.device, dtype=torch.int32)
        batch_idx = torch.arange(KV_B, device=self.device, dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        # convert block mask and score mod
        converted_block_mask = paged_attention.convert_logical_block_mask(block_mask)
        converted_score_mod = paged_attention.get_score_mod(score_mod)
        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype = torch.float16,
        block_mask: BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, Q_H, Q_S, KV_H, KV_S = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[1],
            k.shape[2],
        )
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        if block_mask is None:
            block_mask = create_block_mask(noop_mask, B, 1, Q_S, KV_S, device=self.device)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod,
            q,
            k,
            v,
            block_mask,
            dtype,
            block_mask.BLOCK_SIZE[1],
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        return_lse = True
        if test_inference_only:
            return_lse = False
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
            )

        else:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
            )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Callable | None = _identity,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
        block_mask: BlockMask | None = None,
    ):
        assert Q_H % KV_H == 0
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        q = torch.randn((Q_B, Q_H, Q_S, QK_D), dtype=dtype, device=self.device, requires_grad=False)
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device=self.device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S, device=self.device)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        golden_out, golden_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(score_mod, q, k, v, dtype, block_mask)
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
            is_paged_attention=True,
        )

        if not test_inference_only:
            self._check_out(
                golden_lse,
                ref_lse,
                compiled_lse,
                is_paged_attention=True,
            )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
    ):
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = sdpa_call(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)
        if test_inference_only:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn((Q_B, Q_H, Q_S, V_D), dtype=dtype, device=self.device)

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def run_dynamic_test(
        self,
        score_mask_mod: tuple[Callable, Callable],
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        score_mod, mask_mod = score_mask_mod

        # First batch with original dimensions (B, H, S, D)
        block_mask1 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)

        q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1(q1_gold, k1_gold, v1_gold)

        backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out1.backward(backward_grad1.to(torch.float64))
        ref_out1.backward(backward_grad1)

        # Second batch with modified dimensions (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)

        q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2(q2_gold, k2_gold, v2_gold)

        backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out2.backward(backward_grad2.to(torch.float64))
        ref_out2.backward(backward_grad2)

        # Third batch with modified dimensions (B * 2, H, S / 4, D)
        S = int(S / 2)
        block_mask3 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)

        q3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q3_ref, k3_ref, v3_ref = query_key_value_clones(q3, k3, v3)
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3(q3_gold, k3_gold, v3_gold)

        backward_grad3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out3.backward(backward_grad3.to(torch.float64))
        ref_out3.backward(backward_grad3)

        # Clear dynamo counters
        torch._dynamo.reset()

        # First compilation with original dimensions
        compiled_sdpa1 = torch.compile(sdpa_partial1, dynamic=True)
        compiled_out1 = compiled_sdpa1(q1, k1, v1)
        compiled_out1.backward(backward_grad1)

        self._check_out_and_grad(
            golden_out1,
            ref_out1,
            compiled_out1,
            q1_gold,
            q1_ref,
            q1,
            k1_gold,
            k1_ref,
            k1,
            v1_gold,
            v1_ref,
            v1,
        )
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

        # Second compilation with new dimensions
        compiled_sdpa2 = torch.compile(sdpa_partial2, dynamic=True)
        compiled_out2 = compiled_sdpa2(q2, k2, v2)
        compiled_out2.backward(backward_grad2)

        self._check_out_and_grad(
            golden_out2,
            ref_out2,
            compiled_out2,
            q2_gold,
            q2_ref,
            q2,
            k2_gold,
            k2_ref,
            k2,
            v2_gold,
            v2_ref,
            v2,
        )
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

        # Third compilation with new dimensions
        compiled_sdpa3 = torch.compile(sdpa_partial3, dynamic=True)
        compiled_out3 = compiled_sdpa3(q3, k3, v3)
        compiled_out3.backward(backward_grad3)

        self._check_out_and_grad(
            golden_out3,
            ref_out3,
            compiled_out3,
            q3_gold,
            q3_ref,
            q3,
            k3_gold,
            k3_ref,
            k3,
            v3_gold,
            v3_ref,
            v3,
        )
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

    def run_automatic_dynamic_test(
        self,
        return_lse,
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        MAX_S = S
        block_mask1 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)
        # The first eager batch, shape (B, H, S, D)
        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out1 = sdpa_partial1(q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64))
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out2 = sdpa_partial2(q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64))
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out3 = sdpa_partial3(q3.to(torch.float64), k3.to(torch.float64), v3.to(torch.float64))
        ref_out3 = sdpa_partial3(q3, k3, v3)

        # Need to clear dynamo counters, since flex attention eager mode also uses dynamo tracing.
        # We check dynamo counters["frames"]["ok"] to ensure:
        # 1, the first batch is compiled with static shape
        # 2, the second batch is compiled with dynamic shape
        # 3, no re-compilation in the third batch
        torch._dynamo.reset()

        # Note, it seems like we really are less accurate than the float32
        # computation, likely due to the online softmax
        if dtype == torch.float32:
            fudge_factor = 10.0
        else:
            fudge_factor = 1.1

        # The first batch.
        compiled_out1 = torch.compile(sdpa_partial1)(q1, k1, v1)
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

        # The second batch (automatic dynamic).
        compiled_out2 = torch.compile(sdpa_partial2)(q2, k2, v2)
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)

        # The third batch (no re-compilation).
        compiled_out3 = torch.compile(sdpa_partial3)(q3, k3, v3)
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)

    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("return_lse", [False, True])
    @common_utils.parametrize("block_size", [2, 4])
    @common_utils.parametrize("traning", [True, False])
    @common_utils.parametrize("gqa", [True, False])
    def test_builtin_score_mods(self, dtype: torch.dtype, score_mod: Callable, return_lse, block_size, traning, gqa):
        self.run_test(return_lse, block_size, traning, gqa, score_mod, dtype)
        # self.run_test_with_paged_attention(score_mod, dtype)


common_utils.instantiate_parametrized_tests(TestFlexAttention)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
