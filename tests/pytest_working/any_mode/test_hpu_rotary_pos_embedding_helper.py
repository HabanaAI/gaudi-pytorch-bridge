###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from habana_frameworks.torch.hpex.kernels import (
    RotaryPosEmbeddingHelperV1,
    RotaryPosEmbeddingHelperV2,
    RotaryPosEmbeddingHelperV3,
    RotaryPosEmbeddingMode,
    apply_rotary_pos_emb,
)
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, cpu, hpu, is_gaudi1, is_pytest_mode_compile

apply_rotary_pos_emb_v1_test_case_list = [
    # p_size, cos_sin_size, offset
    ((64, 8, 64), (64, 1, 64), 0),
    ((64, 8, 64), (64, 1, 64), 3),
    ((8, 1, 32, 8), (8, 1, 1, 8), 0),
    ((8, 2, 32, 8), (8, 1, 1, 8), 2),
    ((8, 1, 32, 8), (8, 1, 1, 8), 2),
    ((32, 8, 32), (32, 1, 32), 2),
    ((52, 8, 52), (52, 1, 52), 2),
    ((64, 8, 64), (64, 1, 64), 2),
]

apply_rotary_pos_emb_v2_test_case_list = [
    # p_size, cos_sin_size
    ((1, 32, 133, 32), (1, 1, 4096, 32)),
    ((1, 32, 1, 32), (1, 1, 4096, 32)),
    ((1, 6, 4, 6), (1, 1, 32, 6)),
    ((1, 6, 4, 6), (1, 1, 6, 6)),
    ((2, 32, 108, 128), (1, 1, 108, 128)),
    ((2, 64, 108, 128), (1, 1, 108, 128)),
]

apply_rotary_pos_emb_gptj_test_case_list = [
    # p_size, cos_sin_size
    ((1, 1, 1, 2), (1, 1, 1)),
    ((2, 4, 2, 8), (1, 4, 4)),
    ((4, 48, 8, 64), (1, 48, 32)),
    ((32, 1, 16, 64), (1, 1, 32)),
]

apply_rotary_pos_emb_diff_dtypes_test_case_list = [
    # p_size, cos_sin_size
    ((1, 6, 4, 6), (1, 1, 32, 6)),
]

apply_rotary_pos_emb_chatglm_test_case_list = [
    # p_size, cos_sin_size
    ((2, 4, 2, 8), (1, 2, 4)),
    ((32, 4, 8, 32), (8192, 8)),
    ((32, 4, 8, 32), (8192, 16)),
]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_v1_ref(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """
    Based on apply_rotary_pos_emb() from the GPT-NeoX model in Transformer version 4.27.4 or lower.
    Used, for example, in the LLaMA model.
    """
    cos = cos[..., offset : p.shape[0] + offset]
    sin = sin[..., offset : p.shape[0] + offset]

    return (p * cos) + (rotate_half(p) * sin)


def apply_rotary_pos_emb_v2_ref(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
    squeeze_dims: bool,
) -> torch.Tensor:
    if squeeze_dims:
        """
        Based on apply_rotary_pos_emb() from the LLaMA model in Transformer.
        The first two dimensions of cos and sin are always 1, so we can squeeze them.
        """
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    else:
        """
        Based on apply_rotary_pos_emb() from the GPT-NeoX model in Transformer version greater than 4.27.4
        """
        gather_indices = position_ids[:, None, :, None]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)

    return (p * cos) + (rotate_half(p) * sin)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)

    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb_gptj_ref(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Based on apply_rotary_pos_emb() from the GPTJAttention class in GPT-J model.
    Note that the original version has sin and cos swapped with each other.
    """
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


def apply_rotary_pos_emb_chatglm_ref(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Based on apply_rotary_pos_emb() from ChatGLM model.
    """
    # x: [sq, b, np, hn]
    sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)

    return torch.cat((x_out2, x_pass), dim=-1)


def prepare_test_data(p_size, cos_sin_size, offset, mode):
    if mode == RotaryPosEmbeddingMode.BLOCKWISE:
        p = torch.rand(p_size, requires_grad=True)

        cos_sin_size = cos_sin_size[:-1] + (cos_sin_size[-1] // 2,)
        cos = torch.rand(cos_sin_size, dtype=torch.float32) * 2 - 1
        sin = torch.rand(cos_sin_size, dtype=torch.float32) * 2 - 1

        if offset == 0:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            off_size = (p_size[0],)
            for i in range(len(p_size) - 2):
                off_size = off_size + (1,)
            off_size = off_size + (offset,)

            off = torch.rand(off_size, dtype=torch.float32)
            cos = torch.cat((off, cos, cos), dim=-1)
            sin = torch.cat((off, sin, sin), dim=-1)

        position_ids = torch.randint(0, p_size[2], (p_size[0], p_size[2])).to(torch.long)

        return p, cos, sin, position_ids
    else:
        p = torch.rand(p_size)
        cos = torch.rand(cos_sin_size)
        sin = torch.rand(cos_sin_size)

        output_size = 2 * sin.shape[2]
        sin = torch.repeat_interleave(sin, 2, dim=2, output_size=output_size).unsqueeze(2)
        cos = torch.repeat_interleave(cos, 2, dim=2, output_size=output_size).unsqueeze(2)

    return p, cos, sin


@pytest.mark.parametrize(
    "p_size, cos_sin_size, offset",
    apply_rotary_pos_emb_v1_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_v1_fwd_bwd(p_size, cos_sin_size, offset, dtype):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("Half is not supported on Gaudi.")

    torch.manual_seed(12345)

    p, cos, sin, _ = prepare_test_data(p_size, cos_sin_size, offset, RotaryPosEmbeddingMode.BLOCKWISE)

    # Compute reference gradients on CPU using autograd
    p_embed_ref = apply_rotary_pos_emb_v1_ref(p, cos, sin, offset)
    loss_ref = p_embed_ref.sum()
    loss_ref.backward()

    grad_p_ref = p.grad.clone().detach()

    # Compute gradients on HPU
    p_hpu = p.clone().to(dtype).to(hpu)
    p_hpu.retain_grad()
    cos_hpu = cos.to(dtype).to(hpu)
    sin_hpu = sin.to(dtype).to(hpu)

    output_fwd = RotaryPosEmbeddingHelperV1.apply
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        output_fwd = torch.compile(RotaryPosEmbeddingHelperV1.apply, backend="hpu_backend")

    p_embed = output_fwd(p_hpu, cos_hpu, sin_hpu, offset)
    loss = p_embed.sum()
    loss.backward()

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol)

    torch.testing.assert_close(p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"rotary_pos_embedding", "rotary_pos_embedding_backward"})


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_v2_test_case_list,
)
@pytest.mark.parametrize("squeeze_dims", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
class TestHpuApplyRotaryPosEmbV2FwdBwd:
    @classmethod
    def setup_class(self):
        # Index requires fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_apply_rotary_pos_emb_v2_fwd_bwd(p_size, cos_sin_size, squeeze_dims, dtype):
        if is_gaudi1() and dtype == torch.float16:
            pytest.skip("Half is not supported on Gaudi.")

        torch.manual_seed(12345)

        # Initial shapes for p, cos/sin, position_ids
        # query_shape=[bs, num_attention_heads, seq_len, rotary_ndim]
        # cos_shape=[1, 1, max_position_embeddings, rotary_ndim]
        # position_ids_shape=[bs, seq_len]
        p, cos, sin, position_ids = prepare_test_data(p_size, cos_sin_size, 0, RotaryPosEmbeddingMode.BLOCKWISE)

        # Compute reference gradients on CPU using autograd
        p_embed_ref = apply_rotary_pos_emb_v2_ref(p, cos, sin, position_ids, squeeze_dims)
        loss_ref = p_embed_ref.sum()
        loss_ref.backward()

        grad_p_ref = p.grad.clone().detach()

        # Compute gradients on HPU
        p_hpu = p.clone().to(dtype).to(hpu)
        p_hpu.retain_grad()
        cos_hpu = cos.to(dtype).to(hpu)
        sin_hpu = sin.to(dtype).to(hpu)
        position_ids_hpu = position_ids.to(hpu)

        output_fwd = RotaryPosEmbeddingHelperV2.apply
        if is_pytest_mode_compile():
            clear_t_compile_logs()
            torch._dynamo.reset()
            output_fwd = torch.compile(RotaryPosEmbeddingHelperV2.apply, backend="hpu_backend")

        p_embed = output_fwd(p_hpu, cos_hpu, sin_hpu, position_ids_hpu)
        loss = p_embed.sum()
        loss.backward()

        if dtype == torch.float32:
            tol = 0.001
        else:
            tol = 0.012

        torch.testing.assert_close(p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol)

        torch.testing.assert_close(p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol)

        if is_pytest_mode_compile():
            check_ops_executed_in_jit_ir(
                {"rotary_pos_embedding", "rotary_pos_embedding_backward"}, {"index", "index_1"}
            )


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_gptj_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_gptj_fwd(p_size, cos_sin_size, dtype):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("Half is not supported on Gaudi.")

    torch.manual_seed(12345)

    p, cos, sin = prepare_test_data(p_size, cos_sin_size, 0, RotaryPosEmbeddingMode.PAIRWISE)

    # Compute reference output values
    output_ref = apply_rotary_pos_emb_gptj_ref(p, cos, sin)

    # Compute output values on HPU
    p_hpu = p.to(dtype).to(hpu)
    cos_hpu = cos.to(dtype).to(hpu)
    sin_hpu = sin.to(dtype).to(hpu)

    output_fwd = apply_rotary_pos_emb
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        output_fwd = torch.compile(apply_rotary_pos_emb, backend="hpu_backend")

    output_hpu = output_fwd(p_hpu, cos_hpu, sin_hpu, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(output_hpu.to(torch.float32).to(cpu), output_ref, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("rotary_pos_embedding")


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_diff_dtypes_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cos_dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sin_dtype", [torch.float16, torch.float32, torch.bfloat16])
class TestHpuApplyRotaryPosEmbDiffDTypes:
    @classmethod
    def setup_class(self):
        # Index requires fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_apply_rotary_pos_emb_diff_dtypes(p_size, cos_sin_size, dtype, cos_dtype, sin_dtype):
        if is_gaudi1() and (dtype == torch.float16 or cos_dtype == torch.float16 or sin_dtype == torch.float16):
            pytest.skip("Half is not supported on Gaudi.")

        torch.manual_seed(12345)

        p, cos, sin, position_ids = prepare_test_data(p_size, cos_sin_size, 0, RotaryPosEmbeddingMode.BLOCKWISE)

        # Compute reference gradients on CPU using autograd
        p_embed_ref = apply_rotary_pos_emb_v2_ref(p, cos, sin, position_ids, False)
        loss_ref = p_embed_ref.sum()
        loss_ref.backward()

        grad_p_ref = p.grad.clone().detach()

        # Compute gradients on HPU
        p_hpu = p.clone().to(dtype).to(hpu)
        p_hpu.retain_grad()
        cos_hpu = cos.to(cos_dtype).to(hpu)
        sin_hpu = sin.to(sin_dtype).to(hpu)
        position_ids_hpu = position_ids.to(hpu)

        output_fwd = RotaryPosEmbeddingHelperV2.apply
        if is_pytest_mode_compile():
            clear_t_compile_logs()
            torch._dynamo.reset()
            output_fwd = torch.compile(RotaryPosEmbeddingHelperV2.apply, backend="hpu_backend")

        p_embed = output_fwd(p_hpu, cos_hpu, sin_hpu, position_ids_hpu)
        loss = p_embed.sum()
        loss.backward()

        if dtype == torch.float32:
            tol = 0.002
        else:
            tol = 0.012

        torch.testing.assert_close(p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol)

        torch.testing.assert_close(p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol)

        if is_pytest_mode_compile():
            check_ops_executed_in_jit_ir(
                {"rotary_pos_embedding", "rotary_pos_embedding_backward"}, {"index", "index_1"}
            )


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_chatglm_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_chatglm_fwd(p_size, cos_sin_size, dtype):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("Half is not supported on Gaudi.")

    torch.manual_seed(12345)

    # Prepare input data
    p = torch.rand(p_size, dtype=torch.float32)
    cos = torch.rand(cos_sin_size, dtype=torch.float32)
    sin = torch.rand(cos_sin_size, dtype=torch.float32)
    rope_cache = torch.stack((cos, sin), dim=-1)

    # Compute reference output values
    output_ref = apply_rotary_pos_emb_chatglm_ref(p, rope_cache)

    # Compute output values on HPU
    p_hpu = p.to(dtype).to(hpu)
    rope_cache_hpu = rope_cache.to(hpu)

    output_fwd = apply_rotary_pos_emb
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        output_fwd = torch.compile(apply_rotary_pos_emb, backend="hpu_backend")

    output_hpu = output_fwd(p_hpu, rope_cache_hpu)

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(output_hpu.to(torch.float32).to(cpu), output_ref, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("rotary_pos_embedding")


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_chatglm_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_chatglm_fwd_bwd(p_size, cos_sin_size, dtype):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("Half is not supported on Gaudi.")

    torch.manual_seed(12345)

    # Prepare input data
    p = torch.rand(p_size, dtype=torch.float32, requires_grad=True)
    cos = torch.rand(cos_sin_size, dtype=torch.float32)
    sin = torch.rand(cos_sin_size, dtype=torch.float32)
    rope_cache = torch.stack((cos, sin), dim=-1)

    # Compute reference gradients on CPU using autograd
    p_embed_ref = apply_rotary_pos_emb_chatglm_ref(p, rope_cache)
    loss_ref = p_embed_ref.sum()
    loss_ref.backward()

    grad_p_ref = p.grad.clone().detach()

    # Compute gradients on HPU
    p_hpu = p.clone().to(dtype).to(hpu)
    p_hpu.retain_grad()
    rope_cache_hpu = rope_cache.to(dtype).to(hpu)

    output_fwd = RotaryPosEmbeddingHelperV3.apply
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        output_fwd = torch.compile(RotaryPosEmbeddingHelperV3.apply, backend="hpu_backend")

    p_embed = output_fwd(p_hpu, rope_cache_hpu)
    loss = p_embed.sum()
    loss.backward()

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol)

    torch.testing.assert_close(p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"rotary_pos_embedding", "rotary_pos_embedding_backward"})
