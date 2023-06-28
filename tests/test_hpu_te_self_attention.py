import math
import torch
import pytest
from test_utils import is_gaudi1
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
from habana_frameworks.torch.hpex.experimental.transformer_engine import fp8_autocast
from habana_frameworks.torch.hpex.experimental.transformer_engine import SelfAttentionScoresAndValue, SelfAttentionContext


class BertSelfAttention(torch.nn.Module):
    # Reference implementation from Bert script
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if config.use_hpu_transformer_engine:
            # TE matmuls are stateful modules (they keep historical inputs' statistics),
            # so we need to create separate variable for every matmul operation
            self.attention_scores_matmul = te.MatMul()
            self.context_layer_matmul = te.MatMul()
            linear = te.Linear
        else:
            self.attention_scores_matmul = torch.matmul
            self.context_layer_matmul = torch.matmul
            linear = torch.nn.Linear

        self.query = linear(config.hidden_size, self.all_head_size)
        self.key = linear(config.hidden_size, self.all_head_size)
        self.value = linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        import torch.nn.functional as F
        import math
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.attention_scores_matmul(query_layer, key_layer)
        attention_scores = torch.mul(attention_scores, 1 / math.sqrt(self.attention_head_size))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.context_layer_matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


class TESelfAttention(torch.nn.Module):
    # TE implementation
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.phase1 = SelfAttentionScoresAndValue(config.hidden_size, config.num_attention_heads)
        self.phase2 = SelfAttentionContext(config.hidden_size, config.num_attention_heads)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_scores, mixed_value_layer = self.phase1(hidden_states)

        attention_scores = torch.mul(attention_scores, 1 / math.sqrt(self.attention_head_size))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.phase2(attention_probs, mixed_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor, return_angle=True) -> float:
    cs_ = tensor1.flatten() @ tensor2.flatten() / \
        (torch.norm(tensor1) * torch.norm(tensor2))
    cs_ = torch.clamp(cs_, -1.0, 1.0)
    if return_angle:
        return (torch.arccos(cs_) / torch.pi * 180).item()
    else:
        return cs_.item()


def prepare_fp8_recipe():
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe
    fp8_format = recipe.Format.E5M2 # No stochastic rounding on bwd - for deterministic tests
    fp8_margin = 0
    fp8_interval = 1
    return recipe.DelayedScaling(
        margin=fp8_margin,
        interval=fp8_interval,
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        reduce_amax=False,
    )


def self_attention(input, attention_mask, config, device, self_attention_class=TESelfAttention, self_attention_obj=None):
    fp8_recipe = prepare_fp8_recipe()
    extended_attention_mask = (1.0 - attention_mask) * -10000.0

    if self_attention_obj is not None:
        self_attention = self_attention_obj
    else:
        self_attention = self_attention_class(config).to(device)

    if config.use_hpu_transformer_engine:
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = self_attention(input, extended_attention_mask).cpu()
    else:
        out = self_attention(input, extended_attention_mask).cpu()

    loss = out.sum()
    loss.backward()
    grad = input.grad.cpu()

    return out, grad, self_attention


def self_attention_n_times(inputs, config, device, self_attention_class=TESelfAttention):
    batch = inputs[0].shape[0]
    qkv_dim = inputs[0].shape[1]
    te_attention_mask = torch.ones(
        (batch, 1, qkv_dim, qkv_dim), device=device)
    te_attention_mask[0][0][0][0] = 0

    self_attention_obj = None
    outputs = []
    grads = []

    for input in inputs:
        if self_attention_obj is None:
            out, grad, self_attention_obj = self_attention(
                input, te_attention_mask, config, device, self_attention_class=self_attention_class)
        else:
            out, grad, self_attention_obj = self_attention(
                input, te_attention_mask, config, device, self_attention_obj=self_attention_obj)
        outputs.append(out)
        grads.append(grad)

    return outputs, grads, self_attention_obj


def setup_test():
    class Config(object):
        pass

    config = Config()
    config.hidden_size = 64
    config.num_attention_heads = 16
    config.use_hpu_transformer_engine = True
    config.attention_probs_dropout_prob = 0.1
    batch = 2
    qkv_dim = 32

    return config, batch, qkv_dim

@pytest.mark.xfail(reason="Results mismatch")
def test_bert_self_attention():
    device = torch.device("hpu:0")
    config, batch, qkv_dim = setup_test()

    # Reference implementation
    ref_input = torch.ones(
        (batch, qkv_dim, config.hidden_size), device=device, requires_grad=True, dtype=torch.bfloat16)
    ref_attention_mask = torch.ones(
        (batch, 1, qkv_dim, qkv_dim), device=device)
    ref_attention_mask[0][0][0][0] = 0

    out_ref, grad_ref, _ = self_attention(
        ref_input, ref_attention_mask, config, device, self_attention_class=BertSelfAttention)

    # TE implementation
    te_input = torch.ones((batch, qkv_dim, config.hidden_size),
                          device=device, requires_grad=True, dtype=torch.bfloat16)
    te_attention_mask = torch.ones(
        (batch, 1, qkv_dim, qkv_dim), device=device)
    te_attention_mask[0][0][0][0] = 0
    out_te, grad_te, _ = self_attention(
        te_input, te_attention_mask, config, device, self_attention_class=TESelfAttention)

    print(
        f"Max diff between ref and te outputs: {torch.max(torch.abs(out_ref - out_te))}")

    assert torch.equal(out_ref, out_te)
    print("Forward success")

    abs_diff = torch.abs(grad_ref - grad_te)
    max_diff_abs = torch.max(abs_diff)
    max_diff_rel = torch.max(
        abs_diff / torch.max(torch.abs(grad_ref), torch.abs(grad_te)))

    print(f"Max absolute diff between ref and te grad: {max_diff_abs}")
    print(f"Max relative diff between ref and te grad: {max_diff_rel}")
    cs = cosine_similarity(grad_ref, grad_te)
    print(f"Cosine similarity: {cs}")
    assert max_diff_abs == 0 or cs < 0.1
    print("Backward success")

@pytest.mark.xfail(reason="NaN of inf")
def test_self_attention_scales():
    device = torch.device("hpu:0")

    config, batch, qkv_dim = setup_test()

    # First run - small input
    te_input = torch.full((batch, qkv_dim, config.hidden_size), 1.,
                          device=device, requires_grad=True, dtype=torch.bfloat16)
    te_attention_mask = torch.ones(
        (batch, 1, qkv_dim, qkv_dim), device=device)
    te_attention_mask[0][0][0][0] = 0
    out_te1, _, self_attention_obj = self_attention(
        te_input, te_attention_mask, config, device, self_attention_class=TESelfAttention)

    # Second run - bigger input
    te_input = torch.full((batch, qkv_dim, config.hidden_size), 10.,
                          device=device, requires_grad=True, dtype=torch.bfloat16)
    te_attention_mask = torch.ones(
        (batch, 1, qkv_dim, qkv_dim), device=device)
    te_attention_mask[0][0][0][0] = 0
    out_te2, _, _ = self_attention(
        te_input, te_attention_mask, config, device, self_attention_obj=self_attention_obj)

    def has_infs_or_nans(x):
        return torch.logical_or(torch.any(torch.isinf(x)), torch.any(torch.isnan(x)))

    assert (has_infs_or_nans(out_te2))

@pytest.mark.xfail
@pytest.mark.skipif(is_gaudi1(), reason="fp8 is unsupported on G1")
def test_self_attention_scales_2():
    device = torch.device("hpu:0")

    config, batch, qkv_dim = setup_test()

    # Reference implementation
    ref_inputs = []
    ref_inputs.append(torch.full((batch, qkv_dim, config.hidden_size), 1.,
                                 device=device, requires_grad=True, dtype=torch.bfloat16))
    ref_inputs.append(torch.full((batch, qkv_dim, config.hidden_size), 0.4,
                                 device=device, requires_grad=True, dtype=torch.bfloat16))
    ref_outputs, ref_grads, ref_sa = self_attention_n_times(
        ref_inputs, config, device, self_attention_class=BertSelfAttention)

    # TE implementation
    te_inputs = []
    te_inputs.append(torch.full((batch, qkv_dim, config.hidden_size), 1.,
                                device=device, requires_grad=True, dtype=torch.bfloat16))
    te_inputs.append(torch.full((batch, qkv_dim, config.hidden_size), 0.4,
                                device=device, requires_grad=True, dtype=torch.bfloat16))
    te_outputs, te_grads, te_sa = self_attention_n_times(
        te_inputs, config, device, self_attention_class=TESelfAttention)

    # Verify forward
    for i in range(len(ref_outputs)):
        ref_out = ref_outputs[i]
        te_out = te_outputs[i]
        print(
            f"[{i}] Max diff between ref and te outputs: {torch.max(torch.abs(ref_out - te_out))}")
        assert torch.equal(ref_out, te_out)

    print("Forward success\n")

    # Verify backward
    for i in range(len(ref_grads)):
        ref_grad = ref_grads[i]
        te_grad = te_grads[i]

        abs_diff = torch.abs(ref_grad - te_grad)
        max_diff_abs = torch.max(abs_diff)
        max_diff_rel = torch.max(
            abs_diff / torch.max(torch.abs(ref_grad), torch.abs(te_grad)))

        print(f"[{i}] Max absolute diff between ref and te grad: {max_diff_abs}")
        print(f"[{i}] Max relative diff between ref and te grad: {max_diff_rel}")
        cs = cosine_similarity(ref_grad, te_grad)
        print(f"[{i}] Cosine similarity: {cs}")
        assert max_diff_abs == 0 or cs < 0.1
        print(f"Iteration {i} success\n")

    print("Backward success")