import math
import time

import habana_frameworks.torch as ht
import pytest
import torch
import torch.nn.functional as F
from torch import nn

hpu = torch.device("hpu")
lazy_mode = True
profile_mode = False


class BertSelfAttention(nn.Module):
    __constants__ = [
        "hidden_size",
        "num_attention_heads",
        "attention_probs_dropout_prob",
        "attention_head_size",
        "all_head_size",
    ]

    def __init__(self):
        super(BertSelfAttention, self).__init__()
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.attention_probs_dropout_prob = 0.1
        self.attention_head_size = int(1024 / 16)  # 64
        self.all_head_size = 16 * int(1024 / 16)  # 1024

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = context_layer.flatten(start_dim=2, end_dim=3)
        return context_layer


class BertSelfOutput(nn.Module):
    __constants__ = ["hidden_size", "num_attention_heads", "hidden_dropout_prob"]

    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.hidden_dropout_prob = 0.1
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class ModelUnderTest(nn.Module):
    def __init__(self) -> None:
        super(ModelUnderTest, self).__init__()
        self.g = ht.hpu.HPUGraph()
        self.s = ht.hpu.Stream()
        self.model = BertAttention()
        self.model = self.model.to(hpu)

    def warmup(self, hid, attn, lazy_mode=True):
        with ht.hpu.stream(self.s):
            self.g.capture_begin()
            self.model(hid, attn)
            self.g.capture_end()

    def train_loop(self, hid, attn, lazy_mode=True):
        self.g.replay()


@pytest.mark.skip(reason="There is no assert in this test")
def test_multi_model():
    batch_size, seq_len, hidden_dim, attn_dim = 64, 128, 1024, 1

    hid_nc1 = torch.rand((batch_size, seq_len, hidden_dim), dtype=torch.float)
    attn_nc1 = torch.rand((batch_size, attn_dim, attn_dim, seq_len), dtype=torch.float)
    hid_nc1_hpu = hid_nc1.to(hpu)
    attn_nc1_hpu = attn_nc1.to(hpu)

    hid_nc2 = torch.rand((batch_size, seq_len, hidden_dim), dtype=torch.float)
    attn_nc2 = torch.rand((batch_size, attn_dim, attn_dim, seq_len), dtype=torch.float)
    hid_nc2_hpu = hid_nc2.to(hpu)
    attn_nc2_hpu = attn_nc2.to(hpu)

    m_model1 = ModelUnderTest()
    m_model1.warmup(hid=hid_nc1_hpu, attn=attn_nc1_hpu)

    m_model2 = ModelUnderTest()
    m_model2.warmup(hid=hid_nc2_hpu, attn=attn_nc2_hpu)

    num_w_batches = 500
    step = 0
    start_time = time.time()

    for _ in range(num_w_batches):
        m_model1.train_loop(hid=hid_nc1_hpu, attn=attn_nc1_hpu)
        m_model2.train_loop(hid=hid_nc2_hpu, attn=attn_nc2_hpu)
        step = step + 1

    end_time = time.time()
    total_time = end_time - start_time
    print("Total batches '{}'".format(step))
    print("Total time '{}'".format(total_time))
    print("Img/s '{}'".format(num_w_batches * batch_size / total_time))
