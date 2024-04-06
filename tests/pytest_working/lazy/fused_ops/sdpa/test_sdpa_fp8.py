import math  # for ceil etc
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
from test_utils import compare_tensors, is_gaudi3

DBG_FLAG_verbose_print = False
print_max_diff = False


from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

# LNEG = float('-inf')
LNEG = -1e9
attention_scale = None


def vb_print(*args, **kwargs):
    if DBG_FLAG_verbose_print:
        print(*args, **kwargs)


def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


def is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o):
    if inference:
        if is_amax_s == False:
            return True

        if is_amax_o == True:
            return False
        return False
    else:
        return True


class TestModel(torch.nn.Module):
    def __init__(
        self,
        d_scale_q=None,
        d_scale_k=None,
        d_scale_v=None,
        q_scale_s=None,
        q_scale_o=None,
        d_scale_s=None,
        is_amax_s=False,
        is_amax_o=False,
        inference=False,
    ):
        super(TestModel, self).__init__()
        if inference:  # make scales/descales nn Parameter so that they can be made constants later
            self.d_scale_q = torch.nn.Parameter(d_scale_q) if d_scale_q is not None else None
            self.d_scale_k = torch.nn.Parameter(d_scale_k) if d_scale_k is not None else None
            self.d_scale_v = torch.nn.Parameter(d_scale_v) if d_scale_v is not None else None
            self.q_scale_s = torch.nn.Parameter(q_scale_s) if q_scale_s is not None else None
            self.q_scale_o = torch.nn.Parameter(q_scale_o) if q_scale_o is not None else None
            self.d_scale_s = torch.nn.Parameter(d_scale_s) if d_scale_s is not None else None
        else:  # Training case; No need to make scales/descales nn Parameter
            self.d_scale_q = d_scale_q
            self.d_scale_k = d_scale_k
            self.d_scale_v = d_scale_v
            self.q_scale_s = q_scale_s
            self.q_scale_o = q_scale_o
            self.d_scale_s = d_scale_s
        self.is_amax_s = is_amax_s
        self.is_amax_o = is_amax_o

    def forward(self, q_hpu, k_hpu, v_hpu, attn_mask=None, dropout_p=0.0, is_causal=False, softmax_mode="None"):

        O_hpu, amax_s, amax_o = fp8_fused_sdpa(
            q_hpu,
            k_hpu,
            v_hpu,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            softmax_mode=softmax_mode,
            scale=attention_scale,
            d_scale_q=self.d_scale_q,
            d_scale_k=self.d_scale_k,
            d_scale_v=self.d_scale_v,
            q_scale_s=self.q_scale_s,
            q_scale_o=self.q_scale_o,
            d_scale_s=self.d_scale_s,
            is_amax_s=self.is_amax_s,
            is_amax_o=self.is_amax_o,
        )
        return O_hpu, amax_s, amax_o


# ****************************************************************************************
# ***********************************Test code Follows************************************
# ****************************************************************************************


# reference code from : pytorch/test/test_transformers.py and modified
def create_attention_mask_for_test(batch_size, q_heads, seq_len_N_t, seq_len_N_s, dtype, shape, float_mask=True):
    attn_mask = torch.randint(0, 2, (seq_len_N_s,)).float()
    if float_mask:
        attn_mask = attn_mask.masked_fill(attn_mask == 0, LNEG).masked_fill(attn_mask == 1, float(0.0))
    attn_mask = attn_mask.to(dtype)

    if shape == "Bx1x1xN":
        if q_heads == 0:
            mask_shape = (batch_size, 1, seq_len_N_s)
        else:
            mask_shape = (batch_size, 1, 1, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    else:
        if q_heads == 0:
            mask_shape = (batch_size, seq_len_N_t, seq_len_N_s)
        else:
            mask_shape = (batch_size, q_heads, seq_len_N_t, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    return attn_mask


def vanilla_attention_impl_for_test(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, is_amax_s=False
):

    sqrt_dim_head = query.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale == None:
        scores = scores / sqrt_dim_head
    else:
        scores = scores * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(attn_mask == False, -float("inf"))
        else:
            scores = scores + attn_mask
    elif is_causal:
        seq_len_N_t = query.shape[-2]
        seq_len_N_s = key.shape[-2]
        attn_mask = torch.ones(seq_len_N_t, seq_len_N_s, dtype=torch.bool).tril(diagonal=0)
        scores.masked_fill_(attn_mask == False, LNEG)

    weight = F.softmax(scores, dim=-1)
    fwd_out = torch.matmul(weight, value)

    if is_amax_s:
        return fwd_out, torch.max(torch.abs(weight)).to(torch.float32)
    else:
        return fwd_out, None


def gaudi_llama_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    vb_print("IS GQA? : ", gqa)
    return gqa


def is_mqa(q, k):
    mqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        mqa = (q_heads != kv_heads) and kv_heads == 1
    vb_print("IS MQA? : ", mqa)
    return mqa


def is_param_combo_valid(
    batch_size,
    q_heads,
    kv_heads,
    seq_len_N_t,
    seq_len_N_s,
    head_dim_qk,
    head_dim_v,
    dropout_p,
    use_attn_mask,
    use_float_mask,
    enable_autocast,
    is_causal,
    recompute,
    rhslice,
    inference,
    softmax_mode,
    is_amax_s,
    is_amax_o,
    fp8_run_out_type,
):

    # if inference:
    #    return False

    if not inference:  # limiting tests Temporarily for training
        if q_heads != kv_heads:
            return False  # no MQA/GQA test for now

    # fp8 mode supports only inference in Triangular and Non-Triangular mask mode
    # But training is supported only in Triangular mask mode as of now.
    if not inference:
        if is_causal == False:
            return False

    # fp8 mode supports only recompute mode as of now.
    if not recompute:
        return False

    if is_causal:
        if use_attn_mask:
            return False

    if dropout_p != 0.0:
        return False

    # Fp8 measurement or  supported only if tensors are bf16 before convert to fp8
    if enable_autocast == False:
        return False

    is_amax = is_amax_s or is_amax_o

    fp8_run = is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o)
    # if fp8_run: return False

    if inference:
        # inference does not have amax_o
        if is_amax_o:
            return False

        if is_amax_s:
            if fp8_run == True:
                return False

        if fp8_run:
            if is_amax_s == True:
                return False
            # fp8 run in inference has fast softmax internally.
            # So do not set from test.
            # TODO: See if we should accept this config and ignore.
            if softmax_mode != "None":
                return False
    else:
        # TODO: modify this later
        # Currently support only fast softmax in training measurement/run
        if softmax_mode != "fast":
            return False

    return True


tc_list3 = [
    (  # train run
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "fast",  # "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        "bf16",  # "fp8_143", #"bf16",  # fp8_run_out_type
    ),
]
tc_list4 = [
    (  # Train Meas
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        "bf16",  # "fp8_143", #"bf16",  # fp8_run_out_type
    ),
]

tc_list = tc_list3 + tc_list4

# DONOT remove next line:Disable black formatting for easier parameter update
# fmt: off

"""
@pytest.mark.parametrize(
    "batch_size",
    (
        3,
    ),
    ids=lambda batch_size: f"batch_size-{batch_size}"
    )
@pytest.mark.parametrize(
    "q_heads",
    (
        4,
    ),
    ids=lambda q_heads: f"q_heads-{q_heads}"
)
@pytest.mark.parametrize(
    "kv_heads",
    (
        4, # same kv heads as q
        1, # MQA
        2, # GQA
    ),
    ids=lambda kv_heads: f"kv_heads-{kv_heads}"
)
@pytest.mark.parametrize(
    "seq_len_N_t",
    (
        16,
    ),
    ids=lambda seq_len_N_t: f"seq_len_N_t-{seq_len_N_t}"
    )
@pytest.mark.parametrize(
    "seq_len_N_s",
    (
        32,
    ),
    ids=lambda seq_len_N_s: f"seq_len_N_s-{seq_len_N_s}"
    )
    

@pytest.mark.parametrize(
    "head_dim_qk",
    (
        8,
    ),
    ids=lambda head_dim_qk: f"head_dim_qk-{head_dim_qk}"
    )
@pytest.mark.parametrize(
    "head_dim_v",
    (
        8,
    ),
    ids=lambda head_dim_v: f"head_dim_v-{head_dim_v}"
    )
        
@pytest.mark.parametrize(
    "dropout_p",
    (
        0.0,
    ),
    ids=lambda dropout_p: f"dropout_p-{dropout_p}"
    )
        

@pytest.mark.parametrize(
    "use_attn_mask",
    (
        True,
        False,
    ),
    ids=lambda use_attn_mask: f"use_attn_mask-{use_attn_mask}"
    )
@pytest.mark.parametrize(
    "use_float_mask",
    (
        True,
        #False,  # enable for detailed test
    ),
    ids=lambda use_float_mask: f"use_float_mask-{use_float_mask}"
    )
@pytest.mark.parametrize(
    "enable_autocast",
    (
        True,
        #False, # not applicable for fp8 measurement or run
    ),
    ids=lambda enable_autocast: f"enable_autocast-{enable_autocast}"
    )
    
@pytest.mark.parametrize(
    "is_causal",
    (
        True,
        False,
    ),
    ids=lambda is_causal: f"is_causal-{is_causal}"
    )

@pytest.mark.parametrize(
    "recompute",
    (
        True,
        #False # not  supported for fp8 as of now
    ),
    ids=lambda recompute: f"recompute-{recompute}"
    )
@pytest.mark.parametrize(
    "rhslice",
    (
        True,
        False,
    ),
    ids=lambda rhslice: f"rhslice-{rhslice}"
    )

@pytest.mark.parametrize(
    "inference",
    (
        True,
        False,
    ),
    ids=lambda inference: f"inference-{inference}"
    )
@pytest.mark.parametrize(
    "softmax_mode",
    (
        "None",
        "fast",
    ),
    ids=lambda softmax_mode: f"softmax_mode-{softmax_mode}"
    )
@pytest.mark.parametrize(
    "is_amax_s",
    (
        True,
        False,
    ),
    ids=lambda is_amax_s: f"is_amax_s-{is_amax_s}"
)
@pytest.mark.parametrize(
    "is_amax_o",
    (
        True,
        False,
    ),
    ids=lambda is_amax_o: f"is_amax_o-{is_amax_o}"
)
@pytest.mark.parametrize(
    "fp8_run_out_type",
    (
        "fp8_143",
        "bf16",
        #"None", # Not an fp8 run; can be a run for amax measurement
    ),
    ids=lambda fp8_run_out_type: f"fp8_run_out_type-{fp8_run_out_type}"
)
"""
# DONOT remove following line: re-enable black formatting
# fmt: on


@pytest.mark.parametrize(
    "batch_size,q_heads,kv_heads,seq_len_N_t,seq_len_N_s,head_dim_qk,head_dim_v,dropout_p,use_attn_mask,use_float_mask,enable_autocast,is_causal,recompute,rhslice,inference,softmax_mode,is_amax_s,is_amax_o,fp8_run_out_type",
    tc_list,
)

# @pytest.mark.xfail(reason="Temporarily disabled")
def test_sdpa(
    batch_size,
    q_heads,
    kv_heads,
    seq_len_N_t,
    seq_len_N_s,
    head_dim_qk,
    head_dim_v,
    dropout_p,
    use_attn_mask,
    use_float_mask,
    enable_autocast,
    is_causal,
    recompute,
    rhslice,
    inference,
    softmax_mode,
    is_amax_s,
    is_amax_o,
    fp8_run_out_type,
):
    test_case_valid = is_param_combo_valid(
        batch_size,
        q_heads,
        kv_heads,
        seq_len_N_t,
        seq_len_N_s,
        head_dim_qk,
        head_dim_v,
        dropout_p,
        use_attn_mask,
        use_float_mask,
        enable_autocast,
        is_causal,
        recompute,
        rhslice,
        inference,
        softmax_mode,
        is_amax_s,
        is_amax_o,
        fp8_run_out_type,
    )
    if not is_gaudi3():
        pytest.skip("Fp8 training tests not currently supported on G1 or G2")

    # print("test_case_valid = ", test_case_valid)
    if not test_case_valid:
        pytest.skip("This testcase is not valid for fp8 measurement or run")

    if inference:
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "1"
        htcore.hpu_set_inference_env()

    torch.manual_seed(1234567)

    dtype = torch.float32
    rtol = 1e-3
    atol = 1e-3

    if enable_autocast:
        dtype = torch.bfloat16
        grad_dtype = torch.bfloat16
        rtol = 1e-3
        atol = 0.08

    fp8_run = is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o)

    if is_amax_s and inference:
        assert fp8_run == False, "Fp8 measurement and run can not be True at the same time in inference"

    if fp8_run:
        rtol = 1e-3
        atol = 0.3
        amax_o_atol = 2.0

    attn_mask_shape = "Bx1x1xN"
    if use_float_mask:
        mask_dtype = dtype
    else:
        mask_dtype = torch.bool

    attn_scale = attention_scale
    vb_print("\nbatch_size = ", batch_size)
    vb_print("num_q_heads = ", q_heads)
    vb_print("num_kv_heads = ", kv_heads)
    vb_print("seq_len_N_s = ", seq_len_N_s)
    vb_print("head dim q k = ", head_dim_qk)
    vb_print("head dim v = ", head_dim_v)

    vb_print("dropout probability = ", dropout_p)
    vb_print("Using float attention mask = ", use_float_mask)

    vb_print("softmax mode = ", softmax_mode)
    vb_print("is_amax_s = ", is_amax_s)

    if q_heads == 0:  # special meaning ; no multi head attn . i.e, use 3d tensors
        q_shape = (batch_size, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, seq_len_N_t, head_dim_v)
    else:  # Multi head attn with q_heads
        q_shape = (batch_size, q_heads, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, q_heads, seq_len_N_t, head_dim_v)

    vb_print("q shape = ", q_shape)
    vb_print("k shape = ", k_shape)
    vb_print("v shape = ", v_shape)

    q = torch.randn(q_shape).to(dtype).detach()

    # kk = torch.eye(seq_len_N_s)*0.5
    # k = kk.expand_as(q).to(dtype).detach()

    # vv = torch.eye(seq_len_N_s)
    # v = vv.expand_as(q).to(dtype).detach()
    k = torch.randn(k_shape).to(dtype).detach()
    v = torch.randn(v_shape).to(dtype).detach()

    scaleQInv_hpu = scaleKInv_hpu = scaleVInv_hpu = scaleSInv_hpu = q_scale_s = q_scale_o = None

    q_t = q.clone().detach()
    k_t = k.clone().detach()
    v_t = v.clone().detach()

    if not inference:
        q_t = q_t.requires_grad_()
        k_t = k_t.requires_grad_()
        v_t = v_t.requires_grad_()

    q_hpu = q.to("hpu").detach()
    k_hpu = k.to("hpu").detach()
    v_hpu = v.to("hpu").detach()

    if not inference:
        q_hpu = q_hpu.requires_grad_()
        k_hpu = k_hpu.requires_grad_()
        v_hpu = v_hpu.requires_grad_()

    if use_attn_mask:
        attn_mask = create_attention_mask_for_test(
            batch_size, q_heads, seq_len_N_t, seq_len_N_s, mask_dtype, attn_mask_shape, float_mask=use_float_mask
        )
        attn_mask_hpu = attn_mask.to("hpu")
    else:
        attn_mask = None
        attn_mask_hpu = None

    if use_attn_mask:
        assert is_causal == False, " use_attn_mask and is_causal can not be True at the same time"

    # Set the env. var to enable batchsize/Num heads slicing if needed.
    if rhslice:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "1"
    else:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "0"

    # ------------------------------- Vanilla SDPA implementation on CPU for test----------------------------

    is_mqa(q_t, k_t)  # Just for info: For printing on console.

    if is_gqa(q_t, k_t):
        num_key_value_groups_ = q_heads // kv_heads
        k_t = gaudi_llama_repeat_kv(k_t, num_key_value_groups_)
        v_t = gaudi_llama_repeat_kv(v_t, num_key_value_groups_)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_ref, amax_s_ref = vanilla_attention_impl_for_test(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=attn_scale,
            is_causal=is_causal,
            is_amax_s=True,
        )

    vb_print("amax_s_ref = ", amax_s_ref)
    amax_o_ref = torch.max(O_ref).to(torch.float32)
    vb_print("amax_o_ref = ", amax_o_ref)

    def get_scale_values(name, t, is_t_amax=False, scale_limit=None):

        FP8_MAX_143 = 240 * 0.9
        if is_t_amax == False:
            maxT = torch.max(torch.abs(t)).to(torch.float).item()
        else:
            maxT = t.item()
        scaleT = FP8_MAX_143 / maxT

        lg2 = math.log2(scaleT)
        lg2_int = int(lg2)
        scaleT_pow2 = 2.0**lg2_int

        scaleTInv = 1.0 / scaleT_pow2
        vb_print(name, ": scale", scaleT)
        vb_print(name, ": scale pow2", scaleT_pow2)
        vb_print(name, ": Inv scale", scaleTInv)

        if scale_limit != None:
            scaleT_pow2 = scale_limit
            scaleTInv = 1.0 / scaleT_pow2
            vb_print(name, ": after limiting : scale pow2", scaleT_pow2)
            vb_print(name, ": after limiting : Inv scale", scaleTInv)

        scaleT_cpu = torch.tensor(scaleT_pow2, dtype=torch.float)
        scaleTInv_cpu = torch.tensor(scaleTInv, dtype=torch.float)
        scaleT_hpu = scaleT_cpu.to("hpu")
        scaleTInv_hpu = scaleTInv_cpu.to("hpu")
        return scaleT_hpu, scaleTInv_hpu

    if fp8_run:
        vb_print("TESTING fp8")
        fp8_dtype = torch.float8_e4m3fn

        scaleQ_hpu, scaleQInv_hpu = get_scale_values("q", q)
        scaleK_hpu, scaleKInv_hpu = get_scale_values("k", k)
        scaleV_hpu, scaleVInv_hpu = get_scale_values("v", v)

        q_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(q_hpu, scaleQ_hpu, False, False, fp8_dtype)
        k_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(k_hpu, scaleK_hpu, False, False, fp8_dtype)
        v_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(v_hpu, scaleV_hpu, False, False, fp8_dtype)

        # scaleS_hpu = torch.tensor(1.0, dtype = torch.float32).to("hpu")
        scaleS_hpu, scaleSInv_hpu = get_scale_values("s", amax_s_ref, is_t_amax=True, scale_limit=128)
        q_scale_s = scaleS_hpu
        scaleSInv_hpu = scaleSInv_hpu
        if not inference:
            scaleSInv_hpu = None

        if fp8_run_out_type == "fp8_143":
            scaleO_hpu, _ = get_scale_values("o", O_ref)
            q_scale_o = scaleO_hpu

        # Let fp8 conversions and scale transfer to HPU be in a separate graph
        htcore.mark_step()

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------
    if recompute:
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
            # Use ht.sdp_kernel() context manager to enable/disable recompute based on pytest recompute parameter
            with ht.sdp_kernel(enable_recompute=recompute):
                model = TestModel(
                    d_scale_q=scaleQInv_hpu,
                    d_scale_k=scaleKInv_hpu,
                    d_scale_v=scaleVInv_hpu,
                    q_scale_s=q_scale_s,
                    q_scale_o=q_scale_o,
                    d_scale_s=scaleSInv_hpu,
                    is_amax_s=is_amax_s,
                    is_amax_o=is_amax_o,
                    inference=inference,
                )

                if inference:
                    # make scale tensors constant
                    _mark_params_as_const(model)
                    _check_params_as_const(model)

                O_hpu, amax_s, amax_o = model(
                    q_hpu,
                    k_hpu,
                    v_hpu,
                    attn_mask=attn_mask_hpu,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    softmax_mode=softmax_mode,
                )

    else:
        assert False, " F8 is supported only in recompute mode now"

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------

    htcore.mark_step()

    # ------------------------------- Test Results Comparison ----------------------------
    vb_print("\n")
    O_hpu_c = O_hpu.detach().to("cpu")
    vb_print("DPA output dtype from HPU = ", O_hpu_c.dtype)
    if fp8_run and fp8_run_out_type == "fp8_143":
        O_hpu_c = O_hpu_c.to(q_t.dtype) / q_scale_o.to("cpu").to(q_t.dtype)

    if is_amax_s:
        amax_s_hpu_c = amax_s.detach().to("cpu")
        vb_print("cpu amax_s = ", amax_s_ref)
        vb_print("hpu amax_s = ", amax_s_hpu_c)
        vb_print(
            "Vanilla SDPA amax_s Ref vs FSDPA amax_s match? = ",
            torch.allclose(amax_s_ref, amax_s_hpu_c, rtol=rtol, atol=atol),
        )
        if print_max_diff:
            vb_print(
                "Max diff Vanilla SDPA amax_s Ref vs FSDPA amax_s = ", torch.max(torch.abs(amax_s_ref - amax_s_hpu_c))
            )
        compare_tensors(amax_s_ref, amax_s_hpu_c, atol=atol, rtol=rtol)

    if is_amax_o and inference == False:
        amax_o_hpu_c = amax_o.detach().to("cpu")
        vb_print("cpu amax_o = ", amax_o_ref)
        vb_print("hpu amax_o = ", amax_o_hpu_c)
        vb_print(
            "Vanilla SDPA amax_o Ref vs FSDPA amax_o match? = ",
            torch.allclose(amax_o_ref, amax_o_hpu_c, rtol=rtol, atol=amax_o_atol),
        )
        if print_max_diff:
            vb_print(
                "Max diff Vanilla SDPA amax_o Ref vs FSDPA amax_o = ", torch.max(torch.abs(amax_o_ref - amax_o_hpu_c))
            )
        compare_tensors(amax_o_ref, amax_o_hpu_c, atol=amax_o_atol, rtol=rtol)

    vb_print("Vanilla SDPA FWD Ref vs FSDPA match? = ", torch.allclose(O_ref, O_hpu_c, rtol=rtol, atol=atol))
    vb_print("\n")
    if print_max_diff:
        vb_print("Max diff Vanilla SDPA FWD Ref vs FSDPA = ", torch.max(torch.abs(O_ref - O_hpu_c)))

    compare_tensors(O_ref, O_hpu_c, atol=atol, rtol=rtol)

    if inference:
        htcore.hpu_teardown_inference_env()
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "0"
