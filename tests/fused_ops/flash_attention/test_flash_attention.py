import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as rand_hpu
from habana_frameworks.torch.hpex.kernels import FlashAttnFunc
import math # for ceil etc
import os
import sys

#Large -ve value ; Using -inf can cause issues when softmax soft max is taken over a section that hasll -inf on the row
#this can happen since we operate on slices. So use a large -ve value other than -inf at leaset in the flash impl.

#LNEG = float('-inf')
LNEG = -1e9

def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1 :
        env_var_set = True
    return env_var_set

def create_dropout_mask(input, shape, p, generator=None):
    assert generator is None
    t = torch.rand(shape, dtype=input.dtype, layout=input.layout, device=input.device)
    mask = (t < p).to(dtype=torch.uint8)
    return mask


def dropout_with_mask(input, p, mask):
    dropout_scaling = 1.0 / (1.0 - p)
    #RTC: should type_as be doone ouside to avoid this being done within the loop in BWD
    # because this may have d2d copy. but then it can increase temp. mem if converted
    # upfront to float.
    res = mask.type_as(input) * input * dropout_scaling
    return res

# Debug Wrapper for dropout incase we want to call nn.functional.dropout
# instead of dropout with given mask
def dropout_wrapper(x, p, mask = None):

    if mask is not None:
       return dropout_with_mask(x, p, mask)
    else:
        return F.dropout(x, p=p)

#****************************************************************************************
#***********************************Test code Follows************************************
#****************************************************************************************

#RTC : creates 4d attn mask. If Q/K/V shape is different, need to create mask in that shape for detailed testing
# reference code from : pytorch/test/test_transformers.py and modified
def create_attention_mask_for_test(batch_size, n_heads, seq_len_N, dtype, shape, float_mask = True): #RTC: cross attention will need to use source and target seq lens
    attn_mask = torch.randint(0, 2, (seq_len_N,)).float()
    if float_mask :
        attn_mask = attn_mask.masked_fill(attn_mask == 0, LNEG).masked_fill(attn_mask == 1, float(0.0))
    attn_mask = attn_mask.to(dtype)
    if shape == 'Bx1x1xN':
        attn_mask_4d = attn_mask.expand(batch_size, 1, 1, seq_len_N)
    else:
        attn_mask_4d = attn_mask.expand(batch_size, n_heads, seq_len_N, seq_len_N)
    return attn_mask_4d

def vanilla_attention_impl_for_test(query, key, value, attn_mask = None, dropout_p=0.0, dbg_dropout_mask = None):

    sqrt_dim_head = query.shape[-1]**0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(attn_mask == False, -float('inf'))
        else:
            scores = scores + attn_mask

    weight = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        if dbg_dropout_mask is not None :
            weight = dropout_wrapper(weight, dropout_p, mask = dbg_dropout_mask)
        else:
            mask = None
            if not DBG_FLAG_use_func_drpout:
                mask = create_dropout_mask(weight, weight.shape, dropout_p)

            weight = dropout_wrapper(weight, dropout_p, mask = mask)
    return torch.matmul(weight, value)


def perf_cmp_flash_vs_vanilla_attn(g_hpu, q_hpu, k_hpu, v_hpu, attn_mask = None, dropout_p=0.0):
    is_perf_run = True
    if check_dbg_env_var('FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH'):
        print("Perf cmp run: Flash")
        os.environ['FLASH_ATTN_DBG_USE_DROPOUT_STUB'] = '0'
        O_hpu = FlashAttnFunc.apply(q_hpu,k_hpu,v_hpu, attn_mask_hpu, dropout_p)
    elif check_dbg_env_var('FLASH_ATTN_DBG_PERF_CMP_RUN_VANILLA'):
        print("Perf cmp run: Vanilla")
        O_hpu = vanilla_attention_impl_for_test(q_hpu,k_hpu,v_hpu, attn_mask_hpu, dropout_p = dropout_p)
    else:
        is_perf_run = False
        return is_perf_run

    #htcore.mark_step() # if FWD and BWD in two graphs
    #O = O_hpu.to("cpu")
    O_hpu.backward(g_hpu)

    htcore.mark_step()
    #O = O_hpu.to("cpu") No need to take out FWD pass o/p
    q_grad = q_hpu.grad.to("cpu")
    k_grad = k_hpu.grad.to("cpu")
    v_grad = v_hpu.grad.to("cpu")
    return is_perf_run



torch.manual_seed(1234567)

batch_size = 8
seq_len_N = 128
embed_dim = 768
n_heads = 12
"""
batch_size = 1
seq_len_N = 4
embed_dim = 2
#head_dim  = 64
n_heads = 1
"""
dropout_p = 0.4

#debug flags
DBG_FLAG_use_func_drpout = False

dtype = torch.float32
grad_dtype = torch.float32

use_float_mask = True
enable_autocast = False
attn_mask_shape = 'Bx1x1xN'
if use_float_mask:
    mask_dtype = dtype
else :
    mask_dtype = torch.bool

neg_inf = -float('inf') # for filling m

head_dim = embed_dim // n_heads

rtol = 1e-3
atol = 1e-3

print("batch_size = ", batch_size)
print("num_heads = ", n_heads)
print("seq_len_N = ", seq_len_N)
print("head dim = ", head_dim)

print("dropout probability = ", dropout_p)
print("Using float attention mask = ", use_float_mask)

#RTC :  Need to  test the 3D case
q_k_v_shape = (batch_size, n_heads, seq_len_N, head_dim)
print("q_k_v_shape = ", q_k_v_shape)
q = torch.randn(q_k_v_shape).to(dtype).detach().requires_grad_()
k = torch.randn(q_k_v_shape).to(dtype).detach().requires_grad_()
v = torch.randn(q_k_v_shape).to(dtype).detach().requires_grad_()
g = torch.ones(q_k_v_shape).to(grad_dtype)

q_t = q.clone().detach().requires_grad_()
k_t = k.clone().detach().requires_grad_()
v_t = v.clone().detach().requires_grad_()
g_t = g.clone()
q_hpu = q.to("hpu").detach().requires_grad_()
k_hpu = k.to("hpu").detach().requires_grad_()
v_hpu = v.to("hpu").detach().requires_grad_()
g_hpu = g.to("hpu")


attn_mask = create_attention_mask_for_test(batch_size, n_heads, seq_len_N, mask_dtype, attn_mask_shape, float_mask = use_float_mask)
attn_mask_hpu = attn_mask.to("hpu")

DBG_ONLY_dropout_mask_g = None

if dropout_p == 0.0:
    os.environ['FLASH_ATTN_DBG_USE_DROPOUT_STUB'] = '0'

# ------------------------------- if Perf Run, run it first and return-----------------------------
perf_run_count = 1
profile_step = -1
profile_api = False

if check_dbg_env_var('FLASH_ATTN_DBG_PERF_CMP_RUN_CYCLES'):
    perf_run_count = 6
    profile_step = 4

if profile_step != -1:
    try:
        sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
        from topologies.tools import SynapseProfilerApi, TraceType
    except ImportError:
        print("Failed to import profiling tools")
        profile_step = -1
        pass

if profile_step != -1:
    profile_api = SynapseProfilerApi()
    trace_type = TraceType.TraceDevice
    profile_dev_id = 0

for i in range(perf_run_count) :
    if profile_api and i == profile_step:
        profile_api.profiler_start(trace_type, profile_dev_id)
    perf_run = perf_cmp_flash_vs_vanilla_attn(g_hpu, q_hpu, k_hpu, v_hpu, attn_mask = attn_mask_hpu, dropout_p=dropout_p)
    if profile_api and i == profile_step:
        profile_api.profiler_sync(profile_dev_id)
        profile_api.profiler_stop(trace_type, profile_dev_id)
        profile_api.profiler_get_trace_json(trace_type, profile_dev_id)

if perf_run:
    exit(0)
# ----------------------------------HPU flash attention---------------------------------------------
if not check_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_hpu = FlashAttnFunc.apply(q_hpu,k_hpu,v_hpu, attn_mask_hpu, dropout_p)
else:
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_hpu, DBG_ONLY_dropout_mask_g = FlashAttnFunc.apply(q_hpu,k_hpu,v_hpu, attn_mask_hpu, dropout_p)
    DBG_ONLY_dropout_mask_g = DBG_ONLY_dropout_mask_g.to("cpu")

O_hpu.backward(g_hpu)

htcore.mark_step()

# ------------------------------- PT SDP implementation on CPU  for Test ----------------------------
if dropout_p == 0.0:
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
        sdp_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    sdp_ref.backward(g)
else:
    print("\ndropout_p > 0.0; So not running torch.nn.functional.scaled_dot_product_attention for comparison")
    print("Will use vanilla attention implementation for comparison when dropout_p >0.0")

# ------------------------------- Vanilla SDP implementation on CPU for test----------------------------
# Can take dropout mask from HPU flash atten FWD and use in dropout FWD. In this case
# Vanilla SDP and HPU flash attention FWD and BWD results are expected to match.
with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
    print("DBG_ONLY_dropout_mask_g", DBG_ONLY_dropout_mask_g.requires_grad)
    O_ref = vanilla_attention_impl_for_test(q_t, k_t, v_t, attn_mask = attn_mask, dropout_p = dropout_p, dbg_dropout_mask = DBG_ONLY_dropout_mask_g)
O_ref.backward(g_t)

# ------------------------------- Test Results Comparison ----------------------------
print("\n")

print("Vanilla SDP FWD Ref vs Flash HPU match? = ", torch.allclose(O_ref, O_hpu.to("cpu"), rtol=rtol, atol=atol))
print("Vanilla SDP BWD Ref Q grad vs Flash HPU match? = ", torch.allclose(q_t.grad, q_hpu.grad.to("cpu"), rtol=rtol, atol=atol))
print("Vanilla SDP BWD Ref K grad vs Flash HPU match? = ", torch.allclose(k_t.grad, k_hpu.grad.to("cpu"), rtol=rtol, atol=atol))
print("Vanilla SDP BWD Ref V grad vs Flash HPU match? = ", torch.allclose(v_t.grad, v_hpu.grad.to("cpu"), rtol=rtol, atol=atol))
print("\n")
print("Max diff Vanilla SDP FWD Ref vs Flash HPU ", torch.max(torch.abs(O_ref-O_hpu.to("cpu"))))
print("Max diff Vanilla SDP BWD Ref Q grad vs Flash HPU ", torch.max(torch.abs(q_t.grad-q_hpu.grad.to("cpu"))))
print("Max diff Vanilla SDP BWD Ref K grad vs Flash HPU ", torch.max(torch.abs(k_t.grad-k_hpu.grad.to("cpu"))))
print("Max diff Vanilla SDP BWD Ref V grad vs Flash HPU ", torch.max(torch.abs(v_t.grad-v_hpu.grad.to("cpu"))))

if dropout_p == 0.0:
    print("\n")
    print("PT SDP FWD Ref vs Flash HPU match? = ", torch.allclose(sdp_ref.detach(), O_hpu.detach().to("cpu"), rtol=rtol, atol=atol))
    print("PT SDP BWD Ref Q grad vs Flash HPU match? = ", torch.allclose(q.grad, q_hpu.grad.to("cpu"), rtol=rtol, atol=atol))
    print("PT SDP BWD Ref K grad vs Flash HPU match? = ", torch.allclose(k.grad, k_hpu.grad.to("cpu"), rtol=rtol, atol=atol))
    print("PT SDP BWD Ref V grad vs Flash HPU match? = ", torch.allclose(v.grad, v_hpu.grad.to("cpu"), rtol=rtol, atol=atol))

    print("\n")
    print("Max diff PT SDP FWD Ref vs Flash HPU ", torch.max(torch.abs(sdp_ref-O_hpu.to("cpu"))))
    print("Max diff PT SDP BWD Ref Q grad vs Flash HPU ", torch.max(torch.abs(q.grad-q_hpu.grad.to("cpu"))))
    print("Max diff PT SDP BWD Ref K grad vs Flash HPU ", torch.max(torch.abs(k.grad-k_hpu.grad.to("cpu"))))
    print("Max diff PT SDP BWD Ref V grad vs Flash HPU ", torch.max(torch.abs(v.grad-v_hpu.grad.to("cpu"))))

