import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as rand_hpu
import math # for ceil etc
import os

#Large -ve value ; Using -inf can cause issues when softmax soft max is taken over a section that hasll -inf on the row
#this can happen since we operate on slices. So use a large -ve value other than -inf at leaset in the flash impl.

#LNEG = float('-inf')
LNEG = -1e9
def get_dbg_env_var(v, df = 0):
    return int(os.getenv(v, df))

def create_dropout_mask(input, shape, p, generator=None):
    assert generator is None
    p = 1.0 - p
    t = torch.rand(shape, dtype=input.dtype, layout=input.layout, device=input.device)
    mask = (t < p).to(dtype=torch.uint8)
    return mask

def create_dropout_mask_v2(input, shape, p, generator=None):
    assert generator is None
    p = 1.0 - p
    dropout_scaling = 1.0 / p
    t = torch.rand(shape, dtype=input.dtype, layout=input.layout, device=input.device)
    mask = (t < p).to(dtype=input.dtype)
    Z = torch.mul(mask, dropout_scaling)
    return Z

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

# This is algo with No slicing of tensors
def _flash_attn_no_slice_forward(q,k,v, attn_mask=None, dropout_p=0.0, n_iter = None, scale = None ):
    assert q.dim() == 4, " Currently support only 4D"

    dev = q.device
    #print( "Running _flash_attn_no_slice_forward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    dropout_mask = None
    rng_state = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        htcore.mark_step()
        rng_state = rand_hpu.get_rng_state() #RTC: lock needed?
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        htcore.mark_step()
        if n_iter is not None:
            torch.save(dropout_mask, "dm_fwd_i"+str(n_iter)+".pt")
        #print(" dropout_mask FWD = ", dropout_mask)


    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask

    m,_ = torch.max(S, -1, keepdim = True)

    P = torch.exp(S - m)
    P = P.to(q.dtype) # autocast runs exp in fp32. so convert o/p to type of q.
    l = torch.sum(P, -1, keepdim = True)
    P = torch.div(P,l)
    if dropout_p > 0.0 :
        P_dropped = dropout_wrapper(P, dropout_p, dropout_mask)
    else:
        P_dropped = P

    O = torch.matmul(P_dropped, v)

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return O, l, m, rng_state
    else:
        return O, l, m, rng_state, dropout_mask


# This is algo with No slicing of tensors
def _flash_attn_no_slice_backward(dO, q,k,v, O,l,m, attn_mask=None, rng_state = None, n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    #print( "Running _flash_attn_no_slice_backward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    dropout_mask = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)
        rng_state_backup = rand_hpu.get_rng_state() #RTC lock needed?
        htcore.mark_step()
        rand_hpu.set_rng_state(rng_state)
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        htcore.mark_step()
        if n_iter is not None:
            torch.save(dropout_mask, "dm_bwd_i"+str(n_iter)+".pt")
        #print(" dropout_mask BWD = ", dropout_mask)
        rand_hpu.set_rng_state(rng_state_backup)

    dropout_scale = 1.0/(1.0 - dropout_p)


    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale) #RTC should we use inplace op to reduce mem?
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask
    P = torch.div(torch.exp(S - m), l)
    dP_dropped = torch.matmul(dO, v.transpose(-2, -1))
    if dropout_p > 0.0 :
        Z = torch.mul(dropout_mask.type_as(q), dropout_scale)
        P_dropped = torch.mul(P, Z)
        dP = torch.mul(dP_dropped, Z)
    else:
        P_dropped = P
        dP =  dP_dropped
    dV = torch.matmul(P_dropped.transpose(-2, -1), dO)
    D = torch.sum(torch.mul(dO, O),  -1, keepdim = True)
    dS = torch.mul(P, (dP - D))
    dQ = torch.mul(torch.matmul(dS, k), scale)
    dK = torch.mul(torch.matmul(dS.transpose(-2, -1), q), scale)


    return dQ, dK, dV

def _flash_attn_noslice_with_softmax_forward(q,k,v, attn_mask=None, dropout_p=0.0, n_iter = None, scale = None ):
    assert q.dim() == 4, " Currently support only 4D"

    dev = q.device
    #print( "Running _flash_attn_no_slice_forward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    dropout_mask = None
    rng_state = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        htcore.mark_step()
        rng_state = rand_hpu.get_rng_state() #RTC: lock needed?
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        htcore.mark_step()
        if n_iter is not None:
            torch.save(dropout_mask, "dm_fwd_i"+str(n_iter)+".pt")
        #print(" dropout_mask FWD = ", dropout_mask)


    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask

    #m,_ = torch.max(S, -1, keepdim = True)

    #P = torch.exp(S - m)
    #P = P.to(q.dtype) # autocast runs exp in fp32. so convert o/p to type of q.
    #l = torch.sum(P, -1, keepdim = True)
    #P = torch.div(P,l)
    l = None
    m = None
    P = F.softmax(S, dim=-1)
    if dropout_p > 0.0 :
        P_dropped = dropout_wrapper(P, dropout_p, dropout_mask)
    else:
        P_dropped = P

    O = torch.matmul(P_dropped, v)

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return O, l, m, rng_state
    else:
        return O, l, m, rng_state, dropout_mask


# This is algo with No slicing of tensors
def _flash_attn_noslice_with_softmax_backward(dO, q,k,v, O,l,m, attn_mask=None, rng_state = None, n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    #print( "Running _flash_attn_no_slice_backward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    dropout_mask = None
    if dropout_p > 0.0 :
        dropout_scale = 1.0/(1.0 - dropout_p)
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)
        rng_state_backup = rand_hpu.get_rng_state() #RTC lock needed?
        htcore.mark_step()
        rand_hpu.set_rng_state(rng_state)
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        Z = torch.mul(dropout_mask.type_as(q), dropout_scale)
        htcore.mark_step()
        if n_iter is not None:
            torch.save(dropout_mask, "dm_bwd_i"+str(n_iter)+".pt")
        #print(" dropout_mask BWD = ", dropout_mask)
        rand_hpu.set_rng_state(rng_state_backup)



    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale) #RTC should we use inplace op to reduce mem?
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask
    #P = torch.div(torch.exp(S - m), l)
    P = F.softmax(S, dim=-1)
    dP_dropped = torch.matmul(dO, v.transpose(-2, -1))
    if dropout_p > 0.0 :
        P_dropped = torch.mul(P, Z)
        dP = torch.mul(dP_dropped, Z)
    else:
        P_dropped = P
        dP =  dP_dropped
    dV = torch.matmul(P_dropped.transpose(-2, -1), dO)
    D = torch.sum(torch.mul(dO, O),  -1, keepdim = True)
    dS = torch.mul(P, (dP - D))
    dQ = torch.mul(torch.matmul(dS, k), scale)
    dK = torch.mul(torch.matmul(dS.transpose(-2, -1), q), scale)


    return dQ, dK, dV

def _flash_attn_noslice_with_softmax_norngstate_forward(q,k,v, attn_mask=None, dropout_p=0.0, n_iter = None, scale = None ):
    assert q.dim() == 4, " Currently support only 4D"

    dev = q.device
    #print( "Running _flash_attn_noslice_with_softmax_norngstate_forward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    dropout_mask = None
    rng_state = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        rng_state = create_dropout_mask_v2(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask FWD = ", dropout_mask)


    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask

    #m,_ = torch.max(S, -1, keepdim = True)

    #P = torch.exp(S - m)
    #P = P.to(q.dtype) # autocast runs exp in fp32. so convert o/p to type of q.
    #l = torch.sum(P, -1, keepdim = True)
    #P = torch.div(P,l)
    l = None
    m = None
    P = F.softmax(S, dim=-1)
    if dropout_p > 0.0 :
        #P_dropped = dropout_wrapper(P, dropout_p, dropout_mask)
        P_dropped = torch.mul(P, rng_state)
    else:
        P_dropped = P

    O = torch.matmul(P_dropped, v)

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return O, l, m, rng_state
    else:
        dropout_mask = rng_state.detach().clone()
        return O, l, m, rng_state, dropout_mask


# This is algo with No slicing of tensors
def _flash_attn_noslice_with_softmax_norngstate_backward(dO, q,k,v, O,l,m, attn_mask=None, rng_state = None, n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    print( "_flash_attn_noslice_with_softmax_norngstate_backward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    Z = rng_state

    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale) #RTC should we use inplace op to reduce mem?
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask
    #P = torch.div(torch.exp(S - m), l)
    P = F.softmax(S, dim=-1)
    dP_dropped = torch.matmul(dO, v.transpose(-2, -1))
    if dropout_p > 0.0 :
        P_dropped = torch.mul(P, Z)
        dP = torch.mul(dP_dropped, Z)
    else:
        P_dropped = P
        dP =  dP_dropped
    dV = torch.matmul(P_dropped.transpose(-2, -1), dO)
    D = torch.sum(torch.mul(dO, O),  -1, keepdim = True)
    dS = torch.mul(P, (dP - D))
    dQ = torch.mul(torch.matmul(dS, k), scale)
    dK = torch.mul(torch.matmul(dS.transpose(-2, -1), q), scale)


    return dQ, dK, dV

def _flash_attn_noslice_std_attn_forward(q,k,v, attn_mask=None, dropout_p=0.0, n_iter = None, scale = None ):
    assert q.dim() == 4, " Currently support only 4D"

    dev = q.device
    #print( "Running _flash_attn_noslice_std_attn_forward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    batch_size = q.size(0)
    Z = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        Z = create_dropout_mask_v2(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask FWD = ", dropout_mask)


    S = torch.matmul(q, k.transpose(-2, -1))
    S = torch.mul(S, scale)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            S.masked_fill_(attn_mask == False, LNEG)
        else:
            S += attn_mask

    P = F.softmax(S, dim=-1)
    if dropout_p > 0.0 :
        #P_dropped = dropout_wrapper(P, dropout_p, dropout_mask)
        P_dropped = torch.mul(P, Z)
    else:
        P_dropped = P

    O = torch.matmul(P_dropped, v)

    return O, P, P_dropped, Z


# This is algo with No slicing of tensors
def _flash_attn_noslice_std_attn_backward(dO, q,k,v,P, P_dropped, Z, n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    print( "_flash_attn_noslice_std_attn_backward On device : ", dev)
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))


    dP_dropped = torch.matmul(dO, v.transpose(-2, -1))
    if dropout_p > 0.0 :
        dP = torch.mul(dP_dropped, Z)
    else:
        dP =  dP_dropped
    dV = torch.matmul(P_dropped.transpose(-2, -1), dO)
    D = torch.sum(torch.mul(dP, P),  -1, keepdim = True)
    dS = torch.mul(P, (dP - D))
    dQ = torch.mul(torch.matmul(dS, k), scale)
    dK = torch.mul(torch.matmul(dS.transpose(-2, -1), q), scale)


    return dQ, dK, dV

def _flash_attn_sliceQ_forward(q,k,v, attn_mask=None, dropout_p=0.0,n_iter = None, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    neg_inf = -float('inf')
    #print( "Running _flash_attn_sliceQ_forward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    Tr = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TR', 4) # use default Tr = 4
    Br = seq_len_N // Tr
    if (seq_len_N % Tr) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tr", Tr)
        assert "sequence length should be perfectly divisible by Tr"

    print(" Br = ", Br, " Tr = ", Tr)


    O = torch.zeros(q.shape, dtype = q.dtype).to(dev) # Nxd  #RTC: Change o/p Shape to consider embedding dim of value
    # Shape of l and m is same as that of Q except that the last dim should be 1
    lm_shape = list(q.shape)
    lm_shape[q.dim() -1] = 1
    l = torch.zeros(lm_shape, dtype = q.dtype).to(dev) # ( N x 1)
    m = torch.full(lm_shape, neg_inf,  dtype = q.dtype).to(dev) # ( N x 1)

    batch_size = q.size(0)
    dropout_mask = None
    rng_state = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        rng_state = rand_hpu.get_rng_state() #RTC: lock needed?
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask FWD = ", dropout_mask)


    split_dim = -2 # should be the dim corr. seq_len_N
    Q_blocks = torch.split(q, Br, dim = split_dim) # splitting q into Tr blocks of size Br

    O_blocks = list(torch.split(O, Br, dim = split_dim)) # splitting O into Tr blocks of size Br
    l_blocks = torch.split(l, Br, dim = split_dim) # splitting l into Tr blocks of size Br
    m_blocks = torch.split(m, Br, dim = split_dim) # splitting m into Tr blocks of size Br

    assert Tr == len(Q_blocks)
    assert Tr == len(O_blocks)
    assert Tr == len(l_blocks)
    assert Tr == len(m_blocks)

    attn_mask_split_Br_dim = False
    if attn_mask is not None:
        if attn_mask.size(-2) != 1 :
            attn_mask_split_Br_dim = True
            Attn_mask_blocks = torch.split(attn_mask, Br, dim = -2)
    if dropout_p > 0.0 :
        Dropout_mask_blocks = torch.split(dropout_mask, Br, dim = -2)

    for i in range(Tr):
        Qi = Q_blocks[i]
        Si = torch.matmul(Qi, k.transpose(-2, -1))
        Si = torch.mul(Si, scale)
        #print("Si = ", Si.to("cpu"))
        if attn_mask is not None:
            if attn_mask_split_Br_dim == False:
                amski = attn_mask
            else:
                amski = Attn_mask_blocks[i]
            if attn_mask.dtype == torch.bool:
                Si.masked_fill_(amski == False, LNEG)
            else:
                Si += amski

        mi,_ = torch.max(Si, -1, keepdim = True) # (Br x 1)

        Pi = torch.exp(Si - mi)
        Pi = Pi.to(q.dtype) # autocast runs exp in fp32. so convert o/p to type of q.
        li = torch.sum(Pi, -1, keepdim = True) # (Br, 1)
        Pi = torch.div(Pi, li)

        if dropout_p > 0.0 :
            dmski = Dropout_mask_blocks[i]
            Pi_dropped = dropout_wrapper(Pi, dropout_p, dmski)
        else:
            Pi_dropped = Pi

        t = torch.matmul(Pi_dropped, v)
        Oi = O_blocks[i]    # WA
        t = torch.add(Oi,t) # WA:

        O_blocks[i].copy_(t) #RTC : can we conver to inplce op?
        l_blocks[i].copy_(li)
        m_blocks[i].copy_(mi)

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return O, l, m, rng_state
    else:
        return O, l, m, rng_state, dropout_mask


def _flash_attn_sliceQ_backward(dO, q,k,v, O,l,m, attn_mask=None, rng_state = None,n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    #print( "Running _flash_attn_sliceQ_backward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn
    Tr = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TR', 4) # use default Tr = 4
    Br = seq_len_N // Tr
    if (seq_len_N % Tr) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tr", Tr)
        assert "sequence length should be perfectly divisible by Tr"

    print(" Br = ", Br, " Tr = ", Tr)

    dQ = torch.zeros(q.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor
    dK = torch.zeros(k.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor
    dV = torch.zeros(v.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor

    batch_size = q.size(0)
    dropout_mask = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)
        rng_state_backup = rand_hpu.get_rng_state() #RTC lock needed?
        rand_hpu.set_rng_state(rng_state)
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask BWD = ", dropout_mask)
        rand_hpu.set_rng_state(rng_state_backup)

    split_dim = -2 # should be the dim corr. seq_len_N
    Q_blocks = torch.split(q, Br, dim = split_dim) # splitting q into Tr blocks of size Br

    dQ_blocks = list(torch.split(dQ, Br, dim = split_dim)) # splitting dQ into Tr blocks of size Br

    O_blocks = list(torch.split(O, Br, dim = split_dim)) # splitting O into Tr blocks of size Br
    dO_blocks = list(torch.split(dO, Br, dim = split_dim)) # splitting dO into Tr blocks of size Br
    l_blocks = list(torch.split(l, Br, dim = split_dim)) # splitting l into Tr blocks of size Br
    m_blocks = list(torch.split(m, Br, dim = split_dim)) # splitting m into Tr blocks of size Br
    attn_mask_split_Br_dim = False
    if attn_mask is not None:
        if attn_mask.size(-2) != 1 :
            attn_mask_split_Br_dim = True
            Attn_mask_blocks = torch.split(attn_mask, Br, dim = -2)
    if dropout_p > 0.0 :
        Dropout_mask_blocks = torch.split(dropout_mask, Br, dim = -2)
    assert Tr == len(Q_blocks)
    assert Tr == len(O_blocks)
    assert Tr == len(l_blocks)
    assert Tr == len(m_blocks)
    #RTC : may add asserts for other tensors splits
    dropout_scale = 1.0/(1.0 - dropout_p)


    for i in range(Tr):
        Qi = Q_blocks[i]
        Oi = O_blocks[i]
        dOi = dO_blocks[i]
        #dQi = dQ_blocks[i]
        li = l_blocks[i]
        mi = m_blocks[i]

        Si = torch.matmul(Qi, k.transpose(-2, -1)) # (Br x d ) (d x Bc) = (Br x Bc)
        Si = torch.mul(Si, scale) #RTC should we use inplace op to reduce mem?
        if attn_mask is not None:
            if attn_mask_split_Br_dim == False:
                amski = attn_mask
            else:
                amski = Attn_mask_blocks[i]
            if attn_mask.dtype == torch.bool:
                Si.masked_fill_(amski == False, LNEG)
            else:
                Si += amski
        #print(" Sij = ", Sij)
        Pi = torch.div(torch.exp(Si - mi), li)
        if dropout_p > 0.0 :
            dmski = Dropout_mask_blocks[i]
            Zi = torch.mul(dmski.type_as(q), dropout_scale)
            Pi_dropped = torch.mul(Pi, Zi)
        else:
            Pi_dropped = Pi
        dV.add_(torch.matmul(Pi_dropped.transpose(-2, -1), dOi))
        dPi_dropped = torch.matmul(dOi, v.transpose(-2, -1))
        if dropout_p > 0.0 :
            dPi = torch.mul(dPi_dropped, Zi)
        else:
            dPi =  dPi_dropped
        Di = torch.sum(torch.mul(dOi, Oi),  -1, keepdim = True)
        dSi = torch.mul(Pi, (dPi - Di))
        t1 = torch.mul(torch.matmul(dSi, k), scale)
        #dQi.add_(t1) # RTC the mul by scale can be absorbed into the "value " in add just to reduce FE ops
        dQ_blocks[i].copy_(t1)
        t2 = torch.mul(torch.matmul(dSi.transpose(-2, -1), Qi), scale)
        dK.add_(t2) # RTC the mul by scale can be absorbed into the "value " in add

    return dQ, dK, dV

def _flash_attn_paper_forward(q,k,v, attn_mask=None, dropout_p=0.0, n_iter = None, scale = None):
    assert q.dim() == 4, " Currently support only 4D"

    dev = q.device
    #print( "Running _flash_attn_paper_forward On device : ", dev)
    neg_inf = -float('inf')
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    Tr = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TR', 4) # use default Tr = 4
    Br = seq_len_N // Tr
    if (seq_len_N % Tr) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tr", Tr)
        assert "sequence length should be perfectly divisible by Tr"

    Tc = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TC', 4) # use default Tc = 4
    Bc = seq_len_N // Tc
    if (seq_len_N % Tc) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tc", Tc)
        assert "sequence length should be perfectly divisible by Tc"

    print(" Bc = ", Bc, " Br = ", Br, " Tc = " , Tc, " Tr = ", Tr)

    O = torch.zeros(q.shape, dtype = q.dtype).to(dev) # Nxd  #RTC: Change o/p Shape to consider embedding dim of value
    # Shape of l and m is same as that of Q except that the last dim should be 1
    lm_shape = list(q.shape)
    lm_shape[q.dim() -1] = 1
    l = torch.zeros(lm_shape, dtype = q.dtype).to(dev) # ( N x 1)
    m = torch.full(lm_shape, neg_inf,  dtype = q.dtype).to(dev) # ( N x 1)

    batch_size = q.size(0)
    dropout_mask = None
    rng_state = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)

        rng_state = rand_hpu.get_rng_state() #RTC: lock needed?
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask FWD = ", dropout_mask)


    split_dim = -2 # should be the dim corr. seq_len_N
    Q_blocks = torch.split(q, Br, dim = split_dim) # splitting q into Tr blocks of size Br
    K_blocks = torch.split(k, Bc, dim = split_dim) # splitting k into Tc blocks of size Bc
    V_blocks = torch.split(v, Bc, dim = split_dim) # splitting v into Tc blocks of size Bc

    O_blocks = list(torch.split(O, Br, dim = split_dim)) # splitting O into Tr blocks of size Br
    l_blocks = torch.split(l, Br, dim = split_dim) # splitting l into Tr blocks of size Br
    m_blocks = torch.split(m, Br, dim = split_dim) # splitting m into Tr blocks of size Br

    # in self attention case, mask is an NxN tensor. It needs to be sliced into
    # Br x Bc sections. So first split along the last dim int Tc blocks of size N x Bc
    # then slice each such block along last but one dim into Tr blocks of size Br x Bc
    if attn_mask is not None:
        Attn_mask_blocksj = torch.split(attn_mask, Bc, dim = -1)
    if dropout_p > 0.0 :
        Dropout_mask_blocksj = torch.split(dropout_mask, Bc, dim = -1)
    assert Tr == len(Q_blocks)
    assert Tc == len(K_blocks)
    assert Tc == len(V_blocks)
    assert Tr == len(O_blocks)
    assert Tr == len(l_blocks)
    assert Tr == len(m_blocks)


    for j in range(Tc):
        Kj = K_blocks[j]
        Vj = V_blocks[j]
        if attn_mask is not None:
            Attn_mask_blocksij = torch.split(Attn_mask_blocksj[j], Br, dim = -2)
        if dropout_p > 0.0 :
            Dropout_mask_blocksij = torch.split(Dropout_mask_blocksj[j], Br, dim = -2)

        for i in range(Tr):
            Qi = Q_blocks[i]
            Oi = O_blocks[i]
            li = l_blocks[i]
            mi = m_blocks[i]


            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) # (Br x d ) (d x Bc) = (Br x Bc)
            Sij = torch.mul(Sij, scale)
            if attn_mask is not None:
                amski = Attn_mask_blocksij[i]
                if attn_mask.dtype == torch.bool:
                    Sij.masked_fill_(amski == False, LNEG)
                else:
                    Sij += amski

            mij_tld,_ = torch.max(Sij, -1, keepdim = True) # (Br x 1)

            Pij_tld = torch.exp(Sij - mij_tld) # (Br x Bc) - (Br x 1) = (Br x Bc)
            lij_tld = torch.sum(Pij_tld, -1, keepdim = True) # (Br, 1)

            mi_new = torch.max(mi, mij_tld) # (Br x 1)
            ui = torch.exp(mi - mi_new) # (Br x 1)
            uij_tld = torch.exp(mij_tld - mi_new) # (Br x 1)

            li_new = torch.mul(ui, li) + torch.mul(uij_tld, lij_tld)
            if dropout_p > 0.0 :
                dmski = Dropout_mask_blocksij[i]
                Pij_tld_dropped = dropout_wrapper(Pij_tld, dropout_p, dmski)
            else:
                Pij_tld_dropped = Pij_tld

            t1 = torch.mul(ui, Oi) # (Br x 1) . (Br x d) = (Br x d)
            t2 = torch.matmul(Pij_tld_dropped, Vj) # (Br x Bc) @ (Bc x d) = (Br x d)
            t3 = torch.mul(uij_tld, t2) #  (Br x 1) . (Br x d) = (Br x d)
            t4 = torch.mul(li, t1)  # (Br x 1) . (Br x d) = (Br x d)
            t5 = t3 + t4
            Oi = torch.div(t5, li_new)  # RTC : should we check for div by zero?

            O_blocks[i].copy_(Oi) #RTC : can we conver to inplce op?
            l_blocks[i].copy_(li_new)
            m_blocks[i].copy_(mi_new)

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return O, l, m, rng_state
    else:
        return O, l, m, rng_state, dropout_mask


def _flash_attn_paper_backward(dO, q,k,v, O,l,m, attn_mask=None, rng_state = None, n_iter = None, dropout_p=0.0, scale = None):
    assert q.dim() == 4, " Currently support only 4D"
    dev = q.device
    #print( "Running _flash_attn_paper_backward On device : ", dev)
    head_dim = q.size(-1) #RTC : Can head dim be diff inf in cross attn?
    if scale == None:
        scale = 1 / math.sqrt(q.size(-1))
    seq_len_N = q.size(-2) #RTC : seq_len_N can be diff in cross attn

    Tr = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TR', 4) # use default Tr = 4
    Br = seq_len_N // Tr
    if (seq_len_N % Tr) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tr", Tr)
        assert "sequence length should be perfectly divisible by Tr"

    Tc = get_dbg_env_var('FLASH_ATTN_ALGO_CFG_TC', 4) # use default Tc = 4
    Bc = seq_len_N // Tc
    if (seq_len_N % Tc) :
        print("sequence length ", seq_len_N, "should be perfectly divisible by Tc", Tc)
        assert "sequence length should be perfectly divisible by Tc"

    print(" Bc = ", Bc, " Br = ", Br, " Tc = " , Tc, " Tr = ", Tr)

    dQ = torch.zeros(q.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor
    dK = torch.zeros(k.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor
    dV = torch.zeros(v.shape, dtype = q.dtype).to(dev) # Nxd : RTC take dtype from tensor

    batch_size = q.size(0)
    dropout_mask = None
    if dropout_p > 0.0 :
        if q.dim() == 4:
            n_heads = q.size(1)
            dropout_mask_shape = (batch_size, n_heads, seq_len_N, seq_len_N)
        else:
            dropout_mask_shape = (batch_size, seq_len_N, seq_len_N)
        rng_state_backup = rand_hpu.get_rng_state() #RTC lock needed?
        rand_hpu.set_rng_state(rng_state)
        dropout_mask = create_dropout_mask(q, dropout_mask_shape, dropout_p)
        #print(" dropout_mask BWD = ", dropout_mask)
        rand_hpu.set_rng_state(rng_state_backup)

    split_dim = -2 # should be the dim corr. seq_len_N
    Q_blocks = torch.split(q, Br, dim = split_dim) # splitting q into Tr blocks of size Br
    K_blocks = torch.split(k, Bc, dim = split_dim) # splitting k into Tc blocks of size Bc
    V_blocks = torch.split(v, Bc, dim = split_dim) # splitting v into Tc blocks of size Bc

    dQ_blocks = list(torch.split(dQ, Br, dim = split_dim)) # splitting dQ into Tr blocks of size Br
    dK_blocks = list(torch.split(dK, Bc, dim = split_dim)) # splitting dK into Tc blocks of size Bc
    dV_blocks = list(torch.split(dV, Bc, dim = split_dim)) # splitting dV into Tc blocks of size Bc

    O_blocks = list(torch.split(O, Br, dim = split_dim)) # splitting O into Tr blocks of size Br
    dO_blocks = list(torch.split(dO, Br, dim = split_dim)) # splitting dO into Tr blocks of size Br
    l_blocks = list(torch.split(l, Br, dim = split_dim)) # splitting l into Tr blocks of size Br
    m_blocks = list(torch.split(m, Br, dim = split_dim)) # splitting m into Tr blocks of size Br
    # in self attention case, mask is an NxN tensor. It needs to be sliced into
    # Br x Bc sections. So first split along the last dim int Tc blocks of size N x Bc
    # then slice each such block along last but one dim into Tr blocks of size Br x Bc
    if attn_mask is not None:
        Attn_mask_blocksj = torch.split(attn_mask, Bc, dim = -1)
    if dropout_p > 0.0 :
        Dropout_mask_blocksj = torch.split(dropout_mask, Bc, dim = -1)
    assert Tr == len(Q_blocks)
    assert Tc == len(K_blocks)
    assert Tc == len(V_blocks)
    assert Tr == len(O_blocks)
    assert Tr == len(l_blocks)
    assert Tr == len(m_blocks)
    #RTC : may add asserts for other tensors splits
    dKj_tld = torch.empty_like(K_blocks[0])
    dVj_tld = torch.empty_like(V_blocks[0])
    dropout_scale = 1.0/(1.0 - dropout_p)

    for j in range(Tc):
        Kj = K_blocks[j]
        Vj = V_blocks[j]
        if attn_mask is not None:
            Attn_mask_blocksij = torch.split(Attn_mask_blocksj[j], Br, dim = -2)
        if dropout_p > 0.0 :
            Dropout_mask_blocksij = torch.split(Dropout_mask_blocksj[j], Br, dim = -2)
        dKj = dK_blocks[j]
        dVj = dV_blocks[j]



        dKj_tld.fill_(0.0)
        dVj_tld.fill_(0.0)

        for i in range(Tr):
            Qi = Q_blocks[i]
            Oi = O_blocks[i]
            dOi = dO_blocks[i]
            dQi = dQ_blocks[i]
            li = l_blocks[i]
            mi = m_blocks[i]

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) # (Br x d ) (d x Bc) = (Br x Bc)
            Sij = torch.mul(Sij, scale) #RTC should we use inplace op to reduce mem?
            if attn_mask is not None:
                amski = Attn_mask_blocksij[i]
                if attn_mask.dtype == torch.bool:
                    Sij.masked_fill_(amski == False, LNEG)
                else:
                    Sij += amski
            #print(" Sij = ", Sij)
            Pij = torch.div(torch.exp(Sij - mi), li)
            if dropout_p > 0.0 :
                dmski = Dropout_mask_blocksij[i]
                Zij = torch.mul(dmski.type_as(q), dropout_scale)
                Pij_dropped = torch.mul(Pij, Zij)
            else:
                Pij_dropped = Pij
            dVj_tld.add_(torch.matmul(Pij_dropped.transpose(-2, -1), dOi))
            dPij_dropped = torch.matmul(dOi, Vj.transpose(-2, -1))
            if dropout_p > 0.0 :
                dPij = torch.mul(dPij_dropped, Zij)
            else:
                dPij =  dPij_dropped
            Di = torch.sum(torch.mul(dOi, Oi),  -1, keepdim = True)
            dSij = torch.mul(Pij, (dPij - Di))
            t1 = torch.mul(torch.matmul(dSij, Kj), scale)
            dQi.add_(t1) # RTC the mul by scale can be absorbed into the "value " in add just to reduce FE ops
            t2 = torch.mul(torch.matmul(dSij.transpose(-2, -1), Qi), scale)
            dKj_tld.add_(t2) # RTC the mul by scale can be absorbed into the "value " in add

        dKj.copy_(dKj_tld)
        dVj.copy_(dVj_tld)

    return dQ, dK, dV

#================================================================================================

def flash_attn_forward(ctx, q, k, v, attn_mask = None, dropout_p=0.0, n_iter = None, softmax_scale = None):
    #rng_state = torch.xxxx.get_rng_state() if dropout_p > 0 else None

    #RTC: Should we move rng_get/set state to here from _flash_attn_no_slice_forward/_flash_attn_no_slice_backward?
    fwd = _flash_attn_no_slice_forward
    if get_dbg_env_var('FLASH_ATTN_ALGO_PAPER'):
        fwd = _flash_attn_paper_forward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE'):
        fwd = _flash_attn_no_slice_forward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_Q_SLICE'):
        fwd = _flash_attn_sliceQ_forward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX'):
        fwd = _flash_attn_noslice_with_softmax_forward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX_NORNG'):
        fwd = _flash_attn_noslice_with_softmax_norngstate_forward

    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        out, l,m, rng_state = fwd(q, k, v, attn_mask, dropout_p, n_iter, softmax_scale)
    else:
        out, l,m, rng_state, DBG_ONLY_dropout_mask = fwd(q, k, v, attn_mask, dropout_p, n_iter, softmax_scale)
    if rng_state is not None:
        ctx.save_for_backward(q, k, v, out, l, m, attn_mask, rng_state)
    else:
        ctx.save_for_backward(q, k, v, out, l, m, attn_mask)

    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.n_iter = n_iter
    #ctx.causal = causal
    if not get_dbg_env_var('FLASH_ATTN_DBG_USE_DROPOUT_STUB'):
        return out
    else:
        return out, DBG_ONLY_dropout_mask

def flash_attn_backward(ctx, dout, *args):
    bwd = _flash_attn_no_slice_backward
    if get_dbg_env_var('FLASH_ATTN_ALGO_PAPER'):
        bwd = _flash_attn_paper_backward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE'):
        bwd = _flash_attn_no_slice_backward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_Q_SLICE'):
        bwd = _flash_attn_sliceQ_backward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX'):
        bwd = _flash_attn_noslice_with_softmax_backward
    elif get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX_NORNG'):
        bwd = _flash_attn_noslice_with_softmax_norngstate_backward

    if ctx.dropout_p > 0.0:
        q, k, v, out, l, m, attn_mask, rng_state = ctx.saved_tensors
    else:
        q, k, v, out, l, m, attn_mask = ctx.saved_tensors
        rng_state = None


    dq, dk, dv = bwd(
        dout, q, k, v, out, l, m, attn_mask, rng_state, ctx.n_iter,  ctx.dropout_p, ctx.softmax_scale)
    return dq, dk, dv, None, None, None, None

def flash_attn_noslice_std_attn_fwd_wrapper(ctx, q, k, v, attn_mask = None, dropout_p=0.0, n_iter = None, softmax_scale = None):
    fwd = _flash_attn_noslice_std_attn_forward
    out, P, P_dropped, Z = fwd(q, k, v, attn_mask, dropout_p, n_iter, softmax_scale)
    ctx.save_for_backward(q, k, v, out, P, P_dropped, Z)

    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.n_iter = n_iter
    #ctx.causal = causal
    return out

def flash_attn_noslice_std_attn_bwd_wrapper(ctx, dout, *args):
    bwd = _flash_attn_noslice_std_attn_backward
    q, k, v, out, P, P_dropped, Z = ctx.saved_tensors
    dq, dk, dv = bwd(dout,q,k,v,P, P_dropped, Z, ctx.n_iter,  ctx.dropout_p, ctx.softmax_scale)
    return dq, dk, dv, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask = None, dropout_p=0.0, n_iter = None, softmax_scale = None):
        if get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_STD_ATTN'):
            return flash_attn_noslice_std_attn_fwd_wrapper(ctx, q, k, v, attn_mask = attn_mask, dropout_p=dropout_p, n_iter = n_iter, softmax_scale = softmax_scale)
        else:
            return flash_attn_forward(ctx, q, k, v, attn_mask = attn_mask, dropout_p=dropout_p, n_iter = n_iter, softmax_scale = softmax_scale)


    @staticmethod
    def backward(ctx, dout, *args):
        if get_dbg_env_var('FLASH_ATTN_ALGO_NO_SLICE_STD_ATTN'):
            return flash_attn_noslice_std_attn_bwd_wrapper(ctx, dout, *args)
        else:
            return flash_attn_backward(ctx, dout, *args)


