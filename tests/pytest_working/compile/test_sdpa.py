import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.xfail(reason="Temporarily disabled")
@pytest.mark.parametrize("dropout", [0.1, 0.0])
@pytest.mark.parametrize("requires_backward", [True, False])
def test_sdpa_recompute(dropout, requires_backward):
    def fn(query, key, value, am):
        is_causal = False
        scale = 1.0
        fast_softmax_mode = "None"
        result = torch.ops.hpu.sdpa_recomp_fwd(
            query, key, value, am, dropout, scale, is_causal, requires_backward, fast_softmax_mode
        )
        if requires_backward:
            result = torch.ops.hpu.sdpa_recomp_bwd(
                result[0],
                query,
                key,
                value,
                am,
                result[1],
                result[2],
                result[3],
                is_causal,
                dropout,
                scale,
                fast_softmax_mode,
            )
        return result

    query, key, value = (
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
    )
    am = torch.rand(1, 2, 8, 8, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward)
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    result = compiled_fn(query, key, value, am)
    print(len(result))
    print(result[0].shape)
    print(result[0].cpu())


@pytest.mark.xfail(reason="Temporarily disabled")
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("requires_backward", [False, True])
def test_sdpa(dropout, requires_backward):
    def fn(query, key, value, am):
        is_causal = False
        scale = 1.0
        fast_softmax_mode = "None"
        result = torch.ops.hpu.sdpa_fwd(query, key, value, am, dropout, scale, is_causal, fast_softmax_mode)
        if requires_backward:
            result = torch.ops.hpu.sdpa_bwd(
                result[0], query, key, value, result[1], result[2], is_causal, dropout, scale
            )
        return result

    query, key, value = (
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
        torch.rand(3, 2, 8, 4, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward),
    )
    am = torch.rand(1, 2, 8, 8, dtype=torch.bfloat16, device="hpu", requires_grad=requires_backward)
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    result = compiled_fn(query, key, value, am)
    print(len(result))
    print(result[0].shape)
    print(result[0].cpu())
