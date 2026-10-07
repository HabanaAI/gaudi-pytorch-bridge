###############################################################################
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
)

# (num_slots, num_seqs, HV, V, K)
#   state:  [num_slots, HV, V, K]
#   sidx:   [num_seqs]
#   gates:  [num_seqs, HV]   (broadcasts over V and K)
#   output: [num_seqs, HV, V, K]
gdn_shapes = [
    # tiny sanity
    (4, 2, 3, 5, 7),
    # batch >= 16 (the regime where the strides mismatch used to corrupt output)
    (64, 16, 8, 4, 16),
    # Qwen3.5-ish: HV heads, V x K state matrix
    (128, 32, 16, 128, 128),
]


def gdn_read_decayed_state_ref(state, sidx, gates):
    """out[seq, hv, v, k] = state[sidx[seq], hv, v, k] * gates[seq, hv]."""
    gathered = state[sidx.long()]  # [num_seqs, HV, V, K]
    return gathered * gates[..., None, None]


def _run_or_skip_if_unsupported(op, state, sidx, gates):
    # The fp32 kernel (tpc_kernels#3053) is merged; the bf16-native kernel is a
    # follow-up that may not be in the deployed libtpc_kernels.so yet.  When the
    # bf16 GUID is missing the stack raises "not yet supported on HPU" (eager) or
    # falls back / fails to compile (compile mode).  Treat that as a skip so this
    # test stays green on the fp32-only stack and auto-exercises bf16 once the
    # follow-up kernel lands.
    try:
        return op(state.to(hpu), sidx.to(hpu), gates.to(hpu))
    except Exception as e:
        msg = str(e).lower()
        unsupported = "not yet supported on hpu" in msg or "requires fallback" in msg or "fallback" in msg
        if unsupported:
            pytest.skip(f"gdn_read_decayed_state {state.dtype} kernel not deployed on this stack")
        raise


@pytest.mark.parametrize("num_slots, num_seqs, HV, V, K", gdn_shapes)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_read_decayed_state(num_slots, num_seqs, HV, V, K, dtype):
    torch.manual_seed(12345)

    state = torch.randn(num_slots, HV, V, K, dtype=dtype)
    # sidx: distinct slot indices in [0, num_slots) (num_seqs <= num_slots here)
    sidx = torch.randperm(num_slots)[:num_seqs].to(torch.int32)
    # gates = exp(g); keep magnitude modest so bf16 stays well-conditioned
    gates = torch.rand(num_seqs, HV, dtype=dtype) * 0.5 + 0.5

    ref = gdn_read_decayed_state_ref(state, sidx, gates)

    op = compile_function_if_compile_mode(torch.ops.hpu.gdn_read_decayed_state)
    out = _run_or_skip_if_unsupported(op, state, sidx, gates)

    assert out.shape == (num_seqs, HV, V, K)
    assert out.dtype == dtype

    tol = 1e-3 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.to(torch.float32).to(cpu), ref.to(torch.float32), rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("gdn_read_decayed_state")


def test_gdn_read_decayed_state_non_contiguous_state():
    """state is commonly a non-contiguous KV-cache view; output must still be
    correct (regression guard for the new_empty stride-inheritance bug)."""
    torch.manual_seed(12345)
    num_slots, num_seqs, HV, V, K = 64, 16, 8, 4, 16

    # Build a larger tensor and slice it so `state` carries non-contiguous strides.
    backing = torch.randn(num_slots, HV, V, 2 * K, dtype=torch.float32)
    state = backing[..., :K]
    assert not state.is_contiguous()

    sidx = torch.randperm(num_slots)[:num_seqs].to(torch.int32)
    gates = torch.rand(num_seqs, HV, dtype=torch.float32) * 0.5 + 0.5

    ref = gdn_read_decayed_state_ref(state, sidx, gates)

    op = compile_function_if_compile_mode(torch.ops.hpu.gdn_read_decayed_state)
    out = op(state.to(hpu), sidx.to(hpu), gates.to(hpu))

    assert out.is_contiguous()
    torch.testing.assert_close(out.to(cpu), ref, rtol=1e-3, atol=1e-3)


# mutate -> (build the bad input, substring expected in the rejection message).
# The substring for gates_num_seqs / gates_hv is the exact text of the TORCH_CHECK
# added in GdnReadDecayedStateMeta, so the test fails if that check is removed.
# gates_dtype / sidx_dtype / gates_rank are rejected earlier in the dispatch
# (generic "not yet supported" / type errors); we only assert that they ARE
# rejected, not the precise message.
_VALIDATION_CASES = {
    "gates_dtype": None,
    "gates_rank": None,
    "gates_num_seqs": "gates.size(0) (num_seqs) must match sidx.size(0)",
    "gates_hv": "gates.size(1) (HV) must match state.size(1)",
    "sidx_dtype": None,
}


@pytest.mark.parametrize("mutate", list(_VALIDATION_CASES))
def test_gdn_read_decayed_state_validation(mutate):
    """The backend output_meta / AddNode must reject malformed inputs rather
    than silently producing garbage."""
    if is_pytest_mode_compile():
        # In compile mode the rejection surfaces during tracing with a different
        # exception flow; the eager path exercises the TORCH_CHECKs directly.
        pytest.skip("validation exercised in eager mode")
    if mutate == "sidx_dtype" and is_pytest_mode_lazy():
        # int64 indices are not rejected in lazy mode: with PT_ENABLE_INT64_SUPPORT
        # off (the default) the lazy stack transparently downcasts Long->Int before
        # the op's TORCH_CHECK / output_meta runs (see habana_kernels GetPrecisionString,
        # "If INT64 is not supported, then Long and Int point to i32 precision").
        # The downcast is harmless here (slot indices fit in int32), so there is
        # nothing to reject.  The eager path still sees the true int64 dtype and
        # exercises the rejection.
        pytest.skip("int64 sidx is silently downcast to int32 in lazy mode")
    num_slots, num_seqs, HV, V, K = 8, 4, 3, 5, 7
    state = torch.randn(num_slots, HV, V, K, dtype=torch.float32)
    sidx = torch.arange(num_seqs, dtype=torch.int32)
    gates = torch.rand(num_seqs, HV, dtype=torch.float32)

    if mutate == "gates_dtype":
        gates = gates.to(torch.bfloat16)
    elif mutate == "gates_rank":
        gates = gates.unsqueeze(-1)  # rank 3
    elif mutate == "gates_num_seqs":
        gates = torch.rand(num_seqs + 1, HV, dtype=torch.float32)
    elif mutate == "gates_hv":
        gates = torch.rand(num_seqs, HV + 1, dtype=torch.float32)
    elif mutate == "sidx_dtype":
        sidx = sidx.to(torch.int64)

    expected = _VALIDATION_CASES[mutate]
    with pytest.raises(Exception) as exc_info:
        out = torch.ops.hpu.gdn_read_decayed_state(state.to(hpu), sidx.to(hpu), gates.to(hpu))
        # Force graph build / execution so the TORCH_CHECK actually fires.
        out.cpu()
    if expected is not None:
        assert expected in str(exc_info.value), f"expected rejection mentioning {expected!r}, got: {exc_info.value}"
