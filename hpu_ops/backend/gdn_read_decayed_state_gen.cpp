/**
 * Copyright (c) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hpu_ops/gdn_read_decayed_state.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

// Output meta:
//   shape = [num_seqs, HV, V, K]
//         = [sidx.size(0), state.size(1), state.size(2), state.size(3)]
//   dtype = state.dtype (state, gates and output share dtype)
//
// Shapes (PyTorch order; gates broadcasts over V and K):
//   state: [num_slots, HV, V, K]   sidx: [num_seqs]   gates: [num_seqs, HV]
OutputMetaDataVector GdnReadDecayedStateMeta(const at::Stack& stack) {
  const auto& state = stack.at(0).toTensor();
  const auto& sidx = stack.at(1).toTensor();
  const auto& gates = stack.at(2).toTensor();

  TORCH_CHECK(
      state.dim() == 4,
      "gdn_read_decayed_state: state must be rank 4 [num_slots, HV, V, K], got ",
      state.dim());
  TORCH_CHECK(
      sidx.dim() == 1,
      "gdn_read_decayed_state: sidx must be rank 1 [num_seqs], got ",
      sidx.dim());
  TORCH_CHECK(
      gates.dim() == 2,
      "gdn_read_decayed_state: gates must be rank 2 [num_seqs, HV], got ",
      gates.dim());
  TORCH_CHECK(
      gates.size(0) == sidx.size(0),
      "gdn_read_decayed_state: gates.size(0) (num_seqs) must match sidx.size(0), got ",
      gates.size(0),
      " vs ",
      sidx.size(0));
  TORCH_CHECK(
      gates.size(1) == state.size(1),
      "gdn_read_decayed_state: gates.size(1) (HV) must match state.size(1), got ",
      gates.size(1),
      " vs ",
      state.size(1));

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = {sidx.size(0), state.size(1), state.size(2), state.size(3)};
  meta.dtype = state.scalar_type();
  return metaVec;
}

SharedMetaDataVector GdnReadDecayedStateSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& state = stack.at(0).toTensor();
  const auto& sidx = stack.at(1).toTensor();
  const auto& gates = stack.at(2).toTensor();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& m = meta.emplace_back("gdn_read_decayed_state_fwd");
  m.inputs_data = {
      getSharedMetaFromTensor(state),
      getSharedMetaFromTensor(sidx),
      getSharedMetaFromTensor(gates),
  };
  m.outputs_data = {
      // output: rank=4, dtype = state.dtype
      {4, state.scalar_type()},
  };
  return meta;
}

void GdnReadDecayedState::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "GdnReadDecayedState::AddNode");
  auto state = stackGetter.getNextInput<TensorsPair>();
  auto sidx = stackGetter.getNextInput<TensorsPair>();
  auto gates = stackGetter.getNextInput<TensorsPair>();

  const auto state_dtype = state.pt_t.scalar_type();

  TORCH_CHECK(
      state_dtype == c10::ScalarType::Float ||
          state_dtype == c10::ScalarType::BFloat16,
      "gdn_read_decayed_state: state must be fp32 or bf16, got ",
      state_dtype);
  TORCH_CHECK(
      sidx.pt_t.scalar_type() == c10::ScalarType::Int,
      "gdn_read_decayed_state: sidx must be int32, got ",
      sidx.pt_t.scalar_type());
  TORCH_CHECK(
      gates.pt_t.scalar_type() == state_dtype,
      "gdn_read_decayed_state: gates dtype must match state dtype (",
      state_dtype,
      "), got ",
      gates.pt_t.scalar_type());

  const auto meta = GdnReadDecayedStateMeta(stack);

  using namespace std::literals;
  auto node = BuildNode(
      this,
      graph,
      {get_guid_with_precision("gdn_read_decayed_state_fwd"sv, state_dtype),
       {state.syn_t, sidx.syn_t, gates.syn_t},
       {{meta[0].shape, meta[0].dtype, 0}}});
  syn_out(0) = std::move(node[0]);
}

} // namespace habana
