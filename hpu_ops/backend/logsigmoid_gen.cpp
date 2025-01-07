/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "generated/backend/log_sigmoid_backward.h"
#include "generated/backend/log_sigmoid_forward.h"

namespace habana {

OutputMetaDataVector LogSigmoidFwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta, meta};
}

SharedMetaDataVector LogSigmoidFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  auto rank = self.dim();
  auto dtype = self.scalar_type();

  SharedMetaDataVector metaVec;
  metaVec.reserve(6);
  SharedMetaTensor commonTensor = {rank, dtype};

  SharedMetaData negSharedMeta{"neg_fwd"};
  negSharedMeta.inputs_data = {commonTensor};
  negSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(negSharedMeta);

  SharedMetaData maxSharedMeta{"max_fwd"};
  maxSharedMeta.inputs_data = {commonTensor, commonTensor};
  maxSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(maxSharedMeta);

  SharedMetaData expSharedMeta{"exp_fwd"};
  expSharedMeta.inputs_data = {commonTensor};
  expSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(expSharedMeta);

  SharedMetaData subSharedMeta{"sub_fwd"};
  subSharedMeta.inputs_data = {commonTensor, commonTensor};
  subSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(subSharedMeta);

  SharedMetaData addSharedMeta{"add_fwd"};
  addSharedMeta.inputs_data = {commonTensor, commonTensor};
  addSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(addSharedMeta);

  SharedMetaData logSharedMeta{"log_fwd"};
  logSharedMeta.inputs_data = {commonTensor};
  logSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(logSharedMeta);

  return metaVec;
}

SharedMetaDataVector LogSigmoidBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto selfRank = self.dim();
  SharedMetaDataVector metaVec;
  metaVec.reserve(selfRank > 1 ? 8 : 7);
  SharedMetaTensor commonTensor = {selfRank, grad.scalar_type()};

  if (selfRank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data = {commonTensor};
    metaVec.push_back(constantSharedMeta);
  }

  SharedMetaData lessSharedMeta{"less_fwd"};
  lessSharedMeta.inputs_data = {commonTensor, commonTensor};
  lessSharedMeta.outputs_data.emplace_back(selfRank, c10::ScalarType::Bool);
  metaVec.push_back(lessSharedMeta);

  SharedMetaData negSharedMeta{"neg_fwd"};
  negSharedMeta.inputs_data = {commonTensor};
  negSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(negSharedMeta);

  SharedMetaData whereSharedMeta{"where_fwd"};
  whereSharedMeta.inputs_data = {
      lessSharedMeta.outputs_data[0], commonTensor, commonTensor};
  whereSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(whereSharedMeta);

  SharedMetaData subSharedMeta{"sub"};
  subSharedMeta.inputs_data = {commonTensor, commonTensor};
  subSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(subSharedMeta);

  SharedMetaData divSharedMeta{"div"};
  divSharedMeta.inputs_data = {commonTensor, commonTensor};
  divSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(divSharedMeta);

  SharedMetaData multSharedMeta{"mult"};
  multSharedMeta.inputs_data = {commonTensor, commonTensor};
  multSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(multSharedMeta);

  SharedMetaData addSharedMeta{"add"};
  addSharedMeta.inputs_data = {commonTensor, commonTensor};
  addSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(addSharedMeta);

  return metaVec;
}

void LogSigmoidForward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = LogSigmoidFwdMeta(stack)[0];

  // negitive(input)
  auto neg = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", meta.dtype),
      {syn_in(0)},
      {{meta.shape, meta.dtype}});

  // zero constant
  auto zero = ConstantHelper(graph, 0, meta.dtype);

  // max(neg(input), 0)
  auto max_vec = BuildOp(
      graph,
      get_guid_with_precision("max_fwd", meta.dtype),
      {neg[0].get(), zero.get()},
      {{meta.shape, meta.dtype}});

  // neg(max(neg(input), 0))
  auto buffer_neg = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", meta.dtype),
      {max_vec[0].get()},
      {{meta.shape, meta.dtype}});

  // exp(neg(max(neg(input), 0)))
  auto buffer_left = BuildOp(
      graph,
      get_guid_with_precision("exp_fwd", meta.dtype),
      {buffer_neg[0].get()},
      {{meta.shape, meta.dtype}});

  // sub(neg(input), max(neg(input), 0))
  auto buffer_right_input = BuildOp(
      graph,
      get_guid_with_precision("sub", meta.dtype),
      {neg[0].get(), max_vec[0].get()},
      {{meta.shape, meta.dtype}});

  // exp(buffer_right_input)
  auto buffer_right = BuildOp(
      graph,
      get_guid_with_precision("exp_fwd", meta.dtype),
      {buffer_right_input[0].get()},
      {{meta.shape, meta.dtype}});

  // add(left_buffer, right_buffer)
  auto buffer = BuildOp(
      graph,
      get_guid_with_precision("add", meta.dtype),
      {buffer_right[0].get(), buffer_left[0].get()},
      {{meta.shape, meta.dtype, 1}});

  // log(buffer)
  auto log = BuildOp(
      graph,
      get_guid_with_precision("log_fwd", meta.dtype),
      {buffer[0].get()},
      {{meta.shape, meta.dtype}});

  // add(log, max_vec)
  auto max_vec_log = BuildOp(
      graph,
      get_guid_with_precision("add", meta.dtype),
      {max_vec[0].get(), log[0].get()},
      {{meta.shape, meta.dtype}});

  // neg( add(log, max_vec))
  auto result = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", meta.dtype),
      {max_vec_log[0].get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(buffer[0]);
}

void LogSigmoidBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();

  auto zero_vec = ConstantHelper(graph, 0, ScalarType(), inputshape);

  // input < zero_vec
  auto mask = BuildOp(
      graph,
      get_guid_with_precision("less_fwd", ScalarType()),
      {syn_in(1), zero_vec.get()},
      {{inputshape, ScalarType()}});

  // one vector
  auto one_vec = ConstantHelper(graph, 1, ScalarType(), inputshape);

  // neg(one_vec)
  auto one_vec_neg = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", ScalarType()),
      {one_vec.get()},
      {{inputshape, ScalarType()}});

  // where(mask, neg(one_vec), zero_vec)
  auto max_deriv_vec = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", ScalarType()),
      {mask[0].get(), one_vec_neg[0].get(), zero_vec.get()},
      {{inputshape, ScalarType()}});

  // where(mask, one_vec, neg(one_vec))
  auto sign_vec = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", ScalarType()),
      {mask[0].get(), one_vec.get(), one_vec_neg[0].get()},
      {{inputshape, ScalarType()}});

  // sub(buffer, one_vec)
  auto sub = BuildOp(
      graph,
      get_guid_with_precision("sub", ScalarType()),
      {syn_in(2), one_vec.get()},
      {{inputshape, ScalarType()}});

  // sub(buffer, one_vec) / buffer
  auto o_div = BuildOp(
      graph,
      get_guid_with_precision("div", ScalarType()),
      {sub[0].get(), syn_in(2)},
      {{inputshape, ScalarType()}});

  // mult(sing_vec, (sub(buffer, one_vec) / buffer))
  auto o_mult = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {sign_vec[0].get(), o_div[0].get()},
      {{inputshape, ScalarType()}});

  // max_drive_vec + (sing_vec * (sub(buffer, one_vec) / buffer))
  auto o_add = BuildOp(
      graph,
      get_guid_with_precision("add", ScalarType()),
      {max_deriv_vec[0].get(), o_mult[0].get()},
      {{inputshape, ScalarType()}});

  // neg(o_add)
  auto o_neg = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", ScalarType()),
      {o_add[0].get()},
      {{inputshape, ScalarType()}});

  // mult(neg(o_add), grad_output)
  auto output = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {o_neg[0].get(), syn_in(0)},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana
