/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "generated/backend/cholesky_inverse.h"
#include "generated/backend/linalg_cholesky_ex.h"
#include "habana_helpers/logging.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector CholeskyMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> selfShape = self.sizes().vec();
  const size_t selfRank = self.dim();
  std::vector<int64_t> infoShape(
      std::max<int64_t>(0, static_cast<int64_t>(selfShape.size()) - 2));

  for (size_t i = 0; i < infoShape.size(); ++i) {
    infoShape[i] = selfShape[i];
  }

  HABANA_ASSERT(
      selfRank >= 2,
      "CholeskyInverse: Input tensor must have at least 2 dimensions.");

  HABANA_ASSERT(
      selfShape[selfRank - 1] == selfShape[selfRank - 2],
      "CholeskyInverse: The last two dimensions of the input tensor must be equal.");

  return OutputMetaDataVector{
      OutputMetaData(self.scalar_type(), selfShape),
      OutputMetaData(torch::kInt32, infoShape)};
}

SharedMetaDataVector CholeskySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto inputRank = self.dim();
  const auto outputRank = std::max<int64_t>(1, inputRank - 2);
  const auto dtype = self.scalar_type();

  SharedMetaData choleskySharedMeta{"cholesky_fwd"};
  choleskySharedMeta.inputs_data.emplace_back(inputRank, dtype);
  choleskySharedMeta.outputs_data.emplace_back(outputRank, dtype);

  SharedMetaDataVector metaVec{choleskySharedMeta};
  const auto has_two_outputs = (stack.size() >= 3) && stack.at(2).isBool();
  if (has_two_outputs && outputRank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(
        outputRank, c10::ScalarType::Int);
    metaVec.push_back(constantSharedMeta);
  }

  return metaVec;
}

SharedMetaDataVector CholeskyInverseSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto selfRank = self.dim();
  const auto dtype = self.scalar_type();

  SharedMetaData choleskyInverseSharedMeta{"cholesky_inverse_fwd"};
  choleskyInverseSharedMeta.inputs_data.emplace_back(selfRank, dtype);
  choleskyInverseSharedMeta.outputs_data.emplace_back(selfRank, dtype);

  return {choleskyInverseSharedMeta};
}

synapse_helpers::tensor performTranspose(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    const at::IntArrayRef selfShape,
    const at::ScalarType selfDtype,
    std::optional<int> i = std::nullopt) {
  synTransposeParams transposeParams{};
  transposeParams.tensorDim = selfShape.size();
  for (size_t i = 0; i < selfShape.size(); ++i) {
    transposeParams.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(transposeParams.permutation[0], transposeParams.permutation[1]);

  auto transposed = OpBackend::BuildNode(
      op,
      graph,
      {"transpose",
       {input},
       {{selfShape, selfDtype, i}},
       &transposeParams,
       sizeof(transposeParams)});

  return std::move(transposed[0]);
}

void Cholesky::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool upper = stack.at(1).toBool();

  auto meta = CholeskyMeta(stack);
  const auto& selfShape = meta[0].shape;

  ns_EluKernel::Params params{};
  std::optional<int> choleskyResultIndex{0};
  if (upper) {
    choleskyResultIndex = std::nullopt;
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{selfShape, meta[0].dtype, choleskyResultIndex}},
      &params,
      sizeof(params));

  if (upper) {
    syn_out(0) = std::move(performTranspose(
        this, graph, result[0].get(), selfShape, meta[0].dtype, 0));
  } else {
    syn_out(0) = std::move(result[0]);
  }

  // Check if at::cholesky or at::linalg_cholesky_ex
  auto has_two_outputs = (stack.size() >= 3) && stack.at(2).isBool();
  if (has_two_outputs) {
    // Because tpc kernel do not support checking if matrix is a real symmetric
    // positive-definite matrix we leave it zeros for now.
    auto info = BuildConstant(this, graph, 0, meta[1].dtype, meta[1].shape, 1);
    syn_out(1) = std::move(info);
  }
}

void CholeskyInverse::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const bool upper = stack.at(1).toBool();
  const auto& selfShape = self.sizes();
  const size_t selfRank = self.dim();
  std::optional<sh::tensor> transposed{};

  HABANA_ASSERT(
      selfRank >= 2,
      "CholeskyInverse: Input tensor must have at least 2 dimensions.");

  HABANA_ASSERT(
      selfShape[selfRank - 1] == selfShape[selfRank - 2],
      "CholeskyInverse: The last two dimensions of the input tensor must be equal.");

  if (!upper) {
    transposed =
        performTranspose(this, graph, syn_in(0), selfShape, self.scalar_type());
  }

  auto result = BuildOp(
      graph,
      guid_,
      {transposed.has_value() ? transposed.value().get() : syn_in(0)},
      {{selfShape, self.scalar_type(), 0}});

  syn_out(0) = std::move(result[0]);
}

} // namespace habana
