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

#include "generated/backend/cholesky.h"
#include "generated/backend/linalg_cholesky_ex.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector CholeskyMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> selfShape = self.sizes().vec();
  std::vector<int64_t> infoShape(
      std::max<int64_t>(0, static_cast<int64_t>(selfShape.size()) - 2));

  for (size_t i = 0; i < infoShape.size(); ++i) {
    infoShape[i] = selfShape[i];
  }

  return OutputMetaDataVector{
      OutputMetaData(self.scalar_type(), selfShape),
      OutputMetaData(torch::kInt32, infoShape)};
}

void Cholesky::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool upper = stack.at(1).toBool();

  auto meta = CholeskyMeta(stack);
  const auto& selfShape = meta[0].shape;

  ns_EluKernel::Params params{};
  c10::optional<int> choleskyResultIndex{0};
  if (upper) {
    choleskyResultIndex = c10::nullopt;
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{selfShape, meta[0].dtype, choleskyResultIndex}},
      &params,
      sizeof(params));

  if (upper) {
    synTransposeParams transposeParams{};
    transposeParams.tensorDim = selfShape.size();
    for (size_t i = 0; i < selfShape.size(); ++i) {
      transposeParams.permutation[i] = static_cast<TransposePermutationDim>(i);
    }
    std::swap(transposeParams.permutation[0], transposeParams.permutation[1]);

    auto transposed = BuildOp(
        graph,
        "transpose",
        {result[0].get()},
        {{selfShape, meta[0].dtype, 0}},
        &transposeParams,
        sizeof(transposeParams));

    syn_out(0) = std::move(transposed[0]);
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
} // namespace habana