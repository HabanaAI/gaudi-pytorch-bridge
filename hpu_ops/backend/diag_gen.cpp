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

#include "generated/backend/diag.h"

namespace habana {

std::shared_ptr<void> FillDiagParams(const at::Stack& stack, size_t& size) {
  auto diagonal = stack.at(1).toInt();
  PARAMS_STUB(ns_MatrixDiag::Params);
  params->kMin = diagonal;
  params->kMax = diagonal;
  return params;
}

OutputMetaDataVector DiagMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto sizes = self.sizes().vec();
  auto diagonal = stack.at(1).toInt();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  // https://jira.habana-labs.com/browse/SW-42950

  TORCH_CHECK(self.dim() <= 2, "Input tensor should have a dimension 1 or 2");

  TORCH_CHECK(
      (self.dim() == 1 || self.dim() == 2),
      "Invalid Input size",
      self.sizes().vec())
  if (self.dim() == 1) {
    meta.shape.push_back(self.sizes().vec()[0] + abs(diagonal));
    meta.shape.push_back(self.sizes().vec()[0] + abs(diagonal));
  } else if (self.dim() == 2) {
    int64_t m = self.sizes().vec()[0];
    int64_t n = self.sizes().vec()[1];
    int size;
    if (diagonal == 1) { // diagonal=1
      if (m >= n) { // R>=C
        size = n - abs(diagonal);
      } else { // R<C
        size = m;
      }
    } else if (diagonal == 0) { // diagonal = 0 R>C/ R=C/ R<C
      size = std::min(m, n) -
          abs(diagonal); // https://jira.habana-labs.com/browse/SW-65273 (R>C)
    } else if (diagonal > 0) { // diagonal > 0 R>C/ R=C/ R<C
      size = n - diagonal;
    } else { // diagonal < 0 R>C/ R=C/ R<C
      size = m + diagonal;
    }
    meta.shape.push_back(size);
  }
  return {meta};
}

void Diag::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  auto meta = DiagMeta(stack)[0];

  size_t size = 0;
  auto params = FillDiagParams(stack, size);
  std::string guid;
  if (self.dim() == 1) {
    guid = "matrix_diagonal_fwd";
  } else {
    guid = "matrix_diag_part_fwd";
  }
  auto result = BuildOp(
      graph,
      get_guid_with_precision(guid, meta.dtype),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
