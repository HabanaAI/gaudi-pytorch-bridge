/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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

sizes_vec DiagOutShape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto sizes = self.sizes().vec();
  auto diagonal = stack.at(1).toInt();

  std::vector<int64_t> output_shape;
  // https://jira.habana-labs.com/browse/SW-42950

  TORCH_CHECK(self.dim() <= 2, "Input tensor should have a dimension 1 or 2");

  TORCH_CHECK(
      (self.dim() == 1 || self.dim() == 2),
      "Invalid Input size",
      self.sizes().vec())
  if (self.dim() == 1) {
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
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
    output_shape.push_back(size);
  }
  return {output_shape};
}

void Diag::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  auto out_shape = DiagOutShape(stack)[0];

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
      get_guid_with_precision(guid, ScalarType()),
      {syn_in(0)},
      {{out_shape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
