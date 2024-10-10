/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/tril.h"
#include "generated/backend/tril_indices.h"
#include "generated/backend/triu.h"
#include "generated/backend/triu_indices.h"

namespace habana {
std::shared_ptr<void> FillTriuParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();

  params->numLower = diagonal;
  params->numUpper = INT_MAX;
  params->excludeDiag = 1;
  return params;
}

std::shared_ptr<void> FillTrilParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();

  params->numLower = INT_MIN;
  params->numUpper = diagonal;
  params->excludeDiag = 1;
  return params;
}

std::shared_ptr<void> FillTriluIndicesParams(
    const at::Stack& stack,
    size_t& size,
    const bool lowerTriangle) {
  PARAMS_STUB(ns_TriluIndicesKernel::Params);
  params->row = stack.at(0).toInt();
  params->col = stack.at(1).toInt();
  params->offset = stack.at(2).toInt();
  params->lowerTriangle = lowerTriangle;
  return params;
}

std::shared_ptr<void> FillTrilIndicesParams(
    const at::Stack& stack,
    size_t& size) {
  return FillTriluIndicesParams(stack, size, true);
}

std::shared_ptr<void> FillTriuIndicesParams(
    const at::Stack& stack,
    size_t& size) {
  return FillTriluIndicesParams(stack, size, false);
}

inline int GetTrilNumel(int row, int col, int offset) {
  // If either dimension is 0 then the there is no tril
  if (row == 0 || col == 0) {
    return 0;
  }
  // number of elements in the first row of the tril
  auto mFirstRow = offset > 0 ? std::min<int>(col, 1 + offset)
                              : // upper bounded by col
      row + offset > 0; // either 0 or 1
  // number of elements in the last row of the tril, bounded by [0, col]
  auto mLastRow = std::max<int>(0, std::min<int>(col, row + offset));
  // number of rows, bounded by [0, row]
  auto nRowAll = std::max<int>(0, std::min<int>(row, row + offset));
  auto nRowTrapezoid = (mLastRow - mFirstRow + 1);

  // calculate # of elements in the top trapezoid
  auto trilNumel = (mFirstRow + mLastRow) * nRowTrapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diffRow = nRowAll - nRowTrapezoid;
  if (diffRow > 0) {
    trilNumel += diffRow * col;
  }
  return trilNumel;
}

inline int GetTriuNumel(int row, int col, int offset) {
  return row * col - GetTrilNumel(row, col, offset - 1);
}

OutputMetaDataVector TriluIndicesMeta(
    const at::Stack& stack,
    const bool lowerTriangle) {
  const int row = stack.at(0).toInt();
  const int col = stack.at(1).toInt();
  const int offset = stack.at(2).toInt();
  const auto out_dtype =
      stack.at(3).toOptional<at::ScalarType>().value_or(at::ScalarType::Long);

  TORCH_CHECK((row > 0 && col > 0), "row and col must be greater than 0");
  TORCH_CHECK(
      (out_dtype == at::ScalarType::Long || out_dtype == at::ScalarType::Int),
      "tri(l/u)_indices output must be either int32 or int64");

  const auto numel = lowerTriangle ? GetTrilNumel(row, col, offset)
                                   : GetTriuNumel(row, col, offset);

  std::vector<int64_t> out_shape = {2, numel};

  OutputMetaData meta;
  meta.shape = out_shape;
  meta.dtype = out_dtype;
  return {meta};
}

OutputMetaDataVector TrilIndicesMeta(const at::Stack& stack) {
  return TriluIndicesMeta(stack, true);
}

OutputMetaDataVector TriuIndicesMeta(const at::Stack& stack) {
  return TriluIndicesMeta(stack, false);
}

void TriluIndices::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  auto meta = OutputMeta(stack)[0];

  if (meta.shape[1] == 0) {
    // Skip pointless calculations
    auto emptyOutput = ConstantHelper(graph, 0, meta.dtype, meta.shape, 0);
    syn_out(0) = std::move(emptyOutput);
  } else {
    update_guid_dtype(guid_, meta.dtype);
    auto triluIndices = BuildOp(
        graph,
        guid_,
        {},
        {{meta.shape, meta.dtype, 0}},
        params.get(),
        sizeof(params));
    syn_out(0) = std::move(triluIndices[0]);
  }
}

} // namespace habana
