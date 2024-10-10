/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "hpu_ops/common/batched_matmul_output_shape.h"

#include <c10/core/SymInt.h>
#include <sstream>

namespace habana {

// From documentation of torch.matmul
// If both tensors are 1-dimensional, the dot product (scalar) is returned.
// If both arguments are 2-dimensional, the matrix-matrix product is returned.
// If the first argument is 1-dimensional and the second argument is
// 2-dimensional, a 1 is prepended to its dimension for the purpose of the
// matrix multiply. After the matrix multiply, the prepended dimension is
// removed.
// If both arguments are at least 1-dimensional and at least one
// argument is N-dimensional (where N > 2), then a batched matrix multiply is
// returned. The non-matrix (i.e. batch) dimensions are broadcasted (and thus
// must be broadcastable).
template <class DimT>
ShapeVecT<DimT> getBatchMatmulOutShape(
    ShapeRefT<DimT> inShapeA,
    ShapeRefT<DimT> inShapeB,
    bool transposeA,
    bool transposeB) {
  ShapeVecT<DimT> outputShape;
  const auto rankA = inShapeA.size();
  const auto rankB = inShapeB.size();

  size_t commonDimA = 0;
  size_t commonDimB = 0;

  if (rankB > 1) {
    auto dimB = transposeB ? rankB - 2 : rankB - 1;
    commonDimB = transposeB ? rankB - 1 : rankB - 2;
    outputShape.push_back(inShapeB[dimB]);
  }
  if (rankA > 1) {
    auto dimA = transposeA ? rankA - 1 : rankA - 2;
    commonDimA = transposeA ? rankA - 2 : rankA - 1;
    outputShape.push_back(inShapeA[dimA]);
  }

  auto commonSizeA = inShapeA[commonDimA];
  auto commonSizeB = inShapeB[commonDimB];

  if (commonSizeA != commonSizeB) {
    std::stringstream errorMsg;
    errorMsg
        << "Common dimension sizes of matmul inputs should be the same. Got "
        << commonSizeA << " and " << commonSizeB;
    throw std::invalid_argument(errorMsg.str());
  }

  auto maxRank = std::max(rankA, rankB);
  for (size_t i = 3; i <= maxRank; i++) {
    DimT dimA = i > rankA ? 1 : inShapeA[rankA - i];
    DimT dimB = i > rankB ? 1 : inShapeB[rankB - i];
    if (dimA != dimB and dimA != 1 and dimB != 1) {
      std::stringstream errorMsg;
      errorMsg
          << "Batch dimension " << maxRank - i
          << " of matmul inputs should be the same or at least one of them should be equal to 1. Got "
          << dimA << " and " << dimB;
      throw std::invalid_argument(errorMsg.str());
    }
    outputShape.push_back(dimA == dimB ? dimA : dimA * dimB);
  }
  std::reverse(outputShape.begin(), outputShape.end());

  return outputShape;
}

#define INSTANTIATE(DimT)                          \
  template ShapeVecT<DimT> getBatchMatmulOutShape( \
      ShapeRefT<DimT> inShapeA,                    \
      ShapeRefT<DimT> inShapeB,                    \
      bool transposeA,                             \
      bool transposeB)

INSTANTIATE(int64_t);
INSTANTIATE(c10::SymInt);

} // namespace habana