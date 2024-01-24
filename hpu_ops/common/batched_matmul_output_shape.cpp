/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <sstream>

namespace habana {

// Calculations imported from:
// npu-stack/complex_guid_lib/mlir/lib/Optimizer/Transforms/ComplexGuid/ArithmEngine/CalcOutputShape.cpp

static ShapeRefT extendShape(
    ShapeVecT& extendedShape,
    ShapeRefT inShape,
    int shapeId) {
  switch (inShape.size()) {
    case 0:
      extendedShape = {1, 1};
      return extendedShape;
    case 1:
      if (shapeId == 0)
        extendedShape = {1, inShape[0]};
      else
        extendedShape = {inShape[0], 1};
      return extendedShape;
    default:
      return inShape;
  }
}

static void addShapeToBcastShape(ShapeVecT& bcastShape, ShapeRefT inputShape) {
  size_t offset = bcastShape.size() - inputShape.size();
  for (size_t dim = 0; dim < inputShape.size(); ++dim) {
    size_t bcastI = offset + dim;
    if (bcastShape[bcastI]) {
      if ((bcastShape[bcastI] == 1) || (inputShape[dim] == 0))
        bcastShape[bcastI] = inputShape[dim];
      else if (
          (bcastShape[bcastI] != inputShape[dim]) && (inputShape[dim] != 1)) {
        std::stringstream errorMsg;
        errorMsg << "Broadcast of shape " << inputShape
                 << " not possible at index " << dim << ". Dimension "
                 << inputShape[dim]
                 << " incompatible with output shape dimension "
                 << bcastShape[bcastI];
        throw std::invalid_argument(errorMsg.str());
      }
    }
  }
}

ShapeVecT getBatchMatmulOutShape(
    ShapeRefT inShapeA,
    ShapeRefT inShapeB,
    bool transposeA,
    bool transposeB) {
  // From documentation of numpy.matmul:
  // If both arguments are 2-D they are multiplied like conventional matrices.
  // If either argument is N-D, N > 2, it is treated as a stack of matrices
  // residing in the last two indexes and broadcast accordingly.
  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1
  // to its dimensions. After matrix multiplication the prepended 1 is removed.
  // If the second argument is 1-D, it is promoted to a matrix by appending a 1
  // to its dimensions. After matrix multiplication the appended 1 is removed.
  ShapeVecT extendedShapeA;
  ShapeRefT shapeA = extendShape(extendedShapeA, inShapeA, 0);
  ShapeVecT extendedShapeB;
  ShapeRefT shapeB = extendShape(extendedShapeB, inShapeB, 1);

  ShapeVecT batchShape;
  if ((shapeA.size() > 2) || (shapeB.size() > 2)) {
    ShapeVecT batchShapeA = ShapeVecT(shapeA.begin(), shapeA.end() - 2);
    ShapeVecT batchShapeB = ShapeVecT(shapeB.begin(), shapeB.end() - 2);

    if (batchShapeA.size() >= batchShapeB.size()) {
      addShapeToBcastShape(batchShapeA, batchShapeB);
      batchShape = std::move(batchShapeA);
    } else {
      addShapeToBcastShape(batchShapeB, batchShapeA);
      batchShape = std::move(batchShapeB);
    }
  }

  constexpr size_t ULTIMATE_DIM_OFFSET = 1;
  constexpr size_t PENULTIMATE_DIM_OFFSET = 2;
  size_t outputDimArevIndex =
      transposeA ? ULTIMATE_DIM_OFFSET : PENULTIMATE_DIM_OFFSET;
  size_t outputDimBrevIndex =
      transposeB ? PENULTIMATE_DIM_OFFSET : ULTIMATE_DIM_OFFSET;
  size_t commonDimArevIndex =
      transposeA ? PENULTIMATE_DIM_OFFSET : ULTIMATE_DIM_OFFSET;
  size_t commonDimBrevIndex =
      transposeB ? ULTIMATE_DIM_OFFSET : PENULTIMATE_DIM_OFFSET;

  size_t outputDimAindex = shapeA.size() - outputDimArevIndex;
  size_t outputDimBindex = shapeB.size() - outputDimBrevIndex;

  size_t commonDimAindex = shapeA.size() - commonDimArevIndex;
  size_t commonDimBindex = shapeB.size() - commonDimBrevIndex;

  if (shapeA[commonDimAindex] != shapeB[commonDimBindex]) {
    std::stringstream errorMsg;
    errorMsg << "Matmul common dims incompatible: " << shapeA[commonDimAindex]
             << " vs " << shapeB[commonDimBindex];
    throw std::invalid_argument(errorMsg.str());
  }

  ShapeVecT outputShape;
  outputShape.reserve(batchShape.size() + 2);
  outputShape.insert(outputShape.end(), batchShape.begin(), batchShape.end());
  outputShape.emplace_back(shapeA[outputDimAindex]);
  outputShape.emplace_back(shapeB[outputDimBindex]);

  if (batchShape.empty()) {
    if (shapeA.size() == 1)
      outputShape.erase(outputShape.begin());
    else if (shapeB.size() == 1)
      outputShape.pop_back();
  }

  return outputShape;
}

} // namespace habana