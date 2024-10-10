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
#pragma once

#include <c10/util/ArrayRef.h>
#include <variant>

namespace habana {

template <class DimT>
using ShapeVecT = std::vector<DimT>;

template <class DimT>
using ShapeRefT = c10::ArrayRef<DimT>;

template <class DimT>
ShapeVecT<DimT> getBatchMatmulOutShape(
    ShapeRefT<DimT> inShapeA,
    ShapeRefT<DimT> inShapeB,
    bool transposeA,
    bool transposeB);

} // namespace habana