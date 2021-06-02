/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <iostream>
#include "habana_helpers/tensor_shape.h"

namespace habana {

using NameShapeMap =
    std::unordered_map<std::string, habana_helpers::TensorShape>;

struct ShapeInference {
  enum class InferencePass {
    OUTPUT_SHAPE = 0,
    MIN_SHAPE = 1,
    MAX_SHAPE = 2,
    INVALID = 3
  };

  InferencePass m_pass;
  NameShapeMap m_min_shapes;
  NameShapeMap m_max_shapes;
  NameShapeMap m_actual_shapes;

  virtual ~ShapeInference() {
    m_pass = InferencePass::INVALID;
    m_min_shapes.clear();
    m_max_shapes.clear();
    m_actual_shapes.clear();
  }
};

inline std::ostream& operator<<(
    std::ostream& stream,
    const ShapeInference::InferencePass& value) {
  switch (value) {
    case ShapeInference::InferencePass::OUTPUT_SHAPE:
      stream << "OUTPUT_SHAPE";
      return stream;
    case ShapeInference::InferencePass::MIN_SHAPE:
      stream << "MIN_SHAPE";
      return stream;
    case ShapeInference::InferencePass::MAX_SHAPE:
      stream << "MAX_SHAPE";
      return stream;
    default:
      stream << "INVALID";
      return stream;
  }
}

}; // namespace habana