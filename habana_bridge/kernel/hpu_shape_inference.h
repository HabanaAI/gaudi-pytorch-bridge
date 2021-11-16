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
#include "synapse_helpers/graph.h"

namespace habana {

using NameShapeMap =
    std::unordered_map<std::string, habana_helpers::TensorShape>;

class ShapeInfo {
 public:
  enum class InferencePass {
    OUTPUT_SHAPE = 0,
    MIN_SHAPE = 1,
    MAX_SHAPE = 2,
    INVALID = 3
  };

  virtual ~ShapeInfo() {
    m_pass = InferencePass::INVALID;
    m_min_shapes.clear();
    m_max_shapes.clear();
    m_actual_shapes.clear();
  }

  InferencePass m_pass;
  NameShapeMap m_min_shapes;
  NameShapeMap m_max_shapes;
  NameShapeMap m_actual_shapes;
};

class ShapeInference {
 public:
  /*
   * Sets the structure where min & max shapes are set for
   * capture
   */
  static void Capture(ShapeInfo* shape_info);

  /*
   * Reset the shape_info structure
   */
  static void Reset();

  /*
   * Reset the m_shape_info->m_min_shapes structure
   */
  static void ResetMin();

  /*
   * Reset the m_shape_info->m_max_shapes structure
   */
  static void ResetMax();

  /*
   * Reset the m_shape_info->m_actual_shapes structure
   */
  static void ResetActual();
  /*
   * Method to update and store the shape information for
   * specified tensor
   */
  static std::string UpdateShapeInfo(
      synapse_helpers::graph& graph,
      const std::vector<int64_t>& sizes,
      const std::string& tensor_name_suffix = std::string());
  /*
   * Get the shape of the Min & Max tensor values for the specified
   * tensor name
   */
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>> GetMinMaxShape(
      const std::string& syn_tensor_name);

 private:
  /*
   * Stores all the shape information
   */
  static ShapeInfo* m_shape_info;
};

inline std::ostream& operator<<(
    std::ostream& stream,
    const ShapeInfo::InferencePass& value) {
  switch (value) {
    case ShapeInfo::InferencePass::OUTPUT_SHAPE:
      stream << "OUTPUT_SHAPE";
      return stream;
    case ShapeInfo::InferencePass::MIN_SHAPE:
      stream << "MIN_SHAPE";
      return stream;
    case ShapeInfo::InferencePass::MAX_SHAPE:
      stream << "MAX_SHAPE";
      return stream;
    default:
      stream << "INVALID";
      return stream;
  }
}

}; // namespace habana