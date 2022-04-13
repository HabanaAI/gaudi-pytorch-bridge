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

using IdShapeMap = std::unordered_map<uint64_t, habana_helpers::TensorShape>;

class ShapeInfTensorId {
 public:
  int64_t next() {
    auto value = unique_id;
    unique_id++;
    return value;
  }

  void reset() {
    unique_id = 0;
  }

 private:
  int64_t unique_id;
};

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
  IdShapeMap m_min_shapes;
  IdShapeMap m_max_shapes;
  IdShapeMap m_actual_shapes;
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
  static uint64_t UpdateShapeInfo(
      synapse_helpers::graph& graph,
      const std::vector<int64_t>& sizes);
  /*
   * Method to update values for a tensor id with new shapes
   */
  static uint64_t UpdateShapeInfo(
      synapse_helpers::graph& graph,
      const uint64_t tensor_id,
      const std::vector<int64_t>& sizes);
  /*
   * Get the shape of the Min & Max tensor values for the specified
   * tensor name
   */
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>> GetMinMaxShape(
      const uint64_t tensor_id);
  /*
   * Returns wheter min-max exist for this tensor id
   */
  static bool HasMinMaxShape(const uint64_t tensor_id) {
    if (ShapeInference::m_shape_info) {
      return ShapeInference::m_shape_info->m_min_shapes.count(tensor_id) &&
          ShapeInference::m_shape_info->m_max_shapes.count(tensor_id);
    }
    return false;
  }

  static ShapeInfo::InferencePass GetCurrentPass() {
    return m_shape_info->m_pass;
  }

  static void ResetSifThreadId() {
    sif_tensor_id.reset();
  }

  static int64_t NextSifThreadId() {
    return sif_tensor_id.next();
  }

 private:
  /*
   * Stores all the shape information
   */
  static thread_local ShapeInfo* m_shape_info;
  static thread_local ShapeInfTensorId sif_tensor_id;
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
