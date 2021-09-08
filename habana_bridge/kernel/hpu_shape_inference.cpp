/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "hpu_shape_inference.h"
#include "synapse_helpers/tensor_builder_base.h"

using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

ShapeInfo* ShapeInference::m_shape_info(nullptr);

void ShapeInference::Capture(ShapeInfo* shape_info) {
  HABANA_ASSERT(shape_info);
  m_shape_info = shape_info;
}

void ShapeInference::Reset() {
  m_shape_info = nullptr;
}

std::string ShapeInference::UpdateShapeInfo(
    synapse_helpers::graph& graph,
    const std::vector<int64_t>& sizes) {
  std::string tensor_name;
  if (graph.is_dynamic_graph()) {
    HABANA_ASSERT(ShapeInference::m_shape_info);
    tensor_name = tensor_name_generator::get_next_tensor_name();
    // We only care about the shape during shape inference, hence
    // passing a dummy type of Undefined when creating the shape tensor
    auto shape = habana_helpers::TensorShape(sizes, c10::ScalarType::Undefined);
    switch (ShapeInference::m_shape_info->m_pass) {
      case ShapeInfo::InferencePass::MIN_SHAPE:
        ShapeInference::m_shape_info->m_min_shapes.insert({tensor_name, shape});
        break;
      case ShapeInfo::InferencePass::MAX_SHAPE:
        ShapeInference::m_shape_info->m_max_shapes.insert({tensor_name, shape});
        break;
      case ShapeInfo::InferencePass::OUTPUT_SHAPE:
        ShapeInference::m_shape_info->m_actual_shapes.insert(
            {tensor_name, shape});
        break;
      default:
        HABANA_ASSERT(0);
    }
  }
  return tensor_name;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> ShapeInference::
    GetMinMaxShape(const std::string& syn_tensor_name) {
  std::vector<int64_t> min, max;
  if (ShapeInference::m_shape_info) {
    if (ShapeInference::m_shape_info->m_min_shapes.count(syn_tensor_name) &&
        ShapeInference::m_shape_info->m_max_shapes.count(syn_tensor_name)) {
      min = ShapeInference::m_shape_info->m_min_shapes.at(syn_tensor_name)
                .get_dims();
      max = ShapeInference::m_shape_info->m_max_shapes.at(syn_tensor_name)
                .get_dims();
    }
  }
  return std::make_tuple(min, max);
}

std::ostream& operator<<(
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