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

thread_local ShapeInfo* ShapeInference::m_shape_info(nullptr);
thread_local ShapeInfTensorId ShapeInference::sif_tensor_id;

void ShapeInference::Capture(ShapeInfo* shape_info) {
  HABANA_ASSERT(shape_info);
  m_shape_info = shape_info;
}

void ShapeInference::Reset() {
  m_shape_info = nullptr;
}

void ShapeInference::ResetMin() {
  m_shape_info->m_min_shapes.clear();
}

void ShapeInference::ResetMax() {
  m_shape_info->m_max_shapes.clear();
}

void ShapeInference::ResetActual() {
  m_shape_info->m_actual_shapes.clear();
}

uint64_t ShapeInference::UpdateShapeInfo(
    synapse_helpers::graph& graph,
    const std::vector<int64_t>& sizes) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  if (graph.is_dynamic_graph()) {
    HABANA_ASSERT(ShapeInference::m_shape_info);
    tensor_id = tensor_name_generator::get_tensor_id();

    // We only care about the shape during shape inference, hence
    // passing a dummy type of Undefined when creating the shape tensor
    auto shape = habana_helpers::TensorShape(sizes, c10::ScalarType::Undefined);
    switch (ShapeInference::m_shape_info->m_pass) {
      case ShapeInfo::InferencePass::MIN_SHAPE:
        ShapeInference::m_shape_info->m_min_shapes.insert({tensor_id, shape});
        break;
      case ShapeInfo::InferencePass::MAX_SHAPE:
        ShapeInference::m_shape_info->m_max_shapes.insert({tensor_id, shape});
        break;
      case ShapeInfo::InferencePass::OUTPUT_SHAPE:
        ShapeInference::m_shape_info->m_actual_shapes.insert(
            {tensor_id, shape});
        break;
      case ShapeInfo::InferencePass::INVALID:
        /*
         * Incase we set to invalid, then we dont need to do anything
         * just dont need to update any info
         */
        break;
      default:
        HABANA_ASSERT("Unidentidied Shape inference pass");
    }
  }
  return tensor_id;
}

uint64_t ShapeInference::UpdateShapeInfo(
    synapse_helpers::graph& graph,
    const uint64_t tensor_id,
    const std::vector<int64_t>& sizes) {
  if (graph.is_dynamic_graph()) {
    HABANA_ASSERT(ShapeInference::m_shape_info);
    // We only care about the shape during shape inference, hence
    // passing a dummy type of Undefined when creating the shape tensor
    auto shape = habana_helpers::TensorShape(sizes, c10::ScalarType::Undefined);
    switch (ShapeInference::m_shape_info->m_pass) {
      case ShapeInfo::InferencePass::MIN_SHAPE:
        ShapeInference::m_shape_info->m_min_shapes.insert_or_assign(
            tensor_id, shape);
        break;
      case ShapeInfo::InferencePass::MAX_SHAPE:
        ShapeInference::m_shape_info->m_max_shapes.insert_or_assign(
            tensor_id, shape);
        break;
      case ShapeInfo::InferencePass::OUTPUT_SHAPE:
        ShapeInference::m_shape_info->m_actual_shapes.insert_or_assign(
            tensor_id, shape);
        break;
      case ShapeInfo::InferencePass::INVALID:
        /*
         * Incase we set to invalid, then we dont need to do anything
         * just dont need to update any info
         */
        break;
      default:
        HABANA_ASSERT("Unidentidied Shape inference pass");
    }
  }
  return tensor_id;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> ShapeInference::
    GetMinMaxShape(const uint64_t tensor_id) {
  std::vector<int64_t> min, max;
  if (ShapeInference::m_shape_info) {
    if (ShapeInference::m_shape_info->m_min_shapes.count(tensor_id) &&
        ShapeInference::m_shape_info->m_max_shapes.count(tensor_id)) {
      min = ShapeInference::m_shape_info->m_min_shapes.at(tensor_id).get_dims();
      max = ShapeInference::m_shape_info->m_max_shapes.at(tensor_id).get_dims();
    }
  }
  return std::make_tuple(min, max);
}
