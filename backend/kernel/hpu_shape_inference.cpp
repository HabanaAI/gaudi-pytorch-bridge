/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "hpu_shape_inference.h"
#include "backend/synapse_helpers/tensor_builder_base.h"

using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

thread_local ShapeInfo* ShapeInference::m_shape_info(nullptr);
thread_local ShapeInfTensorId ShapeInference::sif_tensor_id;
thread_local ShapeInfTensorId ShapeInference::shape_tensor_id;
thread_local std::unordered_set<uint64_t> ShapeInference::backend_ST_TIDs;
thread_local std::unordered_map<uint64_t, uint64_t>
    ShapeInference::st_to_tensor_idx_map;

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
    PT_DYNAMIC_SHAPE_DEBUG(
        "PASS:",
        ShapeInference::m_shape_info->m_pass,
        ", Tensor ID : ",
        tensor_id,
        ", Shape : ",
        shape);
  }
  return tensor_id;
}

uint64_t ShapeInference::UpdateShapeInfoDynamic(
    const uint64_t tensor_id,
    const std::vector<int64_t>& sizes) {
  HABANA_ASSERT(ShapeInference::m_shape_info);
  // We only care about the shape during shape inference, hence
  // passing a dummy type of Undefined when creating the shape tensor
  HABANA_ASSERT(
      ShapeInference::m_shape_info->m_pass ==
      ShapeInfo::InferencePass::OUTPUT_SHAPE);
  auto shape = habana_helpers::TensorShape(sizes, c10::ScalarType::Undefined);
  ShapeInference::m_shape_info->m_actual_shapes.insert_or_assign(
      tensor_id, shape);
  PT_DYNAMIC_SHAPE_DEBUG(
      "PASS:",
      ShapeInference::m_shape_info->m_pass,
      ", Tensor ID : ",
      tensor_id,
      ", Shape : ",
      shape);
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
    PT_DYNAMIC_SHAPE_DEBUG(
        "PASS:",
        ShapeInference::m_shape_info->m_pass,
        ", Tensor ID : ",
        tensor_id,
        ", Shape : ",
        shape);
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
