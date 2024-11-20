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
#pragma once
#include <iostream>
#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/tensor_shape.h"
#include "backend/synapse_helpers/graph.h"

namespace habana {

using IdShapeMap = std::unordered_map<uint64_t, habana_helpers::TensorShape>;

class ShapeInfTensorId {
 public:
  int64_t read_and_increment() {
    auto value = unique_id;
    unique_id++;
    return value;
  }

  int64_t read_and_decrement() {
    auto value = unique_id;
    unique_id--;
    return value;
  }

  int64_t get() {
    return unique_id;
  }

  void set(int64_t id) {
    unique_id = id;
  }

  void increment(int64_t val) {
    unique_id += val;
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
    m_min_policy_inuse = habana_helpers::MIN_POLICY_DEFAULT;
    m_max_policy_inuse = habana_helpers::MAX_POLICY_DEFAULT;

    m_min_shapes.clear();
    m_max_shapes.clear();
    m_actual_shapes.clear();
  }

  InferencePass m_pass;
  habana_helpers::DynamicDimsPolicy m_min_policy_inuse{
      habana_helpers::MIN_POLICY_DEFAULT};
  habana_helpers::DynamicDimsPolicy m_max_policy_inuse{
      habana_helpers::MAX_POLICY_DEFAULT};
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

  static uint64_t UpdateShapeInfoDynamic(
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
    return ShapeInference::m_shape_info
        ? m_shape_info->m_pass
        : habana::ShapeInfo::InferencePass::INVALID;
  }

  static habana_helpers::DynamicDimsPolicy GetMinPolicyInUse() {
    return m_shape_info->m_min_policy_inuse;
  }

  static habana_helpers::DynamicDimsPolicy GetMaxPolicyInUse() {
    return m_shape_info->m_max_policy_inuse;
  }

  static void SetMinMaxPolicyInUse(
      habana_helpers::DynamicDimsPolicy min_policy,
      habana_helpers::DynamicDimsPolicy max_policy) {
    m_shape_info->m_min_policy_inuse = min_policy;
    m_shape_info->m_max_policy_inuse = max_policy;
  }

  static void ResetSifTensorId() {
    sif_tensor_id.reset();
  }

  static int64_t ReadAndIncrementSifTensorId() {
    return sif_tensor_id.read_and_increment();
  }

  static int64_t GetSifTensorId() {
    return sif_tensor_id.get();
  }

  static void SetSifTensorId(int64_t id) {
    sif_tensor_id.set(id);
  }

  static void IncrementSifTensorId(int64_t cnt = 1) {
    sif_tensor_id.increment(cnt);
  }

  static void ResetShapeTensorId() {
    shape_tensor_id.reset();
  }

  static int64_t ReadAndIncrementShapeTensorId() {
    return shape_tensor_id.read_and_increment();
  }

  static int64_t ReadAndDecrementShapeTensorId() {
    return shape_tensor_id.read_and_decrement();
  }

  static int64_t GetShapeTensorId() {
    return shape_tensor_id.get();
  }

  static void SetShapeTensorId(int64_t id) {
    shape_tensor_id.set(id);
  }

  static void IncrementShapeTensorId(int64_t cnt = 1) {
    shape_tensor_id.increment(cnt);
  }

  static void SaveSTAndTensorIdxMapping(uint64_t st_idx, uint64_t t_idx) {
    st_to_tensor_idx_map.insert({st_idx, t_idx});
  }

  static uint64_t GetSTMappedTensorIdx(uint64_t st_idx) {
    return st_to_tensor_idx_map.at(st_idx);
  }

  static void SetTensorMapping(std::unordered_map<uint64_t, uint64_t> map) {
    st_to_tensor_idx_map = map;
  }

  static std::unordered_map<uint64_t, uint64_t> GetTensorMapping() {
    return st_to_tensor_idx_map;
  }

  static void ResetTensorMapping() {
    st_to_tensor_idx_map.clear();
  }

  static void SaveBackendStTid(uint64_t t_idx) {
    backend_ST_TIDs.insert(t_idx);
  }

  static std::unordered_set<uint64_t> GetBackendStTidxList() {
    return backend_ST_TIDs;
  }

  static void ResetBackendStTidxList() {
    backend_ST_TIDs.clear();
  }

 private:
  /*
   * Stores all the shape information
   */
  static thread_local ShapeInfo* m_shape_info;
  static thread_local ShapeInfTensorId sif_tensor_id;
  static thread_local ShapeInfTensorId shape_tensor_id;
  static thread_local std::unordered_set<uint64_t> backend_ST_TIDs;
  static thread_local std::unordered_map<uint64_t, uint64_t>
      st_to_tensor_idx_map;
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
