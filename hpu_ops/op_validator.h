/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include <absl/container/inlined_vector.h>
#include <shared_layer_api.hpp>
#include "backend/helpers/habana_types.h"
#include "hpu_ops/op_backend.h"
#include "hpu_ops/supported_dtypes.h"

#pragma once

namespace habana {

namespace detail {

// Lightweight alternative to at::IValue
// It is either
//   * pointer to at::Tensor
//   * rank of the tensor and its dtype
class TensorDescr {
 public:
  TensorDescr() = default;

  explicit TensorDescr(const at::Tensor* tensor) : m_tensor(tensor) {}

  explicit TensorDescr(const uint32_t rank, const at::ScalarType dtype)
      : m_rank(rank), m_dtype(dtype) {}

  explicit TensorDescr(const OutputMetaData& output_meta)
      : TensorDescr(
            static_cast<uint32_t>(output_meta.shape.size()),
            output_meta.dtype) {
    HABANA_ASSERT(
        output_meta.shape.size() <= std::numeric_limits<uint32_t>::max(),
        "Output meta shape exceeds uint32_t limit");
  }

  explicit TensorDescr(const SharedMetaTensor& shared_meta)
      : TensorDescr(shared_meta.first, shared_meta.second) {}

  uint32_t getRank() const {
    HABANA_ASSERT(
        !m_tensor || m_tensor->dim() <= std::numeric_limits<uint32_t>::max(),
        "Tensor dimension exceeds uint32_t limit");
    return m_tensor ? static_cast<uint32_t>(m_tensor->dim()) : m_rank;
  }

  at::ScalarType getType() const {
    return m_tensor ? m_tensor->scalar_type() : m_dtype;
  }

 private:
  const at::Tensor* m_tensor{};
  uint32_t m_rank{};
  at::ScalarType m_dtype{at::ScalarType::Undefined};
};

// We use small-vector-optimization to describe sequence of inputs/outputs
// for tpc kernel. We intentionally avoid std::vector to save 0.1-0.2us.
//
// We set inlined buffer capacity to 5, it should cover all or almost all
// scenarios.
using TensorDescrArray = absl::InlinedVector<TensorDescr, 5>;

} // namespace detail

using OutputMetaFunc =
    std::function<OutputMetaDataVector(const at::Stack& stack)>;
using SharedMetaFunc = std::function<SharedMetaDataVector(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode)>;

struct CheckNodeWithSharedLayerValidator {
  CheckNodeWithSharedLayerValidator(
      const std::string& opname,
      const std::string& guid,
      const std::vector<int>& resIds,
      const std::vector<int>& scalarIds,
      OutputMetaFunc outputMetaFunc,
      const std::vector<int>& typePromotionIds,
      bool promoteIntToFloat,
      bool safeCastCheck,
      bool isInplace,
      bool isOutFn)
      : m_opname(opname),
        m_guid(guid),
        m_resIds(resIds),
        m_scalarIds(scalarIds),
        m_outputMetaFunc(outputMetaFunc),
        m_typePromotionIds(typePromotionIds),
        m_promoteIntToFloat(promoteIntToFloat),
        m_safeCastCheck(safeCastCheck),
        m_isInplace(isInplace),
        m_isOutFn(isOutFn) {}

  CheckNodeWithSharedLayerValidator(
      const std::string& opname,
      SharedMetaFunc sharedMetaFunc,
      habana_helpers::HabanaExecutionMode executionMode)
      : m_opname(opname),
        m_sharedMetaFunc(sharedMetaFunc),
        m_executionMode(executionMode) {}

  bool Validate(
      const at::Stack& values,
      const bool is_dynamic = false,
      const bool check_st_h2d = false,
      const SharedMetaVector& meta = {},
      std::optional<SharedLayer::DeviceId> device_stub = std::nullopt);

  bool ValidateCustom(
      const at::Stack& values,
      const bool is_dynamic = false,
      const bool check_st_h2d = false,
      std::optional<SharedLayer::DeviceId> device_stub = std::nullopt);

  bool IsRequireH2D() const {
    return m_require_h2d;
  }

  bool IsRequireST() const {
    return m_require_st;
  }

 private:
  at::ScalarType ComputePromotedType(const at::Stack& values);

  detail::TensorDescrArray CreateInputList(
      const at::Stack& values,
      at::ScalarType resultType,
      const size_t outs_num);

  std::string m_opname;
  std::string m_guid;
  std::vector<int> m_resIds;
  std::vector<int> m_scalarIds;
  OutputMetaFunc m_outputMetaFunc;
  SharedMetaFunc m_sharedMetaFunc;
  std::vector<int> m_typePromotionIds;
  habana_helpers::HabanaExecutionMode m_executionMode;
  bool m_promoteIntToFloat;
  bool m_safeCastCheck;
  bool m_isInplace;
  bool m_isOutFn;
  bool m_require_h2d = false;
  bool m_require_st = false;
};

struct SharedLayerGuidValidator {
  SharedLayerGuidValidator(
      const std::string& guid,
      const detail::TensorDescrArray& input_values,
      const detail::TensorDescrArray& output_values,
      const bool is_dynamic = false,
      bool valid_shape_tensor = true,
      bool valid_h2d_tensor = true)
      : m_valid_shape_tensor(valid_shape_tensor),
        m_valid_h2d_tensor(valid_h2d_tensor),
        m_guid(guid),
        m_input_values(input_values),
        m_output_values(output_values),
        m_is_dynamic(is_dynamic) {}
  SharedLayerGuidValidator(
      const std::string& guid,
      const detail::TensorDescrArray& input_values,
      const detail::TensorDescrArray& output_values,
      const SharedMetaData::SharedMetaValidationOptions& options,
      const bool is_dynamic = false,
      bool valid_shape_tensor = true,
      bool valid_h2d_tensor = true)
      : m_valid_shape_tensor(valid_shape_tensor),
        m_valid_h2d_tensor(valid_h2d_tensor),
        m_guid(guid),
        m_input_values(input_values),
        m_output_values(output_values),
        m_options(options),
        m_is_dynamic(is_dynamic) {}

  SharedLayer::Return_t ValidateGuid();
  // please check SharedLayer::Queries_t for bit definition.
  SharedLayer::Return_t QueryGuid(
      const unsigned query_bit_map,
      unsigned* result_bit_map,
      std::optional<SharedLayer::DeviceId> device_stub = std::nullopt);

  bool m_valid_shape_tensor;
  bool m_valid_h2d_tensor;

 private:
  const std::string m_guid;
  const detail::TensorDescrArray m_input_values;
  const detail::TensorDescrArray m_output_values;
  const SharedMetaData::SharedMetaValidationOptions m_options;
  const bool m_is_dynamic;
  bool fillSharedLayerTensorType(
      SharedLayer::Tensor& tensor,
      const at::ScalarType& t);
  bool fillGuidParamInfo(
      SharedLayer::Tensor& tensor,
      const detail::TensorDescr& tensor_descr);
  template <typename T>
  bool fillParam(T& params, SharedLayer::DeviceId deviceId);
};

} // namespace habana
