/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <absl/container/inlined_vector.h>
#include "hpu_ops/op_backend.h"
#include "hpu_ops/supported_dtypes.h"

#pragma once

namespace habana {

namespace detail {

// Lightweight alternative to at::IValue
// It is either
//   * pointer to at::Tensor
//   * array of integers describing tensor shape, the last element of the array
//     is at::ScalarType casted to int64
struct TensorDescr {
  TensorDescr() = default;

  explicit TensorDescr(const at::Tensor* tensor) : m_tensor(tensor) {}

  explicit TensorDescr(std::vector<int64_t>&& dims_and_type)
      : m_dims_and_type(std::move(dims_and_type)) {}

  bool isTensor() const {
    return m_tensor != nullptr;
  }

  const at::Tensor* m_tensor = nullptr;
  std::vector<int64_t> m_dims_and_type;
};

// We use small-vector-optimization to describe sequence of inputs/outputs
// for tpc kernel. We intentionally avoid std::vector to save 0.1-0.2us.
//
// We set inlined buffer capacity to 5, it should cover all or almost all
// scenarios.
using TensorDescrArray = absl::InlinedVector<TensorDescr, 5>;

} // namespace detail

using FillNodeParams =
    std::function<std::shared_ptr<void>(const at::Stack& stack, size_t&)>;
using OutputShapeFunc = std::function<sizes_vec(const at::Stack& stack)>;

struct CheckNodeWithSharedLayerValidator {
  CheckNodeWithSharedLayerValidator(
      const std::string& opname,
      const std::string& guid,
      OutputShapeFunc outputShapeFunc,
      FillNodeParams fillNodeParamsFunc,
      bool typePromotion,
      bool promoteIntToFloat,
      bool safeCastCheck,
      bool isInplace,
      bool isOverload,
      SupportedDtypes supportedDtypes)
      : m_opname(opname),
        m_guid(guid),
        m_outputShapeFunc(outputShapeFunc),
        m_fillNodeParamsFunc(fillNodeParamsFunc),
        m_typePromotion(typePromotion),
        m_promoteIntToFloat(promoteIntToFloat),
        m_safeCastCheck(safeCastCheck),
        m_isInplace(isInplace),
        m_isOutFn(isOverload),
        m_supportedDtypes(std::move(supportedDtypes)) {}

  bool Validate(
      at::ScalarType compute_type,
      const std::vector<at::IValue>& values);
  bool Validate(const at::Tensor&, const std::vector<at::IValue>& values);

 private:
  bool ValidateWithSharedLayer(
      at::ScalarType compute_type,
      const std::vector<at::IValue>& values);
  bool ValidateWithDTypes(
      at::ScalarType compute_type,
      const std::vector<at::IValue>& values);
  at::ScalarType ComputePromotedType(const std::vector<at::IValue>& values);
  detail::TensorDescrArray CreateRegularInputList(
      const std::vector<at::IValue>& values);
  detail::TensorDescrArray CreateTypePromotionInputList(
      const std::vector<at::IValue>& values,
      at::ScalarType resultType);

  detail::TensorDescrArray CreateInputList(
      const std::vector<at::IValue>& values,
      at::ScalarType resultType);

  detail::TensorDescrArray CreateRegularOutputList(
      const std::vector<at::IValue>& values);
  detail::TensorDescrArray CreateTypePromotionOutputList(
      const std::vector<at::IValue>& values,
      at::ScalarType resultType);

  detail::TensorDescrArray CreateOutputList(
      const std::vector<at::IValue>& values,
      at::ScalarType resultType);

  std::string m_opname;
  std::string m_guid;
  OutputShapeFunc m_outputShapeFunc;
  FillNodeParams m_fillNodeParamsFunc;
  bool m_typePromotion;
  bool m_promoteIntToFloat;
  bool m_safeCastCheck;
  bool m_isInplace;
  bool m_isOutFn;
  SupportedDtypes m_supportedDtypes;
};

} // namespace habana
