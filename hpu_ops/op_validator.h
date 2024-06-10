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
#include <absl/container/inlined_vector.h>
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
      : TensorDescr(output_meta.shape.size(), output_meta.dtype) {}

  uint32_t getRank() const {
    return m_tensor ? m_tensor->dim() : m_rank;
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

using FillNodeParams =
    std::function<std::shared_ptr<void>(const at::Stack& stack, size_t&)>;
using OutputMetaFunc =
    std::function<OutputMetaDataVector(const at::Stack& stack)>;

struct CheckNodeWithSharedLayerValidator {
  CheckNodeWithSharedLayerValidator(
      const std::string& opname,
      const std::string& guid,
      const std::vector<int>& resIds,
      const std::vector<int>& scalarIds,
      OutputMetaFunc outputMetaFunc,
      FillNodeParams fillNodeParamsFunc,
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
        m_fillNodeParamsFunc(fillNodeParamsFunc),
        m_typePromotionIds(typePromotionIds),
        m_promoteIntToFloat(promoteIntToFloat),
        m_safeCastCheck(safeCastCheck),
        m_isInplace(isInplace),
        m_isOutFn(isOutFn) {}

  bool Validate(
      at::ScalarType,
      const std::vector<at::IValue>& values,
      bool is_dynamic = false);
  bool Validate(
      const at::Tensor&,
      const std::vector<at::IValue>& values,
      bool is_dynamic = false);

 private:
  bool ValidateWithSharedLayer(
      const std::vector<at::IValue>& values,
      bool is_dynamic = false);
  at::ScalarType ComputePromotedType(const at::Stack& values);

  detail::TensorDescrArray CreateInputList(
      const at::Stack& values,
      at::ScalarType resultType,
      const size_t outs_num);

  detail::TensorDescrArray CreateOutputList(const OutputMetaDataVector& meta);

  std::string m_opname;
  std::string m_guid;
  std::vector<int> m_resIds;
  std::vector<int> m_scalarIds;
  OutputMetaFunc m_outputMetaFunc;
  FillNodeParams m_fillNodeParamsFunc;
  std::vector<int> m_typePromotionIds;
  bool m_promoteIntToFloat;
  bool m_safeCastCheck;
  bool m_isInplace;
  bool m_isOutFn;
};

} // namespace habana
