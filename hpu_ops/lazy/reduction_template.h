/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include "hpu_ops/common/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

template <typename T>
class ReductionFrontendTemplate : public habana_lazy::LazyOp<T> {
  at::optional<uint8_t> m_dtype_index;
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;

 public:
  ReductionFrontendTemplate(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
      : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn, -1) {}

  ReductionFrontendTemplate(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const sizes_vec& out_shapes)
      : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes, -1) {}

  T get_result_overrideable() override;

  void SetReductionVarsIndices(
      at::optional<uint8_t> dim_index,
      at::optional<uint8_t> keepdim_index,
      at::optional<uint8_t> dtype_index) {
    m_dim_index = dim_index;
    m_keepdim_index = keepdim_index;
    m_dtype_index = dtype_index;
  }
};
} // namespace habana
