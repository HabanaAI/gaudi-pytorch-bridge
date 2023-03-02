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
#include <ATen/native/ReduceOpsUtils.h>
#include "habana_helpers/dtype_helpers.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {

class ReductionBackendTemplateCGUID : public OpBackend {
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;
  at::optional<uint8_t> m_dtype_index;

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override;

 public:
  ReductionBackendTemplateCGUID(
      int device_id,
      const std::string& guid,
      at::ScalarType scalar_type,
      std::vector<int> res_ids,
      std::vector<int> inplace_ids,
      std::vector<int> scalar_ids,
      bool is_outfn);

 protected:
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
