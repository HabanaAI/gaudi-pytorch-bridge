/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/Tensor.h>

#include "backend/backend_meta.h"

namespace habana {
namespace graph {
struct PermuteWeightTensor {
  explicit PermuteWeightTensor(const torch::Tensor& weight);
  void PermuteIfNeeded();

 private:
  bool ShouldPermuteWeight();
  template <typename T>
  void PermuteDataToRSCK(const torch::Tensor& weight_cpu);
  template <typename T>
  void PermuteDataToQRSCK(const torch::Tensor& weight_cpu);
  const torch::Tensor& m_weight;
  int64_t m_tensor_dim;
  StorageExtraMeta* m_storage_meta;
};

} // namespace graph
} // namespace habana