/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_kernels/habana_operator.h"
namespace habana {
struct HabanaOperatorHelper : public HabanaOperator {
  HabanaOperatorHelper(int device_id = 0, const std::string& guid = {})
      : HabanaOperator(guid) {
    CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
};

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors);
std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);
} // namespace habana

#define HPU_SUPPORTED_DTYPES(fn, supported_dtypes)                       \
  const static std::unordered_set<c10::ScalarType> fn##_supported_dtypes \
      supported_dtypes;

#define FALLBACK_IF_UNSUPPORTED_DTYPE(tensor, fn, args...)       \
  if (ABSL_PREDICT_FALSE(                                        \
          tensor.defined() &&                                    \
          !fn##_supported_dtypes.count(tensor.scalar_type()))) { \
    return AtenHpuTypeDefault::fn(args);                         \
  }
