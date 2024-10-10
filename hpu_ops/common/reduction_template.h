/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/ReduceOpsUtils.h>
#include "habana_helpers/logging.h"
#include "hpu_ops/op_backend.h"

namespace habana {

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim);

unsigned ReductionMask(const at::Tensor& self, at::optional<int64_t> dimOpt);

at::optional<at::ScalarType> get_dtype(
    const at::Stack& stack,
    at::optional<uint8_t> dtype_index);

std::vector<int64_t> get_dims(
    const at::Stack& stack,
    at::optional<uint8_t> dim_index);

inline bool get_keepdim(
    const at::Stack& stack,
    at::optional<uint8_t> keepdim_index) {
  return keepdim_index.has_value() ? stack.at(keepdim_index.value()).toBool()
                                   : false;
}

inline std::pair<unsigned, int>
getMaskWithBitPosOutInTpcOrderAndBitPosInTpcOrder(int bitPos, int ndims) {
  int bitPosInTpcOrder = ndims - 1 - bitPos;
  unsigned fullMask = (1 << ndims) - 1;

  unsigned maskBitPosInTpcOrder =
      (bitPosInTpcOrder >= 0) ? 1 << bitPosInTpcOrder : 0;
  unsigned maskedOutBitPos = fullMask & ~maskBitPosInTpcOrder;

  return {maskedOutBitPos, bitPosInTpcOrder};
}

inline unsigned getMaskWithBitPosOutInTpcOrder(int bitPos, int ndims) {
  return getMaskWithBitPosOutInTpcOrderAndBitPosInTpcOrder(bitPos, ndims).first;
}

struct CommonReductionFrontendTemplate {
  at::optional<uint8_t> m_dtype_index;
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;

  void SetReductionVarsIndices(
      at::optional<uint8_t> dim_index,
      at::optional<uint8_t> keepdim_index,
      at::optional<uint8_t> dtype_index) {
    m_dim_index = dim_index;
    m_keepdim_index = keepdim_index;
    m_dtype_index = dtype_index;
  }
};

#define HPU_REDUCTION_TEMPLATE_FRONTEND(OpClass)                             \
  template <typename T>                                                      \
  class ReductionFrontendTemplate : public OpClass<T>,                       \
                                    public CommonReductionFrontendTemplate { \
   public:                                                                   \
    ReductionFrontendTemplate(                                               \
        const std::string& qualstring,                                       \
        std::vector<at::IValue>&& inputs,                                    \
        const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)     \
        : OpClass<T>(qualstring, std::move(inputs), out_shapes_fn, -1) {}    \
    ReductionFrontendTemplate(                                               \
        const std::string& qualstring,                                       \
        std::vector<at::IValue>&& inputs,                                    \
        sizes_vec&& out_shapes)                                              \
        : OpClass<T>(                                                        \
              qualstring,                                                    \
              std::move(inputs),                                             \
              std::move(out_shapes),                                         \
              -1) {}                                                         \
    T get_result_overrideable() override;                                    \
  };

#define HPU_REDUCTION_TEMPLATE_FRONTEND_LAZY(OpClass)                        \
  template <typename T>                                                      \
  class ReductionFrontendTemplate : public OpClass<T>,                       \
                                    public CommonReductionFrontendTemplate { \
   public:                                                                   \
    ReductionFrontendTemplate(                                               \
        const std::string& qualstring,                                       \
        const std::vector<at::IValue>& inputs,                               \
        const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)     \
        : OpClass<T>(qualstring, inputs, out_shapes_fn, -1) {}               \
    ReductionFrontendTemplate(                                               \
        const std::string& qualstring,                                       \
        const std::vector<at::IValue>& inputs,                               \
        const sizes_vec& out_shapes)                                         \
        : OpClass<T>(qualstring, inputs, out_shapes, -1) {}                  \
    T get_result_overrideable() override;                                    \
  };

} // namespace habana
