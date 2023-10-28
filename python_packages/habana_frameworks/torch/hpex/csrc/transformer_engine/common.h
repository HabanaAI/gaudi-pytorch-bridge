/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 * Changes:
 * - Modified naming of NVTE_ERROR to HPTE_ERROR
 * - Removed unused function declarations
 ******************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "logging.h"
#include "transformer_engine.h"

namespace transformer_engine {

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax_history;
  at::Tensor amax_history_index;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM2_INPUT = 2,
  GEMM2_WEIGHT = 3,
  GEMM3_INPUT = 4,
  GEMM3_WEIGHT = 5,
  GEMM4_INPUT = 6,
  GEMM4_WEIGHT = 7,
  GEMM5_INPUT = 8,
  GEMM5_WEIGHT = 9,
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_OUTPUT2 = 1,
  GRAD_OUTPUT3 = 2,
  GRAD_OUTPUT4 = 3,
  GRAD_OUTPUT5 = 4,
};

} // namespace transformer_engine

transformer_engine::DType getTransformerEngineFP8Type(
    bool e4m3_if_hybrid,
    const std::string& fp8_recipe);

inline at::ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt32:
    case transformer_engine::DType::kFloat32:
      return at::kFloat;
    case transformer_engine::DType::kFloat16:
      return at::kHalf;
    case transformer_engine::DType::kBFloat16:
      return at::kBFloat16;
    case transformer_engine::DType::kByte:
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
      return at::kByte;
    default:
      HPTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(at::ScalarType t) {
  switch (t) {
    case at::kHalf:
      return transformer_engine::DType::kFloat16;
    case at::kFloat:
      return transformer_engine::DType::kFloat32;
    case at::kBFloat16:
      return transformer_engine::DType::kBFloat16;
    default:
      HPTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

#endif // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
