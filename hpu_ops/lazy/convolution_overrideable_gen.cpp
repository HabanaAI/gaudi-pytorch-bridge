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
 *******************************************************************************
 */
#include "generated/lazy/convolution_overrideable.h"
#include "habana_lazy/permute_tensors.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    ConvolutionOverrideableFE,
    at::Tensor) {
  auto weight = inputs[1].toTensor();
  at::Tensor weight_hpu = weight;

  habana_lazy::PermuteTensors::permuteWeight(weight_hpu);

  if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)) {
    auto bias = inputs[2].toTensor();

    if (!bias.defined()) {
      c10::IntArrayRef rm_size;
      if (weight_hpu.suggest_memory_format() ==
          c10::MemoryFormat::ChannelsLast) {
        rm_size = weight_hpu.sizes()[3];
      } else if (
          weight_hpu.suggest_memory_format() ==
          c10::MemoryFormat::ChannelsLast3d) {
        rm_size = weight_hpu.sizes()[4];
      } else {
        rm_size = weight_hpu.sizes()[0];
      }

      auto options =
          torch::TensorOptions()
              .dtype(
                  weight_hpu.scalar_type() == c10::ScalarType::BFloat16
                      ? c10::ScalarType::BFloat16
                      : c10::ScalarType::Float)
              .device(torch::kCPU)
              .requires_grad(false);
      at::Tensor bias_temp = torch::zeros(rm_size, options);
      bias_temp = bias_temp.to(c10::kHPU, true);

      get_inputs()[2] = bias_temp;
    }
  }
}

} // namespace habana
