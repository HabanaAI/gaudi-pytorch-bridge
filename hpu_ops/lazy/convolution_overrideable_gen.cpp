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
#include "habana_lazy/view_utils.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    ConvolutionOverrideableFE,
    at::Tensor) {
  auto weight = inputs[1].toTensor();
  // WA: detect the pattern and skip view handling and weight permuting to avoid
  // the graph break, see details at: SW-162183.
  // WA: do not perform view handling and weight permuting for half dtype as it
  // causes accuracy issues. At the same time, removing permutation for other
  // dtypes causes perf drops, see details: SW-177711, SW-111276
  bool need_skip = false;
  if (inputs[0].toTensor().scalar_type() == c10::ScalarType::Half) {
    need_skip = true;
  } else if (weight.dim() == 4) {
    auto weight_hb_lazy_tensor = habana_lazy::GetHbLazyTensor(weight);
    auto weight_stride_params =
        weight_hb_lazy_tensor.getDataPtr()->stride_params;
    if (weight_stride_params.has_value() &&
        weight_stride_params.value().optype ==
            habana_lazy::StridedOPType::kStridedOpUnsqueeze) {
      // 3D weight
      auto base = habana_lazy::HbLazyTensorViews::get_recent_base_tensor(
          weight_stride_params.value().base);
      const auto base_hb_lazy_tensor = habana_lazy::GetHbLazyTensor(base);
      const auto& ir_value = base_hb_lazy_tensor.GetIrValue();
      const auto& ir_node = ir_value.mp_node;
      const auto& ir_op = ir_node->op();
      if (strcmp(ir_op.toQualString(), "hpu::cast") == 0) {
        const auto& ir_inputs = ir_node->GetInputs();
        const auto& ir_weight_norm_value = ir_inputs[0];
        const auto& ir_weight_norm_node = ir_weight_norm_value.mp_node;
        const auto& ir_weight_norm_op = ir_weight_norm_node->op();
        if (strcmp(
                ir_weight_norm_op.toQualString(),
                "aten::_weight_norm_interface") == 0) {
          need_skip = true;
          PT_LAYOUTS_DEBUG(
              "Detected pattern, skipping view handling and weight",
              " permuting to avoid graph break.");
        }
      }
    }
  }

  if (!need_skip) {
    at::Tensor weight_hpu =
        habana_lazy::HbLazyTensorViews::HandleViewsD2H(weight);

    habana_lazy::PermuteTensors::permuteWeight(weight_hpu);
  }
}

} // namespace habana
