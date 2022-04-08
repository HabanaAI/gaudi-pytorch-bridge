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

#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class CustomOp : public ir::Node {
  using Inputs = std::vector<c10::IValue>;
  using Outputs = std::vector<c10::IValue>;

 public:
  CustomOp() = delete;
  CustomOp(std::string qual_strings, const Inputs& inputs)
      : Node(c10::Symbol::fromQualString(qual_strings)) {
    std::vector<at::Tensor> input_pt_vec;
    for (auto& input : inputs) {
      if (input.isTensor()) {
        auto lazy_input = habana_lazy::GetHbLazyTensor(input.toTensor());
        lazy_input = HbLazyTensorViews::HandleViewsOrUpdate(
            input.toTensor(), lazy_input);
        AddInput(lazy_input.GetIrValue());
        input_pt_vec.emplace_back(input.toTensor());
      } else if (input.isScalar()) {
        AddInput(GetIrValueForScalar(input.toScalar()));
      } else {
        TORCH_CHECK(false, "Custom op supports only tensor & scalars inputs");
      }
    }
    AddInputPtTensors(input_pt_vec);
  }
};

}; // namespace ir
}; // namespace habana_lazy