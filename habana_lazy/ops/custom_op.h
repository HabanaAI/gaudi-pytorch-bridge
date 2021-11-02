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
        AddInput(lazy_input.GetIrValue());
        input_pt_vec.emplace_back(input.toTensor());
      } else {
        TORCH_CHECK(false, "Custom op supports only Tensor inputs");
      }
    }
    AddInputPtTensors(input_pt_vec);
  }
};

}; // namespace ir
}; // namespace habana_lazy