/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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