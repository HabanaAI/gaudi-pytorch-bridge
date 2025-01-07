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
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class FusedNorm : public ir::Node {
 public:
  enum class FusedNormMeta {
    NORM_TYPE_INDEX = 2,
  };
  FusedNorm() = delete;
  FusedNorm(
      std::vector<at::Tensor>& grad,
      const at::Tensor& max_norm,
      float norm_type,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    AddInputVec(grad);

    auto hl_max_norm = GetOrCreateHbLazyTensor(max_norm, c10::kHPU);
    hl_max_norm = HbLazyTensorViews::HandleViewsOrUpdate(max_norm, hl_max_norm);
    AddInput(hl_max_norm.GetIrValue());

    m_meta_data.set(
        norm_type, static_cast<size_t>(FusedNormMeta::NORM_TYPE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", norm_type = "
       << m_meta_data.get(static_cast<size_t>(FusedNormMeta::NORM_TYPE_INDEX));
    return ss.str();
  }

 private:
  void AddInputVec(std::vector<at::Tensor>& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
      hl_tensor = HbLazyTensorViews::HandleViewsOrUpdate(t, hl_tensor);
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(t);
    }

    auto input = GetIrValueForListConstruct(hl_tensors);
    input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(input);
  }
};

}; // namespace ir
}; // namespace habana_lazy
