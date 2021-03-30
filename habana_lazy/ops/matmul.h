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
#include "habana_helpers/logging.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class MatmulBwd : public Node {
 public:
  MatmulBwd() = delete;
  MatmulBwd(const Tensor& grad_output, const Tensor& self, const Tensor& other)
      : Node(c10::Symbol::fromQualString("aten::matmul_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    AddInput(hl_grad_output.GetIrValue());
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());
    auto hl_other = GetOrCreateHbLazyTensor(other, c10::kHABANA);
    AddInput(hl_other.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, self, other};
    AddInputPtTensors(input_pt_vec);
  }
};

}; // namespace ir
}; // namespace habana_lazy
