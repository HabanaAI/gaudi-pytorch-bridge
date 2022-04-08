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
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class MatmulBwd : public Node {
 public:
  MatmulBwd() = delete;
  MatmulBwd(
      const at::Tensor& grad_output,
      const at::Tensor& self,
      const at::Tensor& other)
      : Node(c10::Symbol::fromQualString("aten::matmul_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHPU);
    hl_grad_output =
        HbLazyTensorViews::HandleViewsOrUpdate(grad_output, hl_grad_output);
    AddInput(hl_grad_output.GetIrValue());
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    AddInput(hl_self.GetIrValue());
    auto hl_other = GetOrCreateHbLazyTensor(other, c10::kHPU);
    hl_other = HbLazyTensorViews::HandleViewsOrUpdate(other, hl_other);
    AddInput(hl_other.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, self, other};
    AddInputPtTensors(input_pt_vec);
  }
};

}; // namespace ir
}; // namespace habana_lazy
