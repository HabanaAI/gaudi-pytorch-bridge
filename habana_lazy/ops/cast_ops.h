/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
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
/*
 TODO: This is the Original implementation for Cast Operator
       where we pass the output of the cast as part of the input args,
       making this as inplace operator. For Resnet we decided
       to make the inplace cast operator as cast out operator.
       The down side of cast out operator is it cannot give us
       the effect of eager mode, cast out operator would create
       a new pytorch output and use that instead of the one that
       is altready created by the .to operator from pytorch.

       Will eventually enable this class as needed going further.

class Cast : public Node {
 public:
  Cast() = delete;
  Cast(const at::Tensor& self, const at::Tensor& src, bool non_blocking)
      : Node(c10::Symbol::fromQualString("hpu::cast")) {
    auto hl_src = GetOrCreateHbLazyTensor(src, c10::kHPU);
    auto ir_value_src = hl_src.GetIrValue();
    AddInput(ir_value_src);
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    auto ir_value_self = hl_self.GetIrValue();
    AddInput(ir_value_self);
    std::vector<at::Tensor> input_pt_vec{src, self};
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    return ss.str();
  }
};*/

class Cast : public Node {
 public:
  Cast() = delete;
  Cast(const at::Tensor& src, c10::ScalarType type, bool non_blocking)
      : Node(c10::Symbol::fromQualString("hpu::cast")) {
    static_cast<void>(non_blocking);
    auto hl_src = GetHbLazyTensor(src);
    hl_src = HbLazyTensorViews::HandleViewsOrUpdate(src, hl_src);

    auto ir_value_src = hl_src.GetIrValue();
    AddInput(ir_value_src);
    std::vector<at::Tensor> input_pt_vec{src};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(type, 1);
  }
};

}; // namespace ir
}; // namespace habana_lazy
