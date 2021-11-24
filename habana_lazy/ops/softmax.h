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

enum class LogSoftMaxParams { DIM_INDEX_FWD = 1, HALF_TO_FLOAT };

struct LogSoftMax : public ir::Node {
  LogSoftMax() = delete;
  LogSoftMax(
      const at::Tensor& self,
      const int64_t dim,
      const bool half_to_float,
      const at::string& aten_op)
      : Node(c10::Symbol::fromQualString(aten_op)) {
    auto hl_self = habana_lazy::GetHbLazyTensor(self);

    hl_self = HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(LogSoftMaxParams::DIM_INDEX_FWD));
    m_meta_data.set(
        half_to_float, static_cast<size_t>(LogSoftMaxParams::HALF_TO_FLOAT));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
      << ", dim= "
      << m_meta_data.get(static_cast<size_t>(LogSoftMaxParams::DIM_INDEX_FWD))
      << ", half_to_float= "
      << m_meta_data.get(static_cast<size_t>(LogSoftMaxParams::HALF_TO_FLOAT));
    return ss.str();
  }
};

struct LogSoftMaxBackward : public ir::Node {
  LogSoftMaxBackward() = delete;
  LogSoftMaxBackward(
      const at::Tensor& grad,
      const at::Tensor& output,
      int64_t dim,
      const at::Tensor& input,
      const at::string& aten_op)
      : Node(c10::Symbol::fromQualString(aten_op)), m_dim_index_bwd{2} {
    auto hl_grad = habana_lazy::GetHbLazyTensor(grad);
    auto hl_output = habana_lazy::GetHbLazyTensor(output);
    auto hl_input = habana_lazy::GetHbLazyTensor(input);

    hl_grad = HandleViewsOrUpdate(grad, hl_grad);
    hl_output = HandleViewsOrUpdate(output, hl_output);
    hl_input = HandleViewsOrUpdate(input, hl_input);

    AddInput(hl_grad.GetIrValue());
    AddInput(hl_output.GetIrValue());
    AddInput(hl_input.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad, output, input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, m_dim_index_bwd);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim= " << m_meta_data.get(m_dim_index_bwd);
    return ss.str();
  }

 private:
  const int m_dim_index_bwd;
};
} // namespace ir
} // namespace habana_lazy
