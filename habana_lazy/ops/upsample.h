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
class UpsampleNearest2d : public Node {
  enum class UpsampleNearest2dIndex { OUTPUT_SIZE_INDEX = 1, SCALE_INDEX };

 public:
  UpsampleNearest2d() = delete;
  UpsampleNearest2d(
      const at::Tensor& input,
      c10::optional<at::IntArrayRef> output_size,
      c10::optional<at::ArrayRef<double>> scale_factors)
      : Node(c10::Symbol::fromQualString("aten::upsample_nearest2d")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HandleViewsOrUpdate(input, hl_input);
    auto ir_value = hl_input.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        output_size,
        static_cast<size_t>(UpsampleNearest2dIndex::OUTPUT_SIZE_INDEX));
    m_meta_data.set(
        scale_factors,
        static_cast<size_t>(UpsampleNearest2dIndex::SCALE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", output_size="
       << m_meta_data.get(
              static_cast<size_t>(UpsampleNearest2dIndex::OUTPUT_SIZE_INDEX))
       << ", scale_factor="
       << m_meta_data.get(
              static_cast<size_t>(UpsampleNearest2dIndex::SCALE_INDEX));
    return ss.str();
  }
};

class UpsampleNearest2dBackward : public Node {
  enum class UpsampleNearest2dBackwardIndex {
    OUTPUT_SIZE_INDEX = 1,
    INPUT_SIZE_INDEX,
    SCALE_INDEX
  };

 public:
  UpsampleNearest2dBackward() = delete;
  UpsampleNearest2dBackward(
      const at::Tensor& grad_output,
      c10::optional<at::IntArrayRef> output_size,
      at::IntArrayRef input_size,
      c10::optional<at::ArrayRef<double>> scale_factors)
      : Node(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)
                ? c10::Symbol::fromQualString(
                      "hpu::upsample_nearest2d_backward")
                : c10::Symbol::fromQualString(
                      "aten::upsample_nearest2d_backward")) {
    auto hl_input = GetOrCreateHbLazyTensor(grad_output, c10::kHPU);
    hl_input = HandleViewsOrUpdate(grad_output, hl_input);
    auto ir_value = hl_input.GetIrValue();
    AddInput(ir_value);
    std::vector<at::Tensor> input_pt_vec{grad_output};

    m_meta_data.set(
        output_size,
        static_cast<size_t>(UpsampleNearest2dBackwardIndex::OUTPUT_SIZE_INDEX));
    m_meta_data.set(
        scale_factors,
        static_cast<size_t>(UpsampleNearest2dBackwardIndex::SCALE_INDEX));

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      auto input_shape = empty_hpu_lazy(
          input_size,
          grad_output.options(),
          grad_output.suggest_memory_format(),
          false,
          SHAPE_TENSOR);
      auto hl_input_shape = GetOrCreateHbLazyTensor(input_shape, c10::kHPU);
      AddInput(hl_input_shape.GetIrValue());
      input_pt_vec.emplace_back(input_shape);
    } else {
      m_meta_data.set(
          input_size,
          static_cast<size_t>(
              UpsampleNearest2dBackwardIndex::INPUT_SIZE_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", output_size="
       << m_meta_data.get(static_cast<size_t>(
              UpsampleNearest2dBackwardIndex::OUTPUT_SIZE_INDEX))
       << ", scale_factor="
       << m_meta_data.get(
              static_cast<size_t>(UpsampleNearest2dBackwardIndex::SCALE_INDEX));

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      HABANA_ASSERT(m_inputs.size() == 2);
      auto& input_shape = m_inputs[1];
      if (input_shape.DataPtrValidAndNotExpired()) {
        std::shared_ptr<Data> data = input_shape.m_data_ptr.lock();
        ss << ", input_size=" << data->sizes;
      }
    } else {
      ss << ", input_size="
         << m_meta_data.get(static_cast<size_t>(
                UpsampleNearest2dBackwardIndex::INPUT_SIZE_INDEX));
    }
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
