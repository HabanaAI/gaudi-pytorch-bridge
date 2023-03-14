/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/common/arange_gen.h"
#include "generated/lazy/arange.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static int64_t get_arange_depth(
    const c10::Scalar _start,
    const c10::Scalar _end,
    const c10::Scalar _step) {
  const float start = _start.to<float>();
  const float end = _end.to<float>();
  const float step = _step.to<float>();

  TORCH_CHECK(step != 0.0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  int64_t num_elements = static_cast<int64_t>(ceil((end - start) / step));
  return num_elements;
}

static bool can_use_dynamic_shapes(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step) {
  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  return (
      (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
       GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR)) &&
      ((start.isIntegral(false) || can_convert(start)) &&
       (end.isIntegral(false) || can_convert(end)) &&
       (step.isIntegral(false) || can_convert(step))));
}

template <>
ArangeInputs<at::Tensor&>::ArangeInputs(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  auto start = inputs[0].toScalar();
  auto end = inputs[1].toScalar();
  auto step = inputs[2].toScalar();
  auto output = inputs[3].toTensor();

  if (can_use_dynamic_shapes(start, end, step)) {
    std::vector<int32_t> params_vec{start.toInt(), end.toInt(), step.toInt()};
    auto params_shape = habana_lazy::empty_hpu_lazy(
        params_vec.size(),
        output.options(),
        output.suggest_memory_format(),
        false,
        HOST_TO_DEVICE_TENSOR);
    auto hl_params_shape =
        habana_lazy::GetOrCreateHbLazyTensor(params_shape, c10::kHPU);

    auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
    habana_lazy::HbInternalTensorImpl* impl =
        habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
    HABANA_ASSERT(impl);
    impl->set_host_data(
        params_vec.data(),
        params_vec.size(),
        sizeof(int),
        habana::HostDataType::INT32_T);

    // Create a dummy shape tensor for the output, this shape tensor is not
    // added to synapse graph, but only ensures that when we match in bucket
    // we are restricted by the size of the output
    int out_depth = get_arange_depth(start, end, step);
    auto out_shape = c10::DimVector({out_depth});
    auto result_shape = habana_lazy::empty_hpu_lazy(
        out_shape,
        output.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    // Mark this front end shape tensor as it does not need synapse tensor
    auto hl_result_shape =
        habana_lazy::GetOrCreateHbLazyTensor(result_shape, c10::kHPU);
    auto hl_result_shape_internal =
        hl_result_shape.CurrentTensorAttached().value();
    auto stImpl =
        habana_lazy::GetHbInternalTensorImpl(hl_result_shape_internal);
    if (stImpl) {
      stImpl->setH2DFrontEndShapeTensor();
    }
    set_inputs({start, end, step, params_shape, result_shape, inputs[3]});
  } else {
    set_inputs({start, end, step, {}, {}, inputs[3]});
  }
}

template <>
at::Tensor& ArangeInputs<at::Tensor&>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return LazyOp<at::Tensor&>::get_result_overrideable();
}
} // namespace habana
