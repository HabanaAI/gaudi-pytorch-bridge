/**
 * Copyright (c) 2025 Intel Corporation
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

#include "habana_kernels/h2d_scales_lazy.h"
#include "backend/backend_meta.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"

namespace habana_lazy {

namespace {

at::Tensor create_h2d_scale(void* scale) {
  auto scale_tensor = habana_lazy::empty_hpu_lazy(
      {1}, at::ScalarType::Float, std::nullopt, false, HOST_TO_DEVICE_TENSOR);

  auto hl_params_shape =
      habana_lazy::GetOrCreateHbLazyTensor(scale_tensor, at::kHPU);
  auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
  auto tmeta{habana::get_tensor_extra_meta(hl_param_internal)};

  tmeta->set_host_data(
      scale, {1}, sizeof(float_t), habana::HostDataType::FLOAT_T);

  return scale_tensor;
}

bool is_cpu_float_0d_tensor(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() and tensor.value().defined() and
      tensor->device().is_cpu() and
      tensor->scalar_type() == at::ScalarType::Float and tensor->dim() == 0;
}

} // namespace

std::optional<at::Tensor> maybe_convert_to_h2d(
    const std::optional<at::Tensor>& tensor,
    const bool enabled,
    const std::string_view op_name) {
  if (enabled) {
    if (is_cpu_float_0d_tensor(tensor)) {
      PT_BRIDGE_DEBUG(
          "CPU scale of op ",
          op_name,
          " was converted to H2D tensor with value=",
          tensor->item().toDouble());
      return create_h2d_scale(tensor->data_ptr());
    } else {
      PT_BRIDGE_WARN(
          "H2D scales flow is enabled, but op ",
          op_name,
          " received non cpu-float-0D scale.");
    }
  }
  return tensor;
}

void verify_no_h2d_scales(
    const std::vector<at::TensorList>& scales_lists,
    std::string_view op_name) {
  for (const auto scales : scales_lists) {
    if (not scales.empty() and scales[0].is_cpu()) {
      HABANA_ASSERT(
          false,
          op_name,
          " doesn't support H2D scales feature yet, but received CPU scales.");
    }
  }
}

} // namespace habana_lazy
