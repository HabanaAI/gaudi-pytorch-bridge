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
#include "backend/habana_device/HPUDevice.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"

namespace habana_lazy {

namespace {

at::Tensor create_h2d_scale_tensor(
    void* scale_ptr,
    at::ScalarType dtype,
    void** alloc_pointer = nullptr,
    void** h2d_pointer = nullptr) {
  auto scale_tensor = habana_lazy::empty_hpu_lazy(
      {1}, dtype, std::nullopt, false, HOST_TO_DEVICE_TENSOR);

  auto hl_scale_tensor =
      habana_lazy::GetOrCreateHbLazyTensor(scale_tensor, at::kHPU);
  auto hl_scale_tensor_internal =
      hl_scale_tensor.CurrentTensorAttached().value();
  auto tmeta{habana::get_tensor_extra_meta(hl_scale_tensor_internal)};
  const auto is_float = dtype == at::ScalarType::Float;
  const auto scale_value_size =
      is_float ? sizeof(float_t) : sizeof(at::BFloat16);
  const auto dt_type = is_float ? habana::HostDataType::FLOAT_T
                                : habana::HostDataType::BFLOAT16_T;

  if (nullptr == alloc_pointer and nullptr == h2d_pointer) {
    // Allocate memory for a single H2D scale tensor. This is the case for scale
    // tensors added to cache in runtime.
    tmeta->set_host_data(scale_ptr, 1, scale_value_size, dt_type);
  } else {
    // Use memory from preallocated chunk. This is the case for the initial
    // cache of H2D scales tensors created at startup. Set the host and compile
    // pointer from preallocated chunk and increment the h2d_pointer to point to
    // end of current H2D.
    const auto host_total_elem = 2 * scale_value_size;

    tmeta->set_host_size(1);
    tmeta->set_host_el_size(scale_value_size);
    tmeta->set_host_dt_type(dt_type);
    tmeta->set_host_total_elem(host_total_elem);
    tmeta->set_alloc_ptr(*alloc_pointer);
    tmeta->set_host_ptr(*h2d_pointer);
    char* ptr = static_cast<char*>(*h2d_pointer) + host_total_elem;
    *h2d_pointer = static_cast<char*>(ptr + host_total_elem);
    tmeta->set_compile_host_ptr(ptr);
    tmeta->update_host_data(scale_ptr, 1, scale_value_size, true);
  }

  return scale_tensor;
}

at::Tensor create_h2d_scale_tensor(const at::Tensor& scale_tensor) {
  return create_h2d_scale_tensor(
      scale_tensor.data_ptr(), scale_tensor.scalar_type());
}

/**
 * Creates H2D tensors with all possible hw-aligned scales and their inversions,
 * so they are picked in runtime by fp8 ops supporting H2D scales instead of
 * being created each time. Default exp_bias is 7 (only torch.float8_e4m3fn is
 * supported). Possible exp_bias for gaudi2 is [3, 7, 11, 15], so -1 is added as
 * an inversion of 15. Possible exp_bias for gaudi3 is [0, 63], so with
 * inversions it's [-49, 63]
 */
bool create_h2d_scale_tensors() {
  const auto is_gaudi2 =
      habana::HPUDeviceContext::get_device().type() == synDeviceGaudi2;
  auto& h2d_scales_map = habana_lazy::get_device_lazy_execution_context()
                             ->getScalarToH2dScalesMapRef();
  static constexpr int default_bias = 7;
  std::vector<int> biases;
  if (is_gaudi2) {
    biases = {-1, 3, 7, 11, 15};
  } else {
    biases.resize(113);
    std::iota(biases.begin(), biases.end(), -49);
  }

  static constexpr size_t float_size = sizeof(float_t);
  static constexpr size_t bfloat_size = sizeof(at::BFloat16);
  const size_t h2d_memory_required =
      4 * biases.size() * (float_size + bfloat_size);

  // Allocate the total H2D required in single chunk.
  void* alloc_pointer{nullptr};
  auto& device = habana::HPUDeviceContext::get_device();
  device.get_host_memory().uncached_malloc(&alloc_pointer, h2d_memory_required);
  void* h2d_pointer{alloc_pointer};

  // Stores vector of preallocated H2D tensors and the idx of tensor that should
  // be used next, for each possible hw-aligned scale value. If necessary, new
  // H2D tensors are added to vector in runtime. After mark_step, indices are
  // updated, so all newly allocated tensors are available to pick. For now we
  // preallocate one H2D tensor for each scale value, but if experiments prove
  // more is needed, it will be increased in future.
  auto insert_scales_into_map = [&h2d_scales_map, &alloc_pointer, &h2d_pointer](
                                    const double scale_value,
                                    void* scale_ptr,
                                    const at::ScalarType dtype) {
    std::pair<double, at::ScalarType> key{scale_value, dtype};
    ScalesIdxPair scales_and_idx{
        {create_h2d_scale_tensor(
            scale_ptr, dtype, &alloc_pointer, &h2d_pointer)},
        0};
    h2d_scales_map.emplace(std::move(key), std::move(scales_and_idx));
  };

  for (const auto bias : biases) {
    double scale_f64 = std::pow(2.0, default_bias - bias);
    auto scale_f32 = static_cast<float>(scale_f64);

    // at::BFloat16 is stored internally in uint16_t.
    auto scale_bf16 = at::BFloat16(scale_f32);

    insert_scales_into_map(scale_f64, &scale_f32, at::ScalarType::Float);
    insert_scales_into_map(scale_f64, &scale_bf16, at::ScalarType::BFloat16);
  }
  return true;
}

at::Tensor create_h2d_scale(const at::Tensor& scale) {
  // This happens only once, during the first execution of create_h2d_scale.
  [[maybe_unused]] static bool h2d_scales_created = create_h2d_scale_tensors();

  const auto dtype = scale.scalar_type();
  auto& h2d_scales_map = habana_lazy::get_device_lazy_execution_context()
                             ->getScalarToH2dScalesMapRef();
  if (h2d_scales_map.count({scale.item().toDouble(), dtype}) == 1) {
    auto& [scales_vec, current_idx] =
        h2d_scales_map.at({scale.item().toDouble(), dtype});
    if (current_idx >= 0) {
      PT_BRIDGE_DEBUG("H2D scale taken from cache, current idx: ", current_idx);
      return scales_vec[current_idx--];
    }
    const auto new_scale = create_h2d_scale_tensor(scale);
    scales_vec.push_back(new_scale);
    PT_BRIDGE_DEBUG(
        "H2D scale added to cache, current size: ", scales_vec.size());
    return new_scale;
  }
  PT_BRIDGE_DEBUG("H2D scale created from non hw-aligned value.");

  return create_h2d_scale_tensor(scale);
}

bool is_cpu_float_bfloat_0d_tensor(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() and tensor.value().defined() and
      tensor->device().is_cpu() and
      (tensor->scalar_type() == at::ScalarType::Float or
       tensor->scalar_type() == at::ScalarType::BFloat16) and
      tensor->dim() == 0;
}

} // namespace

std::optional<at::Tensor> maybe_convert_tensor_to_h2d(
    const std::optional<at::Tensor>& tensor,
    const std::string_view op_name) {
  if (is_cpu_float_bfloat_0d_tensor(tensor)) {
    PT_BRIDGE_DEBUG(
        "CPU scale of op ",
        op_name,
        " was converted to H2D tensor with value=",
        tensor->item().toDouble());
    return create_h2d_scale(tensor.value());
  }
  PT_BRIDGE_WARN(
      "H2D scales flow is enabled, but op ",
      op_name,
      " received non cpu-float-0D scale.");
  return tensor;
}

std::optional<at::Tensor> maybe_convert_tensor_to_h2d(
    const std::optional<at::Tensor>& tensor,
    const bool enabled,
    const std::string_view op_name) {
  if (enabled) {
    return maybe_convert_tensor_to_h2d(tensor, op_name);
  }
  return tensor;
}

std::vector<at::Tensor> maybe_convert_list_to_h2d(
    const at::TensorList& tensors,
    const std::string_view op_name) {
  std::vector<at::Tensor> converted_tensors;
  converted_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (is_cpu_float_bfloat_0d_tensor(tensor)) {
      PT_BRIDGE_DEBUG(
          "CPU scale of op ",
          op_name,
          " was converted to H2D tensor with value=",
          tensor.item().toDouble());
      converted_tensors.push_back(create_h2d_scale(tensor));
    } else {
      PT_BRIDGE_WARN(
          "H2D scales flow is enabled, but op ",
          op_name,
          " received non cpu-float-0D scale.");
      converted_tensors.push_back(tensor);
    }
  }
  return converted_tensors;
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
