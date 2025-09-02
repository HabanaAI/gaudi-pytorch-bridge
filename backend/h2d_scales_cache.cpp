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

#include "h2d_scales_cache.h"
#include "habana_device/HPUDevice.h"
#include "habana_device/HPUGuardImpl.h"
#include "kernel/hpu_habana_launch_op_pt.h"

namespace habana::backend {

/**
 * Creates H2D tensors with all possible hw-aligned scales and their inversions,
 * so they are picked in runtime by fp8 ops supporting H2D scales instead of
 * being created each time. Default exp_bias is 7 (only torch.float8_e4m3fn is
 * supported). Possible exp_bias for gaudi2 is [3, 7, 11, 15], so -1 is added as
 * an inversion of 15. Possible exp_bias for gaudi3 is [0, 63], so with
 * inversions it's [-49, 63]
 */
bool H2dScalesCache::CreateH2dScales() {
  PT_EAGER_TRACE;
  const auto is_gaudi2 =
      habana::HPUDeviceContext::get_device().type() == synDeviceGaudi2;
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
  auto insert_scales_into_map = [this, &alloc_pointer, &h2d_pointer](
                                    const double scale_value,
                                    void* scale_ptr,
                                    const at::ScalarType dtype) {
    std::pair<double, at::ScalarType> key{scale_value, dtype};
    ScalesIdxPair scales_and_idx{
        {CreateH2dTensorScale(scale_ptr, dtype, &alloc_pointer, &h2d_pointer)},
        0};
    h2d_scales_map_.emplace(std::move(key), std::move(scales_and_idx));
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

std::optional<at::Tensor> H2dScalesCache::TryGetH2dScale(
    const at::Tensor& scale_tensor) {
  const auto dtype = scale_tensor.scalar_type();
  auto it = h2d_scales_map_.find({scale_tensor.item().toDouble(), dtype});
  if (it != h2d_scales_map_.end()) {
    auto& [scales_vec, current_idx] = it->second;
    if (current_idx >= 0) {
      PT_BRIDGE_DEBUG("H2D scale taken from cache, current idx: ", current_idx);
      return scales_vec[current_idx--];
    }
    const auto new_scale = CreateH2dTensorScale(scale_tensor.data_ptr(), dtype);
    scales_vec.push_back(new_scale);
    PT_BRIDGE_DEBUG(
        "H2D scale added to cache, current size: ", scales_vec.size());
    return new_scale;
  }
  return std::nullopt;
}

void H2dScalesCache::UpdateCurrentIndicesOfH2dScales() {
  for (auto& [key, scales] : h2d_scales_map_) {
    scales.second = scales.first.size() - 1;
  }
}

at::Tensor H2dScalesCache::CreateH2dTensorScale(
    void* scale_ptr,
    at::ScalarType dtype,
    void** alloc_pointer,
    void** h2d_pointer) {
  at::Tensor scale_tensor =
      createDynamicTensor({1}, HOST_TO_DEVICE_TENSOR, dtype);
  auto tmeta{get_tensor_extra_meta(scale_tensor)};

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
    HABANA_ASSERT(
        alloc_pointer != nullptr and h2d_pointer != nullptr,
        "Expected alloc_pointer and h2d_pointer to be both null or not-null, got: ",
        alloc_pointer,
        " and ",
        h2d_pointer);
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

} // namespace habana::backend
