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
#include "habana_eager/ops/empty.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/helpers/tensor_utils.h"

namespace habana {
namespace eager {
at::Tensor empty(
    at::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  PT_EAGER_TRACE;
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");

  auto device = device_opt.value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());

  auto index = device.index();
  if (index != 0 && index != -1) {
    TORCH_WARN_ONCE(
        "\"hpu:X\" notation is not supported by Gaudi PyTorch "
        "intergration bridge. Please change to \"hpu\" without index");
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == at::Layout::Strided);

  auto allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);
  auto dtype = dtype_or_default(dtype_opt);
  HABANA_ASSERT(habana_helpers::is_supported_type(dtype));

  return at::detail::empty_generic(
      at::asIntArrayRefUnchecked(size),
      allocator,
      hpu_ks,
      dtype,
      memory_format_opt);
}

at::Tensor empty_strided(
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  PT_EAGER_TRACE;
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");

  auto device = device_opt.value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());
  auto index = device.index();
  if (index != 0 && index != -1) {
    TORCH_WARN_ONCE(
        "\"hpu:X\" notation is not supported by Gaudi PyTorch "
        "intergration bridge. Please change to \"hpu\" without index");
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == at::Layout::Strided);

  auto allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);
  auto dtype = dtype_or_default(dtype_opt);
  HABANA_ASSERT(habana_helpers::is_supported_type(dtype));

  return at::detail::empty_strided_symint_generic(
      size, stride, allocator, hpu_ks, dtype);
}

} // namespace eager
} // namespace habana
