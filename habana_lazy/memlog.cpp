/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "memlog.h"
#include <sstream>
#include <utility>
#include "aten_lazy_bridge.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/hlexec.h"

namespace habana_lazy {

const auto MB = 1024 * 1024.;
const auto GB = 1024 * MB;

namespace {
size_t compute_size(const HbLazyTensor& tensor) {
  size_t size = 1;
  for (const auto& i : tensor.GetSizes()) {
    size *= static_cast<size_t>(i);
  }

  return size * c10::scalarTypeToTypeMeta(tensor.dtype()).itemsize();
}

void* get_hb_lazy_data_ptr(HbLazyTensor& hb_tensor) {
  auto hb_tensor_data = hb_tensor.CurrentTensorAttached();
  if (!hb_tensor_data || !hb_tensor_data.value().has_storage() ||
      (GetHbInternalTensorImpl(hb_tensor_data.value()) == nullptr)) {
    return nullptr;
  }

  return hb_tensor_data->data_ptr();
}

} // namespace

// Live tensor collection is not allowed if the launch thread execution is
// in progress.
std::pair<uint64_t, uint32_t> get_future_memory() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    return std::make_pair<uint64_t, uint32_t>(0, 0);
  }
  auto& device = habana::HPUDeviceContext::get_device();
  auto* context = habana_lazy::get_device_lazy_execution_context();

  if (context == nullptr || context->m_launch_thread_handle.valid() ||
      context->m_launch_thread_context) {
    return std::make_pair<uint64_t, uint32_t>(0, 0);
  }

  auto aten_device = habana::HPUDeviceContext::aten_device();

  uint32_t future = 0;
  uint64_t future_bytes = 0;
  auto devctx = habana_lazy::HbContextArena::Get().GetHbContext(aten_device);

  for (auto& uid_wptr : devctx->tensors_data) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();

    if (data == nullptr) {
      continue;
    }
    auto t = HbLazyTensor(std::move(data));
    auto device_ptr =
        reinterpret_cast<synapse_helpers::device_ptr>(get_hb_lazy_data_ptr(t));

    if ((device_ptr == 0U) ||
        device.get_device_memory().is_allocated(device_ptr)) {
      continue;
    }
    ++future;
    future_bytes += compute_size(t);
  }
  return std::make_pair(future_bytes, future);
}

void log_dev_mem_stats(
    std::string_view msg,
    std::string_view name /* = "" */,
    uint64_t size /* = 0 */) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    return;
  }

  static bool s_mem_log_enabled = IS_MEMLOG_DEBUG_ENABLED;
  if (!s_mem_log_enabled) {
    return;
  }

  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed;
  ss << msg;
  if (not name.empty()) {
    ss << " [" << name << "]";
  }
  if (size > 0) {
    ss << ", size " << static_cast<double>(size) / GB << "gb";
  }

  auto& device = habana::HPUDeviceContext::get_device();
  auto& device_memory = device.get_device_memory();
  if (device_memory.get_pool_strategy() !=
      synapse_helpers::pool_allocator::strategy_none) {
    synapse_helpers::MemoryStats stats;
    device_memory.get_memory_stats(&stats);

    // Current device memory stats
    auto used = stats.bytes_in_use;
    auto ws = stats.scratch_mem_in_use;
    auto persistent = (int64_t)used - (int64_t)ws;
    auto max_cntgs_chunk = device_memory.get_max_cntgs_chunk_size();

    ss << ": used " << static_cast<double>(used) / GB << "gb, workspace "
       << static_cast<double>(ws) / GB << "gb, persistent "
       << static_cast<double>(persistent) / GB << "gb, max cntgs chunk "
       << static_cast<double>(max_cntgs_chunk) / GB << "gb";

    auto future = get_future_memory();
    ss << " future " << static_cast<double>(future.first) / GB << "gb ("
       << future.second << ")";
  }

  ss << ", last workspace "
     << static_cast<double>(device.get_real_workspace_size()) / GB << "gb";

  PT_MEMLOG_DEBUG(ss.str());
}
} // namespace habana_lazy
