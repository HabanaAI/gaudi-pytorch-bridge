/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "memlog.h"
#include <sstream>
#include <utility>
#include "absl/types/optional.h"
#include "aten_lazy_bridge.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_lazy/hlexec.h"

namespace habana_lazy {

const auto MB = 1024 * 1024.;
const auto GB = 1024 * MB;

namespace {
int64_t compute_size(const HbLazyTensor& tensor) {
  int64_t size = 1;
  for (const auto& i : tensor.GetSizes()) {
    size *= i;
  }

  return size * c10::scalarTypeToTypeMeta(tensor.dtype()).itemsize();
}

void* get_hb_lazy_data_ptr(HbLazyTensor& hb_tensor) {
  auto hb_tensor_data = hb_tensor.CurrentTensorAttached();
  if (!hb_tensor_data or !hb_tensor_data.has_value() or
      !hb_tensor_data.value().has_storage() or
      !GetHbInternalTensorImpl(hb_tensor_data.value())) {
    return nullptr;
  }

  return hb_tensor_data->data_ptr();
}

} // namespace

// Live tensor collection is not allowed if the launch thread execution is
// in progress.
const std::pair<uint64_t, uint32_t> get_future_memory() {
  auto& device = habana::HPURegistrar::get_device();
  auto context = habana_lazy::get_device_lazy_execution_context();

  if (context == nullptr || context->m_launch_thread_handle.valid() == true ||
      context->m_launch_thread_context == true ||
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    return std::make_pair<uint64_t, uint32_t>(0, 0);
  }

  auto aten_device = device.aten_device();

  uint32_t future = 0;
  uint64_t future_bytes = 0;
  habana_lazy::HbContext* devctx =
      habana_lazy::HbContextArena::Get()->GetHbContext(aten_device);

  for (auto& uid_wptr : devctx->tensors_data) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();

    if (data == nullptr)
      continue;

    auto t = HbLazyTensor(std::move(data));
    auto device_ptr =
        reinterpret_cast<synapse_helpers::device_ptr>(get_hb_lazy_data_ptr(t));

    if (!device_ptr || device.get_device_memory().is_allocated(device_ptr))
      continue;

    ++future;
    future_bytes += compute_size(t);
  }
  return std::make_pair(future_bytes, future);
}

void log_dev_mem_stats(
    std::string_view msg,
    std::string_view name /* = "" */,
    uint64_t size /* = 0 */) {
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
    ss << ", size " << size / GB << "gb";
  }

  auto& device = habana::HPURegistrar::get_device();
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

    ss << ": used " << used / GB << "gb, workspace " << ws / GB
       << "gb, persistent " << persistent / GB << "gb, max cntgs chunk "
       << max_cntgs_chunk / GB << "gb";

    auto future = get_future_memory();
    ss << " future " << future.first / GB << "gb (" << future.second << ")";
  }

  ss << ", last workspace "
     << device.syn_device().get_real_workspace_size() / GB << "gb";

  PT_MEMLOG_DEBUG(ss.str());
}
} // namespace habana_lazy
