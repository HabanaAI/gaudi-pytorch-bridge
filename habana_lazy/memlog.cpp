/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <sstream>

#include "absl/types/optional.h"

#include "aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "memlog.h"

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

  auto& device = synapse_helpers::HPURegistrar::get_device();
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

    // Live tensor collection is not allowed if the launch thread execution is
    // in progress.
    auto context =
        habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
    if (context != nullptr &&
        context->m_launch_thread_handle.valid() == false &&
        context->m_launch_thread_context == false &&
        !(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2)) {
      auto aten_device = SynapseDeviceToAtenDevice(device);

      uint32_t future = 0;
      uint64_t future_bytes = 0;
      HbContext* devctx =
          habana_lazy::HbContextArena::Get()->GetHbContext(aten_device);

      for (auto& uid_wptr : devctx->tensors_data) {
        std::shared_ptr<Data> data = uid_wptr.second.lock();

        if (data != nullptr) {
          auto t = HbLazyTensor(std::move(data));
          auto device_ptr = reinterpret_cast<synapse_helpers::device_ptr>(
              get_hb_lazy_data_ptr(t));
          bool is_allocated = false;

          if (device_ptr) {
            is_allocated = device_memory.is_allocated(device_ptr);
          }

          if (not is_allocated) {
            // Future, not yet allocated tensors
            ++future;
            future_bytes += compute_size(t);
          }
        }
      }
      ss << " future " << future_bytes / GB << "gb (" << future << ")";
    }

    ss << ", last workspace " << device.get_real_workspace_size() / GB << "gb";
  }

  PT_MEMLOG_DEBUG(ss.str());
}

} // namespace habana_lazy
