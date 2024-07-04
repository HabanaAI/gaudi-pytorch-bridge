/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "HPUAllocator.h"
#include <synapse_api.h>
#include "HPUGuardImpl.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/memlog.h"
#include "hpu_cached_devices.h"

namespace habana {

synDeviceId HPUDeviceAllocator::allocator_active_device_id =
    static_cast<synDeviceId>(-1);

static HPUDeviceAllocator hpu_device_allocator;

at::DataPtr CreateDataPtr(void* v_ptr, size_t num_bytes) {
  if (v_ptr != nullptr) {
    // create HPUAllocationContext only for non-empty allocation
    auto ctx = new HPUAllocationContext;
    ctx->data_ptr = v_ptr;
    ctx->num_bytes = num_bytes;
    PT_EAGER_DEBUG(
        "Created HPUAllocationContext ",
        ctx,
        " for data_ptr ",
        ctx->data_ptr,
        " and num_bytes ",
        ctx->num_bytes);
    return {
        v_ptr,
        ctx,
        &HPUDeviceAllocator::deleter,
        at::Device(
            at::DeviceType::HPU,
            HPUDeviceAllocator::allocator_active_device_id)};
  } else {
    PT_EAGER_DEBUG(
        "Created DataPtr without HPUAllocationContext due to v_ptr=",
        v_ptr,
        " for num_bytes ",
        num_bytes);
    return {
        v_ptr,
        v_ptr,
        &HPUDeviceAllocator::deleter,
        at::Device(
            at::DeviceType::HPU,
            HPUDeviceAllocator::allocator_active_device_id)};
  }
}

at::Allocator* getHABANADeviceAllocator() {
  HABANAGuardImpl h;
  h.getDevice();
  return &hpu_device_allocator;
}
} // namespace habana

// TODO: it might be not the best place to put this macro. I am confused how
// allocators are registered.

namespace at {
REGISTER_ALLOCATOR(DeviceType::HPU, &habana::hpu_device_allocator);
} // namespace at

namespace detail {
C10_REGISTER_GUARD_IMPL(HPU, habana::HABANAGuardImpl);
} // namespace detail

namespace habana {

static synStatus waitTillRecipeExecution(
    synDeviceId device_id,
    size_t num_bytes,
    void*& v_ptr) {
  synStatus status{synStatus::synFail};
  auto& device = HPURegistrar::get_device(device_id).syn_device();
  // Allocation has failed, if there are still recipies in queue to execute,
  // there is a chance to recover. Wait for next recipe to finish and try to
  // allocate again, continue until malloc succeeds, or there are no more
  // recipes executing (unrecoverable case).
  auto& recipe_counter = device.get_active_recipe_counter();
  uint32_t counter_state{0};
  if (!recipe_counter.is_zero()) {
    do {
      counter_state = recipe_counter.wait_for_next_decrease_call();
      PT_DEVICE_DEBUG(
          "retrying memory alloc, ",
          "waiting for recipe launch completion, recipe count ",
          counter_state,
          " requested size ",
          num_bytes);
      if (common::IsRecordStreamEnabled()) {
        status = device.get_device_memory().malloc(
            &v_ptr, num_bytes, c10::hpu::getCurrentHPUStream().stream());
      } else {
        status = device.get_device_memory().malloc(&v_ptr, num_bytes);
      }
      // It is not guaranteed that device will have more memory avaliable at
      // exit point, since framework might called multiple new allocations
      // from other threads, or wakeup might be spurious.
    } while (counter_state > 1 && v_ptr == nullptr);

    habana_lazy::log_dev_mem_stats(
        "Post-Recipe-Decrease-Execution-Done", "", num_bytes);
  }
  return status;
}

HPUDeviceAllocator::HPUDeviceAllocator() {
  allocator_active_device_id = static_cast<synDeviceId>(-1);
}

void HPUDeviceAllocator::deleter(void* ptr) {
  synStatus status;
  auto& device =
      HPURegistrar::get_device(HPUDeviceAllocator::allocator_active_device_id);
  if (ptr != nullptr) {
    auto alloc_ctx = reinterpret_cast<HPUAllocationContext*>(ptr);
    if (common::IsRecordStreamEnabled()) {
      status = device.get_device_memory().free_with_stream(alloc_ctx->data_ptr);
    } else {
      status = device.get_device_memory().free(alloc_ctx->data_ptr);
    }
    delete alloc_ctx;
  } else {
    if (common::IsRecordStreamEnabled()) {
      status = device.get_device_memory().free_with_stream(ptr);
    } else {
      status = device.get_device_memory().free(ptr);
    }
  }
  TORCH_HABANA_CHECK(status, "Device Free failed");
}

#if IS_PYTORCH_AT_LEAST(2, 3)
at::DataPtr HPUDeviceAllocator::allocate(size_t num_bytes) {
#else
at::DataPtr HPUDeviceAllocator::allocate(size_t num_bytes) const {
#endif
  void* v_ptr{nullptr};
  synStatus status{synStatus::synSuccess};

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");

  auto& device = HPURegistrar::get_device(allocator_active_device_id);

  if (num_bytes != 0) {
    if (common::IsRecordStreamEnabled()) {
      status = device.get_device_memory().malloc(
          &v_ptr, num_bytes, c10::hpu::getCurrentHPUStream().stream());
    } else {
      status = device.get_device_memory().malloc(&v_ptr, num_bytes);
    }

    if (v_ptr == nullptr) {
      status =
          waitTillRecipeExecution(allocator_active_device_id, num_bytes, v_ptr);
    }

    if (status != synStatus::synSuccess) {
      uint64_t free_mem, total_mem;
      auto status_mem = synDeviceGetMemoryInfo(
          allocator_active_device_id, &free_mem, &total_mem);
      if (synStatus::synSuccess != status_mem) {
        PT_DEVICE_FATAL(
            Logger::formatStatusMsg(status_mem),
            "device memory size query failed");
      }

      if (num_bytes > free_mem) {
        PT_DEVICE_DEBUG(
            "requested size ",
            num_bytes,
            " is more than avaiable free memory ",
            free_mem);
      } else {
        PT_DEVICE_DEBUG(
            "failed to allocate ",
            num_bytes,
            " although the avaiable free memory is ",
            free_mem,
            " most likely due to fragmentation");
      }

      TORCH_HABANA_CHECK(
          status, "allocate failed to allocate ", num_bytes, " bytes");
    }

    TORCH_CHECK(nullptr != v_ptr, "memory corruption");

    PT_DEVICE_DEBUG("successful memory alloc, requested size ", num_bytes);
  }

  return CreateDataPtr(v_ptr, num_bytes);
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
}

void HPUDeviceAllocator::recordStream(
    const at::DataPtr& ptr,
    c10::hpu::HPUStream stream) {
  if (!common::IsRecordStreamEnabled())
    return;
  // Empty tensor's storage().data() might be a null ptr. As there is no
  // blocks associated with those tensors, it is fine to do nothing here.
  if (!ptr.get()) {
    return;
  }

  // If a tensor is not allocated by this instance, simply skip
  // This usually happens when HPU tensors are shared across processes,
  // we have implemented reference counting based sharing mechanism to
  // guarantee tensors won't be accidentally freed by one process while
  // they are still being used in another
  if (ptr.get_deleter() != &HPUDeviceAllocator::deleter)
    return;

  if (unsigned(-1) == habana::HPUDeviceAllocator::allocator_active_device_id) {
    return;
  }
  auto& device =
      HPURegistrar::get_device(HPUDeviceAllocator::allocator_active_device_id);
  if (ptr.get() != nullptr) {
    device.get_device_memory().recordStream(ptr.get(), stream.stream());
  }
}

void HPUDeviceAllocator::print_memory_stats(const char* msg) {
  if (!GET_ENV_FLAG_NEW(PT_HABANA_MEM_LOG_LEVEL)) {
    if (unsigned(-1) ==
        habana::HPUDeviceAllocator::allocator_active_device_id) {
      return;
    }
    auto& device = HPURegistrar::get_device(allocator_active_device_id);
    if (device.get_device_memory().get_pool_strategy() !=
        synapse_helpers::pool_allocator::strategy_none) {
      synapse_helpers::MemoryStats stats;
      device.get_device_memory().get_memory_stats(&stats);
      std::string updated_msg = msg;
      updated_msg = updated_msg + "\n" + stats.DebugString();
      synapse_helpers::print_live_allocations(updated_msg.c_str());
      device.get_device_memory().clear_memory_stats();
    }
  } else {
    synapse_helpers::print_live_allocations(msg);
  }
}

void HPUDeviceAllocator::memstat_devmem_start_collect(
    const char* msg,
    bool show_leaked_callstacks) {
  if (unsigned(-1) == habana::HPUDeviceAllocator::allocator_active_device_id) {
    return;
  }
  auto& device =
      HPURegistrar::get_device(allocator_active_device_id).syn_device();
  if (device.IsStreamASyncEnabled()) {
    PT_DEVICE_WARN(
        "Warning: Set PT_ENABLE_HABANA_STREAMASYNC=0 for device memory "
        "statistics/leaks collection so that errors due to async behavior can be reduced");
  }

  if (device.get_device_memory().get_pool_strategy() !=
      synapse_helpers::pool_allocator::strategy_none) {
    synapse_helpers::set_memstats_check_flag(true);
    std::string updated_msg = msg;
    updated_msg = updated_msg + "\nMemory statistics collection started!!";
    synapse_helpers::memstats_dump(device, updated_msg.c_str());
    synapse_helpers::set_back_trace(show_leaked_callstacks);
    device.get_device_memory().clear_memory_stats();
  }
}

void HPUDeviceAllocator::memstat_devmem_stop_collect(const char* msg) {
  if (unsigned(-1) == habana::HPUDeviceAllocator::allocator_active_device_id) {
    return;
  }
  auto& device =
      HPURegistrar::get_device(allocator_active_device_id).syn_device();
  if (device.get_device_memory().get_pool_strategy() !=
      synapse_helpers::pool_allocator::strategy_none) {
    std::string updated_msg = msg;
    updated_msg = updated_msg +
        "\nMemory statistics collection stopped and dumping data...";
    synapse_helpers::memstats_dump(device, updated_msg.c_str());
    device.get_device_memory().clear_memory_stats();
  }
}

void HPUDeviceAllocator::dump_memory_reporter() {
  auto& device = HPURegistrar::get_device().syn_device();
  synapse_helpers::memory_reporter_event_create(
      device, synapse_helpers::mem_reporter_type::MEM_REPORTER_USER_CALL);
}

#if IS_PYTORCH_AT_LEAST(2, 3)
void HPUDeviceAllocator::copy_data(
    [[maybe_unused]] void* dest,
    [[maybe_unused]] const void* src,
    [[maybe_unused]] std::size_t count) const {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for HPUDeviceAllocator");
}
#endif
} // namespace habana
