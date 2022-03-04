/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>

#include <synapse_helpers/devmem_logger.h>
#include <synapse_helpers/env_flags.h>
#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "HPUGuardImpl.h"
#include "hpu_cached_devices.h"

bool synapse_helpers::HPURegistrar::initialized_ = false;

// Note the main thread id
const std::thread::id synapse_helpers::HPURegistrar::main_thread_id_ =
    std::this_thread::get_id();

namespace habana {

synDeviceId HPUDeviceAllocator::allocator_active_device_id = -1;
pgmDropCachedRecipe HPUDeviceAllocator::drop_cached_recipe_cb = nullptr;

static HPUDeviceAllocator hpu_device_allocator;

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

HPUAllocator::HPUAllocator(uint32_t device) : device_id(device) {}

void HPUAllocator::reset() {
  PT_DEVICE_WARN("HPUAllocator::reset should not be invoked.");
}

void HPUAllocator::release() {
  device_id = synapse_helpers::device::INVALID_ID;
  PT_DEVICE_WARN("HPUAllocator::release should not be invoked.");
}

static void waitTillRecipeExecutionDone(synDeviceId device_id) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  bool status = device.get_device_memory().is_mem_threshold_hit();
  auto& recipe_counter = device.get_active_recipe_counter();
  uint32_t counter_state = recipe_counter.get_count();

  while (status && (counter_state > 1)) {
    counter_state = recipe_counter.wait_for_next_decrease_call();
    PT_DEVICE_DEBUG(
        "waiting for recipe launch completion, recipe count ", counter_state);
    status = device.get_device_memory().is_mem_threshold_hit();
  }
}

static synStatus waitTillRecipeExecution(
    synDeviceId device_id,
    size_t num_bytes,
    void*& v_ptr) {
  synStatus status{synStatus::synFail};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
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
      status = device.get_device_memory().malloc(&v_ptr, num_bytes);
      // It is not guaranteed that device will have more memory avaliable at
      // exit point, since framework might called multiple new allocations
      // from other threads, or wakeup might be spurious.
    } while (counter_state > 1 && v_ptr == nullptr);
  }
  return status;
}

static synStatus waitTillCachedRecipesDropped(
    synDeviceId device_id,
    size_t num_bytes,
    void*& v_ptr,
    pgmDropCachedRecipe drop_cached_recipe_cb) {
  synStatus status{synStatus::synFail};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t nrecipes{0};
  do {
    bool drop_succeeded{false};

    // last resort to free up memory
    // we will wait for the completion of one recipe
    do {
      drop_succeeded = drop_cached_recipe_cb(nrecipes);

    } while (false == drop_succeeded && nrecipes > 0);

    PT_DEVICE_DEBUG(
        "retrying mem alloc after dropping lru recipe, ",
        "requested size ",
        num_bytes);

    status = device.get_device_memory().malloc(&v_ptr, num_bytes);
  } while (v_ptr == nullptr && nrecipes > 0);

  return status;
}

void* HPUAllocator::alloc(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  synStatus status{synStatus::synSuccess};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  waitTillRecipeExecutionDone(device_id);
  void* v_ptr{nullptr};
  status = device.get_device_memory().malloc(&v_ptr, num_bytes);

  if (v_ptr == nullptr) {
    status = waitTillRecipeExecution(device_id, num_bytes, v_ptr);
  }

  if (v_ptr == nullptr) {
    TORCH_HABANA_CHECK(
        status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");
  }

  return v_ptr;
}

void HPUAllocator::free(void* ptr) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto status{device.get_device_memory().free(ptr)};
  TORCH_HABANA_CHECK(status, "Device Free failed");
}

HPUDeviceAllocator::HPUDeviceAllocator() {
  allocator_active_device_id = -1;
}

HPUDeviceAllocator::~HPUDeviceAllocator() {
  flush_stream_events();
}

void HPUDeviceAllocator::deleter(void* ptr) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
  auto status{device.get_device_memory().free(ptr)};
  TORCH_HABANA_CHECK(status, "Device Free failed");
}

at::DataPtr HPUDeviceAllocator::allocate(size_t num_bytes) const {
  void* v_ptr{nullptr};
  synStatus status{synStatus::synSuccess};

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");

  auto& device =
      synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);

  waitTillRecipeExecutionDone(allocator_active_device_id);
  if (num_bytes != 0) {
    status = device.get_device_memory().malloc(&v_ptr, num_bytes);

    if (v_ptr == nullptr) {
      status =
          waitTillRecipeExecution(allocator_active_device_id, num_bytes, v_ptr);
    }

    if (v_ptr == nullptr && drop_cached_recipe_cb != nullptr) {
      status = waitTillCachedRecipesDropped(
          allocator_active_device_id, num_bytes, v_ptr, drop_cached_recipe_cb);
    }

    if (status != synStatus::synSuccess) {
      uint64_t free_mem, total_mem;
      auto status_mem = synDeviceGetMemoryInfo(
          allocator_active_device_id, &free_mem, &total_mem);
      if (synStatus::synSuccess != status_mem) {
        PT_DEVICE_FATAL("device memory size query failed with ", status_mem);
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

  return {
      v_ptr,
      v_ptr,
      &HPUDeviceAllocator::deleter,
      at::Device(at::DeviceType::HPU, allocator_active_device_id)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
}

void HPUDeviceAllocator::flush_stream_events() const {
  if (unsigned(-1) == habana::HPUDeviceAllocator::allocator_active_device_id) {
    PT_DEVICE_DEBUG(
        "Invalid Device::",
        habana::HPUDeviceAllocator::allocator_active_device_id);
    return;
  }

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");

  auto& device =
      synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
  device.flush_stream_events();
}

void HPUDeviceAllocator::print_memory_stats(const char* msg) {
  if (!GET_ENV_FLAG_NEW(PT_HABANA_MEM_LOG_LEVEL)) {
    if (unsigned(-1) ==
        habana::HPUDeviceAllocator::allocator_active_device_id) {
      return;
    }
    auto& device =
        synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
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
      synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
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
      synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
  if (device.get_device_memory().get_pool_strategy() !=
      synapse_helpers::pool_allocator::strategy_none) {
    std::string updated_msg = msg;
    updated_msg = updated_msg +
        "\nMemory statistics collection stopped and dumping data...";
    synapse_helpers::memstats_dump(device, updated_msg.c_str());
    device.get_device_memory().clear_memory_stats();
  }
}

} // namespace habana

namespace synapse_helpers {

/**
  The HPURegistrar object is created once for the first time
  at::detail::HABANAGuardImpl::getDevice is called.
  Since HPURegistrar object is a static, it gets destroyed when
  main thread exits via exit_handler.
  However, synapse has a few objects (like KernelDB) that are
  thread_local static. They are created once per thread, and they
  get destroyed from the main thread before the static objects are
  destroyed. These synapse objects are required to be present when
  the synapse devices are destroyed.
  Currently, here is the sequence of object creation -
   OSAL, KernelDB -> synapse devices -> HPURegistrar
  We expect the destruction order to be -
   ~HPURegistrar -> ~synapse devices -> ~OSAL, ~KernelDB
  The destruction order, when KernelDB is created from more than one
  thread (it gets created if a thread creates a synapse graph and compiles)
   ~KernelDB ->  ~HPURegistrar -> ~synapse devices (This fails)

  Hence, we have a thread_local HPURegistrarPerThreadTracker object.
  This is used to drive the cleanup before the ~KernelDB happens. With this,
  the object construction order -
   OSAL, KernelDB -> synapse devices -> HPURegistrar,
  HPURegistrarPerThreadTracker Object destruction order
   ~HPURegistrarPerThreadTracker -> ~synapse devices -> ~KernelDB ->
  ~HPURegistrar -> ~OSAL
 */
class HPURegistrarPerThreadTracker {
 public:
  HPURegistrarPerThreadTracker() = default;
  ~HPURegistrarPerThreadTracker();
};

HPURegistrar& HPURegistrar::get_hpu_registrar() {
  static HPURegistrar instance;
  thread_local static HPURegistrarPerThreadTracker per_thread_tracker;
  return instance;
}

HPURegistrarPerThreadTracker::~HPURegistrarPerThreadTracker() {
  // Cleanup the synapse devices only for the main thread exit path
  // This ensures synapse devices are removed before thread_local synapse
  // objects (Ex: KernelDB) are gone.
  if (HPURegistrar::getMainThreadId() == std::this_thread::get_id()) {
    HPURegistrar::deleteDevices();
    habana::HPUDeviceAllocator::allocator_active_device_id = -1;
    habana::PinnedMemoryAllocator::allocator_active_device_id = -1;
  }
}

} // namespace synapse_helpers
