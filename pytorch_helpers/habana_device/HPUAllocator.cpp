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

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "HPUGuardImpl.h"
#include "hpu_cached_devices.h"

bool synapse_helpers::HPURegistrar::initialized_ = false;

// Note the main thread id
const std::thread::id synapse_helpers::HPURegistrar::main_thread_id_ =
    std::this_thread::get_id();

namespace at {
namespace habana {

synDeviceId HPUDeviceAllocator::allocator_active_device_id = -1;
pgmDropCachedRecipe HPUDeviceAllocator::drop_cached_recipe_cb = nullptr;

static HPUDeviceAllocator hpu_device_allocator;

at::Allocator* getHABANADeviceAllocator() {
  at::detail::HABANAGuardImpl h;
  h.getDevice();
  return &hpu_device_allocator;
}

// TODO: it might be not the best place to put this macro. I am confused how
// allocators are registered.
REGISTER_ALLOCATOR(DeviceType::HABANA, &at::habana::hpu_device_allocator);
} // namespace habana

namespace detail {

C10_REGISTER_GUARD_IMPL(HABANA, HABANAGuardImpl);

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
  void* v_ptr{nullptr};
  status = device.get_device_memory().malloc(&v_ptr, num_bytes);

  if (v_ptr == nullptr) {
    status = waitTillRecipeExecution(device_id, num_bytes, v_ptr);
  }

  if (v_ptr == nullptr && drop_cached_recipe_cb != nullptr) {
    status = waitTillCachedRecipesDropped(
        device_id, num_bytes, v_ptr, drop_cached_recipe_cb);
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
      Device(DeviceType::HABANA, allocator_active_device_id)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
}

void HPUDeviceAllocator::flush_stream_events() const {
  if (unsigned(-1) == habana::HPUDeviceAllocator::allocator_active_device_id) {
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

} // namespace habana
} // namespace at

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
    at::habana::HPUDeviceAllocator::allocator_active_device_id = -1;
  }
}

} // namespace synapse_helpers
