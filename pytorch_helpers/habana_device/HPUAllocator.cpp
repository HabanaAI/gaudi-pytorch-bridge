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

namespace at {
namespace habana {

synDeviceId HPUDeviceAllocator::allocator_active_device_id = -1;
pgmDropCachedRecipe HPUDeviceAllocator::drop_cached_recipe_cb = nullptr;

static HPUDeviceAllocator hpu_device_allocator;

// pool variables
pool_allocator::SubAllocator* HPUDeviceAllocator::suballoc = nullptr;
void* HPUDeviceAllocator::mem_pool = nullptr;
pool_allocator::PoolStrategyType HPUDeviceAllocator::poolingType =
    pool_allocator::strategy_none;
uint64_t HPUDeviceAllocator::poolSize = DEFAULT_POOL_SIZE;
///

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

static void waitTillRecipeExecution(
    synDeviceId device_id,
    size_t num_bytes,
    void*& v_ptr) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  // Allocation has failed, if there are still recipies in queue to execute,
  // there is a chance to recover. Wait for next recipe to finish and try to
  // allocate again, continue until malloc succeeds, or there are no more
  // recipes executing (unrecoverable case).
  auto& recipe_counter = device.get_active_recipe_counter();
  uint32_t counter_state{0};
  do {
    uint64_t ptr{0};
    counter_state = recipe_counter.wait_for_next_decrease_call();
    PT_DEVICE_DEBUG(
        "retrying memory alloc, waiting for recipes to finish execution, recipe count ",
        counter_state,
        " requested size ",
        num_bytes);
    auto status = synDeviceMalloc(device_id, num_bytes, 0, 0, &ptr);
    if (!status)
      v_ptr = reinterpret_cast<void*>(ptr);
    // It is not guaranteed that device will have more memory avaliable at exit
    // point, since framework might called multiple new allocations from other
    // threads, or wakeup might be spurious.
  } while (counter_state > 0 && v_ptr == nullptr);
}

void* HPUAllocator::alloc(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  uint64_t ptr{0};
  auto status{synDeviceMalloc(device_id, num_bytes, 0, 0, &ptr)};
  void* v_ptr = reinterpret_cast<void*>(ptr);

  if (v_ptr == nullptr) {
    waitTillRecipeExecution(device_id, num_bytes, v_ptr);
    if (v_ptr == nullptr) {
      TORCH_HABANA_CHECK(
          status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");
    }
  }

  return v_ptr;
}

void HPUAllocator::free(void* ptr) {
  if (nullptr == ptr) {
    return;
  }

  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  auto status{synDeviceFree(device_id, ptr_address, 0)};
  TORCH_HABANA_CHECK(status, "synDeviceFree failed");
}

void HPUDeviceAllocator::create_pool(synDeviceId deviceID, uint64_t poolSize) {
  switch (poolingType) {
    case pool_allocator::strategy_bump:
      if (!mem_pool) {
        try {
          PT_DEVICE_DEBUG("strategy_bump with size :: ", poolSize);
          suballoc = new pool_allocator::SubAllocator(
              new pool_allocator::StaticPooling);
          if (suballoc == nullptr) {
            PT_DEVICE_FATAL("unable to create pool allocator");
          }
          mem_pool = suballoc->pool_create(deviceID, poolSize);
          if (mem_pool == nullptr) {
            PT_DEVICE_FATAL("unable to create pool");
          }
        } catch (...) {
          PT_DEVICE_FATAL("unknown pool error ");
        }
      }
      break;
    case pool_allocator::strategy_dynamic:
      if (!suballoc) {
        try {
          PT_DEVICE_DEBUG("strategy_dynamic :: ", poolSize);
          suballoc = new pool_allocator::SubAllocator(
              new pool_allocator::DynamicPooling);
          if (suballoc == nullptr) {
            PT_DEVICE_FATAL("unable to create pool allocator");
          }
          mem_pool = suballoc->pool_create(deviceID, poolSize);
        } catch (...) {
          PT_DEVICE_FATAL("unknown pool error ");
        }
      }
      break;
    case pool_allocator::startegy_static_coalesce:
      if (!mem_pool) {
        try {
          PT_DEVICE_DEBUG("startegy_static_coalesce :: ", poolSize);
          suballoc = new pool_allocator::SubAllocator(
              new pool_allocator::StaticCoalescedPooling);
          if (suballoc == nullptr) {
            PT_DEVICE_FATAL("unable to create pool allocator");
          }
          mem_pool = suballoc->pool_create(deviceID, poolSize);
          if (mem_pool == nullptr) {
            PT_DEVICE_FATAL("unable to create pool");
          }
        } catch (...) {
          PT_DEVICE_FATAL("unknown pool error ");
        }
      }
      break;
    case pool_allocator::strategy_none:
    default:
      suballoc = nullptr;
      mem_pool = nullptr;
      break;
  }
}

void HPUDeviceAllocator::delete_pool() {
  if (suballoc) {
    suballoc->pool_destroy(mem_pool);
    delete suballoc;
  }
  mem_pool = nullptr;
  suballoc = nullptr;
}

HPUDeviceAllocator::HPUDeviceAllocator() {
  mem_pool = nullptr;
  suballoc = nullptr;
  poolingType = get_pooling_strategy();
  poolSize = get_pool_size();
}

HPUDeviceAllocator::~HPUDeviceAllocator() {
  flush_stream_events();
  if (poolingType != pool_allocator::strategy_none) {
    delete_pool();
  }
}

void HPUDeviceAllocator::deleter(void* ptr) {
  if (nullptr == ptr) {
    return;
  }

  if (poolingType != pool_allocator::strategy_none) {
    suballoc->pool_free_chunk(ptr);
  } else {
    uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
    TORCH_CHECK(
        habana::HPUDeviceAllocator::allocator_active_device_id == 0,
        "habana active device: ",
        habana::HPUDeviceAllocator::allocator_active_device_id,
        " != 0");
    auto status{synDeviceFree(allocator_active_device_id, ptr_address, 0)};
    TORCH_HABANA_CHECK(status, "synDeviceFree failed");
  }
}

at::DataPtr HPUDeviceAllocator::allocate(size_t num_bytes) const {
  void* v_ptr{nullptr};
  synStatus status{synStatus::synSuccess};

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");

  if (num_bytes != 0) {
    v_ptr = allocate_impl(num_bytes, status);

    if (v_ptr == nullptr) {
      auto& device =
          synapse_helpers::HPURegistrar::get_device(allocator_active_device_id);
      // Allocation has failed, if there are still recipies in queue to execute,
      // there is a chance to recover. Wait for next recipe to finish and try to
      // allocate again, continue until malloc succeeds, or there are no more
      // recipes executing (unrecoverable case).
      auto& recipe_counter = device.get_active_recipe_counter();
      if (!recipe_counter.is_zero()) {
        uint32_t counter_state{0};
        do {
          counter_state = recipe_counter.wait_for_next_decrease_call();
          PT_DEVICE_DEBUG(
              "retrying memory alloc, ",
              "waiting for recipe launch completion, recipe count ",
              counter_state,
              " requested size ",
              num_bytes);
          v_ptr = allocate_impl(num_bytes, status);
          // It is not guaranteed that device will have more memory avaliable at
          // exit point, since framework might called multiple new allocations
          // from other threads, or wakeup might be spurious.
        } while (counter_state > 1 && v_ptr == nullptr);
      }

      if (v_ptr == nullptr && drop_cached_recipe_cb != nullptr) {
        size_t nrecipes{0};
        do {
          bool drop_succeeded{false};

          // last resort to free up memory
          // we will wait for the completion of one recipe
          do {
            drop_succeeded = drop_cached_recipe_cb(nrecipes);
            if (!drop_succeeded) {
              std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
          } while (false == drop_succeeded && nrecipes > 0);

          PT_DEVICE_DEBUG(
              "retrying mem alloc after dropping lru recipe, ",
              "requested size ",
              num_bytes);

          v_ptr = allocate_impl(num_bytes, status);
        } while (v_ptr == nullptr && nrecipes > 0);
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
            status, "allocate_impl failed to allocate ", num_bytes, " bytes");
      }

      TORCH_CHECK(nullptr != v_ptr, "memory corruption");

      PT_DEVICE_DEBUG(
          "successful memory alloc after retry, requested size ", num_bytes);
    }
  }

  return {
      v_ptr,
      v_ptr,
      &HPUDeviceAllocator::deleter,
      Device(DeviceType::HABANA, allocator_active_device_id)};
}

void* HPUDeviceAllocator::allocate_impl(size_t size, synStatus& status) const {
  size_t num_bytes = size;
  uint64_t ptr{0};
  void* v_ptr = nullptr;

  status = synStatus::synSuccess;

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");

  if (poolingType != pool_allocator::strategy_none) {
    // pool must be created in the constructor
    // device is not yet initialized so creating here

    create_pool(allocator_active_device_id, poolSize);
    ptr = (uint64_t)suballoc->pool_alloc_chunk(mem_pool, num_bytes);

    if ((void*)ptr == nullptr) {
      PT_DEVICE_DEBUG("pooling allocator failed, requested size ", num_bytes);
      status = synFail;
    }

    v_ptr = reinterpret_cast<void*>(ptr);
  } else {
    status = synDeviceMalloc(allocator_active_device_id, num_bytes, 0, 0, &ptr);

    if (synStatus::synSuccess != status) {
      PT_DEVICE_DEBUG("synDeviceMalloc failed, requested size ", num_bytes);
    } else {
      v_ptr = reinterpret_cast<void*>(ptr);
    }
  }

  return v_ptr;
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
}

void HPUDeviceAllocator::flush_stream_events() const {
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

HPURegistrar& HPURegistrar::get_hpu_registrar() {
  static HPURegistrar* instance = new HPURegistrar();
  return *instance;
}

} // namespace synapse_helpers

void print_live_allocations() {
  std::cout
      << "\nNo log for device memory allocation is collected. Use the following commands "
         "to enable allocation tracking and reporting with hb_torch.memstat_livealloc() - \n"
         "HBN_SYNAPSE_LOGGER_COMMANDS=log_device_alloc "
         "LD_PRELOAD=$BUILD_ROOT_LATEST/pytorch_synapse_logger.so a.out...\n\n";
}
