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

static HPUDeviceAllocator hpu_device_allocator;

// pool variables
SubAllocator *HPUDeviceAllocator::suballoc = nullptr;
void* HPUDeviceAllocator::mem_pool = nullptr;
PoolStrategyType HPUDeviceAllocator::poolingType = strategy_none;
size_t HPUDeviceAllocator::poolSize = DEFAULT_POOL_SIZE;
///

at::Allocator* getHABANADeviceAllocator() {
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
        "retrying to memory alloc, Waiting for recipes to finish execution recipe count:",
        counter_state,
        "requested size",
        num_bytes);
    auto status = synDeviceMalloc(device_id, num_bytes, 0, 0, &ptr);
    if (!status)
      v_ptr = reinterpret_cast<void*>(ptr);
    // It is not guaranted that device will have more memory avaliable at exit
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

void HPUDeviceAllocator::create_pool(synDeviceId deviceID, size_t poolSize) {
  switch (poolingType)
  {
    case strategy_bump:
      if (!mem_pool) {
        suballoc = new SubAllocator(new StaticPooling);
        mem_pool = suballoc->pool_create(deviceID, poolSize);
        if (mem_pool == nullptr) {
          PT_DEVICE_FATAL("unable to create pool");
        }
      }
    break;
    case strategy_dynamic:
      if (!suballoc) {
        suballoc = new SubAllocator(new DynamicPooling);
        mem_pool = suballoc->pool_create(deviceID, poolSize);
      }
    break;
    case strategy_none:
    default:
      suballoc = nullptr;
      mem_pool = nullptr;
    break;
  }
}

void HPUDeviceAllocator::delete_pool() {
  suballoc->pool_destroy(mem_pool);
  delete suballoc;
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
  if (poolingType != strategy_none) {
    delete_pool();
  }
}

void HPUDeviceAllocator::deleter(void* ptr) {
  if (nullptr == ptr) {
    return;
  }

  if (poolingType != strategy_none) {
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

at::DataPtr HPUDeviceAllocator::allocate(size_t size) const {
  size_t num_bytes = size;
  uint64_t ptr{0};
  void* v_ptr = nullptr;

  if (num_bytes != 0) {
    TORCH_CHECK(
        habana::HPUDeviceAllocator::allocator_active_device_id == 0,
        "habana active device: ",
        habana::HPUDeviceAllocator::allocator_active_device_id,
        " != 0");
    if (poolingType != strategy_none) {
        create_pool(allocator_active_device_id, poolSize);
        ptr = (uint64_t)suballoc->pool_alloc_chunk(mem_pool, num_bytes);
        if ((void*)ptr == nullptr) {
          PT_DEVICE_FATAL("pooling allocator failed");
        }
        v_ptr = reinterpret_cast<void*>(ptr);
    } else {        
        auto status{
          synDeviceMalloc(allocator_active_device_id, num_bytes, 0, 0, &ptr)};
        v_ptr = reinterpret_cast<void*>(ptr);

        if (v_ptr == nullptr) {
          waitTillRecipeExecution(allocator_active_device_id, num_bytes, v_ptr);
          if (v_ptr == nullptr) {
             TORCH_HABANA_CHECK(
                status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");
          }
        }
    }
  }

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");
  return {v_ptr,
          v_ptr,
          &HPUDeviceAllocator::deleter,
          Device(DeviceType::HABANA, allocator_active_device_id)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
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
