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
#include <synapse_common_types.h>
#include <iterator>
#include <sstream>
#include <utility>

#include <synapse_api.h>
#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/devmem_logger.h"
#include "synapse_helpers/env_flags.h"

namespace synapse_helpers {
device_memory::device_memory(device& device) : device_{device} {
  pool_size_ = GET_ENV_FLAG(PT_HABANA_POOL_SIZE) * 1024 * 1024 * 1024;
  pool_strategy_ =
      (pool_allocator::PoolStrategyType)GET_ENV_FLAG(PT_HPU_POOL_STRATEGY);
  switch (pool_strategy_) {
    case pool_allocator::strategy_bump:
      try {
        PT_SYNHELPER_DEBUG("strategy_bump with size :: ", pool_size_);
        suballoc_ =
            new pool_allocator::SubAllocator(new pool_allocator::StaticPooling);
        if (suballoc_ == nullptr) {
          PT_SYNHELPER_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_SYNHELPER_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::strategy_dynamic:
      try {
        PT_SYNHELPER_DEBUG("strategy_dynamic :: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::DynamicPooling);
        if (suballoc_ == nullptr) {
          PT_SYNHELPER_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_SYNHELPER_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::startegy_static_coalesce:
      try {
        PT_SYNHELPER_DEBUG("startegy_static_coalesce :: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::StaticCoalescedPooling);
        if (suballoc_ == nullptr) {
          PT_SYNHELPER_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_SYNHELPER_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::strategy_none:
    default:
      suballoc_ = nullptr;
      break;
  }
  if (suballoc_ && !suballoc_->pool_create(device_.id(), pool_size_)) {
    PT_SYNHELPER_FATAL("pool creation failed");
  }

  if (pool_strategy_ != pool_allocator::startegy_static_coalesce) {
    std::array<uint64_t, 2> dram_infos = {0, 0};
    uint64_t* dram_info = dram_infos.data();
    std::array<synDeviceAttribute, 2> deviceAttrs = {
        DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS, DEVICE_ATTRIBUTE_DRAM_SIZE};
    synDeviceAttribute* deviceAttr = deviceAttrs.data();
    auto status = synDeviceGetAttribute(dram_info, deviceAttr, 2, device_.id());
    if (synStatus::synSuccess != status) {
      PT_SYNHELPER_FATAL("Cannot obtain dram info. Status: ", status);
    }
    log_DRAM_start(dram_info[0]);
    log_DRAM_size(dram_info[1]);
  }
}

device_memory::~device_memory() {
  if (suballoc_) {
    suballoc_->pool_destroy();
    delete suballoc_;
  }
  suballoc_ = nullptr;
}

synStatus device_memory::malloc(void** v_ptr, uint64_t size) {
  uint64_t ptr{0};
  synStatus status{synStatus::synSuccess};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    ptr = (uint64_t)suballoc_->pool_alloc_chunk(size);

    if ((void*)ptr == nullptr) {
      PT_SYNHELPER_DEBUG("pooling allocator failed, requested size ", size);
      status = synFail;
    }

    *v_ptr = reinterpret_cast<void*>(ptr);
  } else {
    status = synDeviceMalloc(device_.id(), size, 0, 0, &ptr);

    if (synStatus::synSuccess != status) {
      PT_SYNHELPER_DEBUG("synDeviceMalloc failed, requested size ", size);
    } else {
      *v_ptr = reinterpret_cast<void*>(ptr);
    }
  }
  log_synDeviceMalloc(ptr, size, status);

  return status;
}

synStatus device_memory::free(void* ptr) {
  synStatus status{synStatus::synSuccess};
  if (nullptr == ptr) {
    return status;
  }

  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->pool_free_chunk(ptr);
  } else {
    uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
    auto status{synDeviceFree(device_.id(), ptr_address, 0)};
    PT_SYNHELPER_DEBUG("SynDeviceFree Failed.", status);
  }
  log_synDeviceFree(reinterpret_cast<uint64_t>(ptr), status);
  return status;
}

} // namespace synapse_helpers
