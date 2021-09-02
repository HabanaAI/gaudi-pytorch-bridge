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
  enable_mem_threshold_check = false;
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
    case pool_allocator::startegy_static_coalesce_with_memthreshold:
      /* this is additional startegy will be workaround for now,
       * we remove it later and enable for all startegy by default */
      enable_mem_threshold_check = true;
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
    case pool_allocator::startegy_coalesce_stringent:
      try {
        PT_SYNHELPER_DEBUG("startegy_coalesce_stringent:: ", pool_size_);
        uint64_t max_merge_count = GET_ENV_FLAG(PT_HPU_POOL_MAX_MERGE_COUNT);
        bool enable_lfu_merging = GET_ENV_FLAG(PT_HPU_POOL_ENABLE_LFU_MERGE);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::CoalescedStringentPooling(
                max_merge_count, enable_lfu_merging));
        if (suballoc_ == nullptr) {
          PT_SYNHELPER_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_SYNHELPER_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::strategy_none:
      suballoc_ = nullptr;
      break;
    default:
      PT_SYNHELPER_FATAL("unsupported pool strategy");
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
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    handle2pointer_.clear();
    handle_id_generator_.reset();
  }
  if (suballoc_) {
    suballoc_->pool_destroy();
    delete suballoc_;
  }
  suballoc_ = nullptr;
}

// warapper for malloc/free for pool startegy not equal to 5
synStatus device_memory::alloc(void** v_ptr, uint64_t size, bool is_workspace) {
  uint64_t ptr{0};
  synStatus status{synStatus::synSuccess};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    ptr = (uint64_t)suballoc_->pool_alloc_chunk(size, is_workspace);

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

  return status;
}

synStatus device_memory::deallocate(void* ptr) {
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
  return status;
}

synStatus device_memory::malloc(void** v_ptr, uint64_t size) {
  synStatus status{synStatus::synSuccess};
  uint64_t ptr{0};
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool inserted;
    decltype(handle2pointer_)::iterator iter;

    std::tie(iter, inserted) = handle2pointer_.emplace(
        handle_id_generator_.get(), ptr_with_size{nullptr, size});

    if (!inserted) {
      PT_SYNHELPER_FATAL("Handle ", mem_handle(iter->first), " already exists");
    }

    ptr = mem_handle::reinterpret_to_pointer(mem_handle(iter->first));

    *v_ptr = reinterpret_cast<void*>(ptr);
  } else {
    status = alloc((void**)&ptr, size);
    *v_ptr = reinterpret_cast<void*>(ptr);
  }

  log_synDeviceMalloc(ptr, size, status);
  return status;
}

synStatus device_memory::free(void* free_ptr) {
  synStatus status{synStatus::synSuccess};
  if (nullptr == free_ptr) {
    return status;
  }

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    if (reinterpret_cast<uint64_t>(free_ptr) == workspace_allocation_) {
      status = deallocate(free_ptr);
      log_synDeviceFree(reinterpret_cast<uint64_t>(free_ptr), status);
      return status;
    }

    auto h = mem_handle::reinterpret_from_pointer(
        reinterpret_cast<uint64_t>(free_ptr));

    if (h.offset() != 0) {
      PT_SYNHELPER_FATAL("Cannot free offseted handle ", h);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const auto id = h.id();
    auto iter = handle2pointer_.find(id);
    if (iter == handle2pointer_.end()) {
      PT_SYNHELPER_FATAL("Handle ", h, " does not exist");
    }

    void* ptr;
    size_t size;

    std::tie(ptr, size) = iter->second;
    if (ptr != nullptr) {
      status = deallocate(ptr);
    }

    handle2pointer_.erase(iter);
    handle_id_generator_.put(id);
  } else {
    status = deallocate(free_ptr);
  }
  log_synDeviceFree(reinterpret_cast<uint64_t>(free_ptr), status);
  return status;
}

bool device_memory::is_mem_threshold_hit() {
  if (!enable_mem_threshold_check)
    return false;
  if (pool_strategy_ != pool_allocator::strategy_none) {
    return suballoc_->is_mem_threshold_hit();
  }
  return false;
}

void* device_memory::workspace_alloc(
    void* ptr,
    size_t& ws_size,
    size_t req_size) {
  if (pool_strategy_ != pool_allocator::startegy_coalesce_stringent) {
    size_t chunk_size = 128 * 1024 * 1024;
    size_t num_chunks = (req_size / chunk_size) + 1;
    size_t actual_size = num_chunks * chunk_size;
    if (ws_size >= actual_size) {
      return ptr;
    } else if (ws_size < actual_size) {
      auto& recipe_counter = device_.get_active_recipe_counter();
      while (recipe_counter.get_count() > 1) {
        recipe_counter.wait_for_next_decrease_call();
      }
      PT_SYNHELPER_DEBUG(
          "requested size > size, free the buffer and reallocte current size::",
          ws_size,
          " requested size::",
          req_size);

      deallocate(ptr);
    }
    void* v_ptr{nullptr};
    alloc(&v_ptr, actual_size, true);
    ws_size = actual_size;
    return v_ptr;
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ws_size < req_size) {
      void* v_ptr{nullptr};
      v_ptr = suballoc_->extend_high_memory_allocation(req_size);
      workspace_allocation_ = reinterpret_cast<uint64_t>(v_ptr);
      ws_size = req_size;
      return v_ptr;
    }
    return ptr;
  }
}

// special case handling for preallocated buffer
void device_memory::fix_address(void* ptr) {
  if (ptr == nullptr) {
    PT_SYNHELPER_FATAL("fix_address ptr is null");
  }

  auto h =
      mem_handle::reinterpret_from_pointer(reinterpret_cast<uint64_t>(ptr));

  if (h.offset() != 0) {
    PT_SYNHELPER_FATAL("Cannot fix offseted handle ", h);
  }

  get_pointer(h);
}

device_ptr_lock device_memory::lock_addresses(
    const std::vector<device_ptr>& addresses) {
  std::vector<device_ptr> out;
  out.reserve(addresses.size());

  for (const auto address : addresses) {
    if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
      const auto h = mem_handle::reinterpret_from_pointer(address);
      const auto translated_address = get_pointer(h);
      out.emplace_back(translated_address);
    } else {
      out.emplace_back(address);
    }
  }
  return device_ptr_lock(std::move(out));
}

device_ptr device_memory::get_pointer(mem_handle h) {
  if (!h.is_valid()) {
    return device_nullptr;
  }

  auto get_and_alloc_mem = [&]() -> std::pair<void*, size_t> {
    void* ptr = nullptr;
    size_t size = 0;

    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = handle2pointer_.find(h.id());
    if (iter == handle2pointer_.end()) {
      PT_SYNHELPER_FATAL("Handle ", h.unoffseted(), " does not exist");
    }

    std::tie(ptr, size) = iter->second;
    if (ptr == nullptr) {
      alloc(&ptr, size);
      iter->second = ptr_with_size{ptr, size};
    }

    return {ptr, size};
  };

  void* ptr = nullptr;
  size_t size = 0;
  std::tie(ptr, size) = get_and_alloc_mem();

  if (ptr == nullptr) {
    // check and wait for recipe execution to complete
    auto& recipe_counter = device_.get_active_recipe_counter();
    uint32_t counter_state{0};
    if (!recipe_counter.is_zero()) {
      do {
        counter_state = recipe_counter.wait_for_next_decrease_call();
        PT_SYNHELPER_DEBUG(
            "retrying memory alloc, ",
            "waiting for recipe launch completion, recipe count ",
            counter_state,
            " requested size ",
            size);
        std::tie(ptr, size) = get_and_alloc_mem();
      } while (counter_state > 1 && ptr == nullptr);
    }
  }

  if (ptr == nullptr) {
    PT_SYNHELPER_FATAL("Allocation failed for size::", size);
  }

  const auto offset = h.offset();

  if (offset >= size) {
    PT_SYNHELPER_FATAL("Trying to access out of bounds of resource");
  }

  return reinterpret_cast<device_ptr>(ptr) + offset;
}

void device_memory::get_memory_stats(MemoryStats* stats) {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->get_stats(stats);
  }
}

void device_memory::clear_memory_stats() {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->clear_stats();
  }
}

} // namespace synapse_helpers
