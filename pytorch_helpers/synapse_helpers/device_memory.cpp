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
#include <unordered_map>
#include <utility>

#include <synapse_api.h>
#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/devmem_logger.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/memory_defragmentation.h"

namespace synapse_helpers {
device_memory::device_memory(device& device) : device_{device} {
  pool_size_ = GET_ENV_FLAG_NEW(PT_HABANA_POOL_SIZE) * 1024 * 1024 * 1024;
  pool_strategy_ =
      (pool_allocator::PoolStrategyType)GET_ENV_FLAG_NEW(PT_HPU_POOL_STRATEGY);
  enable_mem_threshold_check = false;
  switch (pool_strategy_) {
    case pool_allocator::strategy_bump:
      try {
        PT_DEVMEM_DEBUG("strategy_bump with size :: ", pool_size_);
        suballoc_ =
            new pool_allocator::SubAllocator(new pool_allocator::StaticPooling);
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::strategy_dynamic:
      try {
        PT_DEVMEM_DEBUG("strategy_dynamic :: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::DynamicPooling);
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::startegy_static_coalesce_with_memthreshold:
      /* this is additional startegy will be workaround for now,
       * we remove it later and enable for all startegy by default */
      enable_mem_threshold_check = true;
      try {
        PT_DEVMEM_DEBUG("startegy_static_coalesce :: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::StaticCoalescedPooling);
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::startegy_static_coalesce:
      try {
        PT_DEVMEM_DEBUG("startegy_static_coalesce :: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::StaticCoalescedPooling);
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::startegy_coalesce_stringent:
      try {
        PT_DEVMEM_DEBUG("startegy_coalesce_stringent:: ", pool_size_);
        uint64_t max_merge_count =
            GET_ENV_FLAG_NEW(PT_HPU_POOL_MAX_MERGE_COUNT);
        bool enable_lfu_merging =
            GET_ENV_FLAG_NEW(PT_HPU_POOL_ENABLE_LFU_MERGE);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::CoalescedStringentPooling(
                max_merge_count, enable_lfu_merging));
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error ");
      }
      break;
    case pool_allocator::strategy_none:
      suballoc_ = nullptr;
      break;
    default:
      PT_DEVMEM_FATAL("unsupported pool strategy");
      break;
  }
  if (suballoc_ &&
      !suballoc_->pool_create(device_.id(), block_align(pool_size_))) {
    PT_DEVMEM_FATAL("pool creation failed");
  }

  if (pool_strategy_ != pool_allocator::startegy_static_coalesce) {
    std::array<uint64_t, 2> dram_infos = {0, 0};
    uint64_t* dram_info = dram_infos.data();
    std::array<synDeviceAttribute, 2> deviceAttrs = {
        DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS, DEVICE_ATTRIBUTE_DRAM_SIZE};
    synDeviceAttribute* deviceAttr = deviceAttrs.data();
    auto status = synDeviceGetAttribute(dram_info, deviceAttr, 2, device_.id());
    if (synStatus::synSuccess != status) {
      PT_DEVMEM_FATAL("Cannot obtain dram info. Status: ", status);
    }
    log_DRAM_start(dram_info[0]);
    log_DRAM_size(dram_info[1]);
  }
}

device_memory::~device_memory() {
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    if (!threads_in_defragmenter_critical_section_->empty())
      PT_DEVMEM_DEBUG(
          "Some allocated buffers are in use during device memory destructor call.",
          " It may be caused by device memory leak");
  }
  if (suballoc_) {
    suballoc_->pool_destroy();
    delete suballoc_;
  }
  suballoc_ = nullptr;
}

void device_memory::reset_pool() {
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    if (!threads_in_defragmenter_critical_section_->empty())
      PT_DEVMEM_DEBUG(
          "Some allocated buffers are in use during device memory destructor call."
          "It may be caused by device memory leak");
  }
  if (suballoc_) {
    suballoc_->pool_destroy();
  }
  if (suballoc_ && !suballoc_->pool_create(device_.id(), pool_size_)) {
    PT_DEVMEM_FATAL("pool creation failed");
  }
}

size_t device_memory::block_align(size_t n) {
  return (n + DEFAULT_ALIGNMENT - 1) & ~(DEFAULT_ALIGNMENT - 1);
}

// warapper for malloc/free for pool startegy not equal to 5
synStatus device_memory::alloc(void** v_ptr, uint64_t size, bool is_workspace) {
  uint64_t ptr{0};
  synStatus status{synStatus::synSuccess};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    ptr =
        (uint64_t)suballoc_->pool_alloc_chunk(block_align(size), is_workspace);

    if ((void*)ptr == nullptr) {
      PT_DEVMEM_DEBUG("pooling allocator failed, requested size ", size);
      status = synFail;
    }

    *v_ptr = reinterpret_cast<void*>(ptr);
  } else {
    status = synDeviceMalloc(device_.id(), size, 0, 0, &ptr);

    if (synStatus::synSuccess != status) {
      PT_DEVMEM_DEBUG("synDeviceMalloc failed, requested size ", size);
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
    PT_DEVMEM_DEBUG("SynDeviceFree Failed.", status);
  }
  return status;
}

synStatus device_memory::malloc(void** v_ptr, uint64_t size) {
  synStatus status{synStatus::synSuccess};
  uint64_t ptr{0};
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unique_lock<std::mutex> lock(mutex_);

    ptr = mem_handle::reinterpret_to_pointer(
        mem_handle(handle2pointer_.Insert(size)));

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
      PT_DEVMEM_FATAL("Cannot free offseted handle ", h);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const auto id = h.id();
    auto ptr_and_size = handle2pointer_.GetPtrSize(id);
    handle2pointer_.Erase(id);
    if (ptr_and_size.ptr_ != nullptr) {
      deallocate(ptr_and_size.ptr_);
    }
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
    size_t chunk_size = DEFAULT_ALIGNMENT * 1024 * 1024;
    size_t num_chunks = (req_size / chunk_size) + 1;
    size_t actual_size = num_chunks * chunk_size;
    if (ws_size >= actual_size) {
      return ptr;
    } else if (ws_size < actual_size) {
      auto& recipe_counter = device_.get_active_recipe_counter();
      while (recipe_counter.get_count() > 1) {
        recipe_counter.wait_for_next_decrease_call();
      }
      PT_DEVMEM_DEBUG(
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
    if ((ws_size >= req_size) && (ptr != nullptr)) {
      return ptr;
    } else {
      void* v_ptr{nullptr};
      std::unique_lock<std::mutex> lock(defragmentation_mutex_);

      auto extend_high_memory_alloc = [&](size_t new_workspace_size) -> void* {
        std::unique_lock<std::mutex> lock(mutex_);
        void* v_ptr{nullptr};
        v_ptr = suballoc_->extend_high_memory_allocation(new_workspace_size);
        return v_ptr;
      };

      v_ptr = extend_high_memory_alloc(block_align(req_size));

      bool defragmentation_done = false;
      if (v_ptr == nullptr && device_.IsMemorydefragmentationEnabled()) {
        PT_DEVMEM_WARN(
            "Workspace extension failed. Attempt to defragment memory.");
        defragmentation_done =
            defragment_memory(DEFAULT_ALIGNMENT, req_size, true);
      }

      if (defragmentation_done) {
        v_ptr = extend_high_memory_alloc(req_size);
      }

      if (v_ptr != nullptr) {
        workspace_allocation_ = reinterpret_cast<uint64_t>(v_ptr);
        ws_size = req_size;
      } else {
        workspace_allocation_ = 0;
      }

      return v_ptr;
    }
  }
}

// special case handling for preallocated buffer
void device_memory::fix_address(void* ptr) {
  if (ptr == nullptr) {
    PT_DEVMEM_FATAL("fix_address ptr is null");
  }

  auto h =
      mem_handle::reinterpret_from_pointer(reinterpret_cast<uint64_t>(ptr));

  if (h.offset() != 0) {
    PT_DEVMEM_FATAL("Cannot fix offseted handle ", h);
  }

  handle2pointer_.MarkMemoryFixed(h.id());
  get_pointer(h);
}

/* recipe count is incremented before the allocation
 * of device memory for the the tensor, so the
 * default count is 1 which includes the current recipe
 * we are doing allocation.
 */
#define DEFAULT_RECIPE_COUNT 1

void device_memory::check_and_limit_recipe_execution(size_t size) {
  auto& recipe_counter = device_.get_active_recipe_counter();
  PT_DEVMEM_DEBUG("Recipes in queue", recipe_counter.get_count());
  uint32_t counter_state{0};
  if (recipe_counter.get_count() < DEFAULT_RECIPE_COUNT)
    return;

  if (recipe_counter.get_count() > device_.GetMaxRecipeLimitInQueue() ||
      !suballoc_->is_memory_available(size)) {
    do {
      counter_state = recipe_counter.wait_for_next_decrease_call();
    } while (counter_state > DEFAULT_RECIPE_COUNT);
  }
}

namespace {
namespace defragment {
class Lock : public device_ptr_lock_interface {
 public:
  Lock(std::shared_ptr<synchronous_counter> counter, std::vector<device_ptr>&&);
  ~Lock() override;
  Lock(Lock&&) = delete;
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;
  Lock& operator=(Lock&&) = delete;

  device_ptr_lock_interface::iterator_t begin() const override {
    return locked_addresses_.data();
  }
  device_ptr_lock_interface::iterator_t end() const override {
    return locked_addresses_.data() + locked_addresses_.size();
  }
  device_ptr at(size_t position) const override {
    return locked_addresses_.at(position);
  }

 private:
  std::shared_ptr<synchronous_counter> counter_;
  std::vector<device_ptr> locked_addresses_;
};

Lock::Lock(
    std::shared_ptr<synchronous_counter> counter,
    std::vector<device_ptr>&& addresses)
    : counter_(counter), locked_addresses_(std::move(addresses)) {
  counter_->increment();
}

Lock::~Lock() {
  counter_->decrement();
}

struct HandleMover {
  HandleMover() = default;

  HandleMover(mem_handle::id_t handle, void* source_pointer, size_t size)
      : handle_(handle),
        source_pointer_(source_pointer),
        destination_pointer_(nullptr),
        size_(size) {}

  void* GetDestination() const {
    return destination_pointer_;
  }

  bool moveRequired() const {
    return source_pointer_ != destination_pointer_;
  }

  bool operator<(const HandleMover& rhs) const {
    return source_pointer_ < rhs.source_pointer_;
  }

  void Deallocate(pool_allocator::SubAllocator& allocator) const {
    if (source_pointer_ == nullptr) {
      PT_DEVMEM_FATAL("source_pointer_ not set. Deallocation not possible.");
    }
    allocator.pool_free_chunk(source_pointer_);
  }

  void Allocate(
      pool_allocator::SubAllocator& allocator,
      HandlesMap& h2pMap,
      bool workspace) {
    destination_pointer_ = allocator.pool_alloc_chunk(size_, workspace);
    if (destination_pointer_ == nullptr) {
      PT_DEVMEM_FATAL("destination_pointer_ allocation failed");
    }
    h2pMap.SetPtrSize(
        handle_, HandlesMap::PtrSize(destination_pointer_, size_));
  }

  void MoveData(device& dev) const {
    if (!moveRequired()) {
      return;
    }

    if (destination_pointer_ == nullptr) {
      PT_DEVMEM_FATAL(
          "destination_pointer_ not set. Moving data not possible.");
    }

    uint64_t src_base_addr = reinterpret_cast<uint64_t>(source_pointer_);
    uint64_t dst_base_addr = reinterpret_cast<uint64_t>(destination_pointer_);
    uint64_t src_end_addr = src_base_addr + size_;
    uint64_t dst_end_addr = dst_base_addr + size_;

    if (!(dst_end_addr <= src_base_addr || src_end_addr <= dst_base_addr)) {
      PT_DEVMEM_DEBUG(
          "Address overlapping...",
          "src_base_addr::",
          src_base_addr,
          " src_end_addr::",
          src_end_addr,
          " dst_base_addr::",
          dst_base_addr,
          " dst_end_addr::",
          dst_end_addr);
      size_t size_base = dst_end_addr - src_base_addr;
      src_base_addr = src_base_addr;
      dst_base_addr = dst_base_addr;

      auto status = synMemCopyAsync(
          dev.get_device_to_device_stream(),
          src_base_addr,
          size_base,
          dst_base_addr,
          synDmaDir::DRAM_TO_DRAM);
      if (synStatus::synSuccess != status) {
        PT_DEVMEM_FATAL("synMemCopyAsync failed ", status);
      }

      size_t remaning_size = size_ - size_base;
      src_base_addr = src_base_addr + size_base;
      dst_base_addr = dst_base_addr + size_base;
      status = synMemCopyAsync(
          dev.get_device_to_device_stream(),
          src_base_addr,
          remaning_size,
          dst_base_addr,
          synDmaDir::DRAM_TO_DRAM);
      if (synStatus::synSuccess != status) {
        PT_DEVMEM_FATAL("synMemCopyAsync failed ", status);
      }

    } else {
      auto status = synMemCopyAsync(
          dev.get_device_to_device_stream(),
          reinterpret_cast<uint64_t>(source_pointer_),
          size_,
          reinterpret_cast<uint64_t>(destination_pointer_),
          synDmaDir::DRAM_TO_DRAM);
      if (synStatus::synSuccess != status) {
        PT_DEVMEM_FATAL("synMemCopyAsync failed ", status);
      }
    }
  }

  mem_handle::id_t handle_;
  void* source_pointer_;
  void* destination_pointer_;
  size_t size_;
};
} // namespace defragment
} // namespace

bool device_memory::defragment_memory(
    size_t alignment,
    size_t allocation_size,
    bool workspace_grow) {
  using namespace std::chrono_literals;
  auto timestamp_init = std::chrono::high_resolution_clock::now();

  PT_DEVMEM_DEBUG("Waiting for HPU execution to finish");
  if (synStatus::synSuccess != synDeviceSynchronize(device_.id())) {
    PT_DEVMEM_FATAL("Waiting for HPU execution failed");
  }

  PT_DEVMEM_DEBUG(
      "Waiting for threads to leave critical section ",
      std::hex,
      std::this_thread::get_id());
  if (!threads_in_defragmenter_critical_section_->wait_for(2s)) {
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. Some allocated buffers are in use.",
        "It may be caused by device memory leak");
    return false;
  }

  auto timestamp_wait = std::chrono::high_resolution_clock::now();

  std::unique_lock<std::mutex> lock(mutex_);

  PT_DEVMEM_DEBUG("Collecting memory information");
  defragment_helpers::MemoryDefragementer defragmenter(
      *suballoc_, handle2pointer_, alignment);

  std::vector<defragment_helpers::MemoryBlock> memory_blocks;
  if (!defragmenter.CollectMemoryInformation(memory_blocks)) {
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. Invalid memory information.");
    return false;
  }

  std::unique_ptr<defragment_helpers::Region> region;
  PT_DEVMEM_DEBUG("Looking for regions to defragment");
  if (!defragmenter.Run(
          memory_blocks, workspace_grow, allocation_size, region)) {
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. No region that can be defragmented was found.");
    return false;
  }

  if (not region) {
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. There is not enough free memory.");
    return false;
  }

  std::vector<defragment::HandleMover> movers;
  for (auto it = region->begin_; it != region->end_; ++it) {
    if (it->state_ == defragment_helpers::MemoryState::FIXED) {
      PT_DEVMEM_FATAL(
          "Defragmentation algorithm error. Trying to move fixed memory region");
    }

    if (it->state_ == defragment_helpers::MemoryState::FREE) {
      continue;
    }

    movers.emplace_back(it->handle_, it->ptr_, it->size_);
  }

  if (movers.empty()) {
    PT_DEVMEM_WARN(
        "No defragemtantion was done. Waiting for HPU execution to finish and free resources made enough ",
        "free space for a new allocation");
  }

  if (not movers.empty()) {
    PT_DEVMEM_WARN("Starting memory defragmentation");

    PT_DEVMEM_DEBUG("Deallocating resources");
    for (auto const& mover : movers) {
      mover.Deallocate(*suballoc_);
    }

    PT_DEVMEM_DEBUG("Moving resources, number of resources: ", movers.size());
    void* previous_destination = nullptr;
    auto counter = 0;
    for (auto& mover : movers) {
      PT_DEVMEM_DEBUG("Moving resource ", ++counter, " out of ", movers.size());

      mover.Allocate(*suballoc_, handle2pointer_, workspace_grow);
      void* destination = mover.GetDestination();
      // Check if reallocated positions are in the same order as prior to
      // defragmentation. If that's not the case, data copying could results in
      // overwrites between chunks.
      if (previous_destination > destination) {
        PT_DEVMEM_FATAL(
            "Unordered destination pointers during defragmentation");
      }
      previous_destination = destination;

      mover.MoveData(device_);
    }

    if (synStatus::synSuccess != synDeviceSynchronize(device_.id())) {
      PT_DEVMEM_FATAL("Waiting for Move complete failed");
    }

    PT_DEVMEM_DEBUG("Moving resources finished");
  }

  if (device_.IsMemorydefragmentationInfoEnabled()) {
    auto total_duration =
        std::chrono::high_resolution_clock::now() - timestamp_init;
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_duration)
            .count();

    auto wait_duration = timestamp_wait - timestamp_init;
    auto wait_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(wait_duration)
            .count();

    auto in_use_memory = region->in_use_memory_;
    std::string details;
    details += "Reason: ";
    if (workspace_grow) {
      details += "Workspace extension";
    } else {
      details += "Resource allocation";
    }
    details += ", total duration[ms]: " + std::to_string(total_time);
    details += ", wait duration in total[ms]: " + std::to_string(wait_time);
    details +=
        ", number of moved allocations: " + std::to_string(movers.size());
    details +=
        ", amount of moved memory[bytes]: " + std::to_string(in_use_memory);
    PT_DEVMEM_DEBUG("MemoryDefragmentation details:: ", details);
  }

  return true;
}

size_t device_memory::get_total_memory_required(
    absl::Span<const device_ptr> addresses) {
  size_t total_memory = 0;
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unordered_map<device_ptr, size_t> umap_addr;
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto address : addresses) {
      auto h = mem_handle::reinterpret_from_pointer(address);
      if (!h.is_valid())
        continue;
      auto ptr_size = handle2pointer_.GetPtrSize(h.id());
      if (ptr_size.ptr_ == nullptr) {
        auto found = umap_addr.find(address);
        if (found == umap_addr.end()) {
          umap_addr[address] = ptr_size.size_;
        }
      }
    }
    size_t total_memory = 0;
    for (const auto addr : umap_addr) {
      total_memory += block_align(addr.second);
    }
  }

  return total_memory;
}

device_ptr_lock device_memory::lock_addresses(
    absl::Span<const device_ptr> addresses) {
  auto total_mem = get_total_memory_required(addresses);
  // no of recipes in queue if it exceeds a limit
  // there is unpredictable behaviour because of
  // resource contraint. so limit the recipes in
  // queue.
  if (device_.GetMaxRecipeLimitInQueue() > 0 ||
      (total_mem > DEFAULT_ALIGNMENT &&
       !suballoc_->is_memory_available(total_mem)))
    check_and_limit_recipe_execution(total_mem);
  std::vector<device_ptr> out;
  out.reserve(addresses.size());

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unique_lock<std::mutex> lock(defragmentation_mutex_);
    for (const auto address : addresses) {
      const auto h = mem_handle::reinterpret_from_pointer(address);
      const auto translated_address = get_pointer(h);
      out.emplace_back(translated_address);
    }
    return device_ptr_lock(absl::make_unique<defragment::Lock>(
        threads_in_defragmenter_critical_section_, std::move(out)));
  } else {
    for (const auto address : addresses)
      out.emplace_back(address);
    return device_ptr_lock(absl::make_unique<defragment::Lock>(
        threads_in_defragmenter_critical_section_, std::move(out)));
  }
}

device_ptr device_memory::get_pointer(mem_handle h) {
  if (!h.is_valid()) {
    return device_nullptr;
  }

  auto get_and_alloc_mem = [&]() -> std::pair<void*, size_t> {
    std::unique_lock<std::mutex> lock(mutex_);

    auto ptr_size = handle2pointer_.GetPtrSize(h.id());
    if (ptr_size.ptr_ == nullptr) {
      alloc(&ptr_size.ptr_, ptr_size.size_);
      handle2pointer_.SetPtrSize(h.id(), ptr_size);
    }
    return {ptr_size.ptr_, ptr_size.size_};
  };

  void* ptr = nullptr;
  size_t size = 0;
  std::tie(ptr, size) = get_and_alloc_mem();

  if (ptr == nullptr) {
    synapse_helpers::memstats_dump(
        device_,
        "Allocation failed, stats before waiting for recipies to finish.");

    // check and wait for recipe execution to complete
    auto& recipe_counter = device_.get_active_recipe_counter();
    uint32_t counter_state{0};
    if (!recipe_counter.is_zero()) {
      do {
        counter_state = recipe_counter.wait_for_next_decrease_call();
        PT_DEVMEM_DEBUG(
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
    MemoryStats stats;
    get_memory_stats(&stats);
    PT_DEVMEM_DEBUG("Memory Stats", stats.DebugString());
    PT_DEVMEM_DEBUG("Allocation failed for size::", size);
    /* defragment memory now*/
    bool defragmentation_done = false;
    if (device_.IsMemorydefragmentationEnabled()) {
      PT_DEVMEM_WARN("Memory allocation failed. Attempt to defragment memory.");
      defragmentation_done = defragment_memory(DEFAULT_ALIGNMENT, size, false);
    }
    if (defragmentation_done) {
      std::tie(ptr, size) = get_and_alloc_mem();
    }
  }

  if (ptr == nullptr) {
    synapse_helpers::memstats_dump(device_, "Allocation failed.");
    PT_DEVMEM_FATAL(
        "Allocation failed for size::",
        size,
        " (",
        size / (1024 * 1024.),
        ")MB");
  }

  const auto offset = h.offset();

  if (offset >= size) {
    PT_DEVMEM_FATAL("Trying to access out of bounds of resource");
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
