/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <synapse_api.h>
#include <synapse_common_types.h>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <utility>
#include "backend/helpers/event_dispatcher.h"
#include "backend/profiling/trace_sources/sources.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "backend/synapse_helpers/memory_defragmentation.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/towl.h"
#include "habana_lazy/memlog.h"
#include "pool_allocator/CoalescedStringentPoolAllocator.h"

namespace synapse_helpers {

void device_memory::init_hlml_memory() {
#ifdef PT_HLML_ENABLED
  try {
    m_hlml_memory_reporter =
        std::make_shared<HlMlMemoryReporter>(synDeviceId(device_.id()));

    auto get_used_memory = [&] {
      MemoryStats stats;
      get_memory_stats(&stats);
      return stats.bytes_in_use;
    };

    m_hlml_memory_updater = std::make_shared<HlMlMemoryUpdater>(
        m_hlml_memory_reporter, get_used_memory);
    PT_SYNHELPER_DEBUG("HLML memory reporeter initialized");
  } catch (const HlMlMemoryReporter::Error& e) {
    PT_SYNHELPER_WARN("Cannot initialize HLML memory reporter: ", e.what());
  }
#endif
}

device_memory::device_memory(device& device) : device_{device} {
  pool_size_ = GET_ENV_FLAG_NEW(PT_HABANA_POOL_SIZE, 1) * 1024 * 1024 * 1024;
  pool_strategy_ =
      (pool_allocator::PoolStrategyType)GET_ENV_FLAG_NEW(PT_HPU_POOL_STRATEGY);
  enable_mem_threshold_check = false;
  alignment_ = device_.get_device_memory_alignment();
  switch (pool_strategy_) {
    case pool_allocator::startegy_coalesce_stringent:
      try {
        PT_DEVMEM_DEBUG("startegy_coalesce_stringent:: ", pool_size_);
        suballoc_ = new pool_allocator::SubAllocator(
            new pool_allocator::CoalescedStringentPooling(device_));
        if (suballoc_ == nullptr) {
          PT_DEVMEM_FATAL("unable to create pool allocator");
        }
      } catch (...) {
        PT_DEVMEM_FATAL("unknown pool error");
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

  {
    std::array<uint64_t, 2> dram_infos = {0, 0};
    uint64_t* dram_info = dram_infos.data();
    std::array<synDeviceAttribute, 2> deviceAttrs = {
        DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS, DEVICE_ATTRIBUTE_DRAM_SIZE};
    synDeviceAttribute* deviceAttr = deviceAttrs.data();
    auto status = synDeviceGetAttribute(dram_info, deviceAttr, 2, device_.id());
    if (synStatus::synSuccess != status) {
      PT_DEVMEM_FATAL(
          Logger::formatStatusMsg(status), "Cannot obtain dram info.");
    }
    log_DRAM_start(dram_info[0]);
    log_DRAM_size(dram_info[1]);
  }

  init_hlml_memory();
}

device_memory::~device_memory() {
#ifdef PT_HLML_ENABLED
  m_hlml_memory_updater.reset();
  m_hlml_memory_reporter.reset();
#endif

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

// Note:: This is only in used defragementer where
// we wanted to reset and start the degramenter test.
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
  for (auto const& h2p : handle2pointer_) {
    if (h2p.ptr_size_.ptr_ == nullptr)
      continue;
    handle2pointer_.ResetHandlesMap(h2p.id_);
  }

  pool_size_ = GET_ENV_FLAG_NEW(PT_HABANA_POOL_SIZE, 1) * 1024 * 1024 * 1024;
  if (suballoc_ && !suballoc_->pool_create(device_.id(), pool_size_)) {
    PT_DEVMEM_FATAL("pool creation failed");
  }
  MemoryStats stats;
  get_memory_stats(&stats);
  PT_DEVMEM_DEBUG("POOL Creation Stats", stats.DebugString());
}

size_t device_memory::block_align(size_t n) {
  return (n + alignment_ - 1) & ~(alignment_ - 1);
}

bool device_memory::is_allocated(const device_ptr address) const {
  auto h = mem_handle::reinterpret_from_pointer(address);
  if (!h.is_valid()) {
    return false;
  }

  auto ptr_size = handle2pointer_.GetPtrSize(h.id());
  return (ptr_size.ptr_ != nullptr);
}

size_t device_memory::get_max_cntgs_chunk_size() const {
  return suballoc_->get_max_cntgs_chunk_size();
}

#define DEFAULT_RECIPE_COUNT 0

// warapper for malloc/free for pool startegy not equal to 5
synStatus device_memory::alloc(void** v_ptr, uint64_t size, bool is_workspace) {
  uint64_t ptr{0};
  synStatus status{synStatus::synSuccess};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    ptr =
        (uint64_t)suballoc_->pool_alloc_chunk(block_align(size), is_workspace);

    if ((void*)ptr == nullptr) {
      memory_reporter_event_create(device_, MEM_REPORTER_ALLOC_FAILS);
      PT_DEVMEM_DEBUG("pooling allocator failed, requested size ", size);
      status = synFail;
    }
    log_synDeviceMemStats(*this);
    *v_ptr = reinterpret_cast<void*>(ptr);
    PT_DEVMEM_DEBUG(
        "device_memory::allocate ptr=",
        to_hexstring(ptr),
        " aligned size::",
        block_align(size));
  } else {
    status = synDeviceMalloc(device_.id(), size, 0, 0, &ptr);

    if (synStatus::synSuccess != status) {
      PT_DEVMEM_DEBUG(
          Logger::formatStatusMsg(status),
          "synDeviceMalloc failed, requested size ",
          size);
    } else {
      *v_ptr = reinterpret_cast<void*>(ptr);
      log_synDeviceMemStats(*this);
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
    PT_DEVMEM_DEBUG(Logger::formatStatusMsg(status), "SynDeviceFree Failed.");
  }
  log_synDeviceMemStats(*this);
  return status;
}

synStatus device_memory::alloc(
    void** v_ptr,
    uint64_t size,
    hpuStream_t stream) {
  uint64_t ptr{0};
  synStatus status{synStatus::synSuccess};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    ptr = (uint64_t)suballoc_->pool_alloc_chunk(block_align(size), stream);

    if ((void*)ptr == nullptr) {
      memory_reporter_event_create(device_, MEM_REPORTER_ALLOC_FAILS);
      PT_DEVMEM_DEBUG("pooling allocator failed, requested size ", size);
      status = synFail;
    }
    log_synDeviceMemStats(*this);
    *v_ptr = reinterpret_cast<void*>(ptr);
    PT_DEVMEM_DEBUG(
        "device_memory::allocate ptr=",
        to_hexstring(ptr),
        " aligned size::",
        block_align(size));
  } else {
    status = synDeviceMalloc(device_.id(), size, 0, 0, &ptr);

    if (synStatus::synSuccess != status) {
      PT_DEVMEM_DEBUG(
          Logger::formatStatusMsg(status),
          "synDeviceMalloc failed, requested size ",
          size);
    } else {
      *v_ptr = reinterpret_cast<void*>(ptr);
      log_synDeviceMemStats(*this);
    }
  }

  return status;
}

synStatus device_memory::malloc(
    void** v_ptr,
    uint64_t size,
    hpuStream_t stream) {
  synStatus status{synStatus::synSuccess};
  uint64_t ptr{0};
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unique_lock<std::mutex> lock(mutex_);

    ptr = mem_handle::reinterpret_to_pointer(
        mem_handle(handle2pointer_.Insert(size, stream)));

    *v_ptr = reinterpret_cast<void*>(ptr);
  } else {
    status = alloc((void**)&ptr, size);
    *v_ptr = reinterpret_cast<void*>(ptr);
  }
  towl::emitDeviceMemoryAllocated(*v_ptr, size, stream);
  log_synDeviceMalloc(ptr, size, status);
  record(*v_ptr, size, true);
  return status;
}

synStatus device_memory::free_with_stream(void* free_ptr) {
  synStatus status{synStatus::synSuccess};
  if (nullptr == free_ptr) {
    return status;
  }

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    auto h = mem_handle::reinterpret_from_pointer(
        reinterpret_cast<uint64_t>(free_ptr));
    if (h.offset() != 0) {
      PT_DEVMEM_FATAL("Cannot free offseted handle ", h);
    }
    const auto id = h.id();
    std::unique_lock<std::mutex> lock(mutex_);
    if (handle2pointer_.checkIdIsReset(id))
      return status;
    auto ptr_and_size = handle2pointer_.GetPtrSize(id);
    bool is_stream_uses_empty = true;
    if (ptr_and_size.ptr_ != nullptr) {
      is_stream_uses_empty = suballoc_->is_stream_uses_empty(ptr_and_size.ptr_);
      deallocate(ptr_and_size.ptr_);
    }
    if (is_stream_uses_empty) {
      handle2pointer_.Erase(id);
      log_synDeviceMemStats(*this);
      log_synDeviceFree(reinterpret_cast<uint64_t>(free_ptr), status);
      towl::emitDeviceMemoryDeallocated(free_ptr);
      record(free_ptr, 0, false);
    } // TODO fixme if there are other stream, need to erase the h_id
  } else {
    status = deallocate(free_ptr);
    log_synDeviceFree(reinterpret_cast<uint64_t>(free_ptr), status);
    towl::emitDeviceMemoryDeallocated(free_ptr);
    record(free_ptr, 0, false);
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

  towl::emitDeviceMemoryAllocated(*v_ptr, size, 0);
  log_synDeviceMalloc(ptr, size, status);
  record(*v_ptr, size, true);
  return status;
}

synStatus device_memory::free(void* free_ptr) {
  synStatus status{synStatus::synSuccess};
  if (nullptr == free_ptr) {
    return status;
  }

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    auto h = mem_handle::reinterpret_from_pointer(
        reinterpret_cast<uint64_t>(free_ptr));
    if (h.offset() != 0) {
      PT_DEVMEM_FATAL("Cannot free offseted handle ", h);
    }
    const auto id = h.id();
    std::unique_lock<std::mutex> lock(mutex_);
    if (handle2pointer_.checkIdIsReset(id))
      return status;
    auto ptr_and_size = handle2pointer_.GetPtrSize(id);
    handle2pointer_.Erase(id);
    if (ptr_and_size.ptr_ != nullptr) {
      deallocate(ptr_and_size.ptr_);
    }
  } else {
    status = deallocate(free_ptr);
  }
  log_synDeviceFree(reinterpret_cast<uint64_t>(free_ptr), status);
  towl::emitDeviceMemoryDeallocated(free_ptr);
  record(free_ptr, 0, false);
  return status;
}

void device_memory::recordStream(void* ptr, hpuStream_t stream) {
  if (nullptr == ptr) {
    return;
  }
  PT_DEVMEM_DEBUG("record_stream for ptr", reinterpret_cast<uint64_t>(ptr));
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent &&
      common::IsRecordStreamEnabled()) {
    auto h =
        mem_handle::reinterpret_from_pointer(reinterpret_cast<uint64_t>(ptr));
    if (h.offset() != 0) {
      PT_DEVMEM_FATAL("Cannot free offseted handle ", h);
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const auto id = h.id();
    auto ptr_and_size = handle2pointer_.GetPtrSize(id);
    suballoc_->record_stream(ptr_and_size.ptr_, stream);
  }
}

void* device_memory::workspace_alloc(
    void* ptr,
    size_t& ws_size,
    size_t req_size) {
  if (pool_strategy_ != pool_allocator::startegy_coalesce_stringent) {
    size_t chunk_size = alignment_ * 1024 * 1024;
    size_t num_chunks = (req_size / chunk_size) + 1;
    size_t actual_size = num_chunks * chunk_size;
    if (ws_size >= actual_size) {
      return ptr;
    } else if (ws_size < actual_size) {
      auto& recipe_counter = device_.get_active_recipe_counter();
      while (recipe_counter.get_count() > DEFAULT_RECIPE_COUNT) {
        recipe_counter.wait_for_next_decrease_call();
      }

      habana_lazy::log_dev_mem_stats(
          "Post-Recipe-Decrease-Workspace", "", req_size);
      towl::emitDeviceMemorySummary("Post-Recipe-Decrease-Workspace");

      PT_DEVMEM_DEBUG(
          "requested size > size, free the buffer and reallocte current size::",
          ws_size,
          " requested size::",
          req_size);

      deallocate(ptr);
      record(ptr, 0, false);
    }
    void* v_ptr{nullptr};
    alloc(&v_ptr, actual_size, true);
    ws_size = actual_size;
    if (v_ptr == nullptr && pool_strategy_ != pool_allocator::strategy_none) {
      suballoc_->print_pool_stats();
      log_synDeviceAllocFail(device_, true, req_size);
    }
    record(v_ptr, ws_size, true);
    log_synDeviceMemStats(*this);
    return v_ptr;
  } else {
    if ((ws_size >= req_size) && (ptr != nullptr)) {
      return ptr;
    } else {
      void* v_ptr{nullptr};
      std::unique_lock<std::mutex> lock(defragmentation_mutex_);

      auto extend_high_memory_alloc = [&](size_t new_workspace_size,
                                          size_t curr_size) -> void* {
        std::unique_lock<std::mutex> lock(mutex_);
        void* v_ptr{nullptr};
        v_ptr = suballoc_->extend_high_memory_allocation(
            new_workspace_size, curr_size);
        return v_ptr;
      };
      auto& recipe_counter = device_.get_active_recipe_counter();
      v_ptr = extend_high_memory_alloc(block_align(req_size), ws_size);

      if (v_ptr == nullptr) {
        habana_lazy::log_dev_mem_stats("OOM-Workspace", "", req_size);
        towl::emitDeviceMemorySummary("OOM-Workspace");

        while (recipe_counter.get_count() > DEFAULT_RECIPE_COUNT) {
          recipe_counter.wait_for_next_decrease_call();
        }
        PT_DEVMEM_DEBUG(
            "workspace requested size::",
            block_align(req_size),
            " current size::",
            ws_size);
        habana_lazy::log_dev_mem_stats(
            "Post-Recipe-Decrease-Workspace", "", req_size);

        towl::emitDeviceMemorySummary("Post-Recipe-Decrease-Workspace");
        v_ptr = extend_high_memory_alloc(block_align(req_size), ws_size);
      }
      memory_reporter_event_create(device_, MEM_DEFRAGMENT_START);
      bool defragmentation_done = false;
      if (v_ptr == nullptr && device_.IsMemorydefragmentationEnabled()) {
        MemoryStats stats;
        get_memory_stats(&stats);
        PT_DEVMEM_DEBUG(
            "Workspace extension failed. Attempt to defragment memory.");
        PT_DEVMEM_DEBUG(
            "Memory Stats in case workspace failure", stats.DebugString());
        defragmentation_done = defragment_memory(alignment_, req_size, true);
      }

      if (defragmentation_done) {
        v_ptr = extend_high_memory_alloc(block_align(req_size), ws_size);
        memory_reporter_event_create(device_, MEM_DEFRAGMENT_SUCCESS);
      } else {
        memory_reporter_event_create(device_, MEM_DEFRAGMENT_FAIL);
      }

      if (v_ptr != nullptr) {
        ws_size = block_align(req_size);
        record(v_ptr, req_size - ws_size, true);
        log_synDeviceMemStats(*this);
      } else {
        suballoc_->print_pool_stats();
        MemoryStats stats;
        get_memory_stats(&stats);
        PT_DEVMEM_DEBUG(
            "Memory Stats in case workspace failure", stats.DebugString());
        log_synDeviceAllocFail(device_, true, block_align(req_size));
        memory_reporter_event_create(device_, MEM_REPORTER_OOM);
      }

      return v_ptr;
    }
  }
}

synStatus device_memory::workspace_free(void* ptr) {
  auto status = deallocate(ptr);
  log_synDeviceFree(reinterpret_cast<uint64_t>(ptr), status);
  record(ptr, 0, false);
  return status;
}

device_ptr device_memory::fix_address(void* ptr) {
  if (ptr == nullptr) {
    PT_DEVMEM_FATAL("fix_address ptr is null");
  }

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    auto h =
        mem_handle::reinterpret_from_pointer(reinterpret_cast<uint64_t>(ptr));

    if (h.offset() != 0) {
      PT_DEVMEM_FATAL("Cannot fix offseted handle ", h);
    }

    handle2pointer_.MarkMemoryFixed(h.id());
    return get_pointer(h);
  } else {
    return reinterpret_cast<uint64_t>(ptr);
  }
}

void device_memory::record_param(
    const std::string& name,
    const bool is_param,
    const bool is_grad,
    const bool is_optim_state,
    const uint64_t t_start,
    const uint64_t t_end) {
  log_synDeviceRecordTensorInfo(
      name, is_param, is_grad, is_optim_state, false, false, t_start, t_end);
}

void device_memory::check_and_limit_recipe_execution(size_t size) {
  auto& recipe_counter = device_.get_active_recipe_counter();
  PT_DEVMEM_DEBUG("Recipes in queue::", recipe_counter.get_count());
  if (recipe_counter.get_count() > DEFAULT_RECIPE_COUNT &&
      (size > alignment_ && !suballoc_->is_memory_available(size))) {
    uint64_t counter_state{0};
    do {
      counter_state = recipe_counter.wait_for_next_decrease_call();
      if (suballoc_->is_memory_available(size))
        break;
    } while (counter_state > DEFAULT_RECIPE_COUNT);

    habana_lazy::log_dev_mem_stats("Post-Recipe-Decrease-Lock-Addr", "", size);
    towl::emitDeviceMemorySummary("Post-Recipe-Decrease-Lock-Addr");
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

  HandleMover(
      mem_handle::id_t handle,
      void* source_pointer,
      size_t size,
      size_t actual_size,
      hpuStream_t stream)
      : handle_(handle),
        source_pointer_(source_pointer),
        destination_pointer_(nullptr),
        size_(size),
        actual_size_(actual_size),
        stream_(stream) {}

  void* GetSource() const {
    return source_pointer_;
  }

  void* GetDestination() const {
    return destination_pointer_;
  }

  bool moveRequired() const {
    return source_pointer_ != destination_pointer_;
  }

  size_t Size() const {
    return size_;
  }

  size_t ActualSize() const {
    return actual_size_;
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

  void Allocate(pool_allocator::SubAllocator& allocator, HandlesMap& h2pMap) {
    destination_pointer_ =
        allocator.pool_alloc_chunk(actual_size_, stream_, false);
    if (destination_pointer_ == nullptr) {
      PT_DEVMEM_FATAL("destination_pointer_ allocation failed");
    }
    h2pMap.SetPtrSize(
        handle_, HandlesMap::PtrSize(destination_pointer_, size_, stream_));
  }

  mem_handle::id_t handle_;
  void* source_pointer_;
  void* destination_pointer_;
  size_t size_;
  size_t actual_size_;
  hpuStream_t stream_;
};
} // namespace defragment
} // namespace

void device_memory::MoveData(
    device& dev,
    std::vector<std::tuple<uint64_t, uint64_t, size_t>> move_address) {
  std::vector<uint64_t> srcs(move_address.size());
  std::vector<uint64_t> dsts(move_address.size());
  std::vector<uint64_t> lens(move_address.size());
  for (std::size_t i = 0; i < move_address.size(); ++i) {
    srcs[i] = std::get<0>(move_address[i]);
    dsts[i] = std::get<1>(move_address[i]);
    lens[i] = std::get<2>(move_address[i]);
  }

  auto status = synMemCopyAsyncMultiple(
      dev.get_stream(0, DMA_D2D),
      srcs.data(),
      lens.data(),
      dsts.data(),
      synDmaDir::DRAM_TO_DRAM,
      move_address.size());
  if (synStatus::synSuccess != status) {
    PT_DEVMEM_FATAL(Logger::formatStatusMsg(status), "synMemCopyAsync failed");
  }

  auto& handle = device_.get_stream(0, DMA_D2D);
  if (synStatus::synSuccess != synStreamSynchronize(handle)) {
    PT_DEVMEM_FATAL(
        Logger::formatStatusMsg(status), "Waiting for Move complete failed");
  }
}

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
      Logger::_str_wrapper(std::this_thread::get_id()));
  if (!threads_in_defragmenter_critical_section_->wait_for(2s)) {
    PT_DEVMEM_FATAL(
        "Defragmentation cannot be started. Some allocated buffers are in use.",
        "It may be caused by device memory leak");
    return false;
  }

  auto timestamp_wait = std::chrono::high_resolution_clock::now();

  std::unique_lock<std::mutex> lock(mutex_);
  size_t total_moved_memory = 0;
  size_t total_moved_resources = 0;
  PT_DEVMEM_DEBUG("Starting memory defragmentation");
  PT_DEVMEM_DEBUG("Collecting memory information");
  defragment_helpers::MemoryDefragementer defragmenter(
      *suballoc_, handle2pointer_, alignment);

  std::vector<defragment_helpers::MemoryBlock> memory_blocks;
  try {
    if (!defragmenter.CollectMemoryInformation(memory_blocks)) {
      PT_DEVMEM_WARN(
          "Defragmentation cannot be started. Invalid memory information.");
      return false;
    }
  } catch (const std::exception& e) {
    PT_DEVMEM_FATAL("Exception in Collect Memory information...\n", e.what());
  }

  bool defragmentation_needed = true;
  std::unique_ptr<defragment_helpers::Region> region;
  PT_DEVMEM_DEBUG("Looking for regions to defragment");
  bool is_v2;
  if (!defragmenter.Run(
          memory_blocks,
          workspace_grow,
          allocation_size,
          defragmentation_needed,
          region,
          is_v2)) {
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. No region that can be defragmented was found.");
    return false;
  }

  if (not defragmentation_needed) {
    if (workspace_grow) {
      PT_DEVMEM_WARN(
          "There is enough memory free memory to allocate ",
          allocation_size,
          "B. Running defragmentation may indicate a bug");
    } else {
      PT_DEVMEM_WARN(
          "There is enough memory free memory to extend workspace by ",
          allocation_size,
          "B. Running defragmentation may indicate a bug");
    }
    return true;
  }

  if (not region) {
    lock.unlock();
    PT_DEVMEM_WARN(
        "Defragmentation cannot be started. There is not enough free memory.");
    habana_helpers::EmitEvent(
        habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
        habana_helpers::EventDispatcher::EventParams(
            {{"success", std::to_string(0)}}));
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

    movers.emplace_back(
        it->handle_, it->ptr_, it->size_, block_align(it->size_), it->stream_);
  }

  bool alloc_first = workspace_grow && is_v2;

  if (movers.empty()) {
    PT_DEVMEM_WARN("No defragemtantion was done");
  } else {
    PT_DEVMEM_DEBUG("Moving ", movers.size(), " resources");
    suballoc_->set_defragmenter_state(true);
    std::vector<std::tuple<uint64_t, uint64_t, size_t>> move_address;

    if (!alloc_first) {
      for (auto& mover : movers)
        mover.Deallocate(*suballoc_);
    }

    for (auto& mover : movers) {
      mover.Allocate(*suballoc_, handle2pointer_);

      if (not mover.moveRequired()) {
        PT_DEVMEM_DEBUG("Skipping. Resource was not moved in memory");
        continue;
      }
      auto previous_destination = mover.GetSource();
      auto destination = mover.GetDestination();
      if (previous_destination < destination) {
        if (static_cast<void*>(
                static_cast<int8_t*>(previous_destination) +
                mover.ActualSize()) > destination) {
          PT_DEVMEM_FATAL(
              "Defragmentation: New and old resource memory location is overlapping. Cannot move allocation");
        }
      }
      uint64_t src_base_addr = reinterpret_cast<uint64_t>(mover.GetSource());
      uint64_t dst_base_addr =
          reinterpret_cast<uint64_t>(mover.GetDestination());
      size_t size = mover.ActualSize();
      uint64_t src_end_addr = src_base_addr + size;
      uint64_t dst_end_addr = dst_base_addr + size;
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
        move_address.push_back({src_base_addr, dst_base_addr, size_base});

        size_t remaning_size = size - size_base;
        src_base_addr = src_base_addr + size_base;
        dst_base_addr = dst_base_addr + size_base;
        move_address.push_back({src_base_addr, dst_base_addr, remaning_size});
      } else {
        move_address.push_back({src_base_addr, dst_base_addr, size});
      }

      ++total_moved_resources;
      total_moved_memory += mover.ActualSize();
    }
    MoveData(device_, move_address);
    if (alloc_first) {
      for (auto& mover : movers)
        mover.Deallocate(*suballoc_);
    }
    PT_DEVMEM_DEBUG(
        "Move of resources completed no of resources::", movers.size());
    suballoc_->set_defragmenter_state(false);

    auto milliseconds_metric =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - timestamp_init)
            .count();
    lock.unlock();
    habana_helpers::EmitEvent(
        habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
        habana_helpers::EventDispatcher::EventParams(
            {{"success", std::to_string(1)},
             {"milliseconds", std::to_string(milliseconds_metric)}}));
  }
  PT_DEVMEM_DEBUG("defragmentation Done");
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

    std::string details;
    details += "Reason: ";
    if (workspace_grow) {
      details += "Workspace extension";
    } else {
      details += "Resource allocation";
    }
    details += ", total duration[ms]: " + std::to_string(total_time);
    details += ", wait duration in total[ms]: " + std::to_string(wait_time);
    details += ", number of moved allocations: " +
        std::to_string(total_moved_resources);
    details += ", amount of moved memory[bytes]: " +
        std::to_string(total_moved_memory);
    PT_DEVMEM_DEBUG("MemoryDefragmentation details:: ", details);
  }

  return true;
}

size_t device_memory::get_total_memory_required(
    absl::Span<const device_ptr> addresses) {
  size_t total_memory = 0;
  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unordered_map<mem_handle::id_t, size_t> umap_addr;
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto address : addresses) {
      auto h = mem_handle::reinterpret_from_pointer(address);
      if (!h.is_valid())
        continue;
      auto ptr_size = handle2pointer_.GetPtrSize(h.id());
      if (ptr_size.ptr_ == nullptr) {
        auto found = umap_addr.find(h.id());
        if (found == umap_addr.end()) {
          umap_addr[h.id()] = ptr_size.size_;
        }
      }
    }
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
  check_and_limit_recipe_execution(total_mem);

  std::vector<device_ptr> out;
  out.reserve(addresses.size());

  log_synDeviceLockMemory(addresses);

  if (pool_strategy_ == pool_allocator::startegy_coalesce_stringent) {
    std::unique_lock<std::mutex> lock(defragmentation_mutex_);
    for (const auto address : addresses) {
      const auto h = mem_handle::reinterpret_from_pointer(address);
      const auto translated_address = get_pointer(h);
      out.emplace_back(translated_address);
    }

    // update the out vector address to new  one
    // if defragmentor is run as this could change the device
    // address
    if (update_on_defragment_) {
      out.clear();
      for (const auto address : addresses) {
        const auto h = mem_handle::reinterpret_from_pointer(address);
        const auto translated_address = get_pointer(h);
        out.emplace_back(translated_address);
      }
    }
    update_on_defragment_ = false;
    return device_ptr_lock(absl::make_unique<defragment::Lock>(
        threads_in_defragmenter_critical_section_, std::move(out)));
  } else {
    for (const auto address : addresses) {
      out.emplace_back(address);
    }
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
      alloc(&ptr_size.ptr_, ptr_size.size_, ptr_size.stream_);
      handle2pointer_.SetPtrSize(h.id(), ptr_size);
    }
    return {ptr_size.ptr_, ptr_size.size_};
  };

  auto& recipe_counter = device_.get_active_recipe_counter();

  void* ptr = nullptr;
  size_t size = 0;
  std::tie(ptr, size) = get_and_alloc_mem();

  if (ptr == nullptr) {
    synapse_helpers::memstats_dump(
        device_,
        "Allocation failed, stats before waiting for recipies to finish.");

    habana_lazy::log_dev_mem_stats("OOM-Tensor", "", size);
    towl::emitDeviceMemorySummary("OOM-Tensor");

    // check and wait for recipe execution to complete
    uint64_t counter_state{0};
    if (recipe_counter.get_count() > DEFAULT_RECIPE_COUNT) {
      do {
        counter_state = recipe_counter.wait_for_next_decrease_call();
        PT_DEVMEM_DEBUG(
            "retrying memory alloc, ",
            "waiting for recipe launch completion, recipe count ",
            counter_state,
            " requested size ",
            size);
        std::tie(ptr, size) = get_and_alloc_mem();
      } while (counter_state > DEFAULT_RECIPE_COUNT && ptr == nullptr);

      habana_lazy::log_dev_mem_stats("Post-Recipe-Decrease-Tensor", "", size);
      towl::emitDeviceMemorySummary("Post-Recipe-Decrease-Tensor");
    }
  }

  if (ptr == nullptr) {
    MemoryStats stats;
    get_memory_stats(&stats);
    PT_DEVMEM_DEBUG("Memory Stats", stats.DebugString());
    PT_DEVMEM_DEBUG("Allocation failed for size::", size);
    /* defragment memory now*/
    memory_reporter_event_create(device_, MEM_DEFRAGMENT_START);
    bool defragmentation_done = false;
    if (device_.IsMemorydefragmentationEnabled()) {
      PT_DEVMEM_DEBUG(
          "Memory allocation failed. Attempt to defragment memory.");
      defragmentation_done = defragment_memory(alignment_, size, false);
    }
    if (defragmentation_done) {
      std::tie(ptr, size) = get_and_alloc_mem();
      update_on_defragment_ = true;
      memory_reporter_event_create(device_, MEM_DEFRAGMENT_SUCCESS);
    } else {
      memory_reporter_event_create(device_, MEM_DEFRAGMENT_FAIL);
    }

    habana_lazy::log_dev_mem_stats(
        "Post-Defrag", defragmentation_done ? "True" : "False", size);
    towl::emitDeviceMemorySummary("Post-Defrag");
  }

  if (ptr == nullptr) {
    habana_lazy::log_dev_mem_stats("OOM-FATAL", "", size);
    towl::emitDeviceMemorySummary("OOM-FATAL");
    suballoc_->print_pool_stats();
    synapse_helpers::memstats_dump(device_, "Allocation failed.");
    log_synDeviceAllocFail(device_, false, size);
    PT_DEVMEM_FATAL(
        "Allocation failed for size::",
        size,
        " (",
        static_cast<double>(size) / (1024 * 1024.),
        ")MB");
  }

  const auto offset = h.offset();

  if (offset >= size) {
    PT_DEVMEM_FATAL("Trying to access out of bounds of resource");
  }

  if (reinterpret_cast<device_ptr>(ptr) % alignment_ != 0) {
    PT_DEVMEM_FATAL("address not aligned");
  }
  return reinterpret_cast<device_ptr>(ptr) + offset;
}

void device_memory::get_memory_stats(MemoryStats* stats) {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->get_stats(stats);
  }
}

std::vector<std::pair<uint64_t, uint64_t>> device_memory::
    get_occupied_chunk_map() {
  std::vector<std::pair<uint64_t, uint64_t>> occupied_chunk_map{};
  if (pool_strategy_ != pool_allocator::strategy_none) {
    occupied_chunk_map = suballoc_->get_occupied_chunk_map();
  }
  return occupied_chunk_map;
}

void device_memory::clear_memory_stats() {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->clear_stats();
  }
}

void device_memory::reset_peak_memory_stats() {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    suballoc_->reset_peak_mem_stats();
  }
}

bool device_memory::is_memory_available(size_t size) {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    check_and_limit_recipe_execution(size);
    return suballoc_->is_memory_available(size);
  }
  return true;
}

bool device_memory::is_memory_available(
    size_t persistent_size,
    size_t curr_ws_size,
    size_t new_ws_size) {
  if (pool_strategy_ != pool_allocator::strategy_none) {
    return suballoc_->is_memory_available(
        persistent_size, curr_ws_size, new_ws_size);
  }
  return true;
}

synapse_helpers::MemoryReporter* device_memory::get_memory_reporter() {
  return &mem_reporter;
}

void device_memory::record(void* ptr, size_t size, bool alloc) {
  if (habana::profile::memory::enabled()) {
    MemoryStats stats;
    get_memory_stats(&stats);
    if (alloc) {
      habana::profile::memory::recordAllocation(
          reinterpret_cast<uint64_t>(ptr),
          size,
          stats.bytes_in_use + size,
          stats.memory_limit);
    } else {
      habana::profile::memory::recordDeallocation(
          reinterpret_cast<uint64_t>(ptr),
          stats.bytes_in_use,
          stats.memory_limit);
    }
  }
}

} // namespace synapse_helpers
