/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/device.h"

#include <absl/types/variant.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <synapse_api.h>
#include "habana_helpers/logging.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/session.h"
#include "synapse_helpers/util.h"

namespace synapse_helpers {
/**
 * END: These will be removed when all lazy kernels start using shape
 * functions.
 */

std::string get_mem_str(uint64_t nbytes) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(12) << nbytes << " bytes <";
  uint64_t gb{0x40000000};
  if (nbytes > gb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / gb << " GB ";
    nbytes %= gb;
  }
  uint64_t mb{0x100000};
  if (nbytes > mb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / mb << " MB ";
    nbytes %= mb;
  }
  uint64_t kb{0x400};
  if (nbytes > kb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / kb << " KB ";
    nbytes %= kb;
  }
  oss << std::setfill('0') << std::setw(4) << nbytes << " B>";

  return oss.str();
}

// Since computation on stream is asynchronous, in order to share workspace
// buffer, it has to be fixed in size otherwise, there need to be implemented
// mechanism to adjust its size at runtime, but that would require an explcit
// barrier on the computation stream and reallocation of this buffer. For now
// it's fixed to 10GB, since for BERT SQUAD, batch12 on fp32, the largest
// recipe requires WS of size ~9.7GB
// TODO: as a WA for memory issue, modified it to 5GB for the resnet run of
// BS=64,
//       may require changes or revert in future
constexpr std::size_t GLOBAL_WORKSPACE_SIZE = 5e9;

std::weak_ptr<device> device::device_in_use;
std::mutex device::device_mtx;

void active_recipe_counter::increase() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  ++counter_state_;
}

void active_recipe_counter::decrease_and_notify() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  --counter_state_;
  cv_.notify_all();
}

bool active_recipe_counter::is_zero() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  bool zflag = (0 == counter_state_ ? true : false);
  return zflag;
}

uint32_t active_recipe_counter::wait_for_next_decrease_call() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  if (counter_state_ > 0) {
    cv_.wait_for(cond_lock, std::chrono::seconds(1));
  }
  return counter_state_;
}

uint32_t active_recipe_counter::get_count() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  return counter_state_;
}

device::device(
    std::shared_ptr<session> synapse_session,
    synDeviceId device_id,
    synDeviceType device_type,
    const create_allocator_fnc& create_allocator)
    : synapse_session_(std::move(synapse_session)),
      type_{device_type},
      id_{device_id},
      event_handle_cache_{*this, 0},
      time_event_handle_cache_{*this, EVENT_COLLECT_TIME},
      memory_mapper_{*this},
      stream_comp_{*this, stream_flavor::COMPUTE_0},
      // Network collective should not be created without hcl
      stream_network_collective_ptr_{nullptr},
      stream_d2d_{*this, stream_flavor::DMA_D2D},
      stream_h2d_{*this, stream_flavor::DMA_H2D},
      stream_d2h_{*this, stream_flavor::DMA_D2H},
      recipe_handle_cache_{*this},
      host_memory_{*this},
      device_memory_{*this} {
  HABANA_ASSERT(create_allocator != nullptr);
  allocator_ = create_allocator(id_);

  is_hcl_same_addr_enabled_ =
      GET_ENV_FLAG_NEW(PT_ENABLE_HCL_SAME_ADDRESS_RESOLUTION) &&
      GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);

  // use the first allocated buffer always for same_address functionality
  // if each rank uses the same address for the recv/intermediate addresses;
  // then we can use the same address and it will save the address resolution
  // (since the address is known)
  if (is_hcl_same_addr_enabled_ && (std::getenv("ID") != nullptr)) {
    size_t prealloc_size = 2ULL * 1024 * 1024 * 1024; // 2GByte
    void* v_ptr{nullptr};
    device_memory_.malloc(&v_ptr, prealloc_size);
    device_ptr prealloc_addr = reinterpret_cast<device_ptr>(v_ptr);
    device_memory_.fix_address(reinterpret_cast<void*>(prealloc_addr));
    HABANA_ASSERT(prealloc_addr != device_nullptr);
    preallocated_reduction_buffer_ = absl::make_optional<owned_device_ptr>(
        prealloc_addr, prealloc_size, *this);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_INITIAL_WORKSPACE_SIZE) > 0) {
    uint64_t total_memory, free_memory;
    auto status = synDeviceGetMemoryInfo(id_, &free_memory, &total_memory);
    if (synStatus::synSuccess != status) {
      PT_SYNHELPER_FATAL(
          "Cannot obtain device memory size for allocation of global ws buffer");
    }

    // in case of simulator, there might not be 4GB of memory available, so as a
    // fallback solution workspace_buffer_ will be allocated to 70% of free
    // memory on the given device
    size_t global_workspace_size = get_workspace_size();
    auto req_size = free_memory > global_workspace_size ? global_workspace_size
                                                        : 0.7 * free_memory;
    workspace_buffer_ = get_workspace_buffer(req_size);

    PT_SYNHELPER_DEBUG(
        "Allocating static workspace at ",
        (void*)workspace_buffer_,
        " size ",
        synapse_helpers::get_mem_str(workspace_size_));
  }

  is_caching_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_CACHING);
  is_stream_async_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC);
  host_memory_cache_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HOST_MEMORY_CACHE);
  max_dma_copy_retry_count_ =
      GET_ENV_FLAG_NEW(PT_HABANA_MAX_DMA_COPY_RETRY_COUNT);
  dma_copy_retry_delay_ = std::chrono::milliseconds(
      GET_ENV_FLAG_NEW(PT_HABANA_DMA_COPY_RETRY_DELAY));
  max_recipe_limit_in_queue_ =
      GET_ENV_FLAG_NEW(PT_HPU_MAX_RECIPE_SUBMISSION_LIMIT);
}

synapse_error_v<std::shared_ptr<device>> device::get_or_create(
    const std::set<synDeviceType>& allowed_device_types,
    const create_allocator_fnc& allocator) {
  std::lock_guard<std::mutex> lock(device_mtx);
  std::shared_ptr<device> device_ptr = device_in_use.lock();
  if (device_ptr != nullptr) {
    if (!allowed_device_types.count(device_ptr->type())) {
      return synapse_error{
          "Process already acquired device of different type.",
          synDeviceTypeMismatch};
    }
    return device_ptr;
  }

  return device::create(allowed_device_types, allocator);
}

synapse_error_v<std::shared_ptr<device>> device::get_by_id(
    synDeviceId requested_id) {
  std::lock_guard<std::mutex> lock(device_mtx);
  std::shared_ptr<device> device_ptr = device_in_use.lock();
  if (device_ptr != nullptr) {
    if (requested_id == device_ptr->id()) {
      return device_ptr;
    }
  }
  return synapse_error{
      "Device with given id is not open by anyone!", synObjectNotInitialized};
}

synapse_error_v<std::shared_ptr<device>> device::create(
    const std::set<synDeviceType>& allowed_device_types,
    const create_allocator_fnc& create_allocator) {
  PT_SYNHELPER_DEBUG("synHPU Init");
  uint32_t new_device_id;
  synStatus status{synStatus::synSuccess};

  if (create_allocator == nullptr) {
    return synapse_helpers::synapse_error{
        "You should pass non null create_allocator_fnc for device creation.",
        synInvalidArgument};
  }

  auto synapse_session_create_result{synapse_helpers::session::get_or_create()};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          synapse_session_create_result)) {
    auto error = absl::get<synapse_helpers::synapse_error>(
        synapse_session_create_result);
    return error;
  }

  auto synapse_session =
      synapse_helpers::get_value(std::move(synapse_session_create_result));

  synDeviceType acquired_device_type = synDeviceGaudi;
  if (std::getenv("ID") != nullptr) {
    // Required for  multi chip configuration
    status = synDeviceAcquireByModuleId(
        &new_device_id, std::stoll(std::getenv("ID")));
    if (status == synSuccess) {
      synDeviceInfo dinfo;
      auto status_info = synDeviceGetInfo(new_device_id, &dinfo);
      if (status_info != synSuccess) {
        return synapse_error{"Device get info failed.", status_info};
      }
      acquired_device_type = dinfo.deviceType;
    }
  } else {
    for (auto const& device_type : allowed_device_types) {
      status = synDeviceAcquireByDeviceType(&new_device_id, device_type);
      if (status == synSuccess) {
        PT_SYNHELPER_DEBUG(
            "Device acquire successful for device_type: ", device_type);
        acquired_device_type = device_type;
        break;
      } else {
        PT_SYNHELPER_DEBUG(
            "Device acquire failed for device_type: ",
            device_type,
            " with status ",
            status);
      }
    }
  }

  if (status != synSuccess) {
    return synapse_error{"Device acquire failed.", status};
  }

  std::shared_ptr<device> device_ptr{new device(
      synapse_session, new_device_id, acquired_device_type, create_allocator)};

  uint64_t free_mem, total_mem;
  status = synDeviceGetMemoryInfo(device_ptr->id(), &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL("Cannot obtain device memory size. Status: ", status);
  }
  PT_SYNHELPER_DEBUG(
      "Device memory size: total=", total_mem, " free=", free_mem);

  // assign weak_ptr for future gets.
  device_in_use = device_ptr;
  return device_ptr;
}

int device::get_count() {
  int count = 0;
  synStatus status{synStatus::synSuccess};
  status = synDeviceGetCountByDeviceType((uint32_t*)&count, type_);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG("Fail to get device count. Status: ", status);
  }

  return count;
}

// GLOBAL_WORKSPACE_SIZE is set based on PT_HPU_WORKSPACE_SIZE in GB
uint64_t device::get_workspace_size() {
  uint64_t workspaceSize = GLOBAL_WORKSPACE_SIZE;
  auto init_size = GET_ENV_FLAG_NEW(PT_HPU_INITIAL_WORKSPACE_SIZE);
  if (init_size > 0) {
    workspaceSize = init_size * 1024 * 1024 * 1024;
    if (workspaceSize == 0) {
      PT_DEVICE_DEBUG("WorkSpace size not specified, setting default");
      workspaceSize = GLOBAL_WORKSPACE_SIZE;
    }
  }
  PT_DEVICE_DEBUG("WorkSpace size requested for :: ", workspaceSize);
  return workspaceSize;
}

void device::cleanup() {
  if (cleanup_done_) {
    return;
  }
  cleanup_done_ = true;
  flush_stream_events();

  if (is_hcl_same_addr_enabled_ && (std::getenv("ID") != nullptr)) {
    device_ptr prealloc_addr = preallocated_reduction_buffer_->get();
    allocator_->free((void*)prealloc_addr);
  }
  // free workspace buffer
  {
    std::unique_lock<std::mutex> lock(ws_mutex_);
    allocator_->free(reinterpret_cast<void*>(workspace_buffer_));
  }

  framework_specific_cleanup_();

  // We should unmap all buffers BEFORE device is released.
  auto status = memory_mapper_.drop_cache();
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL("memory_mapper::drop_cache() failed. Status: ", status);
  }
}

device::~device() {
  PT_SYNHELPER_DEBUG("Device dectructor entry");
  cleanup();
}

void device::flush_stream_events() {
  for (int id = (int)stream_flavor::_BEGIN; id < (int)stream_flavor::_END;
       id += 1) {
    // do not flush collective, if it was not created before
    if (id == (int)stream_flavor::COLLECTIVE_0 &&
        !stream_network_collective_ptr_)
      continue;
    get_stream(static_cast<stream_flavor>(id)).flush();
  }
  auto start = std::chrono::steady_clock::now();
  while (true) {
    if (sem_.is_flushed())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  PT_SYNHELPER_DEBUG(
      "stream flush completed in ",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

stream& device::get_stream(stream_flavor id) {
  switch (id) {
    case stream_flavor::COMPUTE_0:
      return stream_comp_;
    case stream_flavor::DMA_D2D:
      return stream_d2d_;
    case stream_flavor::DMA_H2D:
      return stream_h2d_;
    case stream_flavor::DMA_D2H:
      return stream_d2h_;
    case stream_flavor::COLLECTIVE_0:
      HABANA_ASSERT(stream_network_collective_ptr_);
      return *stream_network_collective_ptr_;
    default:
      PT_SYNHELPER_FATAL("Invalid stream id ", id);
      std::terminate();
  }
}
std::ostream& operator<<(std::ostream& stream, const device& syn_device) {
  stream << "synDevice at " << &syn_device;
  switch (syn_device.type()) {
    case synDeviceGaudi:
      stream << " Gaudi ";
      break;
    case synDeviceGaudiM:
      stream << " GaudiM ";
      break;
    case synDeviceGaudi2:
      stream << " Gaudi2 ";
      break;
    default:
      stream << " UNKNOWN ";
  }
  auto flag_guard = synapse_helpers::ostream_flag_guard::create(stream);
  stream << std::hex << syn_device.id() << std::dec;
  return stream;
}

device_ptr device::malloc(size_t size) {
  return reinterpret_cast<device_ptr>(allocator_->alloc(size));
}

void device::free(device_ptr ptr) {
  return allocator_->free(reinterpret_cast<void*>(ptr));
}

synapse_error device::copy_data_to_device(
    void* cpu_data,
    device_ptr destination,
    device_ptr event_addr,
    size_t total_bytes,
    const event_done_callback& done_cb,
    bool is_pinned) {
  PT_SYNHELPER_DEBUG(
      "Copy CPU Tensor to Device ",
      cpu_data,
      " to ",
      (void*)destination,
      ", total_bytes=",
      total_bytes);
  synStatus status;

  /* in case of write, we can invoke a fill (compute)
   * stream or via DMA. if we have a fill and a copy
   * Need to wait for the fill compute stream to complete
   * before copy, so wait */
  sem_.enqueue_wait_event(event_addr, stream_h2d_);
  void* mapped_cpu_data = cpu_data;
  uint8_t* dst_ptr;
  if (!is_pinned) {
    status = host_memory_.malloc((void**)&dst_ptr, total_bytes);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN("Host malloc failed: ", status);
      return synapse_error{"Host Malloc failed with status.", status};
    }
    std::copy(
        reinterpret_cast<uint8_t*>(cpu_data),
        reinterpret_cast<uint8_t*>(cpu_data) + total_bytes,
        dst_ptr);
    mapped_cpu_data = dst_ptr;
  } else {
    PT_SYNHELPER_DEBUG("copy_data_to_device uses Pinned memory");
  }

  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_h2d_);

  unsigned attempt = 0;
  do {
    auto locked = lock_addresses(destination);
    status = synMemCopyAsync(
        stream_h2d_,
        reinterpret_cast<uint64_t>(mapped_cpu_data),
        total_bytes,
        locked.at(0),
        synDmaDir::HOST_TO_DRAM);

    if (status == synStatus::synSuccess) {
      if (attempt != 0) {
        PT_SYNHELPER_WARN(
            "DMA to HPU start succeeded on ", attempt + 1, " attempt.");
      }
      break;
    } else if (attempt < max_dma_copy_retry_count_ - 1) {
      PT_SYNHELPER_WARN(
          "DMA to HPU start failed with status ",
          status,
          ". Attempt ",
          attempt + 1,
          "/",
          max_dma_copy_retry_count_,
          ".");
      std::this_thread::sleep_for(dma_copy_retry_delay_);
    } else {
      if (!is_pinned) {
        host_memory_.free((void*)dst_ptr);
      }
      return synapse_error{"DMA to HPU start failed.", status};
    }
  } while (++attempt < max_dma_copy_retry_count_);

  sem_.add_producer(
      {event_addr}, stream_h2d_, [this, dst_ptr, is_pinned, done_cb]() {
        if (!is_pinned)
          host_memory_.free((void*)dst_ptr);
        done_cb();
      });

  return {};
}

synapse_error device::copy_data_to_host(
    device_ptr device_data,
    void* destination,
    device_ptr event_addr,
    size_t total_bytes,
    const event_done_callback& done_cb,
    bool is_pinned) {
  PT_SYNHELPER_DEBUG(
      "Copy Device Tensor to CPU ",
      (void*)device_data,
      " ",
      destination,
      " total_bytes=",
      total_bytes);

  synStatus status;
  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_d2h_);
  sem_.enqueue_wait_event(event_addr, stream_d2h_);

  void* mapped_destination = destination;
  uint8_t* dst_ptr;
  if (!is_pinned) {
    status = host_memory_.malloc((void**)&dst_ptr, total_bytes);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN("Host malloc failed: ", status);
      return synapse_error{"Host Malloc failed with status.", status};
    }
    mapped_destination = dst_ptr;
  } else {
    PT_SYNHELPER_DEBUG("copy_data_to_host uses Pinned memory");
  }

  unsigned attempt = 0;
  do {
    auto locked = lock_addresses(device_data);
    status = synMemCopyAsync(
        stream_d2h_,
        locked.at(0),
        total_bytes,
        reinterpret_cast<uint64_t>(mapped_destination),
        synDmaDir::DRAM_TO_HOST);
    if (status == synStatus::synSuccess) {
      if (attempt != 0) {
        PT_SYNHELPER_WARN(
            "DMA from HPU start succeeded on ", attempt + 1, " attempt.");
      }
      break;
    } else if (attempt < max_dma_copy_retry_count_ - 1) {
      PT_SYNHELPER_WARN(
          "DMA from HPU start failed with status ",
          status,
          ". Attempt ",
          attempt + 1,
          "/",
          max_dma_copy_retry_count_,
          ".");
      std::this_thread::sleep_for(dma_copy_retry_delay_);
    } else {
      if (!is_pinned) {
        host_memory_.free((void*)dst_ptr);
      }
      return synapse_error{"DMA from HPU start failed.", status};
    }
  } while (++attempt < max_dma_copy_retry_count_);

  sem_.add_producer(
      {},
      stream_d2h_,
      [this, done_cb, dst_ptr, total_bytes, destination, is_pinned]() {
        if (!is_pinned) {
          std::copy(
              dst_ptr,
              dst_ptr + total_bytes,
              reinterpret_cast<uint8_t*>(destination));
          host_memory_.free((void*)dst_ptr);
        }
        done_cb();
      });

  return {};
}

synapse_error device::copy_data_within_device(
    device_ptr source,
    device_ptr destination,
    device_ptr src_event_addr,
    device_ptr dst_event_addr,
    size_t total_bytes,
    event_done_callback unref_cb) {
  synStatus status;

  sem_.enqueue_wait_event(src_event_addr, stream_d2d_);
  {
    auto locked = lock_addresses(source, destination);
    status = synMemCopyAsync(
        stream_d2d_,
        locked.at(0),
        total_bytes,
        locked.at(1),
        synDmaDir::DRAM_TO_DRAM);
    if (synStatus::synSuccess != status) {
      return synapse_error{"DMA inside HPU start failed.", status};
    }
  }

  sem_.add_producer({dst_event_addr}, stream_d2d_, std::move(unref_cb));

  return {};
}

synapse_error device::copy_data_within_device(
    transfer_manifest const& transfers,
    event_done_callback unref_cb,
    stream* const next_operation_stream) {
  synStatus status;

  std::vector<std::uint64_t> all_addresses(2 * transfers.size());
  std::vector<std::uint64_t> dsts(transfers.size());
  std::vector<std::uint64_t> lens(transfers.size());
  std::vector<std::uint64_t> dsts_event_addr(transfers.size());

  for (std::size_t i = 0; i < transfers.size(); ++i) {
    sem_.enqueue_wait_event(transfers[i].src_event_addr, stream_d2d_);
    all_addresses[i] = transfers[i].src;
    dsts[i] = all_addresses[i + transfers.size()] = transfers[i].dst;
    lens[i] = transfers[i].bytes_to_transfer;
    dsts_event_addr[i] = transfers[i].dst_event_addr;
  }

  {
    auto locked = lock_addresses(all_addresses);
    absl::Span<device_ptr> locked_srcs{locked.data(), transfers.size()};
    absl::Span<device_ptr> locked_dsts{
        locked.data() + transfers.size(), transfers.size()};

    status = synMemCopyAsyncMultiple(
        stream_d2d_,
        locked_srcs.data(),
        lens.data(),
        locked_dsts.data(),
        synDmaDir::DRAM_TO_DRAM,
        transfers.size());
    if (synStatus::synSuccess != status) {
      return synapse_error{"dma inside hpu start failed.", status};
    }
  }

  if (nullptr == next_operation_stream) {
    sem_.add_producer(
        std::move(dsts_event_addr), stream_d2d_, std::move(unref_cb));
  } else {
    // If next operation stream is known then user wants us to put event on
    // this stream immediately and not pass it into the SEM.
    record_and_wait_for_event(
        stream_d2d_, *next_operation_stream, std::move(unref_cb));
  }
  return {};
}

device_ptr device::get_workspace_buffer(size_t size) {
  std::unique_lock<std::mutex> lock(ws_mutex_);
  if (size == 0) {
    PT_SYNHELPER_DEBUG("workspace Allocation of size is zero");
    return 0;
  }
  void* buffer = device_memory_.workspace_alloc(
      (void*)workspace_buffer_, workspace_size_, size);
  if (buffer == nullptr) {
    PT_SYNHELPER_FATAL("workspace Allocation of size ::", size, " failed!");
  }
  workspace_buffer_ = reinterpret_cast<device_ptr>(buffer);
  PT_SYNHELPER_DEBUG(
      "Allocated workspace buffer at",
      (void*)workspace_buffer_,
      " size ",
      synapse_helpers::get_mem_str(workspace_size_));

  return workspace_buffer_;
}

void device::add_wait_events_on_stream(
    const std::vector<device_ptr>& input_tensors,
    stream& stream) {
  for (const auto& input_addr : input_tensors) {
    PT_SYNHELPER_DEBUG("Wait event address", std::hex, input_addr)
    sem_.enqueue_wait_event(input_addr, stream);
  }
}

void device::add_wait_event_on_stream(
    const std::string& event_id,
    stream& stream) {
  sem_.enqueue_wait_event(event_id, stream);
}

void device::register_producer_on_stream(
    std::vector<device_ptr>&& bound_addresses,
    stream& stream,
    event_done_callback done_cb) {
  sem_.add_producer(std::move(bound_addresses), stream, std::move(done_cb));
}

void device::register_producer_on_stream(
    std::vector<device_ptr>&& bound_addresses,
    const std::string& event_id,
    stream& stream,
    event_done_callback done_cb) {
  sem_.add_producer(
      std::move(bound_addresses), event_id, stream, std::move(done_cb));
}

void device::add_event_id(
    const std::string& event_id,
    const std::string& new_id) {
  sem_.add_event_id(event_id, new_id);
}

void device::wait_until_event_ready(const std::string& event_id) {
  sem_.wait_until_done(event_id);
}

void device::wait_until_address_ready(device_ptr address) {
  sem_.wait_until_done(address);
}

void device::wait_for_event(shared_event& event) {
  sem_.wait_until_done(event);
}

void device::record_and_wait_for_event(
    stream& record_stream,
    stream& other_stream,
    event_done_callback done_callback) {
  auto event_ref = std::make_shared<event>(
      get_event_handle_cache(),
      record_stream,
      std::vector<device_ptr>{},
      "",
      std::move(done_callback));
  record_stream.register_pending_event(event_ref);
  event_ref->stream_wait_event(other_stream);
}

std::set<synDeviceType> device::get_supported_devices() {
  return {
      synDeviceType::synDeviceGaudi,
      synDeviceType::synDeviceGaudiM,
      synDeviceType::synDeviceGaudi2};
}

void device::synchronize() {
  auto status = synDeviceSynchronize(id_);
  if (status != synSuccess) {
    PT_SYNHELPER_FATAL("synDeviceSynchronize failed. Status: ", status);
  }
}

void owned_device_ptr::device_ptr_deleter::operator()(device_ptr* ptr) {
  if (ptr) {
    PT_SYNHELPER_DEBUG(
        "Free buffer ptr ", std::hex, reinterpret_cast<device_ptr>(ptr));
    device_->free(reinterpret_cast<device_ptr>(ptr));
  }
}

device_id::~device_id() {
  if (id_ != device::INVALID_ID) {
    auto status = synDeviceRelease(id_);
    if (status != synSuccess) {
      PT_SYNHELPER_FATAL("synDeviceRelease failed with. Status: ", status);
    }
  }
}

} // namespace synapse_helpers
