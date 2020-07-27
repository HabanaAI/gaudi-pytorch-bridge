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
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include <synapse_api.h>
#include "habana_helpers/logging.h"
#include "synapse_helpers/session.h"
#include "synapse_helpers/util.h"

namespace synapse_helpers {

// Since computation on stream is asynchronous, in order to share workspace
// buffer, it has to be fixed in size otherwise, there need to be implemented
// mechanism to adjust its size at runtime, but that would require an explcit
// barrier on the computation stream and reallocation of this buffer. For now
// it's fixed to 10GB, since for BERT SQUAD, batch12 on fp32, the largest recipe
// requires WS of size ~9.7GB
// TODO: as a WA for memory issue, modified it to 5GB for the resnet run of
// BS=64,
//       may require changes or revert in future
constexpr std::size_t GLOBAL_WORKSPACE_SIZE = 5e9;

std::weak_ptr<device> device::device_in_use;
std::mutex device::device_mtx;

device::device(
    std::shared_ptr<session> synapse_session,
    synDeviceId device_id,
    synDeviceType device_type,
    const create_allocator_fnc& create_allocator)
    : synapse_session_(std::move(synapse_session)),
      type_{device_type},
      id_{device_id},
      event_handle_cache_{*this},
      memory_mapper_{*this},
      stream_comp_{*this, stream_flavor::COMPUTE_0},
      stream_network_collective_{*this, stream_flavor::COLLECTIVE_0},
      stream_d2d_{*this, stream_flavor::DMA_D2D},
      stream_h2d_{*this, stream_flavor::DMA_H2D},
      stream_d2h_{*this, stream_flavor::DMA_D2H},
      recipe_handle_cache_{*this} {
  HABANA_ASSERT(create_allocator != nullptr);
  allocator_ = create_allocator(id_);

  uint64_t total_memory, free_memory;
  auto status = synDeviceGetMemoryInfo(id_, &free_memory, &total_memory);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        "Cannot obtain device memory size for allocation of global ws buffer");
  }
  // in case of simulator, there might not be 4GB of memory available, so as a
  // fallback solution workspace_buffer_ will be allocated to 70% of free memory
  // on the given device
  workspace_size_ = free_memory > GLOBAL_WORKSPACE_SIZE ? GLOBAL_WORKSPACE_SIZE
                                                        : 0.7 * free_memory;
  workspace_buffer_ =
      reinterpret_cast<device_ptr>(allocator_->alloc(workspace_size_));
}

synapse_error_v<std::shared_ptr<device>> device::get_or_create(
    synDeviceType device_type,
    const create_allocator_fnc& allocator) {
  std::lock_guard<std::mutex> lock(device_mtx);
  std::shared_ptr<device> device_ptr = device_in_use.lock();
  if (device_ptr != nullptr) {
    if (device_ptr->type() != device_type) {
      return synapse_error{"Process already acquired device of different type.",
                           synDeviceTypeMismatch};
    }
    return device_ptr;
  }

  return device::create(device_type, allocator);
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
  return synapse_error{"Device with given id is not open by anyone!",
                       synObjectNotInitialized};
}

synapse_error_v<std::shared_ptr<device>> device::create(
    synDeviceType device_type,
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

  if (std::getenv("ID") != nullptr) {
    // Required for  multi chip configuration
    status = synDeviceAcquireByModuleId(
        &new_device_id, std::stoll(std::getenv("ID")));
  } else {
    status = synDeviceAcquireByDeviceType(&new_device_id, device_type);
  }

  if (status != synSuccess) {
    return synapse_error{"Device acquire failed.", status};
  }

  std::shared_ptr<device> device_ptr{new device(
      synapse_session, new_device_id, device_type, create_allocator)};

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

device::~device() {
  PT_SYNHELPER_DEBUG("Device dectructor entry");

  framework_specific_cleanup_();

  // We should unmap all buffers BEFORE device is released.
  auto status = memory_mapper_.drop_cache();
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL("memory_mapper::drop_cache() failed. Status: ", status);
  }
}

void device::flush_stream_events() {
  for (int id = (int)stream_flavor::_BEGIN; id < (int)stream_flavor::_END;
       id += 1) {
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
      return stream_network_collective_;
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

// Note:StreamSync is removed b/w Ops and compute stream sync happens
// before any DMA operation. This can be changed or optimized further
// when we manage the tensor and recipe liftime in the kernel
synapse_error device::copy_data_to_device(
    void* cpu_data,
    device_ptr destination,
    size_t total_bytes,
    const event_done_callback& done_cb) {
  PT_SYNHELPER_DEBUG(
      "Copy CPU Tensor to Device ",
      cpu_data,
      " to ",
      (void*)destination,
      ", total_bytes=",
      total_bytes);
  synStatus status;
  if ((synapse_helpers::IsStreamSyncOptEnabled())) {
    PT_SYNHELPER_DEBUG("Sync on compute stream: ", stream_comp_);
    status = synStreamSynchronize(stream_comp_);
    if (synStatus::synSuccess != status) {
      return synapse_error{"copy_data_to_device: Compute stream sync failed.",
                           status};
    }
  }

  sem_.enqueue_wait_event(destination, stream_h2d_);
  memory_mapper::acquired_entry res{};
  void* mapped_cpu_data = cpu_data;

  res = memory_mapper_.map(total_bytes);
  if (res.status != synStatus::synSuccess) {
    // last resort option to drop cached mapped buffers
    PT_SYNHELPER_WARN(
        "Could not map memory on the device. Dropping cache for mapped buffers.");
    status = memory_mapper_.drop_cache();
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN(
          "Could not drop cache for mapped memory on the device: ", status);
      return synapse_error{
          "Could not drop cache for mapped memory on the device.", status};
    }
    res = memory_mapper_.map(total_bytes);
    if (synStatus::synSuccess != res.status) {
      return synapse_error{
          "Could not allocate and map memory even after cache drop.",
          res.status};
    }
  }

  std::copy(
      reinterpret_cast<uint8_t*>(cpu_data),
      reinterpret_cast<uint8_t*>(cpu_data) + total_bytes,
      res.ptr);
  mapped_cpu_data = res.ptr;

  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_h2d_);

  status = synMemCopyAsync(
      stream_h2d_,
      reinterpret_cast<uint64_t>(mapped_cpu_data),
      total_bytes,
      destination,
      synDmaDir::HOST_TO_DRAM);

  if (synStatus::synSuccess != status) {
    return synapse_error{"DMA to HPU start failed.", status};
  }

  sem_.add_producer({destination}, stream_h2d_, [this, res, done_cb]() {
    memory_mapper_.unmap(res);
    done_cb();
  });

  return {};
}

synapse_error device::copy_data_to_host(
    device_ptr device_data,
    void* destination,
    size_t total_bytes,
    const event_done_callback& done_cb) {
  PT_SYNHELPER_DEBUG(
      "Copy Device Tensor to CPU ",
      (void*)device_data,
      " ",
      destination,
      " total_bytes=",
      total_bytes);

  synStatus status;
  if ((synapse_helpers::IsStreamSyncOptEnabled())) {
    PT_SYNHELPER_DEBUG("Sync on compute stream: ", stream_comp_);
    status = synStreamSynchronize(stream_comp_);
    if (synStatus::synSuccess != status) {
      return synapse_error{"copy_data_to_host:Compute stream sync failed.",
                           status};
    }
  }
  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_d2h_);
  sem_.enqueue_wait_event(device_data, stream_d2h_);

  memory_mapper::acquired_entry res{};
  void* mapped_destination = destination;
  res = memory_mapper_.map(total_bytes);
  if (synStatus::synSuccess != res.status) {
    // last resort option to drop cached mapped buffers

    PT_SYNHELPER_WARN(
        "Could not map memory on the device. Dropping cache for mapped buffers.");
    status = memory_mapper_.drop_cache();
    if (synStatus::synSuccess != status)
      // return synapse_error{
      PT_SYNHELPER_WARN(
          "Could not drop cache for mapped memory on the device.");
    res = memory_mapper_.map(total_bytes);
    if (synStatus::synSuccess != res.status) {
      return synapse_error{
          "Could not allocate and map memory even after cache drop.",
          res.status};
    }
  }
  mapped_destination = res.ptr;

  status = synMemCopyAsync(
      stream_d2h_,
      device_data,
      total_bytes,
      reinterpret_cast<uint64_t>(mapped_destination),
      synDmaDir::DRAM_TO_HOST);
  if (synStatus::synSuccess != status) {
    return synapse_error{"DMA from HPU start failed.", status};
  }

  sem_.add_producer({}, stream_d2h_, [this, done_cb, res, destination]() {
    std::copy(
        res.ptr,
        res.ptr + res.acquired_size,
        reinterpret_cast<uint8_t*>(destination));
    memory_mapper_.unmap(res);
    done_cb();
  });

  return {};
}

synapse_error device::copy_data_within_device(
    device_ptr source,
    device_ptr destination,
    size_t total_bytes,
    event_done_callback unref_cb) {
  synStatus status;
  if ((synapse_helpers::IsStreamSyncOptEnabled())) {
    PT_SYNHELPER_DEBUG("Sync on compute stream: ", stream_comp_);
    status = synStreamSynchronize(stream_comp_);
    if (synStatus::synSuccess != status) {
      return synapse_error{
          "copy_data_within_device-Compute stream sync failed.", status};
    }
  }

  sem_.enqueue_wait_event(source, stream_d2d_);
  status = synMemCopyAsync(
      stream_d2d_, source, total_bytes, destination, synDmaDir::DRAM_TO_DRAM);
  if (synStatus::synSuccess != status) {
    return synapse_error{"DMA inside HPU start failed.", status};
  }

  sem_.add_producer({destination}, stream_d2d_, std::move(unref_cb));

  return {};
}

synapse_error device::copy_data_within_device(
    transfer_manifest const& transfers,
    event_done_callback unref_cb,
    stream* const next_operation_stream) {
  synStatus status;
  if ((synapse_helpers::IsStreamSyncOptEnabled())) {
    PT_SYNHELPER_DEBUG("Sync on compute stream: ", stream_comp_);
    status = synStreamSynchronize(stream_comp_);
    if (synStatus::synSuccess != status) {
      return synapse_error{
          "copy_data_within_device:Compute stream sync failed.", status};
    }
  }

  std::vector<std::uint64_t> srcs(transfers.size());
  std::vector<std::uint64_t> dsts(transfers.size());
  std::vector<std::uint64_t> lens(transfers.size());

  for (std::size_t i = 0; i < transfers.size(); ++i) {
    sem_.enqueue_wait_event(transfers[i].src, stream_d2d_);
    srcs[i] = transfers[i].src;
    dsts[i] = transfers[i].dst;
    lens[i] = transfers[i].bytes_to_transfer;
  }

  status = synMemCopyAsyncMultiple(
      stream_d2d_,
      srcs.data(),
      lens.data(),
      dsts.data(),
      synDmaDir::DRAM_TO_DRAM,
      transfers.size());
  if (synStatus::synSuccess != status) {
    return synapse_error{"dma inside hpu start failed.", status};
  }

  if (nullptr == next_operation_stream) {
    sem_.add_producer(std::move(dsts), stream_d2d_, std::move(unref_cb));
  } else {
    // If next operation stream is known then user wants us to put event on this
    // stream immediately and not pass it into the SEM.
    record_and_wait_for_event(
        stream_d2d_, *next_operation_stream, std::move(unref_cb));
  }
  return {};
}

device_ptr device::get_workspace_buffer(std::size_t size) const {
  if (size > workspace_size_) {
    PT_SYNHELPER_FATAL(
        "Requested buffer size for workspace(",
        size,
        ") is bigger than the available workspace size(",
        workspace_size_,
        ")!");
  }

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
