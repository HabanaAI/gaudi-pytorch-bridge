/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "device_context.h"
#include <absl/memory/memory.h>
#include <absl/types/optional.h>
#include <absl/types/variant.h>
#include <synapse_common_types.h>
#include <condition_variable>
#include <iterator>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/stream.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h"
#include "hccl_types.h"
#include "status_conversion.h"
#include "synapse_api.h"

namespace hccl_integration {

device_context::device_context(int device_id) {
  open_device(device_id);
}

device_context::~device_context() {
  if (device_) {
    device_->cleanup();
    device_ = nullptr;
  }
}

hcclResult_t device_context::open_device(int device_id) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::open_device(device_id=", device_id, ")");
  std::lock_guard<std::mutex> guard{access_mutex_};

  auto maybe_dev_handle =
      synapse_helpers::device::get_by_id(static_cast<synDeviceId>(device_id));
  if (!ok(maybe_dev_handle)) {
    RETURN_ON_SYNAPSE_ERROR(get_error(maybe_dev_handle));
  }
  synapse_helpers::device_handle device =
      synapse_helpers::get_value(maybe_dev_handle);
  device_ = device;
  return hcclSuccess;
}

hcclResult_t device_context::acquire_collective_stream(
    synStreamHandle* stream_handle_ptr) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::acquire_collective_stream(stream_handle_ptr=",
      stream_handle_ptr,
      ")");

  std::lock_guard<std::mutex> guard{access_mutex_};

  if (nullptr == stream_handle_ptr) {
    PT_DISTRIBUTED_DEBUG("Unexpected nullptr as parameter!");
    return hcclInvalidArgument;
  }

  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL("Uninitialized device");
    return hcclInvalidArgument;
  }
  synapse_helpers::hpuStream_t stream;
  device_->create_stream(stream, true);

  synapse_helpers::stream& stream_handle =
      device_->get_stream(stream, synapse_helpers::NETWORK);
  HABANA_ASSERT(nullptr != stream_handle);

  stream_objects_[stream_handle] = &stream_handle;
  hpustream_handle_map_[stream_handle] = stream;

  *stream_handle_ptr = stream_handle;
  return hcclSuccess;
}

hcclResult_t device_context::release_stream(synStreamHandle stream_handle) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::release_stream(stream_handle=",
      stream_handle,
      ')');
  std::lock_guard<std::mutex> guard{access_mutex_};
  if (nullptr == stream_handle) {
    PT_DISTRIBUTED_WARN("Stream handle should not be null!");
    return hcclInvalidArgument;
  }
  device_->delete_stream(hpustream_handle_map_[stream_handle]);
  hpustream_handle_map_.erase(stream_handle);

  stream_objects_[stream_handle] = nullptr;
  return hcclSuccess;
}

hcclResult_t device_context::lock_address(
    void* const address,
    void** device_address) {
  std::lock_guard<std::mutex> guard{access_mutex_};
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::lock_address(address=",
      address,
      ", device_address=",
      device_address,
      ")");

  if (nullptr == device_address) {
    PT_DISTRIBUTED_FATAL("Unexpected null pointer passed!");
    return hcclInvalidArgument;
  }

  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before allocating memory.");
    return hcclInvalidUsage;
  }

  synapse_helpers::device_ptr_lock locked{device_->lock_addresses(
      reinterpret_cast<synapse_helpers::device_ptr>(address))};
  auto locked_address = reinterpret_cast<void*>(locked.at(0));
  *device_address = locked_address;
  return hcclSuccess;
}

hcclResult_t device_context::lock_address(
    std::vector<void*> addresses,
    std::unique_ptr<synapse_helpers::device_ptr_lock>& locked) {
  std::lock_guard<std::mutex> guard{access_mutex_};
  PT_DISTRIBUTED_DEBUG("Calling device_context::lock_address");

  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before allocating memory.");
    return hcclInvalidUsage;
  }

  std::vector<synapse_helpers::device_ptr> dev_addresses;
  for (auto& address : addresses)
    dev_addresses.push_back(
        reinterpret_cast<synapse_helpers::device_ptr>(address));

  locked = absl::make_unique<synapse_helpers::device_ptr_lock>(
      device_->lock_addresses(
          absl::Span<const synapse_helpers::device_ptr>(dev_addresses)));

  return hcclSuccess;
}

hcclResult_t device_context::lock_address(
    void* const address,
    void** device_address,
    std::unique_ptr<synapse_helpers::device_ptr_lock>& locked) {
  std::lock_guard<std::mutex> guard{access_mutex_};
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::lock_address(address=",
      address,
      ", device_address=",
      device_address,
      ")");

  if (nullptr == device_address) {
    PT_DISTRIBUTED_FATAL("Unexpected null pointer passed!");
    return hcclInvalidArgument;
  }

  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before allocating memory.");
    return hcclInvalidUsage;
  }

  locked = std::make_unique<synapse_helpers::device_ptr_lock>(
      device_->lock_addresses(
          reinterpret_cast<synapse_helpers::device_ptr>(address)));
  auto locked_address = reinterpret_cast<void*>(locked->at(0));
  *device_address = locked_address;
  return hcclSuccess;
}

synapse_helpers::active_recipe_counter& device_context::
    get_active_recipe_counter() {
  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before get_active_recipe_counter");
  }
  return device_->get_active_recipe_counter();
}

hcclResult_t device_context::prepare_stream(
    synStreamHandle stream_handle,
    synapse_helpers::device_ptr input_address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::prepare_stream(stream_handle=",
      stream_handle,
      ", input_address=",
      (void*)input_address,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized! (", stream_handle, ")");
    return hcclInvalidArgument;
  }

  HABANA_ASSERT(nullptr != device_);

  std::vector<synapse_helpers::device_ptr> addresses;
  addresses.push_back(input_address);

  device_->add_wait_events_on_stream(
      addresses, *stream_objects_[stream_handle]);

  return hcclSuccess;
}

std::vector<synapse_helpers::shared_event> device_context::
    prepare_stream_and_get_events(
        synStreamHandle stream_handle,
        synapse_helpers::device_ptr input_address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::prepare_stream_and_get_events(stream_handle=",
      stream_handle,
      ", input_address=",
      (void*)input_address,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized! (", stream_handle, ")");
  }

  HABANA_ASSERT(nullptr != device_);

  std::vector<synapse_helpers::device_ptr> addresses;
  addresses.push_back(input_address);

  return device_->get_wait_events_on_stream(
      addresses, *stream_objects_[stream_handle]);
}

synapse_helpers::stream& device_context::get_stream_fromhandle(
    synStreamHandle stream_handle) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::prepare_stream_and_get_events(stream_handle=",
      stream_handle,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized! (", stream_handle, ")");
  }

  return *stream_objects_[stream_handle];
}

hcclResult_t device_context::submit_events(
    synStreamHandle stream_handle,
    synapse_helpers::device_ptr output_address,
    const synapse_helpers::event_done_callback& done_callback) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::submit_events(stream_handle=",
      stream_handle,
      ", output_address=",
      (void*)output_address,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized!");
    return hcclInvalidArgument;
  }

  HABANA_ASSERT(nullptr != device_);

  std::vector<synapse_helpers::device_ptr> addresses;
  addresses.push_back(output_address);

  device_->register_producer_on_stream(
      std::move(addresses), *stream_objects_[stream_handle], done_callback);
  return hcclSuccess;
}

hcclResult_t device_context::submit_future(
    synapse_helpers::device_ptr device_addr,
    std::shared_future<bool> fut) {
  HABANA_ASSERT(nullptr != device_);

  device_->submit_future(device_addr, std::move(fut));
  return hcclSuccess;
}

hcclResult_t device_context::stream_synchronize(synStreamHandle stream) {
  return to_hccl_result(synStreamSynchronize(stream));
}

hcclResult_t device_context::synchronize_output(
    synapse_helpers::device_ptr output_address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::synchronize_output(output_address=",
      (void*)output_address,
      ")");

  HABANA_ASSERT(nullptr != device_);
  device_->wait_for_future(output_address);
  return hcclSuccess;
}
hcclResult_t device_context::synchronize_output(
    synapse_helpers::device_ptr output_address,
    synapse_helpers::hpuStream_t current_stream) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::synchronize_output(output_address=",
      (void*)output_address,
      ")");

  synapse_helpers::device_handle dev_handle = device_;
  HABANA_ASSERT(nullptr != dev_handle);
  dev_handle->wait_for_future(output_address);
  dev_handle->add_wait_events_on_stream(
      {output_address}, dev_handle->get_stream(current_stream));
  return hcclSuccess;
}

hcclResult_t device_context::barrier() {
  PT_DISTRIBUTED_BEGIN;
  PT_DISTRIBUTED_DEBUG("[PYT-DIST] barrier");

  PT_DISTRIBUTED_END;
  return {};
}

} // namespace hccl_integration
