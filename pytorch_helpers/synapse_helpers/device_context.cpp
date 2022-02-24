/******************************************************************************
 * Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */
#include <absl/memory/memory.h>
#include <absl/types/optional.h>
#include <absl/types/variant.h>
#include <synapse_common_types.h>
#include <synapse_helpers/synapse_error.h>
#include <condition_variable>
#include <iterator>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

#include "device_context.h"
#include "habana_helpers/logging.h"
#include "hccl_types.h"
#include "status_conversion.h"
#include "synapse_api.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/stream.h"

namespace hccl_integration {

device_context::device_context(int device_id) {
  open_device(device_id);
}

device_context::~device_context() {}

hcclResult_t device_context::open_device(int device_id) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::open_device(device_id=", device_id, ")");
  std::lock_guard<std::mutex> guard{access_mutex_};

  auto maybe_dev_handle = synapse_helpers::device::get_by_id(device_id);
  if (!ok(maybe_dev_handle)) {
    RETURN_ON_SYNAPSE_ERROR(get_error(maybe_dev_handle));
  }
  synapse_helpers::device_handle dev_handle =
      synapse_helpers::get_value(maybe_dev_handle);
  device_ = dev_handle;
  return hcclSuccess;
}

hcclResult_t device_context::acquire_collective_stream(
    hcclStream_t* stream_handle_ptr) {
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
  synapse_helpers::device_handle dev_handle = device_;
  synapse_helpers::stream& stream_handle =
      dev_handle->get_or_create_network_collective_stream();
  HABANA_ASSERT(nullptr != stream_handle);

  stream_objects_[stream_handle] = &stream_handle;

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

  stream_objects_[stream_handle] = nullptr;
  return hcclSuccess;
}

hcclResult_t device_context::free(void* address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::free(address=",
      reinterpret_cast<void*>(address),
      ")");

  if (device_ == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before allocating memory.");
    return hcclInvalidUsage;
  }

  device_->free(reinterpret_cast<synapse_helpers::device_ptr>(address));

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

  synapse_helpers::device_handle device = device_;

  if (device == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device need to be opened and chosen before allocating memory.");
    return hcclInvalidUsage;
  }

  synapse_helpers::device_ptr_lock locked{device->lock_addresses(
      reinterpret_cast<synapse_helpers::device_ptr>(address))};
  auto locked_address = reinterpret_cast<void*>(locked.at(0));

  addresses_locks_[locked_address] =
      absl::make_unique<synapse_helpers::device_ptr_lock>(std::move(locked));

  *device_address = locked_address;
  return hcclSuccess;
}

hcclResult_t device_context::unlock_address(void* const device_address) {
  std::lock_guard<std::mutex> guard{access_mutex_};
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::unlock_address(device_address=",
      device_address,
      ")");

  if ((addresses_locks_.find(device_address) == addresses_locks_.end()) ||
      addresses_locks_.at(device_address) == nullptr) {
    PT_DISTRIBUTED_FATAL(
        "Device address ", device_address, " has not been locked!");
    return hcclInvalidArgument;
  }

  addresses_locks_[device_address] = nullptr;
  return hcclSuccess;
}

hcclResult_t device_context::acquire_copy_stream(
    synStreamHandle* stream_handle_ptr,
    deviceCtxtMemcpyKind_t kind) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::acquire_copy_stream(stream_handle_ptr=",
      stream_handle_ptr,
      ", kind=",
      kind,
      ")");

  std::lock_guard<std::mutex> guard{access_mutex_};

  synapse_helpers::device_handle dev_handle = device_;

  synapse_helpers::stream* stream_handle{nullptr};

  switch (kind) {
    case deviceCtxtMemcpyHostToDevice: {
      stream_handle = &dev_handle->get_host_to_device_stream();
      break;
    }
    case deviceCtxtMemcpyDeviceToHost: {
      stream_handle = &dev_handle->get_device_to_host_stream();
      break;
    }
    case deviceCtxtMemcpyDeviceToDevice: {
      stream_handle = &dev_handle->get_device_to_device_stream();
      break;
    }
    default: {
      stream_handle = nullptr;
      break;
    }
  };

  if (nullptr == stream_handle) {
    PT_DISTRIBUTED_FATAL("Copy stream of kind ", kind, " not available!");
    return hcclInternalError;
  }

  stream_objects_[*stream_handle] = stream_handle;
  *stream_handle_ptr = *stream_handle;
  return hcclSuccess;
}

hcclResult_t device_context::copy_data_within_device(
    synapse_helpers::device_ptr input_address,
    synapse_helpers::device_ptr output_address,
    synapse_helpers::device_ptr input_event_addr,
    synapse_helpers::device_ptr output_event_addr,
    size_t nbytes,
    const event_done_callback& done_callback) {
  synapse_helpers::device_handle dev_handle = device_;
  auto syn_error = dev_handle->copy_data_within_device(
      input_address,
      output_address,
      input_event_addr,
      output_event_addr,
      nbytes,
      done_callback);
  return to_hccl_result(syn_error);
}

hcclResult_t device_context::prepare_stream(
    hcclStream_t stream_handle,
    synapse_helpers::device_ptr input_address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::acquire_copy_stream(stream_handle=",
      stream_handle,
      ", input_address=",
      input_address,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized! (", stream_handle, ")");
    return hcclInvalidArgument;
  }

  synapse_helpers::device_handle dev_handle = device_;
  HABANA_ASSERT(nullptr != dev_handle);

  std::vector<synapse_helpers::device_ptr> addresses;
  addresses.push_back(input_address);

  dev_handle->add_wait_events_on_stream(
      addresses, *stream_objects_[stream_handle]);

  return hcclSuccess;
}

hcclResult_t device_context::submit_events(
    hcclStream_t stream_handle,
    synapse_helpers::device_ptr output_address,
    const synapse_helpers::event_done_callback& done_callback) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::acquire_copy_stream(stream_handle=",
      stream_handle,
      ", output_address=",
      output_address,
      ")");

  if (stream_objects_.find(stream_handle) == stream_objects_.end() ||
      stream_objects_.at(stream_handle) == nullptr) {
    PT_DISTRIBUTED_FATAL("Stream handle not recognized!");
    return hcclInvalidArgument;
  }

  synapse_helpers::device_handle dev_handle = device_;
  HABANA_ASSERT(nullptr != dev_handle);

  std::vector<synapse_helpers::device_ptr> addresses;
  addresses.push_back(output_address);

  dev_handle->register_producer_on_stream(
      std::move(addresses), *stream_objects_[stream_handle], done_callback);
  return hcclSuccess;
}

hcclResult_t device_context::submit_future(
    synapse_helpers::device_ptr device_addr,
    std::future<bool> fut) {
  synapse_helpers::device_handle dev_handle = device_;
  HABANA_ASSERT(nullptr != dev_handle);

  dev_handle->submit_future(device_addr, std::move(fut));
  return hcclSuccess;
}

hcclResult_t device_context::stream_synchronize(hcclStream_t stream) {
  return to_hccl_result(synStreamSynchronize(stream));
}

hcclResult_t device_context::synchronize_output(
    synapse_helpers::device_ptr output_address) {
  PT_DISTRIBUTED_DEBUG(
      "Calling device_context::synchronize_output(output_address=",
      output_address,
      ")");
  synapse_helpers::device_handle dev_handle = device_;
  HABANA_ASSERT(nullptr != dev_handle);
  dev_handle->wait_until_address_ready(output_address);
  return hcclSuccess;
}

hcclResult_t device_context::barrier() {
  PT_DISTRIBUTED_BEGIN;
  PT_DISTRIBUTED_DEBUG("[PYT-DIST] barrier");

  PT_DISTRIBUTED_END;
  return {};
}

} // namespace hccl_integration
