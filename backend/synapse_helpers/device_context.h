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

#pragma once

#include <synapse_api_types.h> // for HCL_Rank
#include <atomic> // for atomic
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t, uintptr_t
#include <functional> // for function
#include <map> // for vector
#include <memory> // for shared_ptr, enable_share...
#include <mutex> // for mutex
#include <unordered_map> // for unordered_map
#include <vector> // for vector
#include "hccl_types.h"

#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event.h" // for synapse_error_o
#include "backend/synapse_helpers/stream.h" // for synapse_error_o
#include "backend/synapse_helpers/synapse_error.h" // for synapse_error_o

namespace hccl_integration {

using event_done_callback = std::function<void()>;

typedef enum {
  deviceCtxtMemcpyHostToHost = 0,
  deviceCtxtMemcpyHostToDevice = 1,
  deviceCtxtMemcpyDeviceToHost = 2,
  deviceCtxtMemcpyDeviceToDevice = 3,
  deviceCtxtMemcpyDefault = 4,
  deviceCtxtNumMemcpyKindTypes
} deviceCtxtMemcpyKind_t;

class device_context : std::enable_shared_from_this<device_context> {
  const size_t MAX_SUPPORTED_MODULE_ID = 1;

 public:
  // Device handle acquired using currently selected device ID.

  device_context(int device_id);
  ~device_context();

  hcclResult_t open_device(int device_id);

  hcclResult_t acquire_collective_stream(synStreamHandle* stream_handle_ptr);
  hcclResult_t release_stream(synStreamHandle stream_handle);

  hcclResult_t acquire_copy_stream(
      synStreamHandle* stream_handle_ptr,
      deviceCtxtMemcpyKind_t kind);

  hcclResult_t lock_address(void* const address, void** device_address_ptr);
  hcclResult_t lock_address(
      void* const address,
      void** device_address_ptr,
      std::unique_ptr<synapse_helpers::device_ptr_lock>& locked);

  hcclResult_t lock_address(
      std::vector<void*> addresses,
      std::unique_ptr<synapse_helpers::device_ptr_lock>& locked);
  synapse_helpers::active_recipe_counter& get_active_recipe_counter();

  hcclResult_t prepare_stream(
      synStreamHandle stream_handle,
      synapse_helpers::device_ptr input_address);

  std::vector<synapse_helpers::shared_event> prepare_stream_and_get_events(
      synStreamHandle stream_handle,
      synapse_helpers::device_ptr input_address);

  synapse_helpers::stream& get_stream_fromhandle(synStreamHandle stream_handle);

  hcclResult_t submit_events(
      synStreamHandle stream_handle,
      synapse_helpers::device_ptr output_address,
      const synapse_helpers::event_done_callback& done_callback = [] {});

  hcclResult_t submit_future(
      synapse_helpers::device_ptr device_addr,
      std::shared_future<bool> fut);
  hcclResult_t stream_synchronize(synStreamHandle stream_handle);

  hcclResult_t synchronize_output(synapse_helpers::device_ptr output_address);
  hcclResult_t synchronize_output(
      synapse_helpers::device_ptr output_address,
      synapse_helpers::hpuStream_t current_stream);
  hcclResult_t barrier();

  void flush_stream_events() {
    if (device_) {
      device_->flush_stream_events();
    }
  }

 private:
  uint32_t get_sync_tag() const {
    sync_tag_++;
    return sync_tag_;
  }
  // API/state mutex.
  std::mutex access_mutex_;
  mutable uint32_t sync_tag_{2020};
  // Device ID currently selected using set_device().
  synapse_helpers::device_handle device_;
  std::map<synStreamHandle, synapse_helpers::stream*> stream_objects_{};
  std::map<synStreamHandle, synapse_helpers::hpuStream_t> hpustream_handle_map_;
};

} // namespace hccl_integration
