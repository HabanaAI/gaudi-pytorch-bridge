/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <synapse_api_types.h>
#include <synapse_common_types.h>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <vector>

#include "absl/types/variant.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/event_handle_cache.h"
#include "synapse_helpers/memory_mapping.h"
#include "synapse_helpers/stream.h"
#include "synapse_helpers/stream_event_manager.h"
#include "synapse_helpers/synapse_error.h"

namespace absl {
template <typename... Ts>
class variant;
}  // namespace absl

namespace synapse_helpers {

class session;

class device_id {
 public:
  explicit device_id(synDeviceId id) : id_{id} {}
  device_id() = delete;
  device_id(const device_id&) = delete;
  device_id& operator=(const device_id&) = delete;
  device_id(device_id&&) = delete;
  device_id& operator=(device_id&&) = delete;
  ~device_id();
  operator synDeviceId() const { return id_; }

 private:
  synDeviceId id_;
};

class device {
 public:
  struct transfer_desc {
    device_ptr src;
    device_ptr dst;
    size_t bytes_to_transfer;
  };

  using transfer_manifest = std::vector<transfer_desc>;

  static const std::uint32_t INVALID_ID = -1;

  static std::weak_ptr<device> device_in_use;
  static std::mutex device_mtx;

  using ref = std::reference_wrapper<synapse_helpers::device>;
  static synapse_error_v<std::shared_ptr<device>> get_or_create(synDeviceType device_type,
                                                                const create_allocator_fnc& create_allocator);

  static synapse_error_v<std::shared_ptr<device>> get_by_id(synDeviceId idtype_t);

  device(const device&) = delete;
  device& operator=(const device&) = delete;
  device(device&&) = delete;
  device& operator=(device&&) = delete;
  ~device();

  void flush_stream_events();

  // Function passed here will be called at the begining od device dtor.
  void register_framework_specific_cleanup(framework_specific_cleanup_fnc cleanup) {
    framework_specific_cleanup_ = std::move(cleanup);
  }

  synDeviceType type() const { return type_; }
  synDeviceId id() const { return id_; }

  friend std::ostream& operator<<(std::ostream& stream, const device& syn_device);

  device_ptr malloc(size_t size);
  void free(device_ptr ptr);
  synapse_error copy_data_to_device(void* cpu_data, device_ptr destination, size_t total_bytes,
                                    const event_done_callback& done_cb);
  synapse_error copy_data_to_host(device_ptr device_data, void* destination, size_t total_bytes,
                                  const event_done_callback& done_cb);
  synapse_error copy_data_within_device(device_ptr source, device_ptr destination, size_t total_bytes,
                                        event_done_callback unref_cb);

  // This function does entire list of transfers within device and records only single event
  // after last of them is scheduled.
  synapse_error copy_data_within_device(transfer_manifest const&, event_done_callback unref_cb);

  stream& get_compute_stream() { return stream_comp_; }
  stream& get_network_collective_stream() { return stream_network_collective_; }
  stream& get_host_to_device_stream() { return stream_h2d_; }
  stream& get_device_to_host_stream() { return stream_d2h_; }

  /** \brief Returns global workspace buffer
   *  \param size checks if given size is bigger than global buffer, if so, logs FATAL
   *  \return pointer to the global buffer
   */
  device_ptr get_workspace_buffer(std::size_t size) const;

  /** \brief Add WaitEvents on a given stream for a list of inputs
   *  \param input_tensors identifiers of Events - tensor pointers in device memory space
   *  \param stream given stream to record WaitEvents
   */
  void add_wait_events_on_stream(const std::vector<device_ptr>& input_tensors, stream& stream);

  /** \brief Add Events on a given stream for a list of outputs of a stream operation.
   *         It forwards the call to internal stream_event_manager object.
   *  \see stream_event_manager::add_producer
   */
  void register_producer_on_stream(const std::vector<device_ptr>& bound_addresses, stream& stream,
                                   event_done_callback done_cb);

  void wait_until_address_ready(const device_ptr& address);

  event_handle_cache& get_event_handle_cache() { return event_handle_cache_; }

 private:
  static synapse_error_v<std::shared_ptr<device>> create(synDeviceType device_type,
                                                         const create_allocator_fnc& create_allocator);
  device(std::shared_ptr<session> synapse_session, synDeviceId device_id, synDeviceType device_type,
         const create_allocator_fnc& create_allocator);

  std::shared_ptr<session> synapse_session_;

  synDeviceType type_;
  device_id id_;

  // WARNING: ordering of members is critical
  // note that there are inter-dependencies between devices' members that require
  // specific order of destruction.
  std::unique_ptr<device_allocator> allocator_;
  size_t workspace_size_;
  device_ptr workspace_buffer_;  // global workspace buffer per device to be used to launch recipes
  event_handle_cache event_handle_cache_;
  memory_mapper memory_mapper_;
  stream stream_comp_;
  stream stream_network_collective_;
  stream stream_d2d_;
  stream stream_h2d_;
  stream stream_d2h_;
  stream_event_manager sem_;

  // Empty be default, framework can register its function to be called before device is released
  framework_specific_cleanup_fnc framework_specific_cleanup_{[] {}};
};

std::ostream& operator<<(std::ostream& stream, const device& syn_device);

}  // namespace synapse_helpers
