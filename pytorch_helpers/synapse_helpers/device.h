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
 *******************************************************************************
 */
#pragma once

#include <synapse_api_types.h>
#include <synapse_common_types.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "synapse_helpers/device_memory.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/event_handle_cache.h"
#include "synapse_helpers/host_memory.h"
#include "synapse_helpers/memory_mapping.h"
#include "synapse_helpers/recipe_handle_cache.h"
#include "synapse_helpers/stream.h"
#include "synapse_helpers/stream_event_manager.h"
#include "synapse_helpers/synapse_error.h"

namespace absl {
template <typename... Ts>
class variant;
} // namespace absl

namespace synapse_helpers {
std::string get_mem_str(size_t nbytes);

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
  operator synDeviceId() const {
    return id_;
  }

 private:
  synDeviceId id_;
};

class active_recipe_counter {
 public:
  void increase();
  void decrease_and_notify();
  bool is_zero();
  uint32_t wait_for_next_decrease_call();
  uint32_t get_count();

 private:
  uint32_t counter_state_{0};
  std::condition_variable cv_;
  std::mutex counter_mutex_;
};

class device {
 public:
  struct transfer_desc {
    device_ptr src;
    device_ptr dst;
    device_ptr src_event_addr;
    device_ptr dst_event_addr;
    size_t bytes_to_transfer;
  };

  using transfer_manifest = std::vector<transfer_desc>;

  static const std::uint32_t INVALID_ID = -1;

  static std::weak_ptr<device> device_in_use;
  static std::mutex device_mtx;

  using ref = std::reference_wrapper<synapse_helpers::device>;
  static synapse_error_v<std::shared_ptr<device>> get_or_create(
      const std::set<synDeviceType>& allowed_device_types,
      const create_allocator_fnc& create_allocator);

  static synapse_error_v<std::shared_ptr<device>> get_by_id(
      synDeviceId idtype_t);

  device(const device&) = delete;
  device& operator=(const device&) = delete;
  device(device&&) = delete;
  device& operator=(device&&) = delete;
  ~device();

  void cleanup();
  void flush_stream_events();

  // Function passed here will be called at the begining od device dtor.
  void register_framework_specific_cleanup(
      framework_specific_cleanup_fnc cleanup) {
    framework_specific_cleanup_ = std::move(cleanup);
  }

  synDeviceType type() const {
    return type_;
  }
  synDeviceId id() const {
    return id_;
  }

  friend std::ostream& operator<<(
      std::ostream& stream,
      const device& syn_device);

  device_ptr malloc(size_t size);
  void free(device_ptr ptr);

  template <typename... DevicePtrT>
  device_ptr_lock lock_addresses(DevicePtrT&&... ptrs) {
    return device_memory_.lock_addresses({std::forward<DevicePtrT>(ptrs)...});
  }

  device_ptr_lock lock_addresses(const std::vector<device_ptr>& ptrs) {
    return device_memory_.lock_addresses(ptrs);
  }

  synapse_error copy_data_to_device(
      void* cpu_data,
      device_ptr destination,
      device_ptr event_addr,
      size_t total_bytes,
      const event_done_callback& done_cb,
      bool is_pinned = false);
  synapse_error copy_data_to_host(
      device_ptr device_data,
      void* destination,
      device_ptr event_addr,
      size_t total_bytes,
      const event_done_callback& done_cb,
      bool is_pinned = false);
  synapse_error copy_data_within_device(
      device_ptr source,
      device_ptr destination,
      device_ptr src_event_addr,
      device_ptr dst_event_addr,
      size_t total_bytes,
      event_done_callback unref_cb);

  /*!
   * \brief Copies data within device
   * This function does entire list of transfers within device and records only
   * single event after last of them is scheduled.
   *
   * \param manifest List of transfers to schedule
   * \param unref_cb Callback for clearing input tensors
   * \param next_operation_stream If this is not nullptr wait event will be
   * signaled on given stream immediately after operation is scheduled on
   * device2device stream. The event will not be tracked in SEM
   */
  synapse_error copy_data_within_device(
      transfer_manifest const& manifest,
      event_done_callback unref_cb,
      stream* const next_operation_stream = nullptr);

  stream& get_compute_stream() {
    return stream_comp_;
  }
  stream& get_or_create_network_collective_stream() {
    if (!stream_network_collective_ptr_) {
      stream_network_collective_ptr_ =
          absl::make_unique<stream>(*this, stream_flavor::COLLECTIVE_0);
    }
    return *stream_network_collective_ptr_;
  }
  stream& get_host_to_device_stream() {
    return stream_h2d_;
  }
  stream& get_device_to_host_stream() {
    return stream_d2h_;
  }
  stream& get_device_to_device_stream() {
    return stream_d2d_;
  };
  stream& get_stream(stream_flavor id);

  /** \brief Returns global workspace buffer
   *  \param size checks if given size is bigger than global buffer, if so, logs
   * FATAL \return pointer to the global buffer
   */
  device_ptr get_workspace_buffer(std::size_t size);

  /** \brief Add WaitEvents on a given stream for a list of inputs
   *  \param input_tensors identifiers of Events - tensor pointers in device
   * memory space \param stream given stream to record WaitEvents
   */
  void add_wait_events_on_stream(
      const std::vector<device_ptr>& input_tensors,
      stream& stream);

  void add_wait_event_on_stream(const std::string& event_id, stream& stream);

  void submit_future(device_ptr device_addr, std::future<bool> fut);

  /** \brief Records event on a given stream and signals wait for this event on
   * the other stream. Bypass sem \param record_stream stream for recording
   * event \param wait_stream stream for signaling wait for recorded event
   *  \param done_callback function to be invoked, once the event is
   * synchronized. Used for releasing ownership of Input Tensors dependant on
   * this event
   */
  void record_and_wait_for_event(
      stream& record_stream,
      stream& wait_stream,
      event_done_callback done_callback);

  /** \brief Add Events on a given stream for a list of outputs of a stream
   * operation. It forwards the call to internal stream_event_manager object.
   *  \see stream_event_manager::add_producer
   */
  void register_producer_on_stream(
      std::vector<device_ptr>&& bound_addresses,
      stream& stream,
      event_done_callback done_cb);

  void register_producer_on_stream(
      std::vector<device_ptr>&& bound_addresses,
      const std::string& event_id,
      stream& stream,
      event_done_callback done_cb);

  void add_event_id(const std::string& event_id, const std::string& new_id);
  void wait_until_address_ready(device_ptr address);
  void wait_until_event_ready(const std::string& event_id);
  void wait_for_event(shared_event& event);

  const absl::optional<owned_device_ptr>& reduction_buffer() {
    return preallocated_reduction_buffer_;
  }

  event_handle_cache& get_event_handle_cache() {
    return event_handle_cache_;
  }

  event_handle_cache& get_time_event_handle_cache() {
    return time_event_handle_cache_;
  }

  CachedEventHandle get_cached_time_event_handle() {
    return CachedEventHandle(time_event_handle_cache_);
  }

  recipe_handle_cache& get_recipe_handle_cache() {
    return recipe_handle_cache_;
  }

  bool IsCachingEnabled() {
    return is_caching_enabled_;
  }

  bool IsStreamASyncEnabled() {
    return is_stream_async_enabled_;
  }

  active_recipe_counter& get_active_recipe_counter() {
    return recipe_counter_;
  }

  int get_count_by_current_type();

  static int get_total_device_count();

  host_memory& get_host_memory() {
    return host_memory_;
  }

  bool HostMemoryCacheEnabled_() {
    return host_memory_cache_enabled_;
  }

  bool IsHCLSameAddressResolutionEnabled() {
    return is_hcl_same_addr_enabled_;
  }

  bool EnableDynamicWorkspace() {
    return enable_dynamic_workspace_;
  }

  device_memory& get_device_memory() {
    return device_memory_;
  }

  uint32_t GetMaxRecipeLimitInQueue() {
    return max_recipe_limit_in_queue_;
  }

  static std::set<synDeviceType> get_supported_devices();

  void synchronize();

 private:
  friend class stream;
  static synapse_error_v<std::shared_ptr<device>> create(
      const std::set<synDeviceType>& allowed_device_types,
      const create_allocator_fnc& create_allocator);
  device(
      std::shared_ptr<session> synapse_session,
      synDeviceId device_id,
      synDeviceType device_type,
      const create_allocator_fnc& create_allocator);

  void synchronize_event(shared_event& event) {
    sem_.synchronize_event(event);
  }

  uint64_t get_workspace_size();

  std::shared_ptr<session> synapse_session_;

  synDeviceType type_;
  device_id id_;

  // WARNING: ordering of members is critical
  // note that there are inter-dependencies between devices' members that
  // require specific order of destruction.
  std::unique_ptr<device_allocator> allocator_;
  size_t workspace_size_{0};
  device_ptr workspace_buffer_{0}; // global workspace buffer per device to be
                                   // used to launch recipes
  std::mutex ws_mutex_;
  event_handle_cache event_handle_cache_;
  event_handle_cache time_event_handle_cache_;
  memory_mapper memory_mapper_;
  stream stream_comp_;
  std::unique_ptr<stream> stream_network_collective_ptr_;
  stream stream_d2d_;
  stream stream_h2d_;
  stream stream_d2h_;
  stream_event_manager sem_;
  recipe_handle_cache recipe_handle_cache_;
  bool is_caching_enabled_;
  bool is_stream_async_enabled_;
  absl::optional<owned_device_ptr> preallocated_reduction_buffer_;
  bool is_hcl_same_addr_enabled_;

  active_recipe_counter recipe_counter_;
  host_memory host_memory_;
  bool host_memory_cache_enabled_;
  unsigned max_dma_copy_retry_count_;
  std::chrono::milliseconds dma_copy_retry_delay_;
  device_memory device_memory_;
  uint32_t max_recipe_limit_in_queue_;

  bool enable_dynamic_workspace_{false};
  bool cleanup_done_{false};

  // Empty be default, framework can register its function to be called before
  // device is released
  framework_specific_cleanup_fnc framework_specific_cleanup_{[] {}};
};

std::ostream& operator<<(std::ostream& stream, const device& syn_device);

} // namespace synapse_helpers
