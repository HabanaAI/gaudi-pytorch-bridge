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
#pragma once

#include <synapse_api_types.h>
#include <synapse_common_types.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/variant.h"
#include "backend/synapse_helpers/device_memory.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event.h"
#include "backend/synapse_helpers/event_handle_cache.h"
#include "backend/synapse_helpers/host_memory.h"
#include "backend/synapse_helpers/memory_mapping.h"
#include "backend/synapse_helpers/recipe_handle_cache.h"
#include "backend/synapse_helpers/stream.h"
#include "backend/synapse_helpers/stream_event_manager.h"
#include "backend/synapse_helpers/synapse_error.h"

namespace synapse_helpers {
std::string get_mem_str(size_t nbytes);

// this enum is used only for the case where non generic stream is used
enum default_stream_type {
  BEGIN_TYPE_ = 0,
  COMPUTE = 0,
  DMA_D2D = 1,
  DMA_H2D = 2,
  DMA_D2H = 3,
  NETWORK = 4,
  END_TYPE_ = 4
};

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
  uint64_t wait_for_next_decrease_call();
  uint64_t get_count();

 private:
  uint64_t counter_state_{0};
  uint64_t total_submitted_{0};
  uint64_t total_freed_{0};
  std::condition_variable cv_;
  std::mutex counter_mutex_;
};

class host_event {
 public:
  void wait_for_event_complete() {
    std::unique_lock<std::mutex> lck(mutex_);
    cv_.wait(lck, [this]() -> bool { return done(); });
  }

  void complete() {
    std::unique_lock<std::mutex> lck(mutex_);
    done_ = true;
    cv_.notify_all();
  }

  bool done() {
    return done_.load();
  }

 private:
  std::condition_variable cv_;
  std::mutex mutex_;
  std::atomic<bool> done_{false};
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

  static const synDeviceId INVALID_ID = static_cast<synDeviceId>(-1);

  static std::weak_ptr<device> device_in_use;
  static std::mutex device_mtx;

  using ref = std::reference_wrapper<synapse_helpers::device>;
  static synapse_error_v<device_handle> get_or_create(
      const std::set<synDeviceType>& allowed_device_types);

  static synapse_error_v<device_handle> get_by_id(synDeviceId idtype_t);

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

  std::string name() const {
    constexpr uint32_t maxStringLength = 1024;
    char deviceName[maxStringLength] = "";
    auto status = synDeviceGetName(deviceName, maxStringLength, id_);
    if (status != synSuccess) {
      PT_SYNHELPER_DEBUG(
          Logger::formatStatusMsg(status),
          "Failed to get device name for id ",
          id_);
      return "";
    }
    return deviceName;
  }

  friend std::ostream& operator<<(
      std::ostream& stream,
      const device& syn_device);

  template <typename... DevicePtrT>
  device_ptr_lock lock_addresses(DevicePtrT... ptrs) {
    return device_memory_.lock_addresses(
        absl::Span<const device_ptr>({static_cast<device_ptr>(ptrs)...}));
  }

  device_ptr_lock lock_addresses(const absl::Span<const device_ptr>& ptrs) {
    return device_memory_.lock_addresses(ptrs);
  }
  device_ptr_lock lock_addresses(const std::vector<device_ptr>& ptrs) {
    return lock_addresses(absl::Span<const device_ptr>{ptrs});
  }

  device_ptr get_fixed_address(void* ptr) {
    return device_memory_.fix_address(ptr);
  }

  void record_param(
      const std::string& name,
      const bool is_param,
      const bool is_grad,
      const bool is_optim_state,
      const uint64_t t_start,
      const uint64_t t_end) {
    device_memory_.record_param(
        name, is_param, is_grad, is_optim_state, t_start, t_end);
  }

  synapse_error copy_data_to_device(
      void* cpu_data,
      device_ptr destination,
      device_ptr event_addr,
      size_t total_bytes,
      const event_done_callback& done_cb,
      bool non_blocking = false,
      bool is_pinned = false,
      hpuStream_t hpu_stream = 0,
      void* host_cpu_data = nullptr);
  synapse_error copy_data_to_host(
      device_ptr device_data,
      void* destination,
      device_ptr event_addr,
      size_t total_bytes,
      const event_done_callback& done_cb,
      bool is_pinned = false,
      hpuStream_t hpu_stream = 0);
  synapse_error copy_data_within_device(
      device_ptr source,
      device_ptr destination,
      device_ptr src_event_addr,
      device_ptr dst_event_addr,
      size_t total_bytes,
      event_done_callback unref_cb,
      hpuStream_t hpu_stream = 0);

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
      stream* const next_operation_stream = nullptr,
      hpuStream_t hpu_stream = 0);

  synapse_error copy_data_to_device(
      transfer_manifest const& transfers,
      event_done_callback unref_cb,
      hpuStream_t hpu_stream = 0);

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

  std::vector<shared_event> get_wait_events_on_stream(
      const std::vector<device_ptr>& input_tensors,
      stream& stream);

  void submit_future(device_ptr device_addr, std::shared_future<bool> fut);

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
  /** \brief Adds specified Event and bounds it with event's mapped address on a
   * given stream. It forwards the call to internal stream_event_manager object.
   *  \param stream Stream for signaling wait for recorded event
   *  \param event Partial event assossiated with address
   *  \see stream_event_manager::add_producer
   */
  void register_producer_on_stream(stream& stream, shared_event event);

  void add_event_id(const std::string& event_id, const std::string& new_id);
  void wait_for_future(device_ptr address);
  void wait_until_address_ready(device_ptr address);
  void wait_until_event_ready(const std::string& event_id);
  void wait_for_event(shared_event& event);

  /** \brief Function creates an event and maps it with specified tensor
   *  \param stream Stream for signaling wait for recorded event
   *  \param recipe_handle Recipe from which the event will be signaled
   *  \param tensor_info External tensor information
   *  \param done_cb function to be invoked, once the event is synchronized.
   * Used for releasing ownership of Input Tensors dependant on this event
   */
  shared_event map_event_to_tensor(
      stream& stream,
      const synRecipeHandle recipe_handle,
      synLaunchTensorInfo* tensor_info,
      event_done_callback done_cb);

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

  static std::shared_ptr<session> get_or_create_session();
  static int get_total_device_count();
  static int get_device_type();

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

  uint64_t GetMaxRecipeLimitInQueue() {
    return max_recipe_limit_in_queue_;
  }

  bool IsMemorydefragmentationEnabled() {
    return enable_memory_defragmentation_;
  }

  bool IsMemorydefragmentationInfoEnabled() {
    return enable_memory_defrag_info_;
  }

  static std::set<synDeviceType> get_supported_devices();

  void synchronize();

  std::string get_device_capability();

  static std::string get_device_properties(unsigned id);

  void release();

  void cleanup_workspace_buffer();

  void create_stream(hpuStream_t& hpu_stream, bool high_priority = false);

  void synchronize_default_stream();

  bool query_default_stream();

  void create_default_stream();

  stream& get_stream(hpuStream_t id, default_stream_type stream_type = COMPUTE);

  void delete_stream(hpuStream_t id);

  hpuEvent_t create_event(bool flags);

  void record_event(hpuEvent_t id, hpuStream_t record_stream);

  void wait_event(hpuEvent_t id, hpuStream_t block_stream);

  void synchronize_event(hpuEvent_t id);

  bool query_event(hpuEvent_t id);

  uint64_t elapsed_time(hpuEvent_t id1, hpuEvent_t id2);

  void delete_event(hpuEvent_t id, bool flags);

  size_t get_real_workspace_size() const {
    return real_workspace_size_;
  }

  size_t get_workspace_size() {
    return workspace_size_;
  }

  hpuEvent_t get_event_index() {
    event_index_++;
    return event_index_.load();
  }

  size_t get_least_workspace_size(
      size_t persistent_size,
      size_t req_workspace_size);

  void register_host_event(uint64_t addr);

  void wait_for_host_event(uint64_t addr);

  void mark_host_event_complete(uint64_t addr);

  size_t get_device_memory_alignment() {
    return device_memory_alignment_;
  }

 private:
  friend class stream;
  static synapse_error_v<device_handle> create(
      const std::set<synDeviceType>& allowed_device_types);
  device(
      std::shared_ptr<session> synapse_session,
      synDeviceId device_id,
      synDeviceType device_type,
      size_t alignment);

  void synchronize_event(shared_event& event) {
    sem_.synchronize_event(event);
  }

  std::shared_ptr<session> synapse_session_;

  synDeviceType type_;
  device_id id_;
  size_t device_memory_alignment_;

  // WARNING: ordering of members is critical
  // note that there are inter-dependencies between devices' members that
  // require specific order of destruction.
  size_t workspace_size_{0};
  size_t real_workspace_size_{0};
  device_ptr workspace_buffer_{0}; // global workspace buffer per device to be
                                   // used to launch recipes
  std::mutex ws_mutex_;
  std::mutex stream_mutex_;
  event_handle_cache event_handle_cache_;
  event_handle_cache time_event_handle_cache_;
  memory_mapper memory_mapper_;
  host_memory host_memory_;
  device_memory device_memory_;
  std::unordered_map<hpuStream_t, std::unique_ptr<stream>> streams_;
  // Only used with old design of stream assignment
  std::unordered_map<default_stream_type, std::unique_ptr<stream>>
      default_streams_;
  stream_event_manager sem_;
  recipe_handle_cache recipe_handle_cache_;
  bool is_caching_enabled_;
  bool is_stream_async_enabled_;
  absl::optional<owned_device_ptr> preallocated_reduction_buffer_;
  bool is_hcl_same_addr_enabled_;

  active_recipe_counter recipe_counter_;
  bool host_memory_cache_enabled_;
  unsigned max_dma_copy_retry_count_;
  std::chrono::milliseconds dma_copy_retry_delay_;
  uint64_t max_recipe_limit_in_queue_;
  bool enable_memory_defragmentation_;
  bool enable_memory_defrag_info_;

  bool enable_dynamic_workspace_{false};
  bool cleanup_done_{false};

  // stream counter
  std::atomic<uint64_t> stream_index_{0};
  // event counter
  std::atomic<uint64_t> event_index_{0};
  std::mutex event_mutex_;

  // private inline method
  inline bool copy_data_to_device_(
      void* cpu_data,
      device_ptr destination,
      device_ptr event_addr,
      size_t total_bytes,
      const event_done_callback& done_cb,
      bool is_pinned,
      hpuStream_t hpu_stream,
      void* host_cpu_data = nullptr);

  uint64_t get_compute_stream_count();
  // Empty be default, framework can register its function to be called before
  // device is released
  framework_specific_cleanup_fnc framework_specific_cleanup_{[] {}};
  std::map<size_t, uint32_t> workspace_usage_;
  std::unordered_map<hpuEvent_t, std::array<synEventHandle, END_TYPE_>>
      user_event_map_;
  std::mutex usr_event_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<host_event>>
      addr_host_event_map_;
  std::mutex host_event_mutex_;
};

std::ostream& operator<<(std::ostream& stream, const device& syn_device);

} // namespace synapse_helpers
