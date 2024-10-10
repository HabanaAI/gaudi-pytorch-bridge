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

#include <absl/container/flat_hash_map.h>
#include <mutex>
#include <vector>

#include <future>
#include "absl/hash/hash.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event.h"

namespace synapse_helpers {
class stream;

//! Class responsible of recording events on any stream
class stream_event_manager {
  absl::flat_hash_map<device_ptr, shared_event> events_by_addr_;
  absl::flat_hash_map<device_ptr, std::shared_future<bool>> future_by_addr_;
  absl::flat_hash_map<std::string, shared_event> events_by_str_;
  std::mutex mut_;
  std::mutex future_mut_;

 public:
  /*! \brief Tries to record Event on a given stream
   *  \param device_addresses identifier of Events - tensor pointers in device
   * memory space that single Event is recorded for \param stream given stream
   * on which Event will be recorded \param done_cb          function to be
   * invoked, once the event is synchronized. Used for releasing ownership of
   * Input Tensors dependant on this event \return True, if WaitForEvent was
   * recorded on stream, false otherwise
   */
  void add_future(device_ptr device_addr, std::shared_future<bool> fut);
  void add_producer(
      std::vector<device_ptr>&& device_addresses,
      stream& stream,
      event_done_callback done_cb);
  void add_producer(
      std::vector<device_ptr>&& device_addresses,
      std::string event_id,
      stream& stream,
      event_done_callback done_cb);
  void add_producer(std::string event_id, stream& stream, shared_event event);
  void add_producer(stream& stream, shared_event event);

  void add_event_id(const std::string& event_id, const std::string& new_id);

  /*! \brief Makes \p stream wait until \p device_address is ready to be used if
   * it wasn't ready already \param device_address tensor pointer in device
   * memory space to wait for \param stream         given stream that should
   * wait for \p device_address
   */
  void enqueue_wait_event(device_ptr device_address, stream& stream);
  void enqueue_wait_event(const std::string& event_id, stream& stream);

  shared_event get_wait_event(device_ptr device_address, stream& stream);

  /*! \brief Invokes blocking EventSynchronize for a given tenor pointer in
   * device memory space \param device_address identifier of Event - tensor
   * pointer in device memory space
   */
  void wait_for_future(device_ptr device_address);
  void wait_until_done(device_ptr device_address);
  void wait_until_done(shared_event& event);
  void wait_until_done(const std::string& event);
  void wait_for_all_futures();

  shared_event map_event_to_tensor(
      stream& stream,
      const synRecipeHandle recipe_handle,
      synLaunchTensorInfo* tensor_info,
      event_done_callback done_cb);

  /*! \brief Returns reference to Event, if exists
   *  \param device_address identifier of Event - tensor pointer in device
   * memory space \return shared_event if exists, nullptr otherwise
   */
  shared_event get_event(device_ptr device_address);

  bool is_flushed();

  friend class device;

 private:
  /*! \brief Waits for event completion and erases all its registrations from
   * the map. \param event event to unmap
   */
  void synchronize_event(shared_event& event);

  std::mutex sync_mut_;
};

} // namespace synapse_helpers
