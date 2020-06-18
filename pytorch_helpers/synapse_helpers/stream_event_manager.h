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

#include <absl/container/flat_hash_map.h>
#include <mutex>
#include <vector>

#include "absl/hash/hash.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"

namespace synapse_helpers {
class stream;

//! Class responsible of recording events on any stream
class stream_event_manager {
  absl::flat_hash_map<device_ptr, shared_event> events_;
  std::mutex mut_;

 public:
  /*! \brief Tries to record Event on a given stream
   *  \param device_addresses identifier of Events - tensor pointers in device
   * memory space that single Event is recorded for \param stream given stream
   * on which Event will be recorded \param done_cb          function to be
   * invoked, once the event is synchronized. Used for releasing ownership of
   * Input Tensors dependant on this event \return True, if WaitForEvent was
   * recorded on stream, false otherwise
   */
  void add_producer(
      const std::vector<device_ptr>& device_addresses,
      stream& stream,
      event_done_callback done_cb);

  /*! \brief Makes \p stream wait until \p device_address is ready to be used if
   * it wasn't ready already \param device_address tensor pointer in device
   * memory space to wait for \param stream         given stream that should
   * wait for \p device_address \return True if \p device_address is ready,
   * false otherwise
   */
  bool record_wait_event(device_ptr device_address, stream& stream);

  /*! \brief Invokes blocking EventSynchronize for a given tenor pointer in
   * device memory space \param device_address identifier of Event - tensor
   * pointer in device memory space
   */
  void wait_until_done(device_ptr device_address);

  /*! \brief Returns reference to Event, if exists
   *  \param device_address identifier of Event - tensor pointer in device
   * memory space \return shared_event if exists, nullptr otherwise
   */
  shared_event get_event(device_ptr device_address);

  /*! \brief Flushes pendings event.
   *
   *  Blocks execution of current thread until all events will be synchronized.
   *  Does not return until list of pending events is empty.
   */
  void flush();

  /*! \brief Cleanup function, that cleans Events that were already done
   */
  void clear_if_done();
};

} // namespace synapse_helpers
