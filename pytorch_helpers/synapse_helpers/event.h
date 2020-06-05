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
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

namespace synapse_helpers {

class event_handle_cache;
class stream;

using event_done_callback = std::function<void()>;

//! Wrapper Class for synEventHandle
class event {
  event_handle_cache& event_handle_cache_;
  synEventHandle handle_{nullptr};
  event_done_callback done_cb_;
  std::mutex mutex_;
  std::mutex sync_mutex_;
  std::atomic<bool> done_{false};

  stream& stream_recorded_;  // used to avoid waiting on the same stream which is forbidden by synapse

 public:
  /*! \brief Constructor        Requests event from event handle cache
   *  \param event_handle_cache cache of events handles
   *  \param stream             stream on which event is going to be recorder
   *  \param done_cb            function to be invoked, once the internal event handle is synchronized
   */
  explicit event(event_handle_cache& event_handle_cache, stream& stream, event_done_callback done_cb);
  ~event();

  event() = delete;
  event(event&&) = delete;
  event(const event&) = delete;
  event& operator=(event&&) = delete;
  event& operator=(const event&) = delete;

  /*! \brief Invokes synStreamSynchronize on its synEventHandle
   */
  synStatus synchronize();

  /*! \brief Invokes synStreamWaitEvent with its synEventHandle on a given stream
   *  \param stream on which WaitEvent is recorded
   *  \param flags - currently not used
   *  \return true if WaitEvent was recorded, false if internal synEventHandle already happened and no wait is needed
   */
  bool streamWaitEvent(stream& stream, uint32_t flags = 0);

  /*! \return true if synEventHandle already happened, false otherwise
   */
  bool done() { return done_; }

  operator synEventHandle() const { return handle_; }
};

using shared_event = std::shared_ptr<event>;

}  // namespace synapse_helpers
