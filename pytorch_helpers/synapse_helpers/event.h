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
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include "synapse_helpers/device_types.h"

namespace synapse_helpers {

class event_handle_cache;
class stream;
class stream_event_manager;

using event_done_callback = std::function<void()>;

/*! \brief Carrier of resources that must be kept alive for the duration of
 * async computation. Events are created through the stream_event_manager.
 * Producer of an event provides a custom cleanup callback, which can be used to
 * release framework-specific resources, while event itself remains
 * framework-agnostic.
 *
 * Events are constructed in _pending_ state, then get transferred to _done_
 * state once completed by the accelerator.
 */
class event {
  event_handle_cache& event_handle_cache_;
  synEventHandle handle_{nullptr};
  event_done_callback done_cb_;
  std::mutex mutex_;
  std::condition_variable ready_var_;
  std::atomic<bool> done_{false};

  std::vector<device_ptr> device_ptrs_{};
  stream& stream_recorded_; // used to avoid waiting on the same stream which is
                            // forbidden by synapse

 public:
  /*! \brief Constructor        Requests event from event handle cache
   *  \param event_handle_cache cache of events handles
   *  \param stream             stream on which event is going to be recorder
   *  \param device_ptrs        device pointers that map to this event in sem.
   *  \param done_cb            function to be invoked, once the internal event
   * handle is synchronized
   */
  explicit event(
      event_handle_cache& event_handle_cache,
      stream& stream,
      std::vector<device_ptr>&& device_ptrs,
      event_done_callback done_cb);
  ~event();

  event() = delete;
  event(event&&) = delete;
  event(const event&) = delete;
  event& operator=(event&&) = delete;
  event& operator=(const event&) = delete;

  /*! \brief Invokes synStreamWaitEvent with its synEventHandle on a given
   * stream \param stream on which WaitEvent is recorded \param flags -
   * currently not used
   */
  void stream_wait_event(stream& stream, uint32_t flags = 0);

  /*! \return true if synEventHandle already happened, false otherwise
   */
  bool done() {
    return done_;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_var_.wait(lock, std::bind(&event::done, this));
  }

  /*! \brief Return and release device pointers mapped to this events. */
  const std::vector<device_ptr>& get_device_ptrs() const {
    return device_ptrs_;
  }

  operator synEventHandle() const {
    return handle_;
  }

  friend class stream_event_manager;

 private:
  /*! \brief blocks thread until event is triggered.
   * This method invokes synEventSynchronize.  It should be called by
   * stream_event_manager exactly once in context of stream GC thread.  Other
   * threads that wish to block until an event is ready, should call wait() via
   * device::wait_for_event().
   */
  void synchronize();

  /*! \brief completes state transition to done
   * Should be called by stream_event_manager exactly once after event is
   * synchronized and SEM completed its bookkeeping.
   */
  void complete();
};

using shared_event = std::shared_ptr<event>;

} // namespace synapse_helpers
