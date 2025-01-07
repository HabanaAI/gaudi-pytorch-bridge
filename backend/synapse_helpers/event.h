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

#include <fmt/format.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include "backend/synapse_helpers/device_types.h"

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
  std::vector<std::string> event_ids_{};
  stream& stream_recorded_; // used to avoid waiting on the same stream which is
                            // forbidden by synapse
  bool is_partial_{};

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
      std::string event_id,
      event_done_callback done_cb);
  ~event();

  event() = delete;
  event(event&&) = delete;
  event(const event&) = delete;
  event& operator=(event&&) = delete;
  event& operator=(const event&) = delete;

  /*! \brief Tells if event is partial (has been used for intra-op signalization
   */
  bool is_partial() const {
    return is_partial_;
  }

  /*! \brief Invokes synStreamWaitEvent with its synEventHandle on a given
   * stream \param stream on which WaitEvent is recorded \param flags -
   * currently not used
   */
  void stream_wait_event(stream& stream, uint32_t flags = 0);

  void push_id(std::string event_id) {
    event_ids_.emplace_back(std::move(event_id));
  }

  /*! \brief Invokes synEventMapTensor with its synEventHandle, recipe and
   * tensor \param recipe_handle recipe from which the event will be signaled
   *  \param tensor_info External tensor information
   */
  void map_event_to_tensor(
      const synRecipeHandle recipe_handle,
      synLaunchTensorInfo* tensor_info);

  /*! \return true if synEventHandle already happened, false otherwise
   */
  bool done() {
    return done_;
  }

  void wait();

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
  void synchronize() const;

  /*! \brief completes state transition to done
   * Should be called by stream_event_manager exactly once after event is
   * synchronized and SEM completed its bookkeeping.
   */
  void complete();

  /*! \brief Return and release device pointers mapped to this events.
   * This is volatile information and SEM is doing properly synchronized use of
   * it in such a way that SEM maps device_ptrs_/event_ids_ are in sync.
   * NOTE: This implementation assumes that all of these three function are
   * called within a single SEM mutex scope*/
  const std::vector<device_ptr>& get_device_ptrs() const {
    return device_ptrs_;
  };
  const std::vector<std::string>& get_event_ids() const {
    return event_ids_;
  };
  void remove_device_ptr(const device_ptr ptr_to_remove) {
    device_ptrs_.erase(
        std::remove(device_ptrs_.begin(), device_ptrs_.end(), ptr_to_remove),
        device_ptrs_.end());
  }
};

using shared_event = std::shared_ptr<event>;

} // namespace synapse_helpers

template <>
struct fmt::formatter<synapse_helpers::event> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(synapse_helpers::event const& x, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", fmt::ptr(synEventHandle(x)));
  }
};
