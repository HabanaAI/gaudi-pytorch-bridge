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
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace synapse_helpers {

class event;

using shared_event = std::shared_ptr<event>;
class device;

/*! Wrapper Class for synStreamHandle
 keeps also cleaning thread (garbage collector thread)
 one instance per synStream
*/
class stream {
  static const int default_flush_timeout_ms = 1000;
  std::queue<shared_event> pending_cleanups_;
  device& device_;
  std::mutex mut_{};
  std::condition_variable cond_var_;
  std::condition_variable cond_var_empty;
  std::thread gc_worker_;

  synStreamHandle handle_;

  /*! \brief Internal garbage collector thread, that collects all the events
   * from the std::deque and tries to synchronize them
   */
  void gc_thread_proc();

 public:
  /*! \brief Constructor
   *  \param id of the device
   *  \param flavor dedicated usage type of the stream
   */
  explicit stream(device& device);

  stream(const stream& other)
      : device_(other.device_), handle_(other.handle_) {}

  ~stream();

  /*! \brief Pushes newely created event to the std::deque, registers it on its
   * stream handle and notifies garbage collector thread \param event to be
   *  \param record_event indicates whether event should be recorded
   * pushed to the queue
   */
  void register_pending_event(
      const shared_event& event,
      bool record_event = true);

  /*! \brief Synchronizes stream. This is blocking call that will return only
   * when on computation is done on device side.
   */
  void synchronize();

  synStatus query();

  /*! \return device
   */
  device& get_device() const {
    return device_;
  }

  operator synStreamHandle() const {
    return handle_;
  }

  bool operator==(const stream& other) const {
    return handle_ == other.handle_;
  }

  bool operator!=(const stream& other) const {
    return handle_ != other.handle_;
  }
  void flush(int timeout_ms = default_flush_timeout_ms);

 private:
  template <typename collection_t>
  void try_sync_events(collection_t& events_to_sync);
};

} // namespace synapse_helpers

template <>
struct fmt::formatter<synapse_helpers::stream> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(synapse_helpers::stream const& x, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", fmt::ptr(synStreamHandle(x)));
  }
};
