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

enum stream_flavor {
  _BEGIN = 0,
  COMPUTE_0 = 0,
  DMA_D2D = 1,
  DMA_H2D = 2,
  DMA_D2H = 3,
  COLLECTIVE_0 = 4,
  _END = 5
};

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
  explicit stream(device& device, stream_flavor flavor);

  ~stream();

  /*! \brief Pushes newely created event to the std::deque, registers it on its
   * stream handle and notifies garbage collector thread \param event to be
   * pushed to the queue
   */
  void register_pending_event(const shared_event& event);

  /*! \brief Synchronizes stream. This is blocking call that will return only
   * when on computation is done on device side.
   */
  void synchronize();

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
