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

#include <future>
#include "absl/hash/hash.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"

namespace synapse_helpers {
class stream;

//! Class responsible of recording events on any stream
class stream_event_manager {
  absl::flat_hash_map<device_ptr, shared_event> events_by_addr_;
  absl::flat_hash_map<device_ptr, std::future<bool>> future_by_addr_;
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
  void add_future(device_ptr device_addr, std::future<bool> fut);
  void add_producer(
      std::vector<device_ptr>&& device_addresses,
      stream& stream,
      event_done_callback done_cb);
  void add_producer(
      std::vector<device_ptr>&& device_addresses,
      std::string event_id,
      stream& stream,
      event_done_callback done_cb);

  void add_event_id(const std::string& event_id, const std::string& new_id);

  /*! \brief Makes \p stream wait until \p device_address is ready to be used if
   * it wasn't ready already \param device_address tensor pointer in device
   * memory space to wait for \param stream         given stream that should
   * wait for \p device_address
   */
  void enqueue_wait_event(device_ptr device_address, stream& stream);
  void enqueue_wait_event(const std::string& event_id, stream& stream);

  /*! \brief Invokes blocking EventSynchronize for a given tenor pointer in
   * device memory space \param device_address identifier of Event - tensor
   * pointer in device memory space
   */
  void wait_for_future(device_ptr device_address);
  void wait_until_done(device_ptr device_address);
  void wait_until_done(shared_event& event);
  void wait_until_done(const std::string& event);

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

  /*! \brief Consider the scenario -
   * Main thread adds some task to the compute stream and adds an event
   * for the output tensors
   * GC thread sees it in pending events and does a synEventSynchronize
   * The main thread may add a wait on the D2H stream with synStreamWaitEvent
   * The task ends in compute stream, the D2H gets the event and starts the DMA
   * There is a race now about when the DMA finishes from D2H and the main
   thread
   * reaches the exit path compared to when the GC thread wakes up from
   * synEventSynchronize and releases the call back tensor handles.
   * If the main thread reaches exit_handler, then it can initiate the
   destructors
   * for statics including the HPUDeviceAllocator::~HPUDeviceAllocator(). This
   will
   * delete the memory pool.
   * At this time, if the GC thread has come out of synEventSynchronize but yet
   to
   * release the call back tensors, there is a problem. When
   synapse_helpers::event::complete
   * starts, if the main thread has released the memory pool, the tensor
   destruction
   * will fail with a SEGV as the pool is now deleted and a nullptr as seen
   below -

      #0  at::habana::pool_allocator::SubAllocator::pool_free_chunk (this=0x0,
   p=0x4659d3e00)
                                                                          ^^^
      #1  at::habana::HPUDeviceAllocator::deleter (ptr=0x4659d3e00)
      #2  c10::TensorImpl::release_resources() [clone .localalias.208] ()
      ...
   * To prevent this, a mutex is used here so that
   HPUDeviceAllocator::~HPUDeviceAllocator()
   * waits if the GC thread is done synEventSynchronize and yet to release the
   * call back tensor handles.
   */
  std::mutex sync_mut_;
};

} // namespace synapse_helpers
