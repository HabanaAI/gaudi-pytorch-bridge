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

#include <hcl_api_types.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "synapse_helpers/device.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/synapse_error.h"

namespace synapse_helpers {
class stream;

inline synapse_error make_synapse_error(HCLStatus hcl_status, std::string msg) {
  synStatus status{synFail};
  switch (hcl_status) {
    case eHCLSuccess:
      status = synSuccess;
      break;
    case eHCLFail:
      status = synFail;
      break;
    case eHCLInvalidArgument:
      status = synInvalidArgument;
      break;
    case eHCLBusy:
      status = synBusy;
      break;
    case eHCLNotSupported:
      status = synUnsupported;
      break;
  }
  return {std::move(msg), status};
}

class hcl_communicator {
 public:
  explicit hcl_communicator(
      synDeviceId device_id,
      HCL_Comm hcl_comm = HCL_COMM_WORLD,
      std::string config_path = "");
  hcl_communicator(hcl_communicator&) = delete;
  hcl_communicator(hcl_communicator&&) = delete;
  hcl_communicator& operator=(hcl_communicator&) = delete;
  hcl_communicator&& operator=(hcl_communicator&&) = delete;
  ~hcl_communicator();

  synapse_error_o allreduce(
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      HCL_Op hclop,
      const event_done_callback& done_callback = [] {});

  synapse_error_o allreduce(
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      const event_done_callback& done_callback = [] {});

  synapse_error_o reduce(
      HCL_Rank dest_rank,
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      HCL_Op hclop,
      const event_done_callback& done_callback = [] {});

  synapse_error_o reduce_scatter(
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      HCL_Op hclop,
      const event_done_callback& done_callback = [] {});

  synapse_error_o alltoall(
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      const event_done_callback& done_callback = [] {});

  synapse_error_o broadcast(
      HCL_Rank root_rank,
      uint64_t address,
      device_ptr event_addr,
      size_t elem_cnt,
      synDataType data_type,
      const event_done_callback& done_callback = [] {});

  synapse_error_o allgather(
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      const event_done_callback& done_callback = [] {});

  synapse_error_o send(
      device_ptr send_buffer,
      device_ptr event_addr,
      size_t sizeInBytes,
      HCL_Rank remoteRank,
      uint32_t tag,
      const event_done_callback& done_callback = [] {});

  synapse_error_o receive(
      device_ptr receive_buffer,
      device_ptr event_addr,
      size_t sizeInBytes,
      HCL_Rank remoteRank,
      uint32_t tag,
      const event_done_callback& done_callback = [] {});

  synapse_error_o barrier();

  void synchronize_output(synapse_helpers::device_ptr output_address);

  HCL_Rank my_hcl_rank() const {
    return my_hcl_rank_;
  };

  device_ptr reduction_buffer_addr();

  bool can_data_fit_preallocated_buffer(
      size_t elem_cnt,
      synDataType data_type,
      HCL_CollectiveOp operation);

  void print_collective_buffer_size(
      size_t elem_cnt,
      synDataType data_type,
      HCL_CollectiveOp operation);

  size_t get_aligned_data_size(size_t elem_cnt, synDataType data_type);

  size_t get_aligned_elem_cnt(size_t elem_cnt) {
    const size_t alignment = 64 * size();
    return ((elem_cnt + alignment - 1) / alignment) * alignment;
  }

  HCL_Rank root_hcl_rank() const;

  HCL_Comm hcl_comm() const {
    return comm_id_;
  };

  synDeviceId my_device_id() const {
    // Note: my_device should never be null as this is checked by assert in
    // hcl_communicator::open implementation.
    return my_device_->id();
  }

  unsigned size() const {
    return size_;
  }

  bool using_streams() const {
    return using_streams_;
  };

  void negotiate_root_rank(int order);

  uint32_t get_sync_tag() const {
    sync_tag_++;
    return sync_tag_;
  }

 private:
  static const HCL_Rank HCL_RANK_UNASSIGNED{0xFFFF};

  using hcl_collective_fnc = std::function<HCLStatus(
      synStreamHandle handle,
      device_ptr input_address,
      device_ptr output_address,
      size_t elem_cnt,
      synDataType data_type,
      device_ptr intermediate_address,
      size_t intermediate_size,
      uint32_t flags)>;
  synapse_error_o execute_collective_with_fusion_buffer(
      const hcl_collective_fnc& collective,
      const HCL_CollectiveOp operation,
      device_ptr input_address,
      device_ptr output_address,
      device_ptr in_event_addr,
      device_ptr out_event_addr,
      size_t elem_cnt,
      synDataType data_type,
      const event_done_callback& done_callback);

  stream* get_collective_stream() const;
  static synStreamHandle get_synapse_stream_handle(stream* maybe_stream);
  void prepare_stream(stream* stream, device_ptr input_address);
  void submit_events(
      stream* stream,
      device_ptr output_address,
      const event_done_callback& done_callback);

  synapse_error_o memcpy_in_interim_buffer(
      device_ptr input_address,
      const void*& fused_input_data,
      void*& buffer_data,
      size_t& buffer_len,
      const owned_device_ptr& reduction_buffer);
  synapse_error_o memcpy_out_interim_buffer(
      const void* buffer_data,
      device_ptr output_address,
      size_t& buffer_len);

  std::shared_ptr<device> my_device_{nullptr};
  std::shared_ptr<owned_device_ptr> intermediate_buffer_{nullptr};
  std::mutex intermediate_buffer_allocation_mtx;
  HCL_Comm comm_id_;
  bool using_streams_;
  mutable uint32_t sync_tag_{2020};
  int size_{0};
  HCL_Rank my_hcl_rank_{HCL_RANK_UNASSIGNED};
  HCL_Rank root_hcl_rank_{HCL_RANK_UNASSIGNED};
};

inline device_ptr hcl_communicator::reduction_buffer_addr() {
  const absl::optional<owned_device_ptr>& maybe_reduction_buff{
      my_device_->reduction_buffer()};
  return maybe_reduction_buff.has_value() ? maybe_reduction_buff.value().get()
                                          : device_nullptr;
}

} // namespace synapse_helpers
