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
  explicit hcl_communicator(synDeviceId device_id, HCL_Comm hcl_comm = HCL_COMM_WORLD, std::string config_path = "");
  hcl_communicator(hcl_communicator&) = delete;
  hcl_communicator(hcl_communicator&&) = delete;
  hcl_communicator& operator=(hcl_communicator&) = delete;
  hcl_communicator&& operator=(hcl_communicator&&) = delete;
  ~hcl_communicator();

  synapse_error_o allreduce(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                            synDataType data_type, const event_done_callback& done_callback = [] {});

  synapse_error_o reduce(HCL_Rank dest_rank, device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                         synDataType data_type, HCL_Op hclop, const event_done_callback& done_callback = [] {});

  synapse_error_o reduce_scatter(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                                 synDataType data_type, const event_done_callback& done_callback = [] {});

  synapse_error_o alltoall(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                           synDataType data_type, const event_done_callback& done_callback = [] {});

  synapse_error_o broadcast(HCL_Rank root_rank, uint64_t address, size_t elem_cnt, synDataType data_type,
                            const event_done_callback& done_callback = [] {});

  synapse_error_o allgather(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                            synDataType data_type, const event_done_callback& done_callback = [] {});

  synapse_error_o send(device_ptr send_buffer, size_t sizeInBytes, HCL_Rank remoteRank, uint32_t tag,
                       const event_done_callback& done_callback = [] {});

  synapse_error_o receive(device_ptr receive_buffer, size_t sizeInBytes, HCL_Rank remoteRank, uint32_t tag,
                          const event_done_callback& done_callback = [] {});

  void synchronize_output(synapse_helpers::device_ptr output_address);

  HCL_Rank my_hcl_rank() const { return my_hcl_rank_; };

  HCL_Rank root_hcl_rank() const;

  HCL_Comm hcl_comm() const { return comm_id_; };

  synDeviceId my_device_id() const {
    // Note: my_device should never be null as this is checked by assert in hcl_communicator::open implementation.
    return my_device_->id();
  }

  unsigned size() const { return size_; }

  bool using_streams() const { return using_streams_; };

  void negotiate_root_rank(int order);

 private:
  static const HCL_Rank HCL_RANK_UNASSIGNED{0xFFFF};

  using hcl_collective_fnc = std::function<HCLStatus(synStreamHandle handle, device_ptr input_address,
                                                     device_ptr output_address, size_t elem_cnt, synDataType data_type,
                                                     device_ptr intermediate_address, size_t intermediate_size)>;
  synapse_error_o execute_collective_with_fusion_buffer(const hcl_collective_fnc& collective,
                                                        const HCL_CollectiveOp operation, device_ptr input_address,
                                                        device_ptr output_address, size_t elem_cnt,
                                                        synDataType data_type,
                                                        const event_done_callback& done_callback);

  stream* get_collective_stream() const;
  static synStreamHandle get_synapse_stream_handle(stream* maybe_stream);
  void prepare_stream(stream* stream, device_ptr input_address);
  void submit_events(stream* stream, device_ptr output_address, const event_done_callback& done_callback);

  std::shared_ptr<device> my_device_{nullptr};
  std::shared_ptr<owned_device_ptr> intermediate_buffer_{nullptr};
  std::mutex intermediate_buffer_allocation_mtx;
  HCL_Comm comm_id_;
  bool using_streams_;
  int size_{0};
  HCL_Rank my_hcl_rank_{HCL_RANK_UNASSIGNED};
  HCL_Rank root_hcl_rank_{HCL_RANK_UNASSIGNED};
};

}  // namespace synapse_helpers
