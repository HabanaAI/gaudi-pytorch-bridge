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
#include <string>

#include "absl/types/variant.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/synapse_error.h"

namespace absl {
template <typename... Ts>
class variant;
} // namespace absl

namespace synapse_helpers {

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
      size_t elem_cnt,
      synDataType data_type,
      const std::function<void()>& tensor_cleanup_callback = [] {});

  synapse_error_o reduce(
      HCL_Rank dest_rank,
      device_ptr input_address,
      device_ptr output_address,
      size_t elem_cnt,
      synDataType data_type,
      const std::function<void()>& tensor_cleanup_callback = [] {});

  synapse_error_o broadcast(
      HCL_Rank root_rank,
      uint64_t address,
      size_t elem_cnt,
      synDataType data_type,
      const std::function<void()>& tensor_cleanup_callback = [] {});

  synapse_error_o allgather(
      device_ptr input_address,
      device_ptr output_address,
      size_t elem_cnt,
      synDataType data_type,
      const std::function<void()>& tensor_cleanup_callback = [] {});

  synapse_error_o reduce_scatter(
      device_ptr input_address,
      device_ptr output_address,
      size_t elem_cnt,
      synDataType data_type,
      const std::function<void()>& tensor_cleanup_callback = [] {});

  synapse_error memcpy_within_device(
      device_ptr source,
      device_ptr destination,
      size_t total_bytes,
      std::function<void()> tensor_cleanup_callback);

  synapse_error memcpy_within_device(
      const device::transfer_manifest& manifest,
      std::function<void()> tensor_cleanup_callback);

  synapse_error memcpy_to_device(
      void* cpu_data,
      device_ptr destination,
      size_t total_bytes,
      const event_done_callback& done_cb);

  synapse_error memcpy_to_host(
      device_ptr device_data,
      void* destination,
      size_t total_bytes,
      const event_done_callback& done_cb);

  synapse_error memcpy_sync_to_host(
      device_ptr device_data,
      void* destination,
      size_t total_bytes);

  HCL_Rank my_hcl_rank() const {
    return my_hcl_rank_;
  };

  HCL_Rank root_hcl_rank() const;

  HCL_Comm hcl_comm() const {
    return comm_name_.c_str();
  };

  synDeviceId my_device_id() const {
    // Note: my_device should never be null as this is checked by assert in
    // hcl_communicator::open implementation.
    return my_device_->id();
  }

  unsigned size() const {
    return size_;
  }

  void negotiate_root_rank(int order);

 private:
  static const HCL_Rank HCL_RANK_UNASSIGNED{0xFFFF};

  synapse_error_v<owned_device_ptr> alloc_intermediate_buffer(
      size_t elem_cnt,
      synDataType elem_type,
      HCL_CollectiveOp operation);

  std::string comm_name_;
  int size_{0};
  std::shared_ptr<device> my_device_{nullptr};
  HCL_Rank my_hcl_rank_{HCL_RANK_UNASSIGNED};
  HCL_Rank root_hcl_rank_{HCL_RANK_UNASSIGNED};
};

} // namespace synapse_helpers
