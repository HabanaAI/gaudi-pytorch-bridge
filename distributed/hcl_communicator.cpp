/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hcl_communicator.h"

#include <atomic>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <type_traits>

#include <absl/types/variant.h>
#include <absl/strings/match.h>
#include <hcl_api.h>

#include "habana_helpers/logging.h"

// At this moment the only thing we can do for collective is waiting for input tensors to be ready (synEventWait) and
// end synchronoulsy when collective operation is done before returning from op.

#include "synapse_helpers/runtime_tracing.h"

namespace synapse_helpers {

#define VERIFY_HCL_STATUS(msg, status)                         \
  {                                                            \
    if ((status) != eHCLSuccess) {                             \
      std::string msg_str(msg);                                \
      msg_str.append(" HclStatus:");                           \
      msg_str.append(std::to_string(status));                  \
      msg_str.append(" ");                                     \
      msg_str.append(__FILE__);                                \
      msg_str.append("::");                                    \
      msg_str.append(std::to_string(__LINE__));                \
      return make_synapse_error((status), std::move(msg_str)); \
    }                                                          \
  }

hcl_communicator::hcl_communicator(synDeviceId device_id, HCL_Comm hcl_comm, std::string config_path)
    : comm_id_(hcl_comm),
      using_streams_(false) {
  // if config path were not passed by parameter try obtain one from environment
  if (config_path.empty()) {
    char* config_json_path = std::getenv("HCL_CONFIG_PATH");
    if (!config_json_path) {
      PT_SYNHELPER_FATAL("Please export HCL_CONFIG_PATH...");
    }
    config_path = config_json_path;
  }

  // Need to use from pytorch_helpers once we move to a common build after Pytorch1.6
  char* stream_enable_flag = std::getenv("HABANA_HCL_STREAM_ENABLE");
  if (stream_enable_flag && *stream_enable_flag) {
    bool true_found = absl::EqualsIgnoreCase(stream_enable_flag, "1") ||
      absl::EqualsIgnoreCase(stream_enable_flag, "true");
    bool false_found = absl::EqualsIgnoreCase(stream_enable_flag, "0") ||
      absl::EqualsIgnoreCase(stream_enable_flag, "false");
    if (true_found)
      using_streams_ = true;
    else if (false_found)
      using_streams_ =  false;
  }

  PT_SYNHELPER_DEBUG("Opening communication. device_id:", device_id, ".");

  auto device_get_result{synapse_helpers::device::get_by_id(device_id)};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(device_get_result)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_get_result);
    PT_SYNHELPER_FATAL(error.error, " Err: ", error.status, "\n");
  }
  my_device_ = synapse_helpers::get_value(device_get_result);
  HABANA_ASSERT(my_device_ != nullptr);

  HCLStatus hcl_status{HCL_Init(device_id, config_path.c_str())};
  HABANA_ASSERT(hcl_status == eHCLSuccess);

  hcl_status = HCL_Comm_Size(HCL_COMM_WORLD, &size_);
  HABANA_ASSERT(hcl_status == eHCLSuccess);
  HABANA_ASSERT(size_ != 0);

  hcl_status = HCL_Comm_Rank(HCL_COMM_WORLD, &my_hcl_rank_);
  HABANA_ASSERT(hcl_status == eHCLSuccess);
  HABANA_ASSERT(my_hcl_rank_ != HCL_RANK_UNASSIGNED);

  PT_SYNHELPER_DEBUG("Init done. Rank: ", my_hcl_rank_, " Size: ", size_, ".");
}  // namespace synapse_helpers

hcl_communicator::~hcl_communicator() {
  PT_SYNHELPER_DEBUG("~hcl_communicator() entry.");
  HCLStatus hcl_status{eHCLSuccess};
  PT_SYNHELPER_DEBUG("Destroying HCL..");
  hcl_status = HCL_Destroy();
  HABANA_ASSERT(hcl_status == eHCLSuccess);
}

synapse_error_o hcl_communicator::allreduce(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                                            synDataType data_type, const event_done_callback& done_callback) {
  // TBD: Add it to the API.  Will do it as separate release as it needs to be synchronized with pytorch-fork
  auto allreduce_function = [this](synStreamHandle collective_stream, device_ptr input_address,
                                   device_ptr output_address, size_t elem_cnt, synDataType data_type,
                                   device_ptr intermediate_address, size_t intermediate_size) {
    return HCL_Allreduce(collective_stream, input_address, output_address, elem_cnt, data_type, intermediate_address,
                         intermediate_size, eHCLSum, hcl_comm(), false);
  };
  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(allreduce_function, eHCLAllReduce, input_address, output_address,
                                                      elem_cnt, data_type, done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::reduce(HCL_Rank dest_rank, device_ptr input_address, device_ptr output_address,
                                         size_t elem_cnt, synDataType data_type, HCL_Op hclop,
					 const event_done_callback& done_callback) {
  auto reduce_function = [this, dest_rank, hclop](synStreamHandle collective_stream, device_ptr input_address,
                                           device_ptr output_address, size_t elem_cnt, synDataType data_type,
                                           device_ptr intermediate_address, size_t intermediate_size) {
    return HCL_Reduce(collective_stream, input_address, output_address, elem_cnt, data_type, intermediate_address,
                      intermediate_size, dest_rank, hclop, hcl_comm(), false);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(reduce_function, eHCLReduce, input_address, output_address, elem_cnt,
                                                      data_type, done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::reduce_scatter(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                                                 synDataType data_type, const event_done_callback& done_callback) {
  auto reduce_scatter_function = [this](synStreamHandle collective_stream, device_ptr input_address,
                                        device_ptr output_address, size_t elem_cnt, synDataType data_type,
                                        device_ptr intermediate_address, size_t intermediate_size) {
    return HCL_Reduce_Scatter(collective_stream, input_address, output_address, elem_cnt, data_type,
                              intermediate_address, intermediate_size, eHCLSum, hcl_comm(), false);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(reduce_scatter_function, eHCLReduceScatter, input_address,
                                                      output_address, elem_cnt, data_type, done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::alltoall(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                                            synDataType data_type, const event_done_callback& done_callback) {
  auto alltoall_function = [this](synStreamHandle collective_stream, device_ptr input_address,
                                  device_ptr output_address, size_t elem_cnt, synDataType data_type,
                                  device_ptr intermediate_address, size_t intermediate_size) {
    return HCL_AlltoAll(collective_stream, input_address, output_address, elem_cnt, data_type,
                        intermediate_address, intermediate_size, hcl_comm(), false);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(alltoall_function, eHCLAll2All, input_address,
                                               output_address, elem_cnt, data_type, done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::broadcast(HCL_Rank root_rank, device_ptr address, size_t elem_cnt,
                                            synDataType data_type, const std::function<void()>& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;

  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);
  // For root (sending) rank address is input - root does not produce output
  prepare_stream(collective_stream, address);
  status = HCL_Bcast(stream_handle, address, address, elem_cnt, data_type, root_rank, hcl_comm(), false);
  VERIFY_HCL_STATUS("HCL_Bcast(...) failed.", status);
  submit_events(collective_stream, address, done_callback);
  PT_DISTRIBUTED_END;
  return {};
};

synapse_error_o hcl_communicator::allgather(device_ptr input_address, device_ptr output_address, size_t elem_cnt,
                                            synDataType data_type, const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  prepare_stream(collective_stream, input_address);
  status = HCL_AllGather(stream_handle, input_address, output_address, elem_cnt, data_type, hcl_comm(), false);
  VERIFY_HCL_STATUS("HCL_AllGather(...) failed", status);
  submit_events(collective_stream, output_address, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}

HCL_Rank hcl_communicator::root_hcl_rank() const {
  HABANA_ASSERT(root_hcl_rank_ != HCL_RANK_UNASSIGNED && "Call negotiate_root_rank() first to access HCL Root Rank.");
  return root_hcl_rank_;
};

synapse_error_o hcl_communicator::send(device_ptr send_buffer, size_t size_in_bytes, HCL_Rank remote_rank,
                                       uint32_t tag, const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  prepare_stream(collective_stream, send_buffer);
  if (using_streams_) {
    status = HCL_Send(stream_handle, send_buffer, size_in_bytes, remote_rank);
  } else {
    status = HCL_Send_Tag(send_buffer, size_in_bytes, remote_rank, tag);
  }
  VERIFY_HCL_STATUS("HCL_Send(...) failed", status);
  submit_events(collective_stream, send_buffer, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}

synapse_error_o hcl_communicator::receive(device_ptr receive_buffer, size_t size_in_bytes, HCL_Rank remote_rank,
                                          uint32_t tag, const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  if (using_streams_) {
    status = HCL_Receive(stream_handle, receive_buffer, size_in_bytes, remote_rank);
  } else {
    status = HCL_Receive_Tag(receive_buffer, size_in_bytes, remote_rank, tag);
  }
  VERIFY_HCL_STATUS("HCL_Receive(...) failed", status);
  submit_events(collective_stream, receive_buffer, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}
// Being called for every process participating in HCL group, the function determines the HCL Rank of the lowest
// 'order' specified. After calling this function it is possible to retrieve the HCL Root Rank using root_hcl_rank()
// function. It is recommended to call this function just right after HCL Communicator creation.
//
void hcl_communicator::negotiate_root_rank(int order) {
  HABANA_ASSERT(root_hcl_rank_ == HCL_RANK_UNASSIGNED && "The function is meant to be called just once.")
  auto num_workers = size();
  HABANA_ASSERT(num_workers >= 1);
  const auto local_hcl_rank = my_hcl_rank();

  if (num_workers == 1) {
    root_hcl_rank_ = local_hcl_rank;
    return;
  }

  // Hack: Assume there are 32 allgather participants, where not all of them are valid.
  num_workers = 32;

  // The input/output entry for All-Gather operation is a tuple of HCL Rank, Order, and simple checksum.
  using Entry = std::tuple<HCL_Rank, int16_t, uint32_t>;

  auto make_checksum = [](HCL_Rank rank, int16_t order) -> uint32_t {
    return (uint32_t) reinterpret_cast<const uint16_t&>(rank) |
           ((int32_t) reinterpret_cast<const uint16_t&>(order) << 16);
  };

  const auto input_buffer = static_cast<device_ptr>(my_device_->malloc(sizeof(Entry)));
  const auto output_buffer = static_cast<device_ptr>(my_device_->malloc(num_workers * sizeof(Entry)));

  {
    std::atomic<bool> done{false};

    const auto input = Entry{local_hcl_rank, order, make_checksum(local_hcl_rank, order)};
    my_device_->copy_data_to_device((void*)&input, input_buffer, sizeof(input), [&done]() { done = true; });

    while (!done) {
      std::this_thread::yield();
    }
  }

  allgather(input_buffer, output_buffer, sizeof(Entry) / sizeof(int32_t), synDataType::syn_type_int32);

  std::vector<Entry> output(num_workers);

  {
    std::atomic<bool> done{false};

    my_device_->copy_data_to_host(output_buffer, output.data(), num_workers * sizeof(Entry),
                                  [&done]() { done = true; });

    while (!done) {
      std::this_thread::yield();
    }
  }

  for (size_t i = 0; i < num_workers; ++i) {
    if (std::get<0>(output[i]) != i ||
        std::get<2>(output[i]) != make_checksum(std::get<0>(output[i]), std::get<1>(output[i]))) {
      output[i] = Entry{i, std::numeric_limits<int16_t>::max(), -1};
    }
  }

  // The index of worker with the lowest 'order' becomes the HCL Root Rank.
  root_hcl_rank_ =
      std::get<0>(*std::min_element(std::begin(output), std::end(output),
                                    [](const Entry& a, const Entry& b) { return std::get<1>(a) < std::get<1>(b); }));

  my_device_->free(input_buffer);
  my_device_->free(output_buffer);
}

// Privates starts here

synapse_error_o hcl_communicator::execute_collective_with_fusion_buffer(
    const hcl_communicator::hcl_collective_fnc& collective, const HCL_CollectiveOp operation, device_ptr input_address,
    device_ptr output_address, size_t elem_cnt, synDataType data_type, const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  uint64_t required_size{0};
  status = HCL_Get_Intermediate_Buffer_size(&required_size, operation, elem_cnt, data_type, hcl_comm());
  VERIFY_HCL_STATUS("HCL_Get_Intermediate_Buffer_size(...) failed.", status);
  HABANA_ASSERT(required_size != 0)

  std::shared_ptr<owned_device_ptr> intermediate_buffer;
  {
    std::lock_guard<std::mutex> lck(intermediate_buffer_allocation_mtx);
    if ((intermediate_buffer_ == nullptr) || (required_size > intermediate_buffer_->size())) {
      // We need bigger intermediate buffer - allocate new one and store reference
      // We can replace that because:
      //  * All collective ops are sharing the same stream - if that change we need to hold buffer allocation per
      //    stream.
      //  * shared ptr for old buffer was captured in callback function for SEM - buffer will not be deleted until work
      //    scheduled on stream is done.
      intermediate_buffer_ =
          std::make_shared<owned_device_ptr>(my_device_->malloc(required_size), required_size, *my_device_);
      HABANA_ASSERT(intermediate_buffer_ != nullptr);
    }
    intermediate_buffer = intermediate_buffer_;
  }

  // check the ptr
  if (device_nullptr == intermediate_buffer->get()) {
    return synapse_error{"Intermediate buffer memory allocation failed.", synFailedToAllocateDeviceMemory};
  }

  // Include ptr to callback - local variable definition is here because it is impossible to capture property by value
  auto new_done_callback = [intermediate_buffer, done_callback]() mutable {
    intermediate_buffer = nullptr;
    done_callback();
  };

  stream* collective_stream = get_collective_stream();
  prepare_stream(collective_stream, input_address);
  status = collective(get_synapse_stream_handle(collective_stream), input_address, output_address, elem_cnt, data_type,
                      intermediate_buffer->get(), intermediate_buffer->size());
  VERIFY_HCL_STATUS("Collective operation failed", status);
  submit_events(collective_stream, output_address, new_done_callback);
  return {};
}

inline stream* hcl_communicator::get_collective_stream() const {
  if (using_streams_) {
    return &my_device_->get_network_collective_stream();
  } else {
    return nullptr;
  }
}

inline synStreamHandle hcl_communicator::get_synapse_stream_handle(stream* maybe_stream) {
  if (maybe_stream) {
    // We want to dereference that
    return *maybe_stream;
  } else {
    return nullptr;
  }
}

inline void hcl_communicator::prepare_stream(stream* maybe_stream, device_ptr input_address) {
  if (maybe_stream) {
    my_device_->add_wait_events_on_stream({input_address}, *maybe_stream);
  } else {
    trace_scope ts("HclCommunicatorWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }
}

inline void hcl_communicator::submit_events(stream* maybe_stream, device_ptr output_address,
                                            const event_done_callback& done_callback) {
  if (maybe_stream) {
    my_device_->register_producer_on_stream({output_address}, *maybe_stream, done_callback);
  } else {
    done_callback();
  }
}

void hcl_communicator::synchronize_output(synapse_helpers::device_ptr output_address) {
  if (using_streams_) {
    my_device_->wait_until_address_ready(output_address);
  } else {
    return;
  }
}


}  // namespace synapse_helpers
