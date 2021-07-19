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
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/strings/match.h>
#include <absl/types/variant.h>
#include <hcl_api.h>

#include "synapse_helpers/env_flags.h"

#include "habana_helpers/logging.h"

// At this moment the only thing we can do for collective is waiting for input
// tensors to be ready (synEventWait) and end synchronoulsy when collective
// operation is done before returning from op.

#include "synapse_helpers/runtime_tracing.h"

namespace synapse_helpers {

#define VERIFY_HCL_STATUS(msg, status)          \
  {                                             \
    if ((status) != eHCLSuccess) {              \
      std::string msg_str(msg);                 \
      msg_str.append(" HclStatus:");            \
      msg_str.append(std::to_string(status));   \
      msg_str.append(" ");                      \
      msg_str.append(__FILE__);                 \
      msg_str.append("::");                     \
      msg_str.append(std::to_string(__LINE__)); \
      HABANA_ASSERT(status == eHCLSuccess);     \
    }                                           \
  }

#define HCL_SYNC()                                                            \
  {                                                                           \
    if (GET_ENV_FLAG(PT_USE_HCL_SYNC) || GET_ENV_FLAG(PT_HPU_USE_HCL_SYNC)) { \
      HCL_Sync(hcl_comm(), get_sync_tag());                                   \
    }                                                                         \
  }

hcl_communicator::hcl_communicator(
    synDeviceId device_id,
    HCL_Comm hcl_comm,
    std::string config_path)
    : comm_id_(hcl_comm), using_streams_(false) {
  // if config path were not passed by parameter try obtain one from environment
  if (config_path.empty()) {
    char* config_json_path = std::getenv("HCL_CONFIG_PATH");
    if (config_json_path) {
      config_path = config_json_path;
    } else {
      PT_SYNHELPER_DEBUG("HCL_CONFIG_PATH is not set");
    }
  }

  using_streams_ = GET_ENV_FLAG(PT_ENABLE_HCL_STREAM);

  PT_SYNHELPER_DEBUG("Opening communication. device_id:", device_id, ".");

  auto device_get_result{synapse_helpers::device::get_by_id(device_id)};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_get_result)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_get_result);
    PT_SYNHELPER_FATAL(error.error, " Err: ", error.status, "\n");
  }
  my_device_ = synapse_helpers::get_value(device_get_result);
  HABANA_ASSERT(my_device_ != nullptr);

  HCLStatus hcl_status{
      HCL_Init(device_id, config_path.empty() ? nullptr : config_path.c_str())};
  HABANA_ASSERT(hcl_status == eHCLSuccess);

  hcl_status = HCL_Comm_Size(hcl_comm, &size_);
  HABANA_ASSERT(hcl_status == eHCLSuccess);
  HABANA_ASSERT(size_ != 0);

  hcl_status = HCL_Comm_Rank(hcl_comm, &my_hcl_rank_);
  HABANA_ASSERT(hcl_status == eHCLSuccess);
  HABANA_ASSERT(my_hcl_rank_ != HCL_RANK_UNASSIGNED);

  PT_SYNHELPER_DEBUG("Init done. Rank: ", my_hcl_rank_, " Size: ", size_, ".");
} // namespace synapse_helpers

hcl_communicator::~hcl_communicator() {
  PT_SYNHELPER_DEBUG("~hcl_communicator() entry.");
  HCLStatus hcl_status{eHCLSuccess};
  get_collective_stream()->synchronize();
  HCL_Sync(hcl_comm(), get_sync_tag());
  PT_SYNHELPER_DEBUG("Destroying HCL..");
  hcl_status = HCL_Destroy();
  HABANA_ASSERT(hcl_status == eHCLSuccess);
}

synapse_error_o hcl_communicator::allreduce(
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    HCL_Op hclop,
    const event_done_callback& done_callback) {
  // TBD: Add it to the API.  Will do it as separate release as it needs to be
  // synchronized with pytorch-fork
  HABANA_ASSERT(hclop == eHCLSum || hclop == eHCLMul)
  auto allreduce_function = [this, hclop](
                                synStreamHandle collective_stream,
                                device_ptr input_address,
                                device_ptr output_address,
                                size_t elem_cnt,
                                synDataType data_type,
                                device_ptr intermediate_address,
                                size_t intermediate_size,
                                uint32_t flags) {
    HCL_SYNC()
    auto status = HCL_Allreduce(
        collective_stream,
        input_address,
        output_address,
        elem_cnt,
        data_type,
        intermediate_address,
        intermediate_size,
        hclop,
        hcl_comm(),
        flags);
    HCL_SYNC()
    return status;
  };
  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(
      allreduce_function,
      eHCLAllReduce,
      input_address,
      output_address,
      in_event_addr,
      out_event_addr,
      elem_cnt,
      data_type,
      done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::allreduce(
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    const event_done_callback& done_callback) {
  return allreduce(
      input_address,
      output_address,
      in_event_addr,
      out_event_addr,
      elem_cnt,
      data_type,
      eHCLSum,
      done_callback);
}

synapse_error_o hcl_communicator::reduce(
    HCL_Rank dest_rank,
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    HCL_Op hclop,
    const event_done_callback& done_callback) {
  auto reduce_function = [this, dest_rank, hclop](
                             synStreamHandle collective_stream,
                             device_ptr input_address,
                             device_ptr output_address,
                             size_t elem_cnt,
                             synDataType data_type,
                             device_ptr intermediate_address,
                             size_t intermediate_size,
                             uint32_t flags) {
    return HCL_Reduce(
        collective_stream,
        input_address,
        output_address,
        elem_cnt,
        data_type,
        intermediate_address,
        intermediate_size,
        dest_rank,
        hclop,
        hcl_comm(),
        flags);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(
      reduce_function,
      eHCLReduce,
      input_address,
      output_address,
      in_event_addr,
      out_event_addr,
      elem_cnt,
      data_type,
      done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::reduce_scatter(
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    HCL_Op hclop,
    const event_done_callback& done_callback) {
  auto reduce_scatter_function = [this, hclop](
                                     synStreamHandle collective_stream,
                                     device_ptr input_address,
                                     device_ptr output_address,
                                     size_t elem_cnt,
                                     synDataType data_type,
                                     device_ptr intermediate_address,
                                     size_t intermediate_size,
                                     uint32_t flags) {
    return HCL_Reduce_Scatter(
        collective_stream,
        input_address,
        output_address,
        elem_cnt,
        data_type,
        intermediate_address,
        intermediate_size,
        hclop,
        hcl_comm(),
        flags);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(
      reduce_scatter_function,
      eHCLReduceScatter,
      input_address,
      output_address,
      in_event_addr,
      out_event_addr,
      elem_cnt,
      data_type,
      done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::alltoall(
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    const event_done_callback& done_callback) {
  auto alltoall_function = [this](
                               synStreamHandle collective_stream,
                               device_ptr input_address,
                               device_ptr output_address,
                               size_t elem_cnt,
                               synDataType data_type,
                               device_ptr intermediate_address,
                               size_t intermediate_size,
                               uint32_t flags) {
    HCL_SYNC()
    return HCL_AlltoAll(
        collective_stream,
        input_address,
        output_address,
        elem_cnt,
        data_type,
        intermediate_address,
        intermediate_size,
        hcl_comm(),
        flags);
  };

  PT_DISTRIBUTED_BEGIN;
  auto status = execute_collective_with_fusion_buffer(
      alltoall_function,
      eHCLAll2All,
      input_address,
      output_address,
      in_event_addr,
      out_event_addr,
      elem_cnt,
      data_type,
      done_callback);
  PT_DISTRIBUTED_END;
  return status;
}

synapse_error_o hcl_communicator::broadcast(
    HCL_Rank root_rank,
    device_ptr address,
    device_ptr event_addr,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;

  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);
  // For root (sending) rank address is input - root does not produce output
  prepare_stream(collective_stream, event_addr);
  HCL_SYNC()
  status = HCL_Bcast(
      stream_handle,
      address,
      address,
      elem_cnt,
      data_type,
      root_rank,
      hcl_comm(),
      0 /*flags*/);
  VERIFY_HCL_STATUS("HCL_Bcast(...) failed.", status);
  submit_events(collective_stream, event_addr, done_callback);
  PT_DISTRIBUTED_END;
  return {};
};

synapse_error_o hcl_communicator::allgather(
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  prepare_stream(collective_stream, in_event_addr);
  HCL_SYNC()
  status = HCL_AllGather(
      stream_handle,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      hcl_comm(),
      0 /*flags*/);
  VERIFY_HCL_STATUS("HCL_AllGather(...) failed", status);
  submit_events(collective_stream, out_event_addr, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}

size_t hcl_communicator::get_aligned_data_size(
    size_t elem_cnt,
    synDataType data_type) {
  HABANA_ASSERT((data_type == syn_type_float) || (data_type == syn_type_bf16));
  size_t elem_size = (data_type == syn_type_bf16) ? 2 : 4;
  return ((elem_size * elem_cnt) + 0xFFFF) & ~0xFFFF;
}

bool hcl_communicator::can_data_fit_preallocated_buffer(
    size_t elem_cnt,
    synDataType data_type,
    HCL_CollectiveOp operation) {
  HABANA_ASSERT(my_device_ != nullptr);
  const absl::optional<owned_device_ptr>& maybe_reduction_buff{
      my_device_->reduction_buffer()};
  if (!maybe_reduction_buff.has_value()) {
    return false;
  }
  auto& reduction_buff = maybe_reduction_buff.value();

  HCLStatus status{eHCLSuccess};
  size_t required_int_buff_size{0};
  elem_cnt = get_aligned_elem_cnt(elem_cnt);
  status = HCL_Get_Intermediate_Buffer_size(
      &required_int_buff_size, operation, elem_cnt, data_type, hcl_comm());
  HABANA_ASSERT(status == eHCLSuccess);
  HABANA_ASSERT(required_int_buff_size != 0);
  size_t data_size = get_aligned_data_size(elem_cnt, data_type);
  // TBD: if we dont fit, we need to fail the feature else there can be a mixup
  // of data
  return (data_size + required_int_buff_size) < reduction_buff.size();
}

HCL_Rank hcl_communicator::root_hcl_rank() const {
  HABANA_ASSERT(
      root_hcl_rank_ != HCL_RANK_UNASSIGNED &&
      "Call negotiate_root_rank() first to access HCL Root Rank.");
  return root_hcl_rank_;
};

synapse_error_o hcl_communicator::send(
    device_ptr send_buffer,
    device_ptr event_addr,
    size_t size_in_bytes,
    HCL_Rank remote_rank,
    uint32_t tag,
    const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  prepare_stream(collective_stream, event_addr);
  if (using_streams_) {
    status = HCL_Send(stream_handle, send_buffer, size_in_bytes, remote_rank);
  } else {
    status = HCL_Send_Tag(send_buffer, size_in_bytes, remote_rank, tag);
  }
  VERIFY_HCL_STATUS("HCL_Send(...) failed", status);
  submit_events(collective_stream, event_addr, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}

synapse_error_o hcl_communicator::receive(
    device_ptr receive_buffer,
    device_ptr event_addr,
    size_t size_in_bytes,
    HCL_Rank remote_rank,
    uint32_t tag,
    const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  PT_DISTRIBUTED_BEGIN;
  stream* collective_stream = get_collective_stream();
  synStreamHandle stream_handle = get_synapse_stream_handle(collective_stream);

  if (using_streams_) {
    status =
        HCL_Receive(stream_handle, receive_buffer, size_in_bytes, remote_rank);
  } else {
    status = HCL_Receive_Tag(receive_buffer, size_in_bytes, remote_rank, tag);
  }
  VERIFY_HCL_STATUS("HCL_Receive(...) failed", status);
  submit_events(collective_stream, event_addr, done_callback);
  PT_DISTRIBUTED_END;
  return {};
}

synapse_error_o hcl_communicator::barrier() {
  PT_DISTRIBUTED_BEGIN;

  if (using_streams_) {
    stream* collective_stream = get_collective_stream();
    synStreamHandle stream_handle =
        get_synapse_stream_handle(collective_stream);
    HCL_Request phRequest;

    HCL_NetworkFlush(&phRequest, stream_handle);
    HCL_Wait(phRequest);
  }

  HCL_Sync(hcl_comm(), get_sync_tag());

  PT_DISTRIBUTED_END;
  return {};
}

// Being called for every process participating in HCL group, the function
// determines the HCL Rank of the lowest 'order' specified. After calling this
// function it is possible to retrieve the HCL Root Rank using root_hcl_rank()
// function. It is recommended to call this function just right after HCL
// Communicator creation.
//
void hcl_communicator::negotiate_root_rank(int order) {
  HABANA_ASSERT(
      root_hcl_rank_ == HCL_RANK_UNASSIGNED &&
      "The function is meant to be called just once.")
  auto num_workers = size();
  HABANA_ASSERT(num_workers >= 1);
  const auto local_hcl_rank = my_hcl_rank();

  if (num_workers == 1) {
    root_hcl_rank_ = local_hcl_rank;
    return;
  }

  // Hack: Assume there are 32 allgather participants, where not all of them are
  // valid.
  num_workers = 32;

  // The input/output entry for All-Gather operation is a tuple of HCL Rank,
  // Order, and simple checksum.
  using Entry = std::tuple<HCL_Rank, int16_t, uint32_t>;

  auto make_checksum = [](HCL_Rank rank, int16_t order) -> uint32_t {
    return (uint32_t) reinterpret_cast<const uint16_t&>(rank) |
        ((int32_t) reinterpret_cast<const uint16_t&>(order) << 16);
  };

  const auto input_buffer =
      static_cast<device_ptr>(my_device_->malloc(sizeof(Entry)));
  const auto output_buffer =
      static_cast<device_ptr>(my_device_->malloc(num_workers * sizeof(Entry)));

  {
    std::atomic<bool> done{false};

    const auto input =
        Entry{local_hcl_rank, order, make_checksum(local_hcl_rank, order)};
    my_device_->copy_data_to_device(
        (void*)&input, input_buffer, input_buffer, sizeof(input), [&done]() {
          done = true;
        });

    while (!done) {
      std::this_thread::yield();
    }
  }

  allgather(
      input_buffer,
      output_buffer,
      input_buffer,
      output_buffer,
      sizeof(Entry) / sizeof(int32_t),
      synDataType::syn_type_int32);

  std::vector<Entry> output(num_workers);

  {
    std::atomic<bool> done{false};

    my_device_->copy_data_to_host(
        output_buffer,
        output.data(),
        output_buffer,
        num_workers * sizeof(Entry),
        [&done]() { done = true; });

    while (!done) {
      std::this_thread::yield();
    }
  }

  for (size_t i = 0; i < num_workers; ++i) {
    if (std::get<0>(output[i]) != i ||
        std::get<2>(output[i]) !=
            make_checksum(std::get<0>(output[i]), std::get<1>(output[i]))) {
      output[i] = Entry{i, std::numeric_limits<int16_t>::max(), -1};
    }
  }

  // The index of worker with the lowest 'order' becomes the HCL Root Rank.
  root_hcl_rank_ = std::get<0>(*std::min_element(
      std::begin(output), std::end(output), [](const Entry& a, const Entry& b) {
        return std::get<1>(a) < std::get<1>(b);
      }));

  my_device_->free(input_buffer);
  my_device_->free(output_buffer);
}

// Privates starts here

synapse_error_o hcl_communicator::memcpy_in_interim_buffer(
    device_ptr input_address,
    const void*& fused_input_data,
    void*& buffer_data,
    size_t& buffer_len,
    const owned_device_ptr& reduction_buffer) {
  buffer_data = reinterpret_cast<void*>(reduction_buffer.get());
  fused_input_data = buffer_data;

  synapse_helpers::device::transfer_manifest transfers;
  transfers.reserve(1);

  size_t offset{0};
  synapse_helpers::device::transfer_desc transfer{};
  transfer.src = reinterpret_cast<synapse_helpers::device_ptr>(input_address);
  transfer.dst = reinterpret_cast<synapse_helpers::device_ptr>(
      (uint8_t*)buffer_data + offset);
  transfer.bytes_to_transfer = buffer_len;
  transfers.emplace_back(transfer);
  offset += buffer_len;

  auto maybe_error{my_device_->copy_data_within_device(
      transfers, [transfers]() { return; })};

  return {};
}

synapse_error_o hcl_communicator::memcpy_out_interim_buffer(
    const void* buffer_data,
    device_ptr output_address,
    size_t& buffer_len) {
  int64_t offset{0};
  synapse_helpers::device::transfer_manifest transfers;
  transfers.reserve(1);
  void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
  synapse_helpers::device::transfer_desc transfer{};
  transfer.src =
      reinterpret_cast<synapse_helpers::device_ptr>(buffer_data_at_offset);
  transfer.dst = reinterpret_cast<synapse_helpers::device_ptr>(output_address);
  transfer.bytes_to_transfer = buffer_len;
  transfers.emplace_back(transfer);
  offset += buffer_len;

  auto maybe_error{my_device_->copy_data_within_device(
      transfers, [transfers]() { return; })};

  return {};
}

void hcl_communicator::print_collective_buffer_size(
    size_t elem_cnt,
    synDataType data_type,
    HCL_CollectiveOp operation) {
  HCLStatus status{eHCLSuccess};
  size_t required_int_buff_size{0};
  elem_cnt = get_aligned_elem_cnt(elem_cnt);
  status = HCL_Get_Intermediate_Buffer_size(
      &required_int_buff_size, operation, elem_cnt, data_type, hcl_comm());
  HABANA_ASSERT(status == eHCLSuccess);
  HABANA_ASSERT(required_int_buff_size != 0);
  size_t data_size = get_aligned_data_size(elem_cnt, data_type);

  if (my_hcl_rank_ == 0) {
    std::cerr << " intermediate_buf_size :: " << required_int_buff_size
              << std::endl;
    std::cerr << " data_size :: " << data_size << std::endl;
  }
}

synapse_error_o hcl_communicator::execute_collective_with_fusion_buffer(
    const hcl_communicator::hcl_collective_fnc& collective,
    const HCL_CollectiveOp operation,
    device_ptr input_address,
    device_ptr output_address,
    device_ptr in_event_addr,
    device_ptr out_event_addr,
    size_t elem_cnt,
    synDataType data_type,
    const event_done_callback& done_callback) {
  HCLStatus status{eHCLSuccess};
  HABANA_ASSERT(nullptr != my_device_);
  const absl::optional<owned_device_ptr>& maybe_reduction_buff{
      my_device_->reduction_buffer()};
  const void* fused_input_data;
  void* fused_output_data = nullptr;
  size_t buffer_len = 0;

  // Use same address if preallocated reduction buffer exists, operation is
  // allreduce and data is in preallocated reduction buffer. It is caller
  // responsibility to check if reduction_buffer has enough size before using
  // its address for allreduce call.
  const bool same_address{
      maybe_reduction_buff.has_value() && (eHCLAllReduce == operation) &&
      (input_address == output_address) /* && (input_address ==
                                           maybe_reduction_buff.value().get())*/
      && can_data_fit_preallocated_buffer(elem_cnt, data_type, eHCLAllReduce)};

  std::shared_ptr<owned_device_ptr> intermediate_buffer;
  device_ptr intermediate_buffer_address{};
  size_t intermediate_buffer_size{0};

  if (same_address) {
    // As intermediate buffer also must have same_address across communicator,
    // we need to use part of reduction_buffer as intermediate - offset of this
    // part is computed using data size and appriopriate alignment.
    const owned_device_ptr& reduction_buffer = maybe_reduction_buff.value();
    elem_cnt = get_aligned_elem_cnt(elem_cnt);
    size_t intermediate_offset = get_aligned_data_size(elem_cnt, data_type);
    intermediate_buffer_address = reduction_buffer.get() + intermediate_offset;
    intermediate_buffer_size = reduction_buffer.size() - intermediate_offset;
    buffer_len = elem_cnt;
    memcpy_in_interim_buffer(
        input_address,
        fused_input_data,
        fused_output_data,
        buffer_len,
        maybe_reduction_buff.value());
  } else {
    uint64_t required_size{0};
    status = HCL_Get_Intermediate_Buffer_size(
        &required_size, operation, elem_cnt, data_type, hcl_comm());
    VERIFY_HCL_STATUS("HCL_Get_Intermediate_Buffer_size(...) failed.", status);
    HABANA_ASSERT(required_size != 0)

    std::shared_ptr<owned_device_ptr> intermediate_buffer;
    {
      std::lock_guard<std::mutex> lck(intermediate_buffer_allocation_mtx);
      if ((intermediate_buffer_ == nullptr) ||
          (required_size > intermediate_buffer_->size())) {
        // We need bigger intermediate buffer - allocate new one and store
        // reference We can replace that because:
        //  * All collective ops are sharing the same stream - if that change we
        //  need to hold buffer allocation per
        //    stream.
        //  * shared ptr for old buffer was captured in callback function for
        //  SEM - buffer will not be deleted until work
        //    scheduled on stream is done.
        intermediate_buffer_ = std::make_shared<owned_device_ptr>(
            my_device_->malloc(required_size), required_size, *my_device_);
        HABANA_ASSERT(intermediate_buffer_ != nullptr);
      }
      intermediate_buffer = intermediate_buffer_;
    }

    // check the ptr
    if (device_nullptr == intermediate_buffer->get()) {
      return synapse_error{
          "Intermediate buffer memory allocation failed.",
          synFailedToAllocateDeviceMemory};
    }
    intermediate_buffer_address = intermediate_buffer->get();
    intermediate_buffer_size = intermediate_buffer->size();
  }

  // Include ptr to callback - local variable definition is here because it is
  // impossible to capture property by value
  auto new_done_callback = [intermediate_buffer, done_callback]() mutable {
    intermediate_buffer = nullptr;
    done_callback();
  };

  stream* collective_stream = get_collective_stream();
  prepare_stream(collective_stream, in_event_addr);
  auto input = input_address;
  auto output = output_address;
  uint32_t flags = 0;

  // TBD: ensure allreduce buffers are not dependant
  if ((eHCLAllReduce == operation) && GET_ENV_FLAG(PT_USE_HCL_OPTS)) {
    // this will aid in pipelining allreduce calls in hcl
    flags = eHCLWeakOrder;
  }

  if (same_address) {
    input = reinterpret_cast<synapse_helpers::device_ptr>(fused_input_data);
    output = reinterpret_cast<synapse_helpers::device_ptr>(fused_output_data);
    flags = (1 << 0); // eHCLSameAddress;
  }
  status = collective(
      get_synapse_stream_handle(collective_stream),
      input,
      output,
      elem_cnt,
      data_type,
      intermediate_buffer_address,
      intermediate_buffer_size,
      flags);
  VERIFY_HCL_STATUS("Collective operation failed", status);
  submit_events(collective_stream, out_event_addr, new_done_callback);
  if (same_address) {
    memcpy_out_interim_buffer(fused_output_data, input_address, buffer_len);
  }
  return {};
}

inline stream* hcl_communicator::get_collective_stream() const {
  if (using_streams_) {
    return &my_device_->get_network_collective_stream();
  } else {
    return nullptr;
  }
}

inline synStreamHandle hcl_communicator::get_synapse_stream_handle(
    stream* maybe_stream) {
  if (maybe_stream) {
    // We want to dereference that
    return *maybe_stream;
  } else {
    return nullptr;
  }
}

inline void hcl_communicator::prepare_stream(
    stream* maybe_stream,
    device_ptr input_address) {
  if (maybe_stream) {
    my_device_->add_wait_events_on_stream({input_address}, *maybe_stream);
  } else {
    trace_scope ts("HclCommunicatorWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }
}

inline void hcl_communicator::submit_events(
    stream* maybe_stream,
    device_ptr output_address,
    const event_done_callback& done_callback) {
  if (maybe_stream) {
    my_device_->register_producer_on_stream(
        {output_address}, *maybe_stream, done_callback);
  } else {
    done_callback();
  }
}

void hcl_communicator::synchronize_output(
    synapse_helpers::device_ptr output_address) {
  // Added to pipeline the lazy host copy operations after
  // communication collective is called.
  if (using_streams_ &&
      (GET_ENV_FLAG(PT_ENABLE_SYNC_OUTPUT_HOST) ||
       GET_ENV_FLAG(PT_HPU_ENABLE_SYNC_OUTPUT_HOST))) {
    my_device_->wait_until_address_ready(output_address);
  } else {
    return;
  }
}

} // namespace synapse_helpers
