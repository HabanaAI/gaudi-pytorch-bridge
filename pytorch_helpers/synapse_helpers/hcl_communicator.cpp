/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/hcl_communicator.h"

#include <atomic>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/types/variant.h>

#include <hcl_api.h>

#include "habana_helpers/logging.h"

// At this moment the only thing we can do for collective is waiting for input
// tensors to be ready (synEventWait) and end synchronoulsy when collective
// operation is done before returning from op.

#define HCL_STREAM_SUPPORT 0

#include "synapse_helpers/runtime_tracing.h"

PtLogger* PtLogger::instance = nullptr;

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

hcl_communicator::hcl_communicator(
    synDeviceId device_id,
    HCL_Comm hcl_comm,
    std::string config_path)
    : comm_name_(hcl_comm) {
  // if config path were not passed by parameter try obtain one from environment
  if (config_path.empty()) {
    char* config_json_path = std::getenv("HCL_CONFIG_PATH");
    if (!config_json_path) {
      PT_SYNHELPER_FATAL("Please export HCL_CONFIG_PATH...");
    }
    config_path = config_json_path;
  }

  PT_SYNHELPER_DEBUG("Opening communication. device_id:", device_id, ".");

  auto device_get_result{synapse_helpers::device::get_by_id(device_id)};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_get_result)) {
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
} // namespace synapse_helpers

synapse_error hcl_communicator::memcpy_within_device(
    device_ptr source,
    device_ptr destination,
    size_t total_bytes,
    std::function<void()> tensor_cleanup_callback) {
  HABANA_ASSERT(my_device_ != nullptr);
  return my_device_->copy_data_within_device(
      source, destination, total_bytes, std::move(tensor_cleanup_callback));
}

synapse_error hcl_communicator::memcpy_within_device(
    const device::transfer_manifest& manifest,
    std::function<void()> tensor_cleanup_callback) {
  HABANA_ASSERT(my_device_ != nullptr);
  return my_device_->copy_data_within_device(
      manifest, std::move(tensor_cleanup_callback));
}

synapse_error hcl_communicator::memcpy_to_device(
    void* cpu_data,
    device_ptr destination,
    size_t total_bytes,
    const event_done_callback& done_cb) {
  HABANA_ASSERT(my_device_ != nullptr);
  return my_device_->copy_data_to_device(
      cpu_data, destination, total_bytes, done_cb);
}

synapse_error hcl_communicator::memcpy_to_host(
    device_ptr device_data,
    void* destination,
    size_t total_bytes,
    const event_done_callback& done_cb) {
  HABANA_ASSERT(my_device_ != nullptr);
  return my_device_->copy_data_to_host(
      device_data, destination, total_bytes, done_cb);
}

synapse_error hcl_communicator::memcpy_sync_to_host(
    device_ptr device_data,
    void* destination,
    size_t total_bytes) {
  HABANA_ASSERT(my_device_ != nullptr);
  std::mutex mtx;
  std::condition_variable cv;
  std::atomic<bool> done{false};

  synapse_error maybe_error{
      my_device_->copy_data_to_host(device_data, destination, total_bytes, [&] {
        // Note: It is ok to capture by reference here, as this function by
        // design, must not exit until callback is called.
        std::unique_lock<std::mutex> lck(mtx);
        done.store(true);
        cv.notify_one();
      })};
  SYNAPSE_RETURN_IF_ERROR(maybe_error);
  std::unique_lock<std::mutex> lck(mtx);
  cv.wait(lck, [&] { return done.load(); });
  return maybe_error;
}

synapse_error_v<owned_device_ptr> hcl_communicator::alloc_intermediate_buffer(
    size_t elem_cnt,
    synDataType elem_type,
    HCL_CollectiveOp operation) {
  PT_SYNHELPER_DEBUG("alloc_intermediate_buffer entry()");
  HCLStatus status{eHCLSuccess};

  uint64_t required_size{0};
  status = HCL_Get_Intermediate_Buffer_size(
      &required_size, operation, elem_cnt, elem_type, hcl_comm());
  VERIFY_HCL_STATUS("HCL_Get_Intermediate_Buffer_size(...) failed.", status);
  HABANA_ASSERT(required_size != 0)

  owned_device_ptr buffer{
      my_device_->malloc(required_size), required_size, *my_device_};

  // check the ptr
  if (device_nullptr == buffer.get()) {
    return synapse_error{"Intermediate buffer memory allocation failed.",
                         synFailedToAllocateDeviceMemory};
  }

  return {std::move(buffer)};
}

synapse_error_o hcl_communicator::reduce_scatter(
    device_ptr input_address,
    device_ptr output_address,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& tensor_cleanup_callback) {
  HCLStatus status{eHCLSuccess};
  trace_start("IntermediateBufferAlloc");
  synapse_error_v<owned_device_ptr> maybe_buffer_ptr{
      alloc_intermediate_buffer(elem_cnt, data_type, eHCLReduceScatter)};
  if (!ok(maybe_buffer_ptr)) {
    synapse_error error = get_error(maybe_buffer_ptr);
    PT_SYNHELPER_WARN(
        "Intermediate buffer allocation failed. ",
        error.error,
        " Err: ",
        error.status,
        ".");
    return error;
  }
  owned_device_ptr intermediate_buffer{std::move(get_value(maybe_buffer_ptr))};
  trace_end("IntermediateBufferAlloc");

#if HCL_STREAM_SUPPORT
  auto& collective_stream = my_device_->get_network_collective_stream();

  my_device_->add_wait_events_on_stream({input_address}, collective_stream);

  status = HCL_Reduce_Scatter(
      collective_stream,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Reduce_Scatter(...) failed.", status);

  my_device_->register_producer_on_stream(
      {output_address}, collective_stream, std::move(tensor_cleanup_callback));
#else
  {
    trace_scope ts("ReduceScatterWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }

  status = HCL_Reduce_Scatter(
      nullptr,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Reduce_Scatter(...) failed.", status);

  tensor_cleanup_callback();
#endif

  return {};
}

synapse_error_o hcl_communicator::reduce(
    HCL_Rank dest_rank,
    device_ptr input_address,
    device_ptr output_address,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& tensor_cleanup_callback) {
  HCLStatus status{eHCLSuccess};
  trace_start("IntermediateBufferAlloc");
  synapse_error_v<owned_device_ptr> maybe_buffer_ptr{
      alloc_intermediate_buffer(elem_cnt, data_type, eHCLReduce)};
  if (!ok(maybe_buffer_ptr)) {
    synapse_error error = get_error(maybe_buffer_ptr);
    PT_SYNHELPER_WARN(
        "Intermediate buffer allocation failed. ",
        error.error,
        " Err: ",
        error.status,
        ".");
    return error;
  }
  owned_device_ptr intermediate_buffer{std::move(get_value(maybe_buffer_ptr))};
  trace_end("IntermediateBufferAlloc");

#if HCL_STREAM_SUPPORT
  auto& collective_stream = my_device_->get_network_collective_stream();

  my_device_->add_wait_events_on_stream({input_address}, collective_stream);

  status = HCL_Reduce(
      collective_stream,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      dest_rank,
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Reduce(...) failed.", status);

  my_device_->register_producer_on_stream(
      {output_address}, collective_stream, std::move(tensor_cleanup_callback));
#else
  {
    trace_scope ts("ReduceWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }

  status = HCL_Reduce(
      nullptr,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      dest_rank,
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Reduce_Scatter(...) failed.", status);

  tensor_cleanup_callback();
#endif

  return {};
}

synapse_error_o hcl_communicator::allreduce(
    device_ptr input_address,
    device_ptr output_address,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& tensor_cleanup_callback) {
  HCLStatus status{eHCLSuccess};
  trace_start("IntermediateBufferAlloc");
  synapse_error_v<owned_device_ptr> intermediate_buffer_v{
      alloc_intermediate_buffer(elem_cnt, data_type, eHCLAllReduce)};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          intermediate_buffer_v)) {
    auto error =
        absl::get<synapse_helpers::synapse_error>(intermediate_buffer_v);
    PT_SYNHELPER_WARN(
        "Intermediate buffer allocation failed. ",
        error.error,
        " Err: ",
        error.status,
        ".");
    return error;
  }
  auto intermediate_buffer{
      absl::get<owned_device_ptr>(std::move(intermediate_buffer_v))};
  trace_end("IntermediateBufferAlloc");

#if HCL_STREAM_SUPPORT
  auto& collective_stream = my_device_->get_network_collective_stream();

  my_device_->add_wait_events_on_stream({input_address}, collective_stream);

  status = HCL_Allreduce(
      collective_stream,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Allreduce(...) failed.", status);

  my_device_->register_producer_on_stream(
      {output_address}, collective_stream, std::move(tensor_cleanup_callback));
#else
  {
    trace_scope ts("AllReduceWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }

  status = HCL_Allreduce(
      nullptr,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      intermediate_buffer.get(),
      intermediate_buffer.size(),
      eHCLSum,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Allreduce(...) failed.", status);

  tensor_cleanup_callback();
#endif

  return {};
}

synapse_error_o hcl_communicator::broadcast(
    HCL_Rank root_rank,
    device_ptr address,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& tensor_cleanup_callback) {
  HCLStatus status{eHCLSuccess};
#if HCL_STREAM_SUPPORT
  auto& collective_stream = my_device_->get_network_collective_stream();

  // For root (sending) rank address is input - root does not produce output
  if (my_hcl_rank() == root_rank) {
    my_device_->add_wait_events_on_stream({address}, collective_stream);
  }

  status = HCL_Bcast(
      collective_stream,
      address,
      address,
      elem_cnt,
      data_type,
      root_rank,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Bcast(...) failed.", status);

  // For non root (recieving) rank address is output.
  if (my_hcl_rank() != root_rank) {
    my_device_->register_producer_on_stream(
        {address}, collective_stream, std::move(tensor_cleanup_callback));
  }

#else
  // For root (sending) rank address is input - root does not produce output
  if (my_hcl_rank() == root_rank) {
    my_device_->wait_until_address_ready(address);
  }

  status = HCL_Bcast(
      nullptr,
      address,
      address,
      elem_cnt,
      data_type,
      root_rank,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_Bcast(...) failed.", status);

  tensor_cleanup_callback();
#endif
  return {};
} // namespace synapse_helpers

synapse_error_o hcl_communicator::allgather(
    device_ptr input_address,
    device_ptr output_address,
    size_t elem_cnt,
    synDataType data_type,
    const std::function<void()>& tensor_cleanup_callback) {
  HCLStatus status{eHCLSuccess};

#if HCL_STREAM_SUPPORT
  auto& collective_stream = my_device_->get_network_collective_stream();
  my_device_->add_wait_events_on_stream({input_address}, collective_stream);

  status = HCL_AllGather(
      collective_stream,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_AllGather(...) failed", status);

  my_device_->register_producer_on_stream(
      {output_address}, collective_stream, std::move(tensor_cleanup_callback));
#else
  {
    trace_scope ts("AllGatherWaitForInputData");
    my_device_->wait_until_address_ready(input_address);
  }
  status = HCL_AllGather(
      nullptr,
      input_address,
      output_address,
      elem_cnt,
      data_type,
      hcl_comm(),
      false);
  VERIFY_HCL_STATUS("HCL_AllGather(...) failed", status);

  tensor_cleanup_callback();
#endif
  return {};
}

hcl_communicator::~hcl_communicator() {
  PT_SYNHELPER_DEBUG("~hcl_communicator() entry.");
  HCLStatus hcl_status{eHCLSuccess};
  PT_SYNHELPER_DEBUG("Destroying HCL..");
  hcl_status = HCL_Destroy();
  HABANA_ASSERT(hcl_status == eHCLSuccess);
}

HCL_Rank hcl_communicator::root_hcl_rank() const {
  HABANA_ASSERT(
      root_hcl_rank_ != HCL_RANK_UNASSIGNED &&
      "Call negotiate_root_rank() first to access HCL Root Rank.");
  return root_hcl_rank_;
};

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
        (void*)&input, input_buffer, sizeof(input), [&done]() { done = true; });

    while (!done) {
      std::this_thread::yield();
    }
  }

  allgather(
      input_buffer,
      output_buffer,
      sizeof(Entry) / sizeof(int32_t),
      synDataType::syn_type_int32);

  std::vector<Entry> output(num_workers);

  {
    std::atomic<bool> done{false};

    my_device_->copy_data_to_host(
        output_buffer, output.data(), num_workers * sizeof(Entry), [&done]() {
          done = true;
        });

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

} // namespace synapse_helpers
