/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "ProcessGroupHCCL.hpp"
#include <pybind11/chrono.h>
#include <unistd.h>
#include <future>
#include <map>
#include "hccl.h"
#include "hccl_types.h"
#include "synapse_helpers/env_flags.h"

#include <pybind11/chrono.h>
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pytorch_helpers/habana_helpers/job_thread.h"
#include "pytorch_helpers/habana_helpers/tensor_utils.h"
#include "pytorch_helpers/synapse_helpers/device_context.h"

using namespace synapse_helpers;
namespace c10d {

namespace {

#define HOST_SYNC()                                   \
  {                                                   \
    if (GET_ENV_FLAG_NEW(PT_HPU_USE_PT_STORE_SYNC)) { \
      hostBarrier();                                  \
    }                                                 \
  }

#define NW_STREAM_SYNC()                               \
  {                                                    \
    if (GET_ENV_FLAG_NEW(PT_HPU_USE_NW_STREAM_SYNC)) { \
      nwStreamSync();                                  \
    }                                                  \
  }

std::map<at::ScalarType, hcclDataType_t> hcclDataType = {
    {at::kByte, hcclUint8},
    {at::kChar, hcclChar},
    {at::kDouble, hcclDouble},
    {at::kFloat, hcclFloat},
    {at::kHalf, hcclHalf},
    {at::kInt, hcclInt32},
    {at::kLong, hcclInt64},
    {at::kBFloat16, hcclBfloat16},
};

// HCCL op mapping
std::map<ReduceOp, hcclRedOp_t> hcclOp = {
    {ReduceOp::MIN, hcclMin},
    {ReduceOp::MAX, hcclMax},
    {ReduceOp::SUM, hcclSum},
    {ReduceOp::PRODUCT, hcclProd},
};

hcclRedOp_t getHCCLReduceOp(const ReduceOp reduceOp) {
  try {
    return hcclOp.at(reduceOp);
  } catch (std::out_of_range& e) {
    TORCH_CHECK(false, "Unsupported ReduceOp for HCCL process group");
  }
}

size_t getHCCLSliceSizeMB() {
  static const size_t slice_size = GET_ENV_FLAG_NEW(PT_HCCL_SLICE_SIZE_MB);
  return slice_size * 1024 * 1024;
}

at::ScalarType getInternalScalarType(at::ScalarType type) {
  type = (type == c10::ScalarType::Long) ? c10::ScalarType::Int : type;
  type = (type == c10::ScalarType::Double) ? c10::ScalarType::Float : type;
  return type;
}

hcclDataType_t getHCCLDataType(at::ScalarType type) {
  type = getInternalScalarType(type);
  auto it = hcclDataType.find(type);
  TORCH_CHECK(
      it != hcclDataType.end(),
      "Input tensor data type is not supported for HCCL process group: ",
      type);
  return it->second;
}

bool is_valid_reduction_dtype(hcclDataType_t data_type) {
  if (data_type == hcclBfloat16 || data_type == hcclFloat) {
    return true;
  }
  return false;
}

bool is_valid_broadcast_dtype(hcclDataType_t data_type) {
  if (hcclBfloat16 == data_type || hcclFloat == data_type ||
      hcclInt32 == data_type || hcclUint8 == data_type ||
      hcclHalf == data_type) {
    return true;
  }
  return false;
}
// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
            "All tensor operands to scatter/gather must have the same size");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

class JobThreadHCCL {
 public:
  static std::shared_ptr<habana_helpers::JobThread> getInstance() {
    static std::shared_ptr<habana_helpers::JobThread> job(
        new habana_helpers::JobThread);
    return job;
  }
};

} // namespace

const int64_t ProcessGroupHCCL::kWatchdogThreadSleepMillis = 40000;
void ProcessGroupHCCL::broadcastUniqueHCCLID(hcclUniqueId* hcclID) {
  auto hccl_rank = getRank();
  std::string storeKey = std::to_string(hcclCommCounter_++);
  if (hccl_rank == 0) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(hcclID),
        reinterpret_cast<uint8_t*>(hcclID) + sizeof(hcclUniqueId));
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    TORCH_CHECK(vec.size() == sizeof(hcclUniqueId));
    std::memcpy(hcclID, vec.data(), vec.size());
  }
}

constexpr int64_t kSynchronizeBusyWaitMillis = 1;
// Minumum three keys are required to avoid race condition
constexpr int64_t kNumBarrierKeys = 3;

void ProcessGroupHCCL::hostBarrier() {
  PT_DISTRIBUTED_BEGIN;

  auto hccl_rank = getRank();
  std::string barrier_key = std::string("HOST_BARRIER:");
  std::string storeKey = std::to_string(barrier_cnt_);
  storeKey += barrier_key;
  storeKey += std::to_string(size_);

  auto first_count = store_->add(storeKey, 1);
  TORCH_CHECK(first_count - 1 < size_, "Host barrier Key error");
  auto worker_count = store_->add(storeKey, 0);
  while (worker_count != size_) {
    worker_count = store_->add(storeKey, 0);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  if (hccl_rank == 0) {
    // Delete the previous key
    std::string storeKey_pre = std::to_string(
        barrier_cnt_ == 0 ? (kNumBarrierKeys - 1) : barrier_cnt_ - 1);
    storeKey_pre += barrier_key;
    storeKey_pre += std::to_string(size_);
    store_->deleteKey(storeKey_pre);
  }

  barrier_cnt_ = (barrier_cnt_ + 1) % kNumBarrierKeys;
  PT_DISTRIBUTED_END;
}

void ProcessGroupHCCL::nwStreamSync() {
  PT_DISTRIBUTED_BEGIN;
  std::vector<int> devices;
  for (auto it = hccl_communicator_.begin(); it != hccl_communicator_.end();
       it++) {
    devices.push_back(it->first);
  }

  auto comms = getCommList(devices);
  auto commStreams = getCommStreams(devices);
  for (size_t i = 0; i < comms.size(); i++) {
    synStreamSynchronize(commStreams[i]);
  }

  PT_DISTRIBUTED_END;
}

std::shared_ptr<hcclComm_t> ProcessGroupHCCL::getComm(int deviceId) {
  if (hccl_communicator_.find(deviceId) == hccl_communicator_.end()) {
    hcclUniqueId hccl_id;
    auto hccl_size = getSize();
    auto hccl_rank = getRank();
    if (hccl_rank == 0) {
      hcclResult_t result{hcclGetUniqueId(&hccl_id)};
      TORCH_CHECK(hcclSuccess == result, "Get HCCL UniqueId Error");
    }
    broadcastUniqueHCCLID(&hccl_id);
    hcclComm_t new_comm;
    hcclResult_t result{
        hcclCommInitRank(&new_comm, hccl_size, hccl_id, hccl_rank)};
    TORCH_CHECK(hcclSuccess == result, "Comm Init Rank Error");
    std::lock_guard<std::mutex> lock(mutex_);
    hccl_communicator_[deviceId] = std::make_shared<hcclComm_t>(new_comm);
    auto deviceCtxt =
        std::make_shared<hccl_integration::device_context>(deviceId);
    device_contexts_[deviceId] = deviceCtxt;

    hcclStream_t collective_stream;
    deviceCtxt->acquire_collective_stream(&collective_stream);
    comm_streams_[deviceId] = collective_stream;
  }
  return hccl_communicator_.find(deviceId)->second;
}

std::shared_ptr<hccl_integration::device_context> ProcessGroupHCCL::
    getDeviceCtxt(int deviceId) {
  return device_contexts_.find(deviceId)->second;
}
// TBD: Store not used for now and config done from file
// Initial support added for multiple devices on a single node
// So using rank as the device id.  This will be enhanced further.
ProcessGroupHCCL::ProcessGroupHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& opTimeout)
    : ProcessGroup(rank, size),
      store_(store),
      hcclCommCounter_(0),
      barrier_cnt_(0),
      stop_(false) {}

ProcessGroupHCCL::~ProcessGroupHCCL() {
  destroy();
}

void ProcessGroupHCCL::destroy() {
  hostBarrier();
  std::string barrier_key = std::string("ProcessGroupHCCL::destroy");
  auto worker_count = store_->add(barrier_key, 1);
  if (getRank() == 0) {
    while (worker_count != size_) {
      worker_count = store_->add(barrier_key, 0);
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }

  for (auto element : hccl_communicator_) {
    hcclCommDestroy(*(element.second));
  }
  hccl_communicator_ = {};
}

void ProcessGroupHCCL::abort() {
  destroy();
}

ProcessGroupHCCL::WorkHCCL::WorkHCCL(
    const std::vector<at::Tensor>& outputs,
    const std::vector<int>& devices,
    std::vector<std::shared_ptr<hcclComm_t>>& hccl_comms,
    std::vector<std::shared_ptr<hccl_integration::device_context>>& deviceCtxts)
    : outputs_(outputs),
      devices_(devices),
      hccl_comms_(hccl_comms),
      deviceCtxts_(deviceCtxts),
      workStartTime_(std::chrono::steady_clock::now()),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted(at::IValue(outputs_));
}

ProcessGroupHCCL::WorkHCCL::~WorkHCCL() {}

bool ProcessGroupHCCL::WorkHCCL::isCompleted() {
  return exception() || wait(); // check for the completion of work;
}

bool ProcessGroupHCCL::WorkHCCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }
  // Add support for query from device
  return true;
}

// Same as calling synchronize().
bool ProcessGroupHCCL::WorkHCCL::wait(
    std::chrono::milliseconds timeout /*=kNoTimeout*/) {
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupHCCL::WorkHCCL::synchronize() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    deviceCtxts_[i]->synchronize_output(
        (synapse_helpers::device_ptr)outputs_[i].storage().data_ptr().get());
  }
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupHCCL::WorkHCCL::
    getFuture() {
  return future_;
}

void ProcessGroupHCCL::WorkHCCL::abort() {
  TORCH_CHECK(false, "ProcessGroupHCCL::WorkHCCL::abort not implemented.");
}

c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> ProcessGroupHCCL::initWork(
    std::vector<at::Tensor>& outputs,
    std::vector<int> devices,
    std::vector<std::shared_ptr<hcclComm_t>>& hccl_comms,
    std::vector<std::shared_ptr<hccl_integration::device_context>>&
        deviceCtxts) {
  return c10::make_intrusive<ProcessGroupHCCL::WorkHCCL>(
      outputs, devices, hccl_comms, deviceCtxts);
}

// Get the list of devices from list of tensors
std::vector<int> ProcessGroupHCCL::getDeviceList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<int> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.get_device());
  }
  return res;
}

std::vector<std::shared_ptr<hcclComm_t>> ProcessGroupHCCL::getCommList(
    const std::vector<int>& devices) {
  std::vector<std::shared_ptr<hcclComm_t>> comms(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    comms[i] = getComm(int(devices[i]));
  }
  return comms;
}

hcclStream_t ProcessGroupHCCL::getCommStream(int device) {
  return comm_streams_.find(device)->second;
}

std::vector<hcclStream_t> ProcessGroupHCCL::getCommStreams(
    const std::vector<int>& devices) {
  std::vector<hcclStream_t> hcclStreams(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    hcclStreams[i] = getCommStream(devices[i]);
  }
  return hcclStreams;
}

std::vector<std::shared_ptr<hccl_integration::device_context>> ProcessGroupHCCL::
    getDeviceCtxtList(const std::vector<int>& devices) {
  std::vector<std::shared_ptr<hccl_integration::device_context>> deviceCtxts(
      devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    deviceCtxts[i] = getDeviceCtxt(devices[i]);
  }
  return deviceCtxts;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors_,
    Fn fn,
    int peerRank,
    PreProcess pre,
    PostProcess post) {
  auto tensors = habana_lazy::UpdateViewDistributed(tensors_);

  hcclResult_t hccl_result{hcclSuccess};

  const auto devices = getDeviceList(tensors);
  auto comms = getCommList(devices);
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto commStreams = getCommStreams(devices);
  auto work = initWork(tensors, devices, comms, deviceCtxts);

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto deviceCtxt = deviceCtxts[i];
    void* tensor_address;
    hcclStream_t collective_stream = commStreams[i];
    synapse_helpers::device_ptr tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensors[i].storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);
    deviceCtxt->lock_address(tensors[i].data_ptr(), &tensor_address);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 tensor = tensors[i],
                 tensor_address = tensor_address,
                 comm = comms[i],
                 collective_stream = collective_stream,
                 deviceCtxt = deviceCtxt,
                 tensor_storage_ptr = tensor_storage_ptr,
                 peerRank = peerRank,
                 pr = pr]() mutable {
      hcclResult_t hccl_result =
          fn(tensor, tensor_address, *comm, collective_stream, peerRank);
      TORCH_CHECK(hcclSuccess == hccl_result, "P2P call returned error");
      deviceCtxt->submit_events(collective_stream, tensor_storage_ptr);
      pr->set_value(hccl_result == hcclSuccess);
      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
    } else {
      JobThreadHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(tensor_storage_ptr, std::move(fut));
    }
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peerRank) {
  // Need to replace int by device work streams
  return pointToPoint(
      tensors,
      fn,
      peerRank,
      [](std::vector<int>&) {},
      [](std::vector<int>&) {});
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post) {
  habana_lazy::HbLazyTensor::StepMarker();

  // Handle views
  auto in_view_vec = habana_lazy::UpdateViewDistributed(inputs);
  auto out_view_vec = habana_lazy::UpdateViewDistributed(outputs);

  const auto devices = getDeviceList(in_view_vec);
  auto comms = getCommList(devices);
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto commStreams = getCommStreams(devices);
  auto work = initWork(out_view_vec, devices, comms, deviceCtxts);

  for (size_t i = 0; i < in_view_vec.size(); ++i) {
    auto deviceCtxt = deviceCtxts[i];
    void* input_address;
    void* output_address;
    hcclStream_t collective_stream = commStreams[i];
    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)in_view_vec[i].storage().data_ptr().get();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)out_view_vec[i].storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    deviceCtxt->lock_address(in_view_vec[i].data_ptr(), &input_address);
    deviceCtxt->lock_address(out_view_vec[i].data_ptr(), &output_address);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 input = in_view_vec[i],
                 output = out_view_vec[i],
                 input_address = input_address,
                 output_address = output_address,
                 comm = comms[i],
                 collective_stream = collective_stream,
                 deviceCtxt = deviceCtxt,
                 output_storage_ptr = output_storage_ptr,
                 pr = pr]() mutable {
      hcclResult_t hccl_result =
          fn(input,
             output,
             input_address,
             output_address,
             *comm,
             collective_stream);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
      deviceCtxt->submit_events(collective_stream, output_storage_ptr);
      pr->set_value(hccl_result == hcclSuccess);
      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
    } else {
      JobThreadHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(output_storage_ptr, std::move(fut));
    }
  }

  for (size_t i = 0; i < in_view_vec.size(); ++i) {
    // Update work
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn) {
  // Need to replace int by device work streams
  return collective(
      inputs, outputs, fn, [](std::vector<int>&) {}, [](std::vector<int>&) {});
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  auto work = collective(
      tensors,
      tensors,
      [rootRank = opts.rootRank, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        auto tensor_data_type = getHCCLDataType(input.scalar_type());
        auto numel = input.numel();
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] broadcast with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            numel,
            " data_type :: ",
            tensor_data_type);
        return hcclBroadcast(
            send_buffer,
            recv_buffer,
            numel,
            tensor_data_type,
            rootRank,
            hccl_comm,
            stream);
      });
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  std::vector<at::Tensor> allreduce_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (is_valid_reduction_dtype(data_type)) {
      allreduce_tensors.push_back(tensors[i]);
    } else {
      allreduce_tensors.push_back(tensors[i].to(c10::ScalarType::Float));
    }
  }

  auto work = collective(
      allreduce_tensors,
      allreduce_tensors,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size =
            c10::elementSize(getInternalScalarType(input.scalar_type()));
        size_t chunk_size = getHCCLSliceSizeMB() / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] allreduce with input_address :: ",
              (send_buffer + data_offset),
              " output_address :: ",
              (recv_buffer + data_offset),
              " elem_cnt :: ",
              num_elements_in_current_chunk,
              " data_type :: ",
              getHCCLDataType(input.scalar_type()));

          hccl_result = hcclAllReduce(
              send_buffer + data_offset,
              recv_buffer + data_offset,
              num_elements_in_current_chunk,
              getHCCLDataType(input.scalar_type()),
              getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset += num_elements_in_current_chunk * element_size;
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });

  for (size_t i = 0; i < tensors.size(); i++) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (!is_valid_reduction_dtype(data_type)) {
      tensors[i].copy_(allreduce_tensors[i].to(tensors[i].scalar_type()));
    }
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with HCCL");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  auto work = collective(
      tensors,
      tensors,
      [root = opts.rootRank * tensors.size() + opts.rootTensor,
       reduceOp = opts.reduceOp](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] reduce with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        return hcclReduce(
            send_buffer,
            recv_buffer,
            input.numel(),
            getHCCLDataType(input.scalar_type()),
            getHCCLReduceOp(reduceOp),
            root,
            hccl_comm,
            stream);
      });

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  // Currently only support for alltoall of same size split supported
  std::vector<at::Tensor> inputTensors;
  std::vector<at::Tensor> outputTensors;
  inputTensors.push_back(inputTensor);
  outputTensors.push_back(outputTensor);

  // This is a workaround to support alltoall using hcclSend and hcclRecv
  // because HCCL library does support alltoall yet.
  // hcclSend and hcclRecv works when the ranks are different. In order to
  // ensure that same rank data is present in output, we are first performing
  // copy_data_within_device
  outputTensor.copy_(inputTensor);

  auto work = collective(
      inputTensors,
      outputTensors,
      [numRanks = getSize(), rank = getRank()](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        size_t count = input.numel() / numRanks;
        size_t rank_offset = count *
            c10::elementSize(getInternalScalarType(input.scalar_type()));
        auto type = getHCCLDataType(input.scalar_type());
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] alltoall with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            count,
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        hcclGroupStart();
        hcclResult_t hccl_result{hcclSuccess};
        for (auto r = 0; r < numRanks; r++) {
          if (r < rank) {
            hcclSend(
                reinterpret_cast<const unsigned char*>(send_buffer) +
                    r * rank_offset,
                count,
                type,
                r,
                hccl_comm,
                stream);
            hcclRecv(
                reinterpret_cast<unsigned char*>(recv_buffer) + r * rank_offset,
                count,
                type,
                r,
                hccl_comm,
                stream);
          } else if (r > rank) {
            hcclRecv(
                reinterpret_cast<unsigned char*>(recv_buffer) + r * rank_offset,
                count,
                type,
                r,
                hccl_comm,
                stream);
            hcclSend(
                reinterpret_cast<const unsigned char*>(send_buffer) +
                    r * rank_offset,
                count,
                type,
                r,
                hccl_comm,
                stream);
          }
        }
        hcclGroupEnd();

        return hccl_result;
      });
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  auto outputFlattened =
      flatten_for_scatter_gather(outputTensors, inputTensors, size_);

  auto work = collective(
      inputTensors,
      outputFlattened,
      [&](at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] allgather with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        auto work = hcclAllGather(
            send_buffer,
            recv_buffer,
            input.numel(),
            getHCCLDataType(input.scalar_type()),
            hccl_comm,
            stream);
        return work;
      });
  // Record even for outputFlattened on ncclStream
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    for (size_t j = 0; j < outputTensors[0].size(); ++j) {
      outputTensors[i][j].copy_(outputFlattened[i][j], true);
    }
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error(
      "allgather_base is currently not supported with HCCL");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupHCCL does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupHCCL does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  auto inputFlattened =
      flatten_for_scatter_gather(inputTensors, outputTensors, size_);
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    for (size_t j = 0; j < inputTensors[0].size(); ++j) {
      inputFlattened[i][j].copy_(inputTensors[i][j], true);
    }
  }
  auto work = collective(
      inputFlattened,
      outputTensors,
      [reduceOp = opts.reduceOp](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        // Wait for event on input
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] reduce_scatter with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            output.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        auto work = hcclReduceScatter(
            send_buffer,
            recv_buffer,
            output.numel(),
            getHCCLDataType(input.scalar_type()),
            getHCCLReduceOp(reduceOp),
            hccl_comm,
            stream);

        return work;
      });
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::HbLazyTensor::StepMarker();
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          const void* send_buff,
          hcclComm_t& hccl_comm,
          hcclStream_t stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            send_buff,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        return hcclSend(
            send_buff,
            input.numel(),
            getHCCLDataType(input.scalar_type()),
            peerRank,
            hccl_comm,
            stream);
      },
      dstRank);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::HbLazyTensor::StepMarker();
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& tensor,
          void* recv_buff,
          hcclComm_t& hccl_comm,
          hcclStream_t stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            recv_buff,
            " elem_cnt :: ",
            tensor.numel(),
            " data_type :: ",
            getHCCLDataType(tensor.scalar_type()));
        return hcclRecv(
            recv_buff,
            tensor.numel(),
            getHCCLDataType(tensor.scalar_type()),
            peerRank,
            hccl_comm,
            stream);
      },
      srcRank);
  PT_DISTRIBUTED_END;
  return work;
}

void ProcessGroupHCCL::groupStart() {
  auto func = []() {
    hcclResult_t hccl_result = hcclGroupStart();
    TORCH_CHECK(hcclSuccess == hccl_result, "Group Start returned error");
    return true;
  };
  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
    func();
  } else {
    JobThreadHCCL::getInstance()->addJob(std::move(func));
  }
}

void ProcessGroupHCCL::groupEnd() {
  auto func = []() {
    hcclResult_t hccl_result = hcclGroupEnd();
    TORCH_CHECK(hcclSuccess == hccl_result, "Group End returned error");
    return true;
  };
  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
    func();
  } else {
    JobThreadHCCL::getInstance()->addJob(std::move(func));
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("ProcessGroupHCCL does not support recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCCL::barrier(
    const BarrierOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  std::vector<int> devices;
  for (auto it = hccl_communicator_.begin(); it != hccl_communicator_.end();
       it++) {
    devices.push_back(it->first);
  }

  auto comms = getCommList(devices);
  auto commStreams = getCommStreams(devices);
  PT_DISTRIBUTED_DEBUG(
      "[PYT-DIST] Host and device barrier from rank :: ", getRank())
  hostBarrier();
  for (size_t i = 0; i < comms.size(); i++) {
    hcclBarrier(*comms[i], commStreams[i]);
  }

  std::vector<int> res;
  std::vector<at::Tensor> outputs;
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto work = initWork(outputs, res, comms, deviceCtxts);
  PT_DISTRIBUTED_END;
  return work;
}

} // namespace c10d

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
