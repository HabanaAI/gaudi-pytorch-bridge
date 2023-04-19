/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "ProcessGroupHCCL.hpp"
#include <pybind11/chrono.h>
#include <unistd.h>
#include <future>
#include <map>
#include "backend/synapse_helpers/env_flags.h"
#include "hccl.h"
#include "hccl_types.h"

#include <pybind11/chrono.h>
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/device_context.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "pytorch_helpers/habana_helpers/job_thread.h"

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

typedef enum {
  collectiveAllReduce = 0,
  collectiveReduce = 1,
  collectiveAllGather = 2,
  collectiveReduceScatter = 3,
  collectiveBroadcast = 4,
  collectiveNone
} collectiveKind_t;

hcclRedOp_t getHCCLReduceOp(const ReduceOp reduceOp) {
  try {
    return hcclOp.at(reduceOp);
  } catch (std::out_of_range& e) {
    TORCH_CHECK(false, "Unsupported ReduceOp for HCCL process group");
  }
}

size_t getHCCLSliceSize(collectiveKind_t kind) {
  size_t slice_size = GET_ENV_FLAG_NEW(PT_HCCL_SLICE_SIZE_MB);
  if (slice_size != DEFAULT_HCCL_SLICE_SIZE_MB) {
    // user has set slicing for tuning
    return slice_size * 1024 * 1024;
  }

  // hccl slicing is static for now and will get updated once SIMB is enabled
  switch (kind) {
    case collectiveAllReduce:
    case collectiveReduceScatter:
    case collectiveBroadcast:
      slice_size = 128;
      break;
    case collectiveReduce:
    case collectiveAllGather:
      slice_size = 16;
      break;
  }
  return slice_size * 1024 * 1024;
}

hcclDataType_t getHCCLDataType(at::ScalarType type) {
  type = habana_helpers::getInternalDtype(type);
  auto it = hcclDataType.find(type);
  TORCH_CHECK(
      it != hcclDataType.end(),
      "Input tensor data type is not supported for HCCL process group: ",
      type);
  return it->second;
}

size_t getHCCLDataSize(hcclDataType_t type) {
  const std::map<hcclDataType_t, size_t> type2size = {
      {hcclBfloat16, 2}, {hcclFloat, 4}};

  auto it = type2size.find(type);
  TORCH_CHECK(
      it != type2size.end(),
      "Getting size for given data type is not supported: ",
      type);
  return it->second;
}

void getCountDatatype(
    c10::ScalarType scalar_type,
    int64_t& numel,
    hcclDataType_t& tensor_data_type) {
  switch (scalar_type) {
    case at::kChar:
    case at::kByte:
      numel = (numel * sizeof(char)) / sizeof(uint16_t);
      tensor_data_type = getHCCLDataType(at::kBFloat16);
      break;
    case at::kInt:
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kLong:
      if (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT)) {
        numel = (numel * 2);
      }
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kDouble:
      numel = (numel * sizeof(float)) / sizeof(float);
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kHalf:
      tensor_data_type = getHCCLDataType(at::kBFloat16);
      break;
    default:
      break;
  }
}

void adjustElementcount_int64(
    c10::ScalarType scalar_type,
    std::vector<size_t>& send_lengths,
    std::vector<size_t>& recv_lengths,
    size_t& ele_size) {
  if (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) && scalar_type == at::kLong) {
    for (size_t i = 0; i < send_lengths.size(); i++) {
      send_lengths[i] = send_lengths[i] * 2;
      recv_lengths[i] = recv_lengths[i] * 2;
    }
  } else {
    if (scalar_type == at::kLong) {
      ele_size = ele_size / 2;
    }
  }
}

bool resizeTensor(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList) {
  bool change = false;
  for (int i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    changed[i] = false;
    if ((at::kChar == btensor_type || at::kByte == btensor_type) &&
        tensors[i].numel() % 2 != 0) {
      changed[i] = true;
      sizeList[i] = tensors[i].sizes().vec();
      strideList[i] = tensors[i].strides().vec();
      tensors[i].resize_(tensors[i].numel() + 1);
      change = true;
    }
  }
  return change;
}

void restoreTensorsize(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList,
    c10::intrusive_ptr<Work>& work) {
  for (int i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    if ((at::kChar == btensor_type || at::kByte == btensor_type)) {
      work->wait();
    }
    if (changed[i] == true) {
      tensors[i].resize_(tensors[i].numel() - 1);
      tensors[i].unsafeGetTensorImpl()->set_sizes_and_strides(
          sizeList[i], strideList[i]);
    }
  }
}
bool is_valid_hccl_dtype(hcclDataType_t data_type) {
  if (data_type == hcclBfloat16 || data_type == hcclFloat) {
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
  if (this->emulate_distributed_) {
    return;
  }
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
  if (this->emulate_distributed_) {
    return;
  }
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
    hcclResult_t result{hcclSuccess};
    if (!this->emulate_distributed_) {
      result = hcclCommInitRank(&new_comm, hccl_size, hccl_id, hccl_rank);
    }
    TORCH_CHECK(hcclSuccess == result, "Comm Init Rank Error");
    std::lock_guard<std::mutex> lock(mutex_);
    hccl_communicator_[deviceId] = std::make_shared<hcclComm_t>(new_comm);
    auto deviceCtxt =
        std::make_shared<hccl_integration::device_context>(deviceId);
    device_contexts_[deviceId] = deviceCtxt;

    synStreamHandle collective_stream;
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
      stop_(false) {
  this->emulate_distributed_ = GET_ENV_FLAG_NEW(PT_HPU_EMULATE_DISTRIBUTED);
}

ProcessGroupHCCL::~ProcessGroupHCCL() {
  destroy();
}

void ProcessGroupHCCL::destroy() {
  hostBarrier();
  device_contexts_.clear();
  if (!this->emulate_distributed_) {
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

synStreamHandle ProcessGroupHCCL::getCommStream(int device) {
  return comm_streams_.find(device)->second;
}

std::vector<synStreamHandle> ProcessGroupHCCL::getCommStreams(
    const std::vector<int>& devices) {
  std::vector<synStreamHandle> hcclStreams(devices.size());
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
c10::intrusive_ptr<Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors_,
    Fn fn,
    int peerRank,
    PreProcess pre,
    PostProcess post) {
  auto tensors =
      habana_lazy::HbLazyTensorViews::UpdateViewDistributed(tensors_);

  hcclResult_t hccl_result{hcclSuccess};

  const auto devices = getDeviceList(tensors);
  auto comms = getCommList(devices);
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto commStreams = getCommStreams(devices);
  auto work = initWork(tensors, devices, comms, deviceCtxts);

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto deviceCtxt = deviceCtxts[i];
    synStreamHandle collective_stream = commStreams[i];
    synapse_helpers::device_ptr tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensors[i].storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 tensor = tensors[i],
                 comm = comms[i],
                 collective_stream = collective_stream,
                 deviceCtxt = deviceCtxt,
                 tensor_storage_ptr = tensor_storage_ptr,
                 peerRank = peerRank,
                 pr = pr]() mutable {
      hcclResult_t hccl_result = hcclSuccess;
      auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

      struct ResourceHolder {
        at::Tensor tensor_;
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
      };
      auto resource_holder = std::make_shared<ResourceHolder>();
      resource_holder->tensor_ = tensor;

      void* tensor_address;
      deviceCtxt->lock_address(
          tensor.data_ptr(), &tensor_address, resource_holder->address_lock);

      hccl_result =
          fn(tensor, tensor_address, *comm, collective_stream, peerRank);
      TORCH_CHECK(hcclSuccess == hccl_result, "P2P call returned error");

      recipe_counter.increase();
      deviceCtxt->submit_events(
          collective_stream,
          tensor_storage_ptr,
          [resource_holder, &recipe_counter]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
          });
      pr->set_value(hccl_result == hcclSuccess);
      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
      if (!GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC)) {
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(syn_result == synSuccess, "synStreamSynchronize failed");
      }
    } else {
      JobThreadHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(tensor_storage_ptr, std::move(fut));
      if (!GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC)) {
        deviceCtxt->synchronize_output(tensor_storage_ptr);
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(syn_result == synSuccess, "synStreamSynchronize failed");
      }
    }
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupHCCL::pointToPoint(
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
c10::intrusive_ptr<Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    bool is_allreduce) {
  auto& device = synapse_helpers::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  HABANA_ASSERT(
      context->getCapturing() == false,
      "collective nonSFG is not supported during hpu graph capturing");
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRADIENT_BUCKET_VIEW) && is_allreduce) {
    habana_lazy::HbLazyTensorViews::StepMarkerAllReduce(inputs);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, false /*async*/);
  }

  // Handle views
  auto in_view_vec =
      habana_lazy::HbLazyTensorViews::UpdateViewDistributed(inputs);
  auto out_view_vec =
      habana_lazy::HbLazyTensorViews::UpdateViewDistributed(outputs);

  const auto devices = getDeviceList(in_view_vec);
  auto comms = getCommList(devices);
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto commStreams = getCommStreams(devices);
  auto work = initWork(out_view_vec, devices, comms, deviceCtxts);

  for (size_t i = 0; i < in_view_vec.size(); ++i) {
    if (in_view_vec[i].numel() == 0) {
      // It is a W/A for SW-140597
      // When empty tensor is passed to collective op, its processing is
      // skipped.
      PT_DISTRIBUTED_DEBUG("Empty tensor, skipping collective");
      continue;
    }

    auto deviceCtxt = deviceCtxts[i];
    synStreamHandle collective_stream = commStreams[i];
    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)in_view_vec[i].storage().data_ptr().get();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)out_view_vec[i].storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    if (input_storage_ptr != output_storage_ptr) {
      deviceCtxt->prepare_stream(collective_stream, output_storage_ptr);
    }

    auto input_shallow_copy = at::Tensor{
        in_view_vec[i].unsafeGetTensorImpl()->shallow_copy_and_detach(
            0, false)};
    auto output_shallow_copy = at::Tensor{
        out_view_vec[i].unsafeGetTensorImpl()->shallow_copy_and_detach(
            0, false)};

    auto pr = std::make_shared<std::promise<bool>>();
    std::shared_future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 input = std::move(input_shallow_copy),
                 output = std::move(output_shallow_copy),
                 comm = comms[i],
                 collective_stream = collective_stream,
                 deviceCtxt = deviceCtxt,
                 output_storage_ptr = output_storage_ptr,
                 pr = pr]() mutable {
      hcclResult_t hccl_result = hcclSuccess;
      auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

      struct ResourceHolder {
        std::vector<at::Tensor> tensors_;
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
      };
      auto resource_holder = std::make_shared<ResourceHolder>();
      resource_holder->tensors_ = {input, output};

      void* input_address;
      void* output_address;
      deviceCtxt->lock_address(
          {input.data_ptr(), output.data_ptr()}, resource_holder->address_lock);
      input_address =
          reinterpret_cast<void*>(resource_holder->address_lock->at(0));
      HABANA_ASSERT(input_address != nullptr, "input_address is null");
      output_address =
          reinterpret_cast<void*>(resource_holder->address_lock->at(1));
      HABANA_ASSERT(output_address != nullptr, "output_address is null");

      hccl_result =
          fn(input,
             output,
             input_address,
             output_address,
             *comm,
             collective_stream);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");

      recipe_counter.increase();
      deviceCtxt->submit_events(
          collective_stream,
          output_storage_ptr,
          [resource_holder, &recipe_counter]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
          });
      pr->set_value(hccl_result == hcclSuccess);
      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
      if (!GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC)) {
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(syn_result == synSuccess, "synStreamSynchronize failed");
      }
    } else {
      JobThreadHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(output_storage_ptr, std::move(fut));
      if (!GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC)) {
        deviceCtxt->synchronize_output(output_storage_ptr);
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(syn_result == synSuccess, "synStreamSynchronize failed");
      }
    }
  }

  for (size_t i = 0; i < in_view_vec.size(); ++i) {
    // Update work
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    bool is_allreduce) {
  // Need to replace int by device work streams
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<int>&) {},
      [](std::vector<int>&) {},
      is_allreduce);
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  auto work = collective(
      tensors,
      tensors,
      [rootRank = opts.rootRank, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        const auto scalar_type = input.scalar_type();
        auto tensor_data_type = getHCCLDataType(scalar_type);
        auto num_elems = input.numel();

        getCountDatatype(scalar_type, num_elems, tensor_data_type);

        size_t element_size = getHCCLDataSize(tensor_data_type);
        size_t chunk_size_in_elems =
            getHCCLSliceSize(collectiveBroadcast) / element_size;

        size_t data_offset = 0;
        hcclResult_t hccl_result{hcclSuccess};

        while (num_elems > 0) {
          size_t num_elements_in_current_chunk =
              (static_cast<size_t>(num_elems) > chunk_size_in_elems)
              ? chunk_size_in_elems
              : num_elems;
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] broadcast with input_address :: ",
              static_cast<const void*>(
                  static_cast<const uint8_t*>(send_buffer) + data_offset),
              " output_address :: ",
              static_cast<void*>(
                  static_cast<uint8_t*>(recv_buffer) + data_offset),
              " elem_cnt :: ",
              num_elements_in_current_chunk,
              " data_type :: ",
              tensor_data_type);

          if (!this->emulate_distributed_) {
            hccl_result = hcclBroadcast(
                static_cast<const uint8_t*>(send_buffer) + data_offset,
                static_cast<uint8_t*>(recv_buffer) + data_offset,
                num_elements_in_current_chunk,
                tensor_data_type,
                rootRank,
                hccl_comm,
                stream);
          }

          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elems -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<at::Tensor> allreduce_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (is_valid_hccl_dtype(data_type)) {
      allreduce_tensors.push_back(tensors[i]);
    } else {
      PT_DISTRIBUTED_DEBUG("[PYT-DIST] allreduce tensors converted to float ");
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
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size = c10::elementSize(
            habana_helpers::getInternalDtype(input.scalar_type()));
        size_t chunk_size =
            getHCCLSliceSize(collectiveAllReduce) / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          const void* offseted_send_buffer = reinterpret_cast<const void*>(
              reinterpret_cast<const char*>(send_buffer) + data_offset);
          void* offseted_recv_buffer = reinterpret_cast<void*>(
              reinterpret_cast<char*>(recv_buffer) + data_offset);
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] allreduce with input_address :: ",
              offseted_send_buffer,
              " output_address :: ",
              offseted_recv_buffer,
              " elem_cnt :: ",
              num_elements_in_current_chunk,
              " data_type :: ",
              getHCCLDataType(input.scalar_type()));

          if (!this->emulate_distributed_) {
            hccl_result = hcclAllReduce(
                offseted_send_buffer,
                offseted_recv_buffer,
                num_elements_in_current_chunk,
                getHCCLDataType(input.scalar_type()),
                getHCCLReduceOp(reduceOp),
                hccl_comm,
                stream);
          }
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      },
      true /*is_allreduce*/);

  for (size_t i = 0; i < tensors.size(); i++) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (!is_valid_hccl_dtype(data_type)) {
      work->wait();
      tensors[i].copy_(allreduce_tensors[i].to(tensors[i].scalar_type()));
    }
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /*tensors*/,
    const AllreduceCoalescedOptions& /*opts*/) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with HCCL");
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<at::Tensor> reduction_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (is_valid_hccl_dtype(data_type)) {
      reduction_tensors.push_back(tensors[i]);
    } else {
      PT_DISTRIBUTED_DEBUG("[PYT-DIST] reduction tensors converted to float ");
      reduction_tensors.push_back(tensors[i].to(c10::ScalarType::Float));
    }
  }
  auto work = collective(
      reduction_tensors,
      reduction_tensors,
      [root = opts.rootRank * reduction_tensors.size() + opts.rootTensor,
       reduceOp = opts.reduceOp,
       this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] reduce with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size = c10::elementSize(
            habana_helpers::getInternalDtype(input.scalar_type()));
        size_t chunk_size = getHCCLSliceSize(collectiveReduce) / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          if (!this->emulate_distributed_) {
            hccl_result = hcclReduce(
                reinterpret_cast<const void*>(
                    reinterpret_cast<const char*>(send_buffer) + data_offset),
                reinterpret_cast<void*>(
                    reinterpret_cast<char*>(recv_buffer) + data_offset),
                num_elements_in_current_chunk,
                getHCCLDataType(input.scalar_type()),
                getHCCLReduceOp(reduceOp),
                root,
                hccl_comm,
                stream);
          }
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });

  for (size_t i = 0; i < tensors.size(); i++) {
    auto data_type = getHCCLDataType(tensors[i].scalar_type());
    if (!is_valid_hccl_dtype(data_type)) {
      work->wait();
      tensors[i].copy_(reduction_tensors[i].to(tensors[i].scalar_type()));
    }
  }

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  auto flattenedIn = newLikeFlat(inputTensors);
  auto flattenedOut = newLikeFlat(outputTensors);
  for (const auto i : c10::irange(inputTensors.size())) {
    flattenedIn[i].copy_(inputTensors.at(i));
  }
  std::vector<at::Tensor> inputTensorsFlat;
  std::vector<at::Tensor> outputTensorsFlat;
  inputTensorsFlat.push_back(flattenedIn);
  outputTensorsFlat.push_back(flattenedOut);
  auto work = collective(
      inputTensorsFlat,
      outputTensorsFlat,
      [&](at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        int64_t count = input.numel();
        auto type = getHCCLDataType(input.scalar_type());
        hcclResult_t hccl_result{hcclSuccess};
        const auto scalar_type = input.scalar_type();
        getCountDatatype(scalar_type, count, type);
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] alltoall with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            count,
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));

        hccl_result = hcclAlltoAll(
            send_buffer, recv_buffer, count, type, hccl_comm, stream);
        return hccl_result;
      });

  work->wait();
  for (const auto i : c10::irange(outputTensors.size())) {
    outputTensors.at(i).copy_(
        flattenedOut[i].to(inputTensors.at(i).scalar_type()));
  }

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  c10::intrusive_ptr<Work> work;
  at::Tensor alltoall_out_tensors;
  at::Tensor alltoall_in_tensors;
  auto out_scalar_t = outputTensor.scalar_type();
  auto data_type = getHCCLDataType(outputTensor.scalar_type());
  if (is_valid_hccl_dtype(data_type) || out_scalar_t == at::kInt ||
      out_scalar_t == at::kLong) {
    alltoall_out_tensors = outputTensor;
    alltoall_in_tensors = inputTensor;
  } else {
    PT_DISTRIBUTED_DEBUG("[PYT-DIST] alltoall tensors converted to float ");
    alltoall_out_tensors = outputTensor.to(c10::ScalarType::Float);
    alltoall_in_tensors = inputTensor.to(c10::ScalarType::Float);
  }

  std::vector<at::Tensor> inputTensors;
  std::vector<at::Tensor> outputTensors;
  inputTensors.push_back(alltoall_in_tensors);
  outputTensors.push_back(alltoall_out_tensors);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    work = collective(
        inputTensors,
        outputTensors,
        [numRanks = getSize(), rank = getRank(), this](
            at::Tensor& input,
            at::Tensor& output,
            const void* send_buffer,
            void* recv_buffer,
            hcclComm_t& hccl_comm,
            synStreamHandle stream) {
          int64_t count = input.numel();
          auto type = getHCCLDataType(input.scalar_type());
          HOST_SYNC()
          NW_STREAM_SYNC()
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] alltoall with input_address :: ",
              send_buffer,
              " output_address :: ",
              recv_buffer,
              " elem_cnt :: ",
              count,
              " data_type :: ",
              getHCCLDataType(input.scalar_type()));
          hcclResult_t hccl_result{hcclSuccess};

          const auto scalar_type = input.scalar_type();
          getCountDatatype(scalar_type, count, type);

          if (!this->emulate_distributed_) {
            hccl_result = hcclAlltoAll(
                send_buffer, recv_buffer, count, type, hccl_comm, stream);
          }
          return hccl_result;
        });
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    size_t numRanks = getSize();
    work = collective(
        inputTensors,
        outputTensors,
        [numRanks = getSize(), inputSplitSizes, outputSplitSizes, this](
            at::Tensor& input,
            at::Tensor& output,
            const void* send_buffer,
            void* recv_buffer,
            hcclComm_t& hccl_comm,
            synStreamHandle stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          const auto scalar_type = input.scalar_type();

          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          int64_t count = input.numel();
          auto type = getHCCLDataType(input.scalar_type());
          HOST_SYNC()
          NW_STREAM_SYNC()
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] alltoall with input_address :: ",
              send_buffer,
              " output_address :: ",
              recv_buffer,
              " elem_cnt :: ",
              count,
              " data_type :: ",
              getHCCLDataType(input.scalar_type()));
          size_t ele_size = input.element_size();
          getCountDatatype(scalar_type, count, type);
          adjustElementcount_int64(
              scalar_type, send_lengths, recv_lengths, ele_size);
          hcclGroupStart();
          hcclResult_t hccl_result{hcclSuccess};
          for (const auto r : c10::irange(numRanks)) {
            if (send_lengths[r] != 0) {
              hccl_result = hcclSend(
                  reinterpret_cast<const unsigned char*>(send_buffer) +
                      send_offsets[r] * ele_size,
                  send_lengths[r],
                  type,
                  r,
                  hccl_comm,
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }

            if (recv_lengths[r] != 0) {
              hccl_result = hcclRecv(
                  reinterpret_cast<unsigned char*>(recv_buffer) +
                      recv_offsets[r] * ele_size,
                  recv_lengths[r],
                  type,
                  r,
                  hccl_comm,
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }
          }
          hcclGroupEnd();
          return hccl_result;
        });
  }

  if (!is_valid_hccl_dtype(data_type) && out_scalar_t != at::kInt &&
      out_scalar_t != at::kLong) {
    work->wait();
    outputTensor.copy_(alltoall_out_tensors.to(outputTensor.scalar_type()));
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  bool change = false;
  size_t tensor_size = outputTensors[0].size();
  std::unique_ptr<std::unique_ptr<bool[]>[]> changed(
      new std::unique_ptr<bool[]>[tensor_size]());
  std::vector<std::vector<std::vector<int64_t>>> sizeList(tensor_size);
  std::vector<std::vector<std::vector<int64_t>>> strideList(tensor_size);
  for (int i = 0; i < outputTensors.size(); i++) {
    changed[i] = std::make_unique<bool[]>(outputTensors[i].size());
    sizeList[i].resize(outputTensors[i].size());
    strideList[i].resize(outputTensors[i].size());
    resizeTensor(outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  size_t in_tensor_size = inputTensors.size();
  size_t element_cout = inputTensors[0].numel();
  std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(tensor_size);
  change = resizeTensor(inputTensors, in_changed, in_sizeList, in_strideList);
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
          synStreamHandle stream) {
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
        auto scalar_type = input.scalar_type();
        auto tensor_data_type = getHCCLDataType(scalar_type);
        auto numel = input.numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclAllGather(
              send_buffer,
              recv_buffer,
              numel,
              tensor_data_type,
              hccl_comm,
              stream);
        }
        return hccl_result;
      });
  // Record even for outputFlattened on ncclStream
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    for (size_t j = 0; j < outputTensors[0].size(); ++j) {
      if (!this->emulate_distributed_) {
        outputTensors[i][j].copy_(outputFlattened[i][j], true);
      } else {
        outputTensors[i][j].copy_(inputTensors[i], true);
      }
    }
  }
  if (change) {
    habana_lazy::HbLazyTensor::StepMarker();
  }
  for (int i = 0; i < outputTensors.size(); i++) {
    restoreTensorsize(
        outputTensors[i], changed[i], sizeList[i], strideList[i], work);
  }
  restoreTensorsize(inputTensors, in_changed, in_sizeList, in_strideList, work);

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  auto work = collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] _allgather_base with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        auto scalar_type = input.scalar_type();
        auto tensor_data_type = getHCCLDataType(scalar_type);
        auto numel = input.numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclAllGather(
              send_buffer,
              recv_buffer,
              numel,
              tensor_data_type,
              hccl_comm,
              stream);
        }
        return hccl_result;
      });

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  habana_lazy::NoAccThread no_acc_thread;
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupHCCL::gather: " + msg);
  };

  std::vector<at::Tensor> outputs;
  c10::intrusive_ptr<Work> work;
  if (getRank() == opts.rootRank) {
    TORCH_CHECK(outputTensors.size() == 1, "Requires a single element list");
    TORCH_CHECK(
        outputTensors[0].size() == getSize(),
        "Output list should be same size as process group");
    assertTypeAndSizesMatch(
        invalidArgument,
        outputTensors[0],
        inputTensors[0].options(),
        inputTensors[0].sizes());
    outputs = outputTensors[0];
    if (!this->emulate_distributed_) {
      groupStart();
    }
    int numRanks = getSize();
    for (int r = 0; r < numRanks; r++) {
      if (r == getRank()) {
        outputs[r].copy_(inputTensors[0]);
        std::vector<int> devices;
        for (auto it = hccl_communicator_.begin();
             it != hccl_communicator_.end();
             it++) {
          devices.push_back(it->first);
        }
        std::vector<int> res;
        std::vector<at::Tensor> outputs;
        auto deviceCtxts = getDeviceCtxtList(devices);
        auto comms = getCommList(devices);
        work = initWork(outputs, res, comms, deviceCtxts);
      } else {
        std::vector<at::Tensor> recvTensor;
        recvTensor.push_back(outputs[r]);
        work = recv(recvTensor, r, 0 /*tag*/);
      }
    }
    if (!this->emulate_distributed_) {
      groupEnd();
    }
  } else {
    TORCH_CHECK(outputTensors.size() == 0, "Requires empty output on non-root");
    work = send(inputTensors, opts.rootRank, 0 /*tag*/);
  }
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::scatter(
    std::vector<at::Tensor>& /*outputTensors*/,
    std::vector<std::vector<at::Tensor>>& /*inputTensors*/,
    const ScatterOptions& /*opts*/) {
  throw std::runtime_error("ProcessGroupHCCL does not support scatter");
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
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
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
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
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclReduceScatter(
              send_buffer,
              recv_buffer,
              output.numel(),
              getHCCLDataType(input.scalar_type()),
              getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
        }
        return hccl_result;
      });
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::_reduce_scatter_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const ReduceScatterOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (input_tensor.numel() != output_tensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  auto work = collective(
      inputs,
      outputs,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        HOST_SYNC()
        NW_STREAM_SYNC()
        // Wait for event on input
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] _reduce_scatter_base with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            output.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclReduceScatter(
              send_buffer,
              recv_buffer,
              output.numel(),
              getHCCLDataType(input.scalar_type()),
              getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
        }
        return hccl_result;
      });
  PT_DISTRIBUTED_END;
  return work;
}

// Sending a tensor doesn't have metadata field, hence we can't send the info if
// tensor is dense or permuted. So for first functional step, we'll always
// permute it back to be dnese before sending it. In future it can be optimized
// if we can send metadata too via send mechanism to provide this info.
void ProcessGroupHCCL::permutedSendTensorsToDense(
    std::vector<at::Tensor>& tensors) {
  habana_lazy::NoAccThread no_acc_thread;
  bool has_tensors_to_dense = false;
  for (auto& tensor : tensors) {
    auto self_hb_tensor = habana_lazy::GetHbLazyTensor(tensor);
    auto self_internal_tensor = self_hb_tensor.EvaluateTensorData();
    std::vector<uint8_t> permutation;
    auto hb_weight_impl =
        habana_lazy::GetHbInternalTensorImpl(self_internal_tensor);
    TORCH_CHECK(
        hb_weight_impl != nullptr,
        "Tensor has to have backend impl before send op");
    permutation = hb_weight_impl->GetMemoryPermutation();
    if (!permutation.empty()) {
      PT_DISTRIBUTED_DEBUG(
          "Tensor: ",
          self_hb_tensor.getTensorUniqueId(),
          " has permutation: ",
          VecToString(permutation),
          " transposing it back to be dense");
      tensor = torch::clone(tensor);
      has_tensors_to_dense = true;
    }
  }
  // Creating a Synapse that of memcpy permuted tensors back to dense.
  if (has_tensors_to_dense) {
    std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend>
        lazy_front_end_info =
            std::make_shared<habana_lazy::HbLazyFrontEndInfoToBackend>();
    lazy_front_end_info->set_is_hccl_send_mark_step(true);
    habana_lazy::HbLazyTensor::StepMarker({}, lazy_front_end_info);
  }
}

// When recieving a tensor we make sure during send it's dense.
// So once we recive a tensor, we clear it's permutation info.
void ProcessGroupHCCL::clearPermutesFromRecvTensors(
    std::vector<at::Tensor>& tensors) {
  for (auto& tensor : tensors) {
    auto self_hb_tensor = habana_lazy::GetHbLazyTensor(tensor);
    auto self_internal_tensor = self_hb_tensor.EvaluateTensorData();
    auto hb_weight_impl =
        habana_lazy::GetHbInternalTensorImpl(self_internal_tensor);
    PT_DISTRIBUTED_DEBUG(
        "recieved tensor: ",
        self_hb_tensor.getTensorUniqueId(),
        " Clearing its permutation");
    hb_weight_impl->SetMemoryPermutation({});
  }
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  habana_lazy::HbLazyTensor::StepMarker();
  permutedSendTensorsToDense(tensors);
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          const void* send_buff,
          hcclComm_t& hccl_comm,
          synStreamHandle stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            send_buff,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            getHCCLDataType(input.scalar_type()));
        auto scalar_type = input.scalar_type();
        auto tensor_data_type = getHCCLDataType(scalar_type);
        auto numel = input.numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclSend(
              send_buff, numel, tensor_data_type, peerRank, hccl_comm, stream);
        }
        return hccl_result;
      },
      dstRank);
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  habana_lazy::HbLazyTensor::StepMarker();
  clearPermutesFromRecvTensors(tensors);
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& tensor,
          void* recv_buff,
          hcclComm_t& hccl_comm,
          synStreamHandle stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            recv_buff,
            " elem_cnt :: ",
            tensor.numel(),
            " data_type :: ",
            getHCCLDataType(tensor.scalar_type()));
        auto scalar_type = tensor.scalar_type();
        auto tensor_data_type = getHCCLDataType(scalar_type);
        auto numel = tensor.numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclRecv(
              recv_buff, numel, tensor_data_type, peerRank, hccl_comm, stream);
        }
        return hccl_result;
      },
      srcRank);
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
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

c10::intrusive_ptr<Work> ProcessGroupHCCL::recvAnysource(
    std::vector<at::Tensor>& /*tensors*/,
    int /*tag*/) {
  throw std::runtime_error("ProcessGroupHCCL does not support recv");
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::barrier(const BarrierOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<int> devices;
  for (auto it = hccl_communicator_.begin(); it != hccl_communicator_.end();
       it++) {
    devices.push_back(it->first);
  }

  auto comms = getCommList(devices);
  auto commStreams = getCommStreams(devices);
  PT_DISTRIBUTED_DEBUG(
      "[PYT-DIST] Host and device barrier from rank :: ", getRank())
  while (JobThreadHCCL::getInstance()->jobCounter() > 0) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }
  hostBarrier();
  if (!this->emulate_distributed_) {
    for (size_t i = 0; i < comms.size(); i++) {
      hcclBarrier(*comms[i], commStreams[i]);
    }
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
