/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "ProcessGroupHCCL.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <unistd.h>
#include <future>
#include <map>
#include <memory>
#include <mutex>

#include <c10/util/intrusive_ptr.h>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/collective_utils.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/device_context.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "process_group_registry.hpp"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_lazy_hccl.hpp"
#include "pytorch_helpers/habana_helpers/job_thread.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"

using namespace synapse_helpers;
namespace c10d {

namespace {

class JobThreadHCCL {
 public:
  static std::shared_ptr<habana_helpers::JobThread> getInstance() {
    static std::shared_ptr<habana_helpers::JobThread> job(
        new habana_helpers::JobThread);
    return job;
  }
};

} // namespace

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

void ProcessGroupHCCL::initializeCommForDevice(int deviceId) {
  if (hccl_communicator_.find(deviceId) == hccl_communicator_.end()) {
    hcclUniqueId hccl_id = {{0}, 0};
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
}

void ProcessGroupHCCL::initComms() {
  std::vector<int> devices;
  for (const auto& it : hccl_communicator_) {
    devices.push_back(it.first);
  }
  auto deviceCtxts = getDeviceCtxtList(devices);
  for (int deviceId : devices) {
    initializeCommForDevice(deviceId);
  }
}

std::shared_ptr<hcclComm_t> ProcessGroupHCCL::getComm(int deviceId) {
  initializeCommForDevice(deviceId);
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
    std::string group_name)
    : ProcessGroupHcclBase(store, rank, size, group_name), hcclCommCounter_(0) {
  PT_DISTRIBUTED_DEBUG(
      "Created ProcessGroupHCCL name:",
      group_name_,
      ", size:",
      size,
      ", rank:",
      rank);
}

ProcessGroupHCCL::~ProcessGroupHCCL() {
  PT_DISTRIBUTED_DEBUG(
      "~ProcessGroupHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);
  habana_helpers::AutoNoGIL gil_release;
  destroy();
}

void ProcessGroupHCCL::destroy() {
  if (is_destroyed_)
    return;
  PT_DISTRIBUTED_DEBUG(
      "Destroy ProcessGroupHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);
  hostBarrier();
  device_contexts_.clear();

  if (!this->emulate_distributed_) {
    for (auto element : hccl_communicator_) {
      hcclCommDestroy(*(element.second));
    }
  }

  hccl_communicator_ = {};
  destroyHandshake();
  is_destroyed_ = true;
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
bool ProcessGroupHCCL::WorkHCCL::wait(std::chrono::milliseconds timeout
                                      [[maybe_unused]]) {
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupHCCL::WorkHCCL::synchronize() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    deviceCtxts_[i]->synchronize_output(
        (synapse_helpers::device_ptr)outputs_[i].storage().data_ptr().get(),
        (c10::hpu::getCurrentHPUStream()).stream());
  }
  outputs_.clear();
  deviceCtxts_.clear();
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

c10::intrusive_ptr<Work> ProcessGroupHCCL::initWork(
    std::vector<at::Tensor>& outputs) {
  std::vector<int> devices;
  for (auto it = hccl_communicator_.begin(); it != hccl_communicator_.end();
       it++) {
    devices.push_back(it->first);
  }
  std::vector<int> res;
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto comms = getCommList(devices);
  return initWork(outputs, res, comms, deviceCtxts);
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

c10::intrusive_ptr<Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors_,
    PointToPointFn fn,
    int peerRank) {
  auto tensors =
      habana_lazy::HbLazyTensorViews::UpdateViewDistributed(tensors_);

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

void ProcessGroupHCCL::groupStart() {
  initComms();

  auto pr = std::make_shared<std::promise<bool>>();
  std::future<bool> fut = pr->get_future();
  auto fn = hcclGroupStart;
  auto func = [fn = fn, pr = pr]() mutable {
    hcclResult_t hccl_result = hcclSuccess;
    hccl_result = hcclGroupStart();
    TORCH_CHECK(
        hcclSuccess == hccl_result, "hcclGroupStart call returned error");
    pr->set_value(hccl_result == hcclSuccess);
    return true;
  };

  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
    func();
  } else {
    JobThreadHCCL::getInstance()->addJob(std::move(func));
  }
}

void ProcessGroupHCCL::groupEnd() {
  auto pr = std::make_shared<std::promise<bool>>();

  std::future<bool> fut = pr->get_future();
  auto fn = hcclGroupStart;
  auto func = [fn = fn, pr = pr]() mutable {
    hcclResult_t hccl_result = hcclSuccess;
    hccl_result = hcclGroupEnd();
    TORCH_CHECK(hcclSuccess == hccl_result, "hcclGroupEnd call returned error");
    pr->set_value(hccl_result == hcclSuccess);
    return true;
  };

  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
    func();
  } else {
    JobThreadHCCL::getInstance()->addJob(std::move(func));
  }
}

void ProcessGroupHCCL::waitForJobCompletion() {
  while (JobThreadHCCL::getInstance()->jobCounter() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

c10::intrusive_ptr<Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    CollectiveFn fn,
    bool is_allreduce) {

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  HABANA_ASSERT(
      context->getCapturing() == false,
      "collective nonSFG is not supported during hpu graph capturing");
  if (is_allreduce) {
    habana_lazy::HbLazyTensorViews::StepMarkerAllReduce(inputs);
  } else {
    PT_IRGRAPH_DEBUG("step marker due to ProcessGroupHCCL::collective");
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

    std::vector<synapse_helpers::shared_event> event_lists = {};
    event_lists = deviceCtxt->prepare_stream_and_get_events(
        collective_stream, input_storage_ptr);
    if (input_storage_ptr != output_storage_ptr) {
      std::vector<synapse_helpers::shared_event> out_event_lists =
          deviceCtxt->prepare_stream_and_get_events(
              collective_stream, output_storage_ptr);
      event_lists.insert(
          event_lists.end(), out_event_lists.begin(), out_event_lists.end());
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
                 event_lists = event_lists,
                 output_storage_ptr = output_storage_ptr,
                 pr = pr]() mutable {
      hcclResult_t hccl_result = hcclSuccess;
      auto& stream = deviceCtxt->get_stream_fromhandle(collective_stream);
      for (auto& event : event_lists) {
        event->stream_wait_event(stream);
      }

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

c10::intrusive_ptr<Work> ProcessGroupHCCL::barrier(const BarrierOptions& opts
                                                   [[maybe_unused]]) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<int> devices;
  for (auto it = hccl_communicator_.begin(); it != hccl_communicator_.end();
       it++) {
    devices.push_back(it->first);
  }

  auto comms = getCommList(devices);
  PT_DISTRIBUTED_DEBUG(
      "[PYT-DIST] Host and device barrier from rank :: ", getRank());
  while (JobThreadHCCL::getInstance()->jobCounter() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  hostBarrier();
  std::vector<int> res;
  std::vector<at::Tensor> outputs;
  auto deviceCtxts = getDeviceCtxtList(devices);
  auto work = initWork(outputs, res, comms, deviceCtxts);
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
    PT_IRGRAPH_DEBUG(
        "step marker due to ProcessGroupHCCL::permutedSendTensorsToDense");
    habana_lazy::HbLazyTensor::StepMarker({}, lazy_front_end_info);
  }
}

// When recieving a tensor we make sure during send it's dense.
// So once we recive a tensor, we clear it's permutation info.
void ProcessGroupHCCL::clearPermutesFromRecvTensors(
    std::vector<at::Tensor>& tensors) {
  for (auto& tensor : tensors) {
    bool is_non_contiguous_view = false;
    auto self_hb_tensor = habana_lazy::GetHbLazyTensor(tensor);
    if (!self_hb_tensor.isStorageAttached()) {
      auto& stride_params_opt = self_hb_tensor.getDataPtr()->stride_params;
      if (stride_params_opt.has_value()) {
        if (tensor.is_contiguous()) {
          auto base = habana_lazy::HbLazyTensorViews::get_recent_base_tensor(
              stride_params_opt.value().base);
          TORCH_CHECK(base.storage(), "base tensor should have valid storage");
          self_hb_tensor = habana_lazy::GetHbLazyTensor(base);
        } else {
          is_non_contiguous_view = true;
        }
      } else {
        TORCH_CHECK(
            0, "Neither storage attached to input tensor, not its view.")
      }
    }
    auto self_internal_tensor = self_hb_tensor.EvaluateTensorData();
    auto hb_weight_impl =
        habana_lazy::GetHbInternalTensorImpl(self_internal_tensor);
    if (is_non_contiguous_view) {
      HABANA_ASSERT(
          hb_weight_impl->GetMemoryPermutation().empty(),
          "Noncontiguous view output of Recv should not have permutation");
      PT_DISTRIBUTED_DEBUG(
          "recieved tensor: ",
          self_hb_tensor.getTensorUniqueId(),
          " is non contiguous view, skipping clearing its permutation");
      continue;
    } else {
      PT_DISTRIBUTED_DEBUG(
          "recieved tensor: ",
          self_hb_tensor.getTensorUniqueId(),
          " Clearing its permutation");
      hb_weight_impl->SetMemoryPermutation({});
    }
  }
}
} // namespace c10d

namespace py = pybind11;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::object backend =
      py::module_::import("torch.distributed").attr("_Backend");

  intrusive_ptr_class_<::c10d::ProcessGroupHCCL> processGroupHccl(
      module, "ProcessGroupHCCL", backend);

  processGroupHccl.def(py::init(
      &c10d::ProcessGroupHCCLRegistry<c10d::ProcessGroupHCCL>::create));
};
