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
#include "process_group_lazy_hccl.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include "backend/helpers/collective_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "process_group_registry.hpp"
#include "pytorch_helpers/habana_helpers/python_utils.h"

namespace c10d {

namespace {

#define HOST_SYNC()                                   \
  {                                                   \
    if (GET_ENV_FLAG_NEW(PT_HPU_USE_PT_STORE_SYNC)) { \
      hostBarrier();                                  \
    }                                                 \
  }

bool resizeOddTensor(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList) {
  bool change = false;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    changed[i] = false;
    if ((at::kChar == btensor_type || at::kByte == btensor_type ||
         at::kBool == btensor_type || at::kFloat8_e5m2 == btensor_type ||
         at::kFloat8_e4m3fn == btensor_type) &&
        (tensors[i].numel() % 2 != 0)) {
      changed[i] = true;
      sizeList[i] = tensors[i].sizes().vec();
      strideList[i] = tensors[i].strides().vec();
      tensors[i] = tensors[i].resize_(tensors[i].numel() + 1);
      change = true;
    }
  }
  return change;
}

bool resizeTensor(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList,
    int extra_num_elems) {
  if (extra_num_elems == 1) {
    return resizeOddTensor(tensors, changed, sizeList, strideList);
  }

  // Below for the case: extra_num_elems > 1
  bool change = false;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    changed[i] = false;
    if (at::kChar == btensor_type || at::kByte == btensor_type ||
        at::kBool == btensor_type || at::kFloat8_e5m2 == btensor_type ||
        at::kFloat8_e4m3fn == btensor_type) {
      changed[i] = true;
      sizeList[i] = tensors[i].sizes().vec();
      strideList[i] = tensors[i].strides().vec();
      tensors[i] = tensors[i].resize_(tensors[i].numel() + extra_num_elems);
      change = true;
    }
  }
  return change;
}

void restoreOddTensorsize(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList) {
  for (size_t i = 0; i < tensors.size(); i++) {
    if (changed[i] == true) {
      tensors[i] = tensors[i].resize_(sizeList[i]);
      tensors[i].unsafeGetTensorImpl()->set_sizes_and_strides(
          sizeList[i], strideList[i]);
    }
  }
}

void restoreTensorsize(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList,
    int extra_num_elems,
    int ori_input_size = -1) {
  if (extra_num_elems == 1) {
    return restoreOddTensorsize(tensors, changed, sizeList, strideList);
  }

  // Below for the case: extra_num_elems > 1
  for (size_t i = 0; i < tensors.size(); i++) {
    if (changed[i] == true) {
      // Here restore logic is like below, typically for output tensor:
      //
      // Considering we have input tensor with shape [63] on two ranks.
      // Originally output tensor should have shape [126]. Hovever, after
      // resize, each input tensor has shape [64] and output tensor shape
      // [128]. So for output tensor, there are two extra elements added,
      // specifically the positions are 63 and 127.
      //
      // For restore stage, simply resizing is not enough since the extra
      // element is at pos 63. Instead separate copy is used below to recover
      // output correctly. To do this, a temporary buffer is required,
      // see `resized_out` in below code. Firstly, copy elements from
      // ori_out[0, 1, ..., 62] to resized_out[0, 1,..., 62]. And then copy
      // elements from ori_out[64, 65, ..., 126] to
      // resized_out[63, 64, ..., 125]. After all these done, copy elements
      // from resized_out back to original output tensor. Now, output tensor
      // should have all updated elements at index 0~125 and is safe to do
      // resize.
      TORCH_CHECK(
          ori_input_size != -1,
          "original input tensor size should be provided.");

      auto resized_out = at::empty_like(tensors[i], tensors[i].scalar_type());
      TORCH_CHECK(tensors[i].sizes().size() == 1, "only support 1D tensor");
      auto resized_input_size = ori_input_size + 1;
      for (int n = 0; n < extra_num_elems; ++n) {
        auto dst = at::as_strided(
            resized_out,
            {ori_input_size},
            resized_out.strides(),
            n * ori_input_size);
        auto src = at::as_strided(
            tensors[i],
            {ori_input_size},
            tensors[i].strides(),
            n * resized_input_size);
        dst.copy_(src);
      }
      tensors[i].copy_(resized_out);

      // need step marker here to ensure later resize_ has no conflict with
      // above two as_strided operations.
      PT_IRGRAPH_DEBUG("step marker due to restore tensor size");
      habana_lazy::HbLazyTensor::StepMarker();

      tensors[i] = tensors[i].resize_(sizeList[i]);
      tensors[i].unsafeGetTensorImpl()->set_sizes_and_strides(
          sizeList[i], strideList[i]);
    }
  }
}
} // namespace

ProcessGroupLazyHCCL::ProcessGroupLazyHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::string group_name)
    : Backend(rank, size),
      store_(store),
      barrier_cnt_(0),
      group_name_(group_name) {
  PT_DISTRIBUTED_DEBUG(
      "Created ProcessGroupLazyHCCL name:",
      group_name_,
      ", size:",
      size,
      ", rank:",
      rank);
  emulate_distributed_ = GET_ENV_FLAG_NEW(PT_HPU_EMULATE_DISTRIBUTED);
  comm_ = habana::HcclCommunicator::Create(
      rank,
      size,
      [store, rank](hcclUniqueId* hcclID) { // Use hcclID as store key?
        if (rank == 0) {
          auto vec = std::vector<uint8_t>(
              reinterpret_cast<uint8_t*>(hcclID),
              reinterpret_cast<uint8_t*>(hcclID) + sizeof(hcclUniqueId));
          store->set("HCCL_GROUP_UNIQUE_ID", vec);
        } else {
          auto vec = store->get("HCCL_GROUP_UNIQUE_ID");
          TORCH_CHECK(vec.size() == sizeof(hcclUniqueId));
          std::memcpy(hcclID, vec.data(), vec.size());
        }
      });
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {

  for (size_t index = 0; index < inputs.size(); ++index) {
    auto data_type = inputs.at(index).scalar_type();
    bool cast_tensor =
        !(data_type == c10::ScalarType::Float ||
          data_type == c10::ScalarType::BFloat16);
    at::Tensor t_updated;
    if (!cast_tensor) {
      habana_lazy::reduce_scatter_hpu_lazy_out(
          inputs.at(index),
          (uint8_t)opts.reduceOp,
          comm_->GetId(),
          outputs.at(index));
    } else {
      t_updated = inputs.at(index).to(c10::ScalarType::Float);
      auto output =
          at::empty_like(outputs.at(index), c10::ScalarType::Float);
      habana_lazy::reduce_scatter_hpu_lazy_out(
          t_updated, (uint8_t)opts.reduceOp, comm_->GetId(), output);
      outputs.at(index).copy_(output.to(data_type));
    }
  }

  auto work =
      c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputs);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
}

ProcessGroupLazyHCCL::~ProcessGroupLazyHCCL() {
  PT_DISTRIBUTED_DEBUG(
      "~ProcessGroupLazyHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);
  habana_helpers::AutoNoGIL gil_release;
  destroy();
  destroyHandshake();
};

void ProcessGroupLazyHCCL::destroy() {
  PT_DISTRIBUTED_DEBUG(
      "Destroy ProcessGroupLazyHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);

  hostBarrier();

  if (comm_) {
    habana_helpers::AutoNoGIL gil_release;
    comm_->flush_stream();
    comm_.reset();
  }
}

ProcessGroupLazyHCCL::WorkLazy::WorkLazy(const std::vector<at::Tensor>& outputs)
    : outputs_(outputs),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted(at::IValue(outputs_));
}

ProcessGroupLazyHCCL::WorkLazy::~WorkLazy() {}

bool ProcessGroupLazyHCCL::WorkLazy::isCompleted() {
  return true;
}

bool ProcessGroupLazyHCCL::WorkLazy::isSuccess() const {
  return true;
}

bool ProcessGroupLazyHCCL::WorkLazy::wait(std::chrono::milliseconds timeout
                                          [[maybe_unused]]) {
  PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::WorkLazy::wait");
  habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  return true;
}

void ProcessGroupLazyHCCL::WorkLazy::abort() {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
}

void ProcessGroupLazyHCCL::WorkLazy::synchronize() {
  PT_IRGRAPH_DEBUG(
      "step marker due to ProcessGroupLazyHCCL::WorkLazy::synchronize");
  habana_lazy::HbLazyTensor::StepMarker();
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupLazyHCCL::WorkLazy::
    getFuture() {
  return future_;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeOddTensor(tensors, changed, sizeList, strideList);
  HOST_SYNC()
  for (auto& t : tensors) {
    habana_lazy::broadcast_hpu_lazy_(t, opts.rootRank, comm_->GetId());
  }
  habana_lazy::HbLazyTensor::StepMarker();
  restoreOddTensorsize(tensors, changed, sizeList, strideList);
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  HOST_SYNC()
  for (auto& t : tensors) {
    auto data_type = t.scalar_type();
    bool cast_tensor =
        !(data_type == c10::ScalarType::Float ||
          data_type == c10::ScalarType::BFloat16);
    at::Tensor t_updated;
    if (!cast_tensor) {
      t_updated = t;
    } else {
      t_updated = t.to(c10::ScalarType::Float);
    }
    habana_lazy::allreduce_hpu_lazy_(
        t_updated, (uint8_t)opts.reduceOp, comm_->GetId());
    if (cast_tensor) {
      t.copy_(t_updated.to(data_type));
    }
  }
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);

  if (coalescing_state_) {
    coalesed_works_->append(work);
  }

  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  return allreduce(tensors, opts);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  for (auto& t : tensors) {
    auto data_type = t.scalar_type();
    bool cast_tensor =
        !(data_type == c10::ScalarType::Float ||
          data_type == c10::ScalarType::BFloat16);
    at::Tensor t_updated;
    if (!cast_tensor) {
      t_updated = t;
    } else {
      t_updated = t.to(c10::ScalarType::Float);
    }
    habana_lazy::reduce_hpu_lazy_(
        t_updated, opts.rootRank, (uint8_t)opts.reduceOp, comm_->GetId());
    if (cast_tensor) {
      t.copy_(t_updated.to(data_type));
    }
  }
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const AllgatherOptions& opts) {
  bool change = false;
  size_t tensor_size = outputTensors[0].size();
  std::unique_ptr<std::unique_ptr<bool[]>[]> changed(
      new std::unique_ptr<bool[]>[tensor_size]());
  std::vector<std::vector<std::vector<int64_t>>> sizeList(tensor_size);
  std::vector<std::vector<std::vector<int64_t>>> strideList(tensor_size);
  for (size_t i = 0; i < outputTensors.size(); i++) {
    changed[i] = std::make_unique<bool[]>(outputTensors[i].size());
    sizeList[i].resize(outputTensors[i].size());
    strideList[i].resize(outputTensors[i].size());
    resizeOddTensor(outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(tensor_size);
  change =
      resizeOddTensor(inputTensors, in_changed, in_sizeList, in_strideList);
  auto output_flattened = habana_helpers::flatten_for_scatter_gather(
      outputTensors, inputTensors, size_);
  HOST_SYNC()
  for (size_t index = 0; index < output_flattened.size(); ++index) {
    habana_lazy::allgather_hpu_lazy_out(
        inputTensors.at(index), comm_->GetId(), output_flattened.at(index));
  }

  // Record even for outputFlattened on ncclStream
  std::vector<at::Tensor> output_list_flat;
  if (!outputTensors.empty()) {
    output_list_flat.reserve(outputTensors.size() * outputTensors.at(0).size());
  }

  for (size_t i = 0; i < outputTensors.size(); ++i) {
    for (size_t j = 0; j < outputTensors.at(i).size(); ++j) {
      outputTensors[i][j].copy_(output_flattened[i][j], true);
      output_list_flat.push_back(outputTensors[i][j]);
    }
  }
  if (change) {
    PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::allgather");
    habana_lazy::HbLazyTensor::StepMarker();
  }
  for (size_t i = 0; i < outputTensors.size(); i++) {
    restoreOddTensorsize(
        outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  auto work =
      c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(output_list_flat);

  if (coalescing_state_) {
    coalesed_works_->append(work);
  }

  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    [[maybe_unused]] const AllgatherOptions& opts) {
  TORCH_CHECK(
      inputBuffer.dtype() == outputBuffer.dtype(), "buffer types don't match");
  TORCH_CHECK(
      inputBuffer.numel() * size_ == outputBuffer.numel(),
      "incompatible buffer sizes");

  auto ori_input_size = inputBuffer.numel();

  // size compatible with resize method api and with singularity of allgather
  // base
  auto inputs = std::vector<at::Tensor>{inputBuffer};
  auto outputs = std::vector<at::Tensor>{outputBuffer};

  auto tensor_size{1};
  std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(tensor_size);

  std::unique_ptr<bool[]> out_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> out_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> out_strideList(tensor_size);

  // Case 1 with even world size:
  //    rank 0: input [63] -> resize to [64]
  //    rank 1: input [63] -> resize to [64]
  //
  //    rank 0: output [126] -> no resize happen
  //    rank 1: output [126] -> no resize happen
  // actually require resize output tensor to [128]
  //
  // Case 2 with odd world size:
  //    rank 0: input [63] -> resize to [64]
  //    rank 1: input [63] -> resize to [64]
  //    rank 2: input [63] -> resize to [64]
  //
  //    rank 0: output [189] -> resize to [190]
  //    rank 1: output [189] -> resize to [190]
  //    rank 2: output [189] -> resize to [190]
  // resize happens, but got wrong size, should be [192] rather than [190]
  bool changed =
      resizeOddTensor(inputs, in_changed, in_sizeList, in_strideList);
  // if no resize on input, keep current logic
  int out_resize_extra_num_elems = changed ? size_ : 1;
  changed |= resizeTensor(
      outputs,
      out_changed,
      out_sizeList,
      out_strideList,
      out_resize_extra_num_elems);

  HOST_SYNC()
  PT_DISTRIBUTED_DEBUG(
      "Calling ProcessGroupLazyHCCL::_allgather_base :: ",
      " in sizes :: ",
      inputBuffer.sizes(),
      " out sizes :: ",
      outputBuffer.sizes());
  habana_lazy::allgather_hpu_lazy_out(
      inputBuffer, comm_->GetId(), outputBuffer);

  if (changed) {
    PT_IRGRAPH_DEBUG(
        "step marker due to ProcessGroupLazyHCCL::_allgather_base");
    habana_lazy::HbLazyTensor::StepMarker();
  }

  restoreOddTensorsize(inputs, in_changed, in_sizeList, in_strideList);
  restoreTensorsize(
      outputs,
      out_changed,
      out_sizeList,
      out_strideList,
      out_resize_extra_num_elems,
      ori_input_size);
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputs);

  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

static constexpr int CoalActive = 0x01;

void ProcessGroupLazyHCCL::groupStart() {
  hcclResult_t hccl_result = hcclSuccess;
  hccl_result = hcclGroupStart();
  TORCH_CHECK(hcclSuccess == hccl_result, "hcclGroupStart call returned error");
}

void ProcessGroupLazyHCCL::groupEnd() {
  hcclResult_t hccl_result = hcclGroupEnd();
  TORCH_CHECK(hcclSuccess == hccl_result, "hcclGroupEnd call returned error");
}

ProcessGroupLazyHCCL::CoalescedWorkHCCL::~CoalescedWorkHCCL() = default;

// Method to append a new Work object to works_
void c10d::ProcessGroupLazyHCCL::CoalescedWorkHCCL::append(
    const c10::intrusive_ptr<Work>& work) {
  works_.push_back(work);
}

// Method to clear the works_ vector
void c10d::ProcessGroupLazyHCCL::CoalescedWorkHCCL::clear() {
  works_.clear();
}

// Same as calling synchronize().
bool c10d::ProcessGroupLazyHCCL::CoalescedWorkHCCL::wait(
    std::chrono::milliseconds timeout [[maybe_unused]]) {
  for (auto& w : works_) {
    w->wait(timeout);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupLazyHCCL::startCoalescing() {
  TORCH_CHECK(
      habana::HPUDeviceContext::is_device_acquired(),
      "HPU Device not initialized! startCoalescing cannot be done without device init!")

  TORCH_CHECK(
      coalescing_state_ == 0,
      "Coalescing is already in progress. Have you invoked startCoalescing again without endCoalescing. BTW nested coalesing is not supported.");

  coalesed_works_ =
      c10::make_intrusive<ProcessGroupLazyHCCL::CoalescedWorkHCCL>();
  coalescing_state_ |= CoalActive;
  coalesed_works_->clear();
  groupStart();
}

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::endCoalescing() {
  TORCH_CHECK(
      coalescing_state_ != 0, "endCoalescing invoked without startCoalescing");

  TORCH_CHECK(
      coalesed_works_ != nullptr, "Error: coalesed_works_ is not initied")

  coalescing_state_ = 0;
  groupEnd();
  return coalesed_works_;
}

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  return allgather(outputTensorLists, inputTensors, opts);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  bool change = false;
  size_t tensor_size = outputTensors.empty() ? 0 : outputTensors[0].size();
  std::unique_ptr<std::unique_ptr<bool[]>[]> changed(
      new std::unique_ptr<bool[]>[tensor_size]());
  std::vector<std::vector<std::vector<int64_t>>> sizeList(tensor_size);
  std::vector<std::vector<std::vector<int64_t>>> strideList(tensor_size);
  for (size_t i = 0; i < outputTensors.size(); i++) {
    changed[i] = std::make_unique<bool[]>(outputTensors[i].size());
    sizeList[i].resize(outputTensors[i].size());
    strideList[i].resize(outputTensors[i].size());
    change = resizeOddTensor(
        outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  size_t in_tensor_size = inputTensors.size();
  std::unique_ptr<bool[]> in_changed(new bool[in_tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(in_tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(in_tensor_size);
  resizeOddTensor(inputTensors, in_changed, in_sizeList, in_strideList);
  HOST_SYNC()

  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupLazyHCCL::gather: " + msg);
  };

  std::vector<at::Tensor> outputs;
  c10::intrusive_ptr<Work> work;
  if (getRank() == opts.rootRank) {
    TORCH_CHECK(outputTensors.size() == 1, "Requires a single element list");
    TORCH_CHECK(
        outputTensors[0].size() == static_cast<size_t>(getSize()),
        "Output list should be same size as process group");
    assertTypeAndSizesMatch(
        invalidArgument,
        outputTensors[0],
        inputTensors[0].options(),
        inputTensors[0].sizes());
    outputs = outputTensors[0];
    int numRanks = getSize();
    for (int r = 0; r < numRanks; r++) {
      if (r == getRank()) {
        outputs[r].copy_(inputTensors[0]);
        work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputs);
      } else {
        std::vector<at::Tensor> recvTensor;
        recvTensor.push_back(outputs[r]);
        work = recv(recvTensor, r, 0 /*tag*/);
      }
    }
  } else {
    TORCH_CHECK(outputTensors.size() == 0, "Requires empty output on non-root");
    work = send(inputTensors, opts.rootRank, 0 /*tag*/);
  }
  if (change) {
    PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::gather");
    habana_lazy::HbLazyTensor::StepMarker();
  }
  for (size_t i = 0; i < outputTensors.size(); i++) {
    restoreOddTensorsize(
        outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputs);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const AllToAllOptions& opts) {
  HABANA_ASSERT(
      inputTensors.size() && outputTensors.size(),
      "ProcessGroupLazyHCCL::alltoall input and output tensors must have at least one element");
  auto data_type = outputTensors[0].scalar_type();

  bool cast_tensor = !(
      data_type == c10::ScalarType::Float ||
      data_type == c10::ScalarType::BFloat16 ||
      data_type == c10::ScalarType::Int || data_type == c10::ScalarType::Long);
  at::Tensor t_output;
  at::Tensor t_input;
  if (!cast_tensor) {
    t_output = {outputTensors[0]};
    t_input = {inputTensors[0]};
  } else {
    t_output = {outputTensors[0].to(c10::ScalarType::Float)};
    t_input = {inputTensors[0].to(c10::ScalarType::Float)};
  }

  size_t tensor_size = inputTensors.size();
  // Collecting 1st tensor dim size to split concatenated output later
  std::vector<int64_t> inputSizeList = {inputTensors[0].size(0)};
  std::vector<int64_t> outputSizeList = {outputTensors[0].size(0)};

  for (size_t i = 1; i < tensor_size; i++) {
    HABANA_ASSERT(
        inputTensors[0].dim() == inputTensors[i].dim(),
        "ProcessGroupLazyHCCL::alltoall input tensors must have matching number of dimensions");
    HABANA_ASSERT(
        outputTensors[0].dim() == outputTensors[i].dim(),
        "ProcessGroupLazyHCCL::alltoall input tensors must have matching number of dimensions");
    HABANA_ASSERT(
        inputTensors[0].scalar_type() == inputTensors[i].scalar_type(),
        "ProcessGroupLazyHCCL::alltoall input tensors must have the same dtype");
    HABANA_ASSERT(
        outputTensors[0].scalar_type() == outputTensors[i].scalar_type(),
        "ProcessGroupLazyHCCL::alltoall input tensors must have the same dtype");
    inputSizeList.push_back(inputTensors[i].size(0));
    outputSizeList.push_back(outputTensors[i].size(0));
    if (!cast_tensor) {
      t_input = at::cat({t_input, inputTensors[i]});
      t_output = at::cat({t_output, outputTensors[i]});
    } else {
      t_input = at::cat({t_input, inputTensors[i].to(c10::ScalarType::Float)});
      t_output =
          at::cat({t_output, outputTensors[i].to(c10::ScalarType::Float)});
    }
  }

  habana_lazy::alltoall_hpu_lazy_out(
      t_input, comm_->GetId(), t_output, outputSizeList, inputSizeList);
  habana_lazy::HbLazyTensor::StepMarker();
  PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::alltoall");
  // torch::tensor_split as second argument takes indexes where the splits
  // should be applied, so we need to pass there split sizes vector without the
  // last element
  std::vector<at::Tensor> outputSplitted = torch::tensor_split(
      t_output,
      std::vector<int64_t>(outputSizeList.begin(), outputSizeList.end() - 1));

  for (size_t idx = 0; idx < outputTensors.size(); idx++) {
    if (!cast_tensor) {
      outputTensors[idx].copy_(outputSplitted[idx]);
    } else {
      outputTensors[idx].copy_(outputSplitted[idx].to(data_type));
    }
  }

  std::vector<at::Tensor> out_tensors = {outputTensors};
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(out_tensors);

  if (coalescing_state_) {
    coalesed_works_->append(work);
  }

  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    [[maybe_unused]] const AllToAllOptions& opts) {
  auto data_type = outputTensor.scalar_type();
  bool cast_tensor = !(
      data_type == c10::ScalarType::Float ||
      data_type == c10::ScalarType::BFloat16 ||
      data_type == c10::ScalarType::Int || data_type == c10::ScalarType::Long ||
      data_type == at::kFloat8_e5m2 || data_type == at::kFloat8_e4m3fn);
  at::Tensor t_output;
  at::Tensor t_input;
  if (!cast_tensor) {
    t_output = outputTensor;
    t_input = inputTensor;
  } else {
    t_output = outputTensor.to(c10::ScalarType::Float);
    t_input = inputTensor.to(c10::ScalarType::Float);
  }
  habana_lazy::alltoall_hpu_lazy_out(
      t_input, comm_->GetId(), t_output, outputSplitSizes, inputSplitSizes);
  if (cast_tensor) {
    PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::alltoall_base");
    habana_lazy::HbLazyTensor::StepMarker();
    outputTensor.copy_(t_output.to(data_type));
  }
  std::vector<at::Tensor> out_tensors = {outputTensor};
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(out_tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::scatter(
    [[maybe_unused]] std::vector<at::Tensor>& outputTensors,
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& inputTensors,
    [[maybe_unused]] const ScatterOptions& opts) {
  throw std::runtime_error("scatter is currently not supported with HCCL");
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  auto input_flattened = habana_helpers::flatten_for_scatter_gather(
      inputTensors, outputTensors, size_);
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    for (size_t j = 0; j < inputTensors[0].size(); ++j) {
      input_flattened[i][j].copy_(inputTensors[i][j], true);
    }
  }

  for (size_t index = 0; index < input_flattened.size(); ++index) {
    auto data_type = input_flattened.at(index).scalar_type();
    bool cast_tensor =
        !(data_type == c10::ScalarType::Float ||
          data_type == c10::ScalarType::BFloat16);
    at::Tensor t_updated;
    if (!cast_tensor) {
      habana_lazy::reduce_scatter_hpu_lazy_out(
          input_flattened.at(index),
          (uint8_t)opts.reduceOp,
          comm_->GetId(),
          outputTensors.at(index));
    } else {
      t_updated = input_flattened.at(index).to(c10::ScalarType::Float);
      auto output =
          at::empty_like(outputTensors.at(index), c10::ScalarType::Float);
      habana_lazy::reduce_scatter_hpu_lazy_out(
          t_updated, (uint8_t)opts.reduceOp, comm_->GetId(), output);
      outputTensors.at(index).copy_(output.to(data_type));
    }
  }

  auto work =
      c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputTensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  habana_lazy::reduce_scatter_hpu_lazy_out(
      inputTensor, (uint8_t)opts.reduceOp, comm_->GetId(), outputTensor);
  std::vector<at::Tensor> out_tensors = {outputTensor};
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(out_tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

void ProcessGroupLazyHCCL::permutedSendTensorsToDense(at::Tensor& tensor) {
  auto self_hb_tensor = habana_lazy::GetHbLazyTensor(tensor);
  auto self_internal_tesor = self_hb_tensor.EvaluateTensorData();
  std::vector<uint8_t> permutation;
  auto hb_weight_impl =
      habana_lazy::GetHbInternalTensorImpl(self_internal_tesor);
  permutation = hb_weight_impl->GetMemoryPermutation();
  if (!permutation.empty()) {
    PT_DISTRIBUTED_DEBUG(
        "Tensor: ",
        self_hb_tensor.getTensorUniqueId(),
        " has permutation: ",
        VecToString(permutation),
        " transposing it back to be dense");
    tensor = torch::clone(tensor);
  }
}

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeOddTensor(tensors, changed, sizeList, strideList);
  for (size_t index = 0; index < tensors.size(); ++index) {
    auto& tensor = tensors[index];
    permutedSendTensorsToDense(tensor);
    habana_lazy::send_hpu_lazy_(tensor, dstRank, tag, comm_->GetId());
  }
  restoreOddTensorsize(tensors, changed, sizeList, strideList);
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeOddTensor(tensors, changed, sizeList, strideList);
  for (size_t index = 0; index < tensors.size(); ++index) {
    habana_lazy::recv_hpu_lazy_(
        tensors.at(index), srcRank, tag, comm_->GetId());
  }
  restoreOddTensorsize(tensors, changed, sizeList, strideList);
  auto work = c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
  if (coalescing_state_) {
    coalesed_works_->append(work);
  }
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::recvAnysource(
    [[maybe_unused]] std::vector<at::Tensor>& tensors,
    [[maybe_unused]] int tag) {
  throw std::runtime_error(
      "recvAnysource is currently not supported with HCCL");
};

void ProcessGroupLazyHCCL::hostBarrier() {
  if (this->emulate_distributed_) {
    return;
  }

  PT_DISTRIBUTED_DEBUG(
      "Enter hostBarrier group_name:", group_name_, ", rank:", rank_);

  constexpr int64_t kSynchronizeBusyWaitMillis = 1;
  // Minumum three keys are required to avoid race condition
  constexpr int64_t kNumBarrierKeys = 3;

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

  PT_DISTRIBUTED_DEBUG(
      "Exit hostBarrier group_name:", group_name_, ", rank:", rank_);
}

void ProcessGroupLazyHCCL::destroyHandshake() {
  /**
   * This handshake ensures that rank 0 that hosts store service finishes its
   * job as last.
   */
  if (this->emulate_distributed_) {
    return;
  }

  PT_DISTRIBUTED_DEBUG(
      "Enter destroyHandshake group_name:", group_name_, ", rank:", rank_);
  std::string barrier_key = std::string("ProcessGroup::destroy");

  auto worker_count = store_->add(barrier_key, 1);
  if (getRank() == 0) {
    while (worker_count != size_) {
      worker_count = store_->add(barrier_key, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  PT_DISTRIBUTED_DEBUG(
      "Exit destroyHandshake group_name:", group_name_, ", rank:", rank_);
}

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::barrier(
    const BarrierOptions& opts [[maybe_unused]]) {
  hostBarrier();
  PT_DISTRIBUTED_DEBUG("step marker due to ProcessGroupLazyHCCL::barrier");
  habana_lazy::HbLazyTensor::StepMarker();

  std::vector<at::Tensor> tensors;
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

} // namespace c10d

namespace py = pybind11;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::object backend =
      (py::object)py::module_::import("torch.distributed").attr("_Backend");
  intrusive_ptr_class_<::c10d::ProcessGroupLazyHCCL> processGroupHccl(
      module, "ProcessGroupHCCL", backend);

  processGroupHccl.def(py::init(
      &c10d::ProcessGroupHCCLRegistry<c10d::ProcessGroupLazyHCCL>::create));
};