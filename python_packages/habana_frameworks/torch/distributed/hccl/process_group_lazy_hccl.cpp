/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

namespace c10d {

namespace {

#define HOST_SYNC()                                   \
  {                                                   \
    if (GET_ENV_FLAG_NEW(PT_HPU_USE_PT_STORE_SYNC)) { \
      hostBarrier();                                  \
    }                                                 \
  }

bool resizeTensor(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList) {
  bool change = false;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    changed[i] = false;
    if ((at::kChar == btensor_type || at::kByte == btensor_type ||
         at::kBool == btensor_type) &&
        tensors[i].numel() % 2 != 0) {
      changed[i] = true;
      sizeList[i] = tensors[i].sizes().vec();
      strideList[i] = tensors[i].strides().vec();
      tensors[i] = tensors[i].resize_(tensors[i].numel() + 1);
      change = true;
    }
  }
  return change;
}

void restoreTensorsize(
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

} // namespace
ProcessGroupLazyHCCL::ProcessGroupLazyHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store), barrier_cnt_(0) {
  PT_LAZY_DEBUG("Create ProcessGroupLazyHCCL, rank = ", rank, " size = ", size);
  comm_ = habana::HcclCommunicator::Create(
      rank,
      size,
      [store, rank](
          int64_t comm_id, hcclUniqueId* hcclID) { // Use hcclID as store key?
        std::string storeKey = std::to_string(comm_id);
        if (rank == 0) {
          auto vec = std::vector<uint8_t>(
              reinterpret_cast<uint8_t*>(hcclID),
              reinterpret_cast<uint8_t*>(hcclID) + sizeof(hcclUniqueId));
          store->set(storeKey, vec);
        } else {
          auto vec = store->get(storeKey);
          TORCH_CHECK(vec.size() == sizeof(hcclUniqueId));
          std::memcpy(hcclID, vec.data(), vec.size());
        }
      });
};

ProcessGroupLazyHCCL::~ProcessGroupLazyHCCL() {
  PT_LAZY_DEBUG("Destroy ProcessGroupLazyHCCL");
  hostBarrier();
  comm_.reset();
};

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
  habana_lazy::HbLazyTensor::StepMarker();
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
  resizeTensor(tensors, changed, sizeList, strideList);
  HOST_SYNC()
  for (auto& t : tensors) {
    habana_lazy::broadcast_hpu_lazy_(t, opts.rootRank, comm_->GetId());
  }
  restoreTensorsize(tensors, changed, sizeList, strideList);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
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
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    [[maybe_unused]] const AllreduceCoalescedOptions& opts) {
  at::TensorList at_tensors(tensors);
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
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
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
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
    resizeTensor(outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(tensor_size);
  change = resizeTensor(inputTensors, in_changed, in_sizeList, in_strideList);
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
    restoreTensorsize(outputTensors[i], changed[i], sizeList[i], strideList[i]);
  }
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(output_list_flat);
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
  HOST_SYNC()
  habana_lazy::allgather_hpu_lazy_out(
      inputBuffer, comm_->GetId(), outputBuffer);
  std::vector<at::Tensor> out_tensors = {outputBuffer};
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(out_tensors);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::allgather_coalesced(
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& outputTensorLists,
    [[maybe_unused]] std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const AllgatherOptions& opts) {
  throw std::runtime_error(
      "allgather_coalesced is currently not supported with HCCL");
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::gather(
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& outputTensors,
    [[maybe_unused]] std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const GatherOptions& opts) {
  throw std::runtime_error("gather is currently not supported with HCCL");
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
      data_type == c10::ScalarType::Int || data_type == c10::ScalarType::Long);
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
    outputTensor.copy_(t_output.to(data_type));
  }
  std::vector<at::Tensor> out_tensors = {outputTensor};
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(out_tensors);
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

  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputTensors);
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
  resizeTensor(tensors, changed, sizeList, strideList);
  for (size_t index = 0; index < tensors.size(); ++index) {
    auto& tensor = tensors[index];
    permutedSendTensorsToDense(tensor);
    habana_lazy::send_hpu_lazy_(tensor, dstRank, tag, comm_->GetId());
  }
  restoreTensorsize(tensors, changed, sizeList, strideList);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  for (size_t index = 0; index < tensors.size(); ++index) {
    habana_lazy::recv_hpu_lazy_(
        tensors.at(index), srcRank, tag, comm_->GetId());
  }
  restoreTensorsize(tensors, changed, sizeList, strideList);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::recvAnysource(
    [[maybe_unused]] std::vector<at::Tensor>& tensors,
    [[maybe_unused]] int tag) {
  throw std::runtime_error(
      "recvAnysource is currently not supported with HCCL");
};

constexpr int64_t kSynchronizeBusyWaitMillis = 1;
// Minumum three keys are required to avoid race condition
constexpr int64_t kNumBarrierKeys = 3;
void ProcessGroupLazyHCCL::hostBarrier() {
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

c10::intrusive_ptr<Work> ProcessGroupLazyHCCL::barrier(
    const BarrierOptions& opts [[maybe_unused]]) {
  hostBarrier();
  PT_IRGRAPH_DEBUG("step marker due to ProcessGroupLazyHCCL::barrier");
  habana_lazy::HbLazyTensor::StepMarker();

  std::vector<at::Tensor> tensors;
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

} // namespace c10d

namespace py = pybind11;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  intrusive_ptr_class_<::c10d::ProcessGroupLazyHCCL, c10d::ProcessGroup>
      processGroupHccl(module, "ProcessGroupHCCL");

  processGroupHccl.def(
      py::init<const c10::intrusive_ptr<c10d::Store>&, int, int>());
};