/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "process_group_lazy_hccl.hpp"
#include "habana_kernels/lazy_kernels_declarations.h"
//#include "hpu_ops/generated/hpu_op.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/synapse_helpers/hccl_communicator.h"

namespace c10d {
ProcessGroupLazyHCCL::ProcessGroupLazyHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout)
    : ProcessGroup(rank, size) {
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

bool ProcessGroupLazyHCCL::WorkLazy::wait(std::chrono::milliseconds timeout) {
  habana_lazy::HbLazyTensor::StepMarker();
  return true;
}

void ProcessGroupLazyHCCL::WorkLazy::abort() {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
}

void ProcessGroupLazyHCCL::WorkLazy::synchronize() {
  habana_lazy::HbLazyTensor::StepMarker();
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupLazyHCCL::WorkLazy::
    getFuture() {
  return future_;
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  for (auto& t : tensors) {
    habana_lazy::broadcast_hpu_lazy_(t, opts.rootRank, comm_->GetId());
  }
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  for (auto& t : tensors) {
    habana_lazy::allreduce_hpu_lazy_(t, (uint8_t)opts.reduceOp, comm_->GetId());
  }
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::
    allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const AllreduceCoalescedOptions& opts) {
  at::TensorList at_tensors(tensors);
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  at::TensorList at_tensors(tensors);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  at::TensorList at_tensors(inputTensors);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(inputTensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  at::TensorList at_tensors({inputBuffer});
  std::vector<at::Tensor> tensors;
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::
    allgather_coalesced(
        std::vector<std::vector<at::Tensor>>& outputTensorLists,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  at::TensorList at_tensors(inputTensors);
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(inputTensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(inputTensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  std::vector<at::Tensor> tensors;
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(
      inputTensors.at(0));
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(outputTensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupLazyHCCL::barrier(
    const BarrierOptions& opts) {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
  std::vector<at::Tensor> tensors;
  return c10::make_intrusive<ProcessGroupLazyHCCL::WorkLazy>(tensors);
};

} // namespace c10d
