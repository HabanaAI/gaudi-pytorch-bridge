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

#pragma once
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <torch_ver/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch_ver/csrc/distributed/c10d/Store.hpp>
#include <torch_ver/csrc/distributed/c10d/Types.hpp>
#include <torch_ver/csrc/distributed/c10d/Utils.hpp>

#include "backend/synapse_helpers/hccl_communicator.h"
#include "habana_kernels/lazy_kernels_declarations.h"

using Work = c10d_ver::Work;

namespace c10d {
class TORCH_API ProcessGroupLazyHCCL : public Backend {
 public:
  ProcessGroupLazyHCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      std::string group_name);
  virtual ~ProcessGroupLazyHCCL();

  class WorkLazy : public Work, public std::enable_shared_from_this<WorkLazy> {
   public:
    WorkLazy(const std::vector<at::Tensor>& outputs);
    WorkLazy(const WorkLazy& w) = delete;
    virtual ~WorkLazy();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    void abort() override;
    void synchronize() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    std::vector<at::Tensor> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupLazyHCCL;
  };

  const std::string getBackendName() const override {
    return std::string("hccl");
  }
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;
  void destroy();

 private:
  void hostBarrier();
  void destroyHandshake();
  void permutedSendTensorsToDense(at::Tensor& tensor);
  c10::intrusive_ptr<Store> store_;
  size_t barrier_cnt_;
  std::string group_name_;
  bool emulate_distributed_;

 protected:
  std::shared_ptr<habana::HcclCommunicator> comm_;
};

} // namespace c10d
