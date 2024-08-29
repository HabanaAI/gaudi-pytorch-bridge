/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/extension.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "backend/synapse_helpers/device_context.h"

using Work = c10d::Work;

namespace c10d {

using CollectiveFn = std::function<hcclResult_t(
    at::Tensor&,
    at::Tensor&,
    const void*,
    void*,
    hcclComm_t&,
    synStreamHandle)>;

using PointToPointFn = std::function<
    hcclResult_t(at::Tensor&, void*, hcclComm_t&, synStreamHandle, int)>;

// Now continue on other work in the current stream.
class TORCH_API ProcessGroupHcclBase : public Backend {
 public:
  ProcessGroupHcclBase(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      std::string group_name);

  virtual ~ProcessGroupHcclBase();
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

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
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
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;


  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  virtual void destroy() = 0;

  class CoalescedWorkHCCL
      : public Work,
        public std::enable_shared_from_this<CoalescedWorkHCCL> {
   public:
    explicit CoalescedWorkHCCL(ProcessGroupHcclBase* pg) : pg_(pg) {}

    ~CoalescedWorkHCCL();

    // Same as calling synchronize() for HCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout);

    // Method to append a new Work object to works_
    void append(const c10::intrusive_ptr<Work>& work);

    // Method to clear the works_ vector
    void clear();

   protected:
    // The cached list of CUDA devices to operate on
    // std::vector<Work> works_;
    std::vector<c10::intrusive_ptr<Work>> works_;

    friend class ProcessGroupHcclBase;

   private:
    ProcessGroupHcclBase* pg_; // Pointer to the enclosing class instance
  };

 protected:
  virtual void groupStart() = 0;

  virtual void groupEnd() = 0;

  virtual void waitForJobCompletion() = 0;

  virtual c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      CollectiveFn fn,
      bool is_allreduce = false) = 0;

  virtual void initComms() = 0;
  virtual c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      PointToPointFn fn,
      int peerRank) = 0;

  c10::intrusive_ptr<Work> _broadcast_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const BroadcastOptions& opts);
  void hostBarrier();
  void destroyHandshake();

  virtual c10::intrusive_ptr<Work> initWork(
      std::vector<at::Tensor>& outputs) = 0;
  virtual void permutedSendTensorsToDense(std::vector<at::Tensor>& tensors) = 0;
  virtual void clearPermutesFromRecvTensors(
      std::vector<at::Tensor>& tensors) = 0;

  bool emulate_distributed_;
  bool always_support_int64_;
  c10::intrusive_ptr<Store> store_;
  size_t barrier_cnt_;
  std::string group_name_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // The latest work used in collective, used for coalese start/end
  c10::intrusive_ptr<CoalescedWorkHCCL> coalesed_works_ = nullptr;
};

} // namespace c10d
