/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch_ver/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch_ver/csrc/distributed/c10d/Store.hpp>
#include <torch_ver/csrc/distributed/c10d/Types.hpp>
#include <torch_ver/csrc/distributed/c10d/Utils.hpp>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "backend/synapse_helpers/device_context.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "process_group_hccl_base.hpp"

using Work = c10d_ver::Work;

namespace c10d {

// Now continue on other work in the current stream.
class TORCH_API ProcessGroupHCCL : public ProcessGroupHcclBase {
 public:
  class WorkHCCL : public Work, public std::enable_shared_from_this<WorkHCCL> {
   public:
    // Constructor takes a list of HABANA devices and communicators
    WorkHCCL(
        const std::vector<at::Tensor>& outputs,
        std::shared_ptr<hcclComm_t> hccl_comm_,
        std::shared_ptr<hccl_integration::device_context> deviceCtx);
    WorkHCCL(const WorkHCCL& w);
    WorkHCCL();

    virtual ~WorkHCCL();

    bool isCompleted() override;

    bool isSuccess() const override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    void synchronize() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    // HCCL runs on a different stream. Hold tensor references which is used
    // to query completion of execution
    std::vector<at::Tensor> outputs_;
    std::shared_ptr<hcclComm_t> hccl_comm_;
    std::shared_ptr<hccl_integration::device_context> deviceCtx_;
    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    c10::intrusive_ptr<Store> store_;
    c10::intrusive_ptr<at::ivalue::Future> future_;

    friend class ProcessGroupHCCL;
  };

  ProcessGroupHCCL(const c10::intrusive_ptr<Store>& store, int rank, int size);

  virtual ~ProcessGroupHCCL();

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Helper function that is called by the destructor
  void destroy() override;

 protected:
  // Helper that encapsulates work shared across all collective communication
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      CollectiveFn fn,
      bool is_allreduce = false) override;

  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      PointToPointFn fn,
      int peerRank) override;

  c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> initWork(
      std::vector<at::Tensor>& outputs,
      std::shared_ptr<hcclComm_t> hccl_comm,
      std::shared_ptr<hccl_integration::device_context> deviceCtxts);

  void permutedSendTensorsToDense(std::vector<at::Tensor>& tensors) override;
  void clearPermutesFromRecvTensors(std::vector<at::Tensor>& tensors) override;
};

} // namespace c10d
