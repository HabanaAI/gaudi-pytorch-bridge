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

#pragma once

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <functional>

#include "backend/synapse_helpers/hccl_communicator.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "process_group_hccl_base.hpp"

using Work = c10d::Work;

namespace c10d {
class TORCH_API ProcessGroupEagerHCCL : public ProcessGroupHcclBase {
 public:
  ProcessGroupEagerHCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      std::string group_name);
  virtual ~ProcessGroupEagerHCCL();

  class WorkEager : public Work,
                    public std::enable_shared_from_this<WorkEager> {
   public:
    WorkEager(
        const std::vector<at::Tensor>& outputs,
        std::shared_ptr<habana::HcclCommunicator> hccl_comm_);
    WorkEager();
    WorkEager(const WorkEager& w) = delete;
    virtual ~WorkEager();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    void abort() override;
    void synchronize() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    std::vector<at::Tensor> outputs_;
    std::shared_ptr<habana::HcclCommunicator> comm_;
    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupEagerHCCL;
  };

  c10::intrusive_ptr<Work> barrier(const BarrierOptions& opts) override;

  void destroy() override;

  void restoreOddSizeSendTensors(std::vector<at::Tensor>& tensors);

 protected:
  void groupStart();

  void groupEnd();

  void waitForJobCompletion(){};

  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      CollectiveFn fn,
      bool is_allreduce = false) override;

  void initComms() override;

  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      PointToPointFn fn,
      int peerRank) override;

  c10::intrusive_ptr<Work> initWork(std::vector<at::Tensor>& outputs) override;
  void permutedSendTensorsToDense(std::vector<at::Tensor>& tensors) override;
  void clearPermutesFromRecvTensors(std::vector<at::Tensor>& tensors) override;

  std::shared_ptr<habana::HcclCommunicator> comm_;
};

} // namespace c10d
