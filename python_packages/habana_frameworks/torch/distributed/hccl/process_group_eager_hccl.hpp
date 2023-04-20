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
class TORCH_API ProcessGroupEagerHCCL : public ProcessGroup {
 public:
  ProcessGroupEagerHCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout);
  virtual ~ProcessGroupEagerHCCL();

  class WorkEager : public Work,
                    public std::enable_shared_from_this<WorkEager> {
   public:
    WorkEager(const std::vector<at::Tensor>& outputs);
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
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupEagerHCCL;
  };

  const std::string getBackendName() const override {
    return std::string("hccl");
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> barrier(const BarrierOptions& opts) override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts) override;

 private:
  c10::intrusive_ptr<Store> store_;
  size_t barrier_cnt_;

  void hostBarrier();

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      bool is_allreduce = false);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      bool is_allreduce);

 protected:
  std::shared_ptr<habana::HcclCommunicator> comm_;
};

} // namespace c10d
