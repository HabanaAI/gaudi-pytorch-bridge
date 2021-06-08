/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once

#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#include <hcl_communicator.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
namespace c10d {

// Now continue on other work in the current stream.
class ProcessGroupHCL : public ProcessGroup {
 public:
  class WorkHCL : public ProcessGroup::Work,
                  public std::enable_shared_from_this<WorkHCL> {
   public:
    // Constructor takes a list of HABANA devices and communicators
    WorkHCL(
        const std::vector<at::Tensor>& inputs,
        const std::vector<int>& devices,
        std::vector<std::shared_ptr<synapse_helpers::hcl_communicator>>&
            hcl_comms_);
    virtual ~WorkHCL();

    bool isCompleted() override;

    bool isSuccess() const override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    void synchronize() override;

   protected:
    // HCL runs on a different stream. Hold tensor references which is used
    // to query completion of execution
    std::vector<at::Tensor> outputs_;
    std::vector<int> devices_;
    std::vector<std::shared_ptr<synapse_helpers::hcl_communicator>> hcl_comms_;
    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    friend class ProcessGroupHCL;
  };

  ProcessGroupHCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& opTimeout);

  virtual ~ProcessGroupHCL();
  void abort();

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  static c10::intrusive_ptr<::c10d::ProcessGroup> createProcessGroupHCL(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout) {
    return c10::make_intrusive<ProcessGroupHCL>(store, rank, size, timeout);
  }

 private:
  // Helper that encapsulates work shared across all collective communication
  template <typename Fn>
  c10::intrusive_ptr<ProcessGroup::Work> hclcollective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<ProcessGroup::Work> hclcollective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post);

  void (*mark_step)();

 protected:
  virtual c10::intrusive_ptr<ProcessGroupHCL::WorkHCL> initWork(
      std::vector<at::Tensor>& outputs,
      std::vector<int> devices,
      std::vector<std::shared_ptr<synapse_helpers::hcl_communicator>>&
          hcl_comms_);

  std::shared_ptr<synapse_helpers::hcl_communicator> getComm(int deviceId);

  std::vector<std::shared_ptr<synapse_helpers::hcl_communicator>> getCommList(
      const std::vector<int>& devices);
  // Helper function that is called by the destructor
  void destroy();
  bool stop_;

  // Maintains the list of communicators associated with the devices.
  std::map<int, std::shared_ptr<synapse_helpers::hcl_communicator>>
      hcl_communicator_;
};

} // namespace c10d
