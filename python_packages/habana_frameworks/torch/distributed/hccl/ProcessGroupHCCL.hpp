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

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "process_group_lazy_hccl.hpp"
#include "pytorch_helpers/synapse_helpers/device_context.h"
namespace c10d {

// Now continue on other work in the current stream.
class TORCH_API ProcessGroupHCCL : public ProcessGroup {
 public:
  class WorkHCCL : public ProcessGroup::Work,
                   public std::enable_shared_from_this<WorkHCCL> {
   public:
    // Constructor takes a list of HABANA devices and communicators
    WorkHCCL(
        const std::vector<at::Tensor>& outputs,
        const std::vector<int>& devices,
        std::vector<std::shared_ptr<hcclComm_t>>& hccl_comms_,
        std::vector<std::shared_ptr<hccl_integration::device_context>>&
            deviceCtxts_);
    WorkHCCL(const WorkHCCL& w);

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
    std::vector<int> devices_;
    std::vector<std::shared_ptr<hcclComm_t>> hccl_comms_;
    std::vector<std::shared_ptr<hccl_integration::device_context>> deviceCtxts_;
    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    c10::intrusive_ptr<Store> store_;
    c10::intrusive_ptr<at::ivalue::Future> future_;

    friend class ProcessGroupHCCL;
  };

  ProcessGroupHCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& opTimeout);

  virtual ~ProcessGroupHCCL();
  void abort();
  const std::string getBackendName() const override {
    return std::string("hccl");
   }

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
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

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
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

  void permutedSendTensorsToDense(std::vector<at::Tensor>& tensors);
  void clearPermutesFromRecvTensors(std::vector<at::Tensor>& tensors);

  static void groupStart();

  static void groupEnd();

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  static c10::intrusive_ptr<::c10d::ProcessGroup> createProcessGroupHCCL(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES)) {
      return c10::make_intrusive<ProcessGroupLazyHCCL>(
          store, rank, size, timeout);
    } else {
      return c10::make_intrusive<ProcessGroupHCCL>(store, rank, size, timeout);
    }
  }

  template <typename T>
  using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

  static const int64_t kWatchdogThreadSleepMillis;
  static void ProcessGroupHCCLConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");

    register_backend(
        "hccl",
        py::cpp_function(
            &c10d::ProcessGroupHCCL::createProcessGroupHCCL,
            py::arg("store"),
            py::arg("rank"),
            py::arg("size"),
            py::arg("timeout") =
                std::chrono::milliseconds(kWatchdogThreadSleepMillis)));

    auto processGroup = module.attr("ProcessGroup");

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES)) {
      auto processGroupHCCL =
          intrusive_ptr_class_<::c10d::ProcessGroupLazyHCCL>(
              module, "ProcessGroupHCCL", processGroup);

      processGroupHCCL.def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      std::chrono::milliseconds timeout) {
            return c10::make_intrusive<::c10d::ProcessGroupLazyHCCL>(
                store, rank, size, timeout);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") =
              std::chrono::milliseconds(kWatchdogThreadSleepMillis));
    } else {
      auto processGroupHCCL = intrusive_ptr_class_<::c10d::ProcessGroupHCCL>(
          module, "ProcessGroupHCCL", processGroup);

      processGroupHCCL.def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      std::chrono::milliseconds timeout) {
            return c10::make_intrusive<::c10d::ProcessGroupHCCL>(
                store, rank, size, timeout);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") =
              std::chrono::milliseconds(kWatchdogThreadSleepMillis));
      processGroupHCCL.def_static(
          "_group_start", []() { ::c10d::ProcessGroupHCCL::groupStart(); });
      processGroupHCCL.def_static(
          "_group_end", []() { ::c10d::ProcessGroupHCCL::groupEnd(); });
    }
  }

 private:
  // Helper that encapsulates work shared across all collective communication
  template <typename Fn>
  c10::intrusive_ptr<ProcessGroup::Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<ProcessGroup::Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post);

  template <typename Fn>
  c10::intrusive_ptr<ProcessGroup::Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      Fn fn,
      int peerRank);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<ProcessGroup::Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      Fn fn,
      int peerRank,
      PreProcess pre,
      PostProcess post);

 protected:
  virtual c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> initWork(
      std::vector<at::Tensor>& outputs,
      std::vector<int> devices,
      std::vector<std::shared_ptr<hcclComm_t>>& hccl_comms_,
      std::vector<std::shared_ptr<hccl_integration::device_context>>&
          deviceCtxts);

  void broadcastUniqueHCCLID(hcclUniqueId* hcclID);
  std::shared_ptr<hcclComm_t> getComm(int deviceId);
  synStreamHandle getCommStream(int deviceId);
  std::shared_ptr<hccl_integration::device_context> getDeviceCtxt(int deviceId);

  std::vector<int> getDeviceList(const std::vector<at::Tensor>& tensors);
  std::vector<std::shared_ptr<hcclComm_t>> getCommList(
      const std::vector<int>& devices);
  std::vector<std::shared_ptr<hccl_integration::device_context>>
  getDeviceCtxtList(const std::vector<int>& devices);
  std::vector<synStreamHandle> getCommStreams(const std::vector<int>& devices);
  // Helper function that is called by the destructor
  void destroy();
  bool stop_;

  uint64_t hcclCommCounter_{0};
  std::mutex mutex_;
  c10::intrusive_ptr<Store> store_;
  void hostBarrier();
  void nwStreamSync();
  size_t barrier_cnt_;

  // Maintains the list of communicators associated with the devices.
  std::map<int, std::shared_ptr<hcclComm_t>> hccl_communicator_;
  std::map<int, std::shared_ptr<hccl_integration::device_context>>
      device_contexts_;
  std::map<int, synStreamHandle> comm_streams_;
};

} // namespace c10d
