/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#include "process_group_eager_hccl.hpp"

#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <utility>
#include <vector>

#include <unistd.h>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/collective_utils.h"
#include "backend/helpers/eager_pipeline.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_hccl_base.hpp"
#include "pytorch_helpers/habana_helpers/python_utils.h"

#include "hpu_ops/op_logger.h"
#include "process_group_registry.hpp"

namespace c10d {

namespace {
class CollectiveContext {
 public:
  CollectiveContext(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      bool is_pipelined = false)
      : inputs_(inputs), outputs_(outputs), is_pipelined_(is_pipelined) {
    TORCH_CHECK(inputs.size() == outputs.size());
    ensure_input_output_tensors_contiguity();
  }
  CollectiveContext(const CollectiveContext&) = delete;
  CollectiveContext(CollectiveContext&&) = delete;
  CollectiveContext& operator=(const CollectiveContext&) = delete;
  CollectiveContext& operator=(CollectiveContext&) = delete;
  CollectiveContext& operator=(CollectiveContext&&) = delete;

  ~CollectiveContext() {
    PT_DISTRIBUTED_DEBUG("~CollectiveContext")
    // if collectives are pipelined, the output tensor restore must be
    // done at the main thread, such that D2D copies are also pipelined.
    if (!is_pipelined_) {
      restore_output_tensors_if_needed();
    }
  }

  CollectiveContext(std::vector<at::Tensor>& tensors)
      : CollectiveContext(tensors, tensors) {}

  std::vector<std::pair<at::Tensor, at::Tensor>>& tensors() {
    return in_out_tensors_contiguous_;
  }

  std::vector<at::Tensor>& outputs() {
    return outputs_;
  }

 private:
  std::vector<std::pair<at::Tensor, at::Tensor>> in_out_tensors_contiguous_;
  std::vector<at::Tensor> inputs_;
  std::vector<at::Tensor> outputs_;
  bool is_pipelined_{false};

  void ensure_input_output_tensors_contiguity() {
    in_out_tensors_contiguous_.resize(inputs_.size());

    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!inputs_[i].is_contiguous() || !outputs_[i].is_contiguous()) {
        PT_DISTRIBUTED_WARN(
            "Provided input/output is not contiguous. Additional tensor copy will"
            " be created prior to collective operation execution, what may impact"
            " performance.");
      }

      // create tensor copy in case if provided input is not contiguous
      PT_DISTRIBUTED_DEBUG(
          "ensure contiguous input tensor ", habana::to_string(inputs_[i]));
      at::Tensor input_contiguous = inputs_[i].contiguous();
      at::Tensor output_contiguous;
      if (inputs_[i].unsafeGetTensorImpl() ==
          outputs_[i].unsafeGetTensorImpl()) {
        // inplace collective: output == intput
        output_contiguous = input_contiguous;
      } else {
        // create tensor copy in case if provided output is not contiguous
        PT_DISTRIBUTED_DEBUG(
            "ensure contiguous output tensor ", habana::to_string(outputs_[i]));
        output_contiguous = outputs_[i].contiguous();
      }
      in_out_tensors_contiguous_[i] =
          std::make_pair(input_contiguous, output_contiguous);
    }
  }

  void restore_output_tensors_if_needed() {
    for (size_t i = 0; i < in_out_tensors_contiguous_.size(); ++i) {
      if (outputs_[i].unsafeGetTensorImpl() !=
          in_out_tensors_contiguous_[i].second.unsafeGetTensorImpl()) {
        // provided output wasn't contiguous, so the result of collective is
        // stored in tensors_contiguous[i]
        PT_DISTRIBUTED_DEBUG(
            "restore output tensor ", habana::to_string(outputs_[i]));
        outputs_[i].copy_(in_out_tensors_contiguous_[i].second, true);
      }
    }
  }
};
} // namespace

ProcessGroupEagerHCCL::ProcessGroupEagerHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::string group_name)
    : ProcessGroupHcclBase(store, rank, size, group_name) {
  PT_DISTRIBUTED_DEBUG(
      "Created ProcessGroupEagerHCCL name:",
      group_name_,
      ", size:",
      size,
      ", rank:",
      rank);
  always_support_int64_ = true;
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

void ProcessGroupEagerHCCL::destroy() {
  PT_DISTRIBUTED_DEBUG(
      "Destroy ProcessGroupEagerHCCL name:",
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

  destroyHandshake();
}

ProcessGroupEagerHCCL::~ProcessGroupEagerHCCL() {
  PT_DISTRIBUTED_DEBUG(
      "~ProcessGroupEagerHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);
  habana_helpers::AutoNoGIL gil_release;
  destroy();
};

ProcessGroupEagerHCCL::WorkEager::WorkEager(
    const std::vector<at::Tensor>& outputs,
    std::shared_ptr<habana::HcclCommunicator> comm)
    : outputs_(outputs),
      comm_(comm),
      workStartTime_(std::chrono::steady_clock::now()),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted(at::IValue(outputs));
}

ProcessGroupEagerHCCL::WorkEager::WorkEager()
    : future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted();
}

ProcessGroupEagerHCCL::WorkEager::~WorkEager() {
  outputs_.clear();
}

bool ProcessGroupEagerHCCL::WorkEager::isCompleted() {
  return exception() || wait(); // check for the completion of work;
}

bool ProcessGroupEagerHCCL::WorkEager::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }
  return true;
}

void ProcessGroupEagerHCCL::WorkEager::synchronize() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    PT_DISTRIBUTED_DEBUG("WorkEager::synchronize()");
    comm_->getDeviceCtxt()->synchronize_output(
        (synapse_helpers::device_ptr)outputs_[i].storage().data_ptr().get(),
        (c10::hpu::getCurrentHPUStream()).stream());
  }
  outputs_.clear();
}

void Synchronize_Execute_Task(
    const std::vector<at::Tensor>& outputs,
    std::shared_ptr<habana::HcclCommunicator> comm,
    synapse_helpers::hpuStream_t stream) {
  PT_DISTRIBUTED_DEBUG("Synchronize_Execute_Task");
  for (size_t i = 0; i < outputs.size(); ++i) {
    comm->getDeviceCtxt()->synchronize_output(
        (synapse_helpers::device_ptr)outputs[i].storage().data_ptr().get(),
        stream);
  }
}

void Synchronize_Empty_Compile_Task(
    const std::vector<at::Tensor>& outputs,
    std::shared_ptr<habana::HcclCommunicator> comm,
    synapse_helpers::hpuStream_t stream) {
  PT_DISTRIBUTED_DEBUG("Synchronize_Empty_Compile_Task");
  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      Synchronize_Execute_Task, std::move(outputs), comm, stream);
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
  }
}

void Synchronize_Empty_Lowering_Task(
    const std::vector<at::Tensor>& outputs,
    std::shared_ptr<habana::HcclCommunicator> comm,
    synapse_helpers::hpuStream_t stream) {
  PT_DISTRIBUTED_DEBUG("Synchronize_Empty_Lowering_Task");
  habana_helpers::Singleton_CompileThreadPool::getInstance().Enqueue(
      Synchronize_Empty_Compile_Task, std::move(outputs), comm, stream);
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_CompileThreadPool::getInstance()
        .JoinPendingThread();
  }
}

bool ProcessGroupEagerHCCL::WorkEager::wait(std::chrono::milliseconds timeout
                                            [[maybe_unused]]) {
  PT_DISTRIBUTED_DEBUG("WorkEager::wait");
  const bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);

  std::vector<at::Tensor> outputs_backend;
  if (pipeline_flag) {
    // Update backend tensors if pipeline is enabled
    outputs_backend.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); i++) {
      outputs_backend.push_back(
          habana::eager::HbEagerTensorPool::get_backend_tensor(outputs_[i]));

      // Set tensor pipeline metadata
      auto output_hb_tmeta{
          habana::get_tensor_extra_meta(outputs_backend.back())};
      output_hb_tmeta->set_tensor_pipelined();
    }
    habana::eager::SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            Synchronize_Empty_Lowering_Task,
            std::move(outputs_backend),
            comm_,
            (c10::hpu::getCurrentHPUStream()).stream());
  } else {
    // Nothing to join pending here
    Synchronize_Execute_Task(
        outputs_, comm_, (c10::hpu::getCurrentHPUStream()).stream());
  }
  outputs_.clear();
  return true;
}

void ProcessGroupEagerHCCL::WorkEager::abort() {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupEagerHCCL::WorkEager::
    getFuture() {
  return future_;
};

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::initWork(
    std::vector<at::Tensor>& outputs) {
  return c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(outputs, comm_);
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    PointToPointFn fn,
    int peerRank) {
  CollectiveContext collective_ctx(tensors);

  habana::eager::JoinPendingPipelineThreads();

  auto deviceCtxt = comm_->getDeviceCtxt();
  for (auto& input_output : collective_ctx.tensors()) {
    at::Tensor& tensor = input_output.first;
    TORCH_CHECK(
        tensor.get_device() == 0,
        "All tensors are expected to be assigned to device with id 0");
    synStreamHandle collective_stream = comm_->getCommStream();

    synapse_helpers::device_ptr tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensor.storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);

    auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

    struct ResourceHolder {
      at::Tensor tensor_;
      std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    };
    auto resource_holder = std::make_shared<ResourceHolder>();
    resource_holder->tensor_ = tensor;

    void* tensor_address;
    deviceCtxt->lock_address(
        tensor.data_ptr(), &tensor_address, resource_holder->address_lock);

    hcclResult_t hccl_result =
        fn(tensor,
           tensor_address,
           *(comm_->GetHcclHandle()),
           collective_stream,
           peerRank);
    TORCH_CHECK(hcclSuccess == hccl_result, "P2P call returned error");

    recipe_counter.increase();
    deviceCtxt->submit_events(
        collective_stream,
        tensor_storage_ptr,
        [resource_holder, &recipe_counter]() mutable {
          resource_holder.reset();
          recipe_counter.decrease_and_notify();
        });
  }

  auto work =
      c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(tensors, comm_);
  return work;
}

void ProcessGroupEagerHCCL::initComms() {
  comm_->getDeviceCtxt();
}

void Collective_Execute_Task(
    std::shared_ptr<habana::HcclCommunicator> comm_,
    std::unique_ptr<CollectiveContext> ctx,
    CollectiveFn&& fn,
    [[maybe_unused]] bool is_allreduce) {
  PT_DISTRIBUTED_DEBUG("Collective_Execute_Task");
  auto deviceCtxt = comm_->getDeviceCtxt();

  for (auto& input_output : ctx->tensors()) {
    at::Tensor& input = input_output.first;
    at::Tensor& output = input_output.second;

    if (input.numel() == 0) {
      // It is a W/A for SW-140597
      // When empty tensor is passed to collective op, its processing is
      // skipped.
      PT_DISTRIBUTED_DEBUG("Empty tensor, skipping collective");
      continue;
    }

    TORCH_CHECK(
        input.get_device() == 0 && output.get_device() == 0,
        "All tensors are expected to be assigned to device with id 0");
    synStreamHandle collective_stream = comm_->getCommStream();

    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)input.storage().data_ptr().get();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)output.storage().data_ptr().get();

    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    if (input_storage_ptr != output_storage_ptr) {
      deviceCtxt->prepare_stream(collective_stream, output_storage_ptr);
    }

    hcclResult_t hccl_result = hcclSuccess;
    auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

    struct ResourceHolder {
      std::vector<at::Tensor> tensors_;
      std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    };
    auto resource_holder = std::make_shared<ResourceHolder>();
    resource_holder->tensors_ = {input, output};

    void* input_address;
    void* output_address;
    deviceCtxt->lock_address(
        {input.data_ptr(), output.data_ptr()}, resource_holder->address_lock);
    input_address =
        reinterpret_cast<void*>(resource_holder->address_lock->at(0));
    HABANA_ASSERT(input_address != nullptr, "input_address is null");
    output_address =
        reinterpret_cast<void*>(resource_holder->address_lock->at(1));
    HABANA_ASSERT(output_address != nullptr, "output_address is null");

    hccl_result =
        fn(input,
           output,
           input_address,
           output_address,
           *(comm_->GetHcclHandle()),
           collective_stream);
    TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");

    recipe_counter.increase();
    deviceCtxt->submit_events(
        collective_stream,
        output_storage_ptr,
        [resource_holder, &recipe_counter]() mutable {
          resource_holder.reset();
          recipe_counter.decrease_and_notify();
        });
  }

  return;
}

void Collective_Empty_Compile_Task(
    std::shared_ptr<habana::HcclCommunicator> comm_,
    std::unique_ptr<CollectiveContext> ctx,
    CollectiveFn&& fn,
    bool is_allreduce) {
  PT_DISTRIBUTED_DEBUG("Collective_Empty_Compile_Task");
  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      Collective_Execute_Task,
      comm_,
      std::move(ctx),
      std::move(fn),
      is_allreduce);

  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
  }
}

void Collective_Empty_Lowering_Task(
    std::shared_ptr<habana::HcclCommunicator> comm_,
    std::unique_ptr<CollectiveContext> ctx,
    CollectiveFn&& fn,
    bool is_allreduce) {
  PT_DISTRIBUTED_DEBUG("Collective_Empty_Lowering_Task");
  habana_helpers::Singleton_CompileThreadPool::getInstance().Enqueue(
      Collective_Empty_Compile_Task,
      comm_,
      std::move(ctx),
      std::move(fn),
      is_allreduce);
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_CompileThreadPool::getInstance()
        .JoinPendingThread();
  }
}

void ProcessGroupEagerHCCL::groupStart() {
  initComms();
  TORCH_CHECK(
      hcclSuccess == hcclGroupStart(), "hcclGroupStart call returned error");
}

void ProcessGroupEagerHCCL::groupEnd() {
  TORCH_CHECK(
      hcclSuccess == hcclGroupEnd(), "hcclGroupEnd call returned error");
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    CollectiveFn fn,
    [[maybe_unused]] bool is_allreduce) {
  PT_DISTRIBUTED_BEGIN;
  PT_DISTRIBUTED_DEBUG("ProcessGroupEagerHCCL::collective");

  TORCH_CHECK(inputs.size() == outputs.size());
  const bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);

  // Update backend tensors if pipeline is enabled
  std::vector<at::Tensor> inputs_backend;
  std::vector<at::Tensor> outputs_backend;
  if (pipeline_flag) {
    inputs_backend.reserve(inputs.size());
    outputs_backend.reserve(outputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_backend.push_back(
          habana::eager::HbEagerTensorPool::get_backend_tensor(inputs[i]));
      outputs_backend.push_back(
          habana::eager::HbEagerTensorPool::get_backend_tensor(outputs[i]));

      // Set tensor pipeline metadata
      auto input_hb_tmeta{habana::get_tensor_extra_meta(inputs_backend.back())};
      auto output_hb_tmeta{
          habana::get_tensor_extra_meta(outputs_backend.back())};
      input_hb_tmeta->set_tensor_pipelined();
      output_hb_tmeta->set_tensor_pipelined();
    }
  }

  auto ctx = pipeline_flag
      ? std::make_unique<CollectiveContext>(
            inputs_backend, outputs_backend, true)
      : std::make_unique<CollectiveContext>(inputs, outputs);

  if (pipeline_flag) {
    const auto input_output_tensors = ctx->tensors();
    habana::eager::SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            Collective_Empty_Lowering_Task,
            comm_,
            std::move(ctx),
            std::move(fn),
            is_allreduce);
    // Restore the output tensors i.e. copy D2D in the main thread
    // So that such copies are also pipelined.
    // ToDo: Refactor it when handling pointToPoint collective pipeline.
    for (size_t i = 0; i < input_output_tensors.size(); ++i) {
      if (outputs_backend[i].unsafeGetTensorImpl() !=
          input_output_tensors[i].second.unsafeGetTensorImpl()) {
        PT_DISTRIBUTED_DEBUG(
            "restore output tensor ", habana::to_string(outputs_backend[i]));
        outputs_backend[i].copy_(input_output_tensors[i].second, true);
      }
    }

  } else {
    habana::eager::JoinPendingPipelineThreads();
    Collective_Execute_Task(comm_, std::move(ctx), std::move(fn), is_allreduce);
  }

  auto work =
      c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(outputs, comm_);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::barrier(
    const BarrierOptions& opts [[maybe_unused]]) {
  PT_DISTRIBUTED_BEGIN;
  hostBarrier();
  PT_DISTRIBUTED_END;
  return c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>();
};

// Sending a tensor doesn't have metadata field, hence we can't send the info if
// tensor is dense or permuted. So for first functional step, we'll always
// permute it back to be dnese before sending it. In future it can be optimized
// if we can send metadata too via send mechanism to provide this info.
void ProcessGroupEagerHCCL::permutedSendTensorsToDense(
    std::vector<at::Tensor>& tensors) {
  for (auto& tensor : tensors) {
    synapse_helpers::layouts::MemoryPermutation permutation;
    std::tie(permutation, std::ignore) =
        habana_helpers::get_tensor_memory_permutation(tensor);
    if (!permutation.empty()) {
      auto t_meta{habana::get_tensor_extra_meta(tensor)};
      PT_DISTRIBUTED_DEBUG(
          "Tensor: ",
          t_meta->get_id(),
          " has permutation: ",
          VecToString(permutation),
          " transposing it back to be dense");
      tensor = torch::clone(tensor);
    }
  }
}

// When recieving a tensor we make sure during send it's dense.
// So once we recive a tensor, we clear it's permutation info.
void ProcessGroupEagerHCCL::clearPermutesFromRecvTensors(
    std::vector<at::Tensor>& tensors) {
  for (auto& tensor : tensors) {
    auto s_meta{habana::get_storage_extra_meta(tensor)};
    if (s_meta) {
      auto t_meta{habana::get_tensor_extra_meta(tensor)};
      PT_DISTRIBUTED_DEBUG(
          "Received tensor: ", t_meta->get_id(), " clearing its permutation.");
      s_meta->set_memory_permutation({});
    }
  }
}

} // namespace c10d

namespace py = pybind11;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::object backend =
      py::module_::import("torch.distributed").attr("_Backend");
  intrusive_ptr_class_<::c10d::ProcessGroupEagerHCCL> processGroupHccl(
      module, "ProcessGroupHCCL", backend);

  processGroupHccl.def(py::init(
      &c10d::ProcessGroupHCCLRegistry<c10d::ProcessGroupEagerHCCL>::create));

  py::cpp_function cleanup = []() {
    {
      // Flushing all streams in order to ensure that all events have been
      // handled (all tensors connected with pending events are deallocated)
      // before Python interpreter finalization. If tensor is deallocated when
      // interpreter is down or is going down (finalizing) then cPython may
      // issue std::terminate (abort), what will be observed in DFA report.

      auto gil_release = pybind11::gil_scoped_release();

      const int hccl_comms_num = habana::HcclCommunicator::Count();
      PT_DISTRIBUTED_DEBUG("PG cleanup: HCCL comms count: ", hccl_comms_num);

      for (int hccl_comm_id = 0; hccl_comm_id < hccl_comms_num;
           hccl_comm_id++) {
        std::shared_ptr<habana::HcclCommunicator> hccl_comm =
            habana::HcclCommunicator::Get(hccl_comm_id);

        if (hccl_comm) {
          PT_DISTRIBUTED_DEBUG(
              "PG cleanup: flushing HCCL comm with id=", hccl_comm_id);
          hccl_comm->flush_stream();
        } else {
          PT_DISTRIBUTED_DEBUG(
              "PG cleanup: HCCL comm with given id has been already destroyed, id=",
              hccl_comm_id);
        }
      }
    }

    // Destroying default PG when it hasn't been destroyed by user.

    py::object dist = py::module_::import("torch.distributed");
    py::object destroy_process_group = dist.attr("destroy_process_group");
    py::object default_pg = dist.attr("GroupMember").attr("WORLD");
    if (!default_pg.is(py::none())) {
      PT_DISTRIBUTED_DEBUG("Destroying process groups at exit");
      destroy_process_group();
    }
  };
  py::module::import("atexit").attr("register")(cleanup);
};
