/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "process_group_eager_hccl.hpp"
#include "habana_eager/ops/eager_op.h"

#include <absl/functional/any_invocable.h>
#include <c10/core/TensorImpl.h>
#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <utility>
#include <vector>

#include <unistd.h>
#include "backend/synapse_helpers/hccl_communicator.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/towl.h"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_hccl_base.hpp"
#include "pytorch_helpers/habana_helpers/python_utils.h"

#include "hpu_ops/op_logger.h"
#include "process_group_registry.hpp"

#include "backend/helpers/generic_resource_holder.h"

namespace c10d {

namespace {
inline void restore_output_tensors(
    const std::vector<std::pair<at::Tensor, at::Tensor>>&
        in_out_tensors_contiguous,
    const std::vector<at::Tensor>& outputs) {
  for (size_t i = 0; i < in_out_tensors_contiguous.size(); ++i) {
    if (outputs[i].unsafeGetTensorImpl() !=
        in_out_tensors_contiguous[i].second.unsafeGetTensorImpl()) {
      // provided output wasn't contiguous, so the result of collective is
      // stored in tensors_contiguous[i]
      PT_DISTRIBUTED_DEBUG(
          "restore output tensor ", habana::to_string(outputs[i]));
      outputs[i].copy_(in_out_tensors_contiguous[i].second, true);
    }
  }
}

class CollectiveContext {
 public:
  CollectiveContext(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      bool is_pipelined = false)
      : inputs_(inputs), outputs_(outputs), is_pipelined_(is_pipelined) {
    HABANA_ASSERT(inputs.size() == outputs.size());
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

  CollectiveContext(std::vector<at::Tensor>& tensors, bool is_pipelined = false)
      : CollectiveContext(tensors, tensors, is_pipelined) {}

  std::vector<std::pair<at::Tensor, at::Tensor>>& tensors() {
    return in_out_tensors_contiguous_;
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
    restore_output_tensors(in_out_tensors_contiguous_, outputs_);
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
          HABANA_ASSERT(vec.size() == sizeof(hcclUniqueId));
          std::memcpy(hcclID, vec.data(), vec.size());
        }
      });
};

// Abort all communicators on this rank
bool ProcessGroupEagerHCCL::abort(std::optional<std::string> abortReason) {
  PT_DISTRIBUTED_DEBUG(
      "Launching ProcessGroupEagerHCCL abort asynchrounously. Abort Reason: ",
      abortReason.value_or(""));
  PT_DISTRIBUTED_DEBUG("hcclCommAbort initiated  commId:", comm_.get());
  // Note: HCCL doesnt support abort operation. Once Hccl Supports, This can be
  // enabled
  //  comm_->hcclCommAbort(abortReason);

  return true;
}

void ProcessGroupEagerHCCL::shutdown(std::optional<std::string> reason) {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.

  // lauch abort asynchrounously and wait for it to complete or timeout
  PT_DISTRIBUTED_DEBUG(
      "Launching ProcessGroupEagerHCCL abort asynchrounously.");

  std::ignore = std::async(
      std::launch::async, [this, &reason]() { return this->abort(reason); });

  PT_DISTRIBUTED_DEBUG("ProcessGroupEagerHCCL aborts successfully.");
}

void ProcessGroupEagerHCCL::destroy() {
  if (is_destroyed_)
    return;
  PT_DISTRIBUTED_DEBUG(
      "Destroy ProcessGroupEagerHCCL name:",
      group_name_,
      ", size:",
      size_,
      ", rank:",
      rank_);

  // Synchronization is needed in case where execute thread have outstanding
  // tasks. In such case destroy may came before tasks are processed. Due to
  // that segfault can be observed because comm_ is destroyed here and passed to
  // task as copy, so we have to make sure that outstanding tasks are processed
  // before destroy. Maybe there's other way to ensure synchronization because
  // now we're blocking main thread.
  habana::eager::JoinPendingPipelineThreads();
  hostBarrier();

  if (comm_) {
    habana_helpers::AutoNoGIL gil_release;
    comm_->flush_stream();
    comm_.reset();
  }

  destroyHandshake();
  is_destroyed_ = true;
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
      future_(
          c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted(at::IValue(outputs));
}

ProcessGroupEagerHCCL::WorkEager::WorkEager()
    : future_(
          c10::make_intrusive<at::ivalue::Future>(
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
  if (outputs_.empty()) {
    return;
  }

  if (auto comm_sp = comm_.lock()) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      PT_DISTRIBUTED_DEBUG("WorkEager::synchronize()");
      comm_sp->getDeviceCtxt()->synchronize_output(
          (synapse_helpers::device_ptr)outputs_[i].storage().data_ptr().get(),
          (c10::hpu::getCurrentHPUStream()).stream());
    }
  } else {
    HABANA_ASSERT(
        comm_sp,
        "Trying to synchronize HCCL task with uninitialized HCCLCommunicator");
  }
  outputs_.clear();
}

void Synchronize_Execute_Task(
    const std::vector<at::Tensor>& outputs,
    std::weak_ptr<habana::HcclCommunicator> comm,
    synapse_helpers::hpuStream_t stream) {
  PT_DISTRIBUTED_DEBUG("Synchronize_Execute_Task");
  if (outputs.empty()) {
    return;
  }

  if (auto comm_sp = comm.lock()) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      // Check if the tensor metadata send org tensor is available.
      // Use the org tensor address for synchronize.
      // Note: This org tensor as a tensor metadata can be set if
      // the previous OP is P2P send i.e. permutedSendTensorsToDense.
      auto output_address = outputs[i].storage().data_ptr().get();
      auto tensor_tmeta{habana::get_tensor_extra_meta(outputs[i])};
      if (auto org_tensor = tensor_tmeta->get_send_org_tensor()) {
        output_address = org_tensor->storage().data_ptr().get();
      }
      comm_sp->getDeviceCtxt()->synchronize_output(
          (synapse_helpers::device_ptr)output_address, stream);
    }
  } else {
    HABANA_ASSERT(
        comm_sp,
        "Trying to synchronize HCCL task with uninitialized HCCLCommunicator");
  }
}

bool ProcessGroupEagerHCCL::WorkEager::wait(
    std::chrono::milliseconds timeout [[maybe_unused]]) {
  if (is_coalescing_fn_()) {
    PT_DISTRIBUTED_DEBUG(
        "WorkEager::wait | Skip, because work is within group, wait have no effect as operation won't start before groupEnd will be called.");
    return false;
  }

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

    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [outputs_backend = std::move(outputs_backend),
         comm = comm_,
         stream = (c10::hpu::getCurrentHPUStream()).stream()]() {
          Synchronize_Execute_Task(outputs_backend, comm, stream);
        });
  } else {
    habana::eager::JoinPendingPipelineThreads();
    Synchronize_Execute_Task(
        outputs_, comm_, (c10::hpu::getCurrentHPUStream()).stream());
  }
  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
  outputs_.clear();
  // NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)
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

void PointToPoint_Execute_Task(
    habana::HcclCommunicator& comm,
    std::unique_ptr<CollectiveContext> ctx,
    PointToPointFn&& fn,
    int peerRank,
    bool is_coalescing,
    std::vector<absl::AnyInvocable<void()>>& group_submit_events_tasks_queue) {
  auto deviceCtxt = comm.getDeviceCtxt();
  for (auto& input_output : ctx->tensors()) {
    at::Tensor& tensor = input_output.first;
    // Check if the tensor metadata send org tensor is available.
    // Use the org tensor for registering events and send it to NIC.
    // Note: This org tensor as a tensor metadata can be set if
    // the previous OP is P2P send i.e. permutedSendTensorsToDense.
    auto tensor_tmeta{habana::get_tensor_extra_meta(tensor)};
    if (auto org_tensor = tensor_tmeta->get_send_org_tensor()) {
      tensor = *org_tensor;
    }
    HABANA_ASSERT(
        tensor.get_device() == 0,
        "All tensors are expected to be assigned to device with id 0");
    synStreamHandle collective_stream = comm.getCommStream();

    auto tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensor.storage().data_ptr().get();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);

    auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

    auto resource_holder = std::make_shared<GenericResourceHolder>();
    if (!(common::IsRecordStreamEnabled() &&
          GET_ENV_FLAG_NEW(PT_HPU_USE_LAUNCH_RECORD_STREAM))) {
      resource_holder->add_tensor(tensor);
    } else {
      auto& device = habana::HPUDeviceContext::get_device();
      synapse_helpers::hpuStream_t hpu_stream;
      deviceCtxt->get_hpu_stream(collective_stream, &hpu_stream);

      void* data_ptr = tensor.data_ptr();
      device.get_device_memory().recordStream(data_ptr, hpu_stream);
    }

    void* tensor_address;
    deviceCtxt->lock_address(
        tensor.data_ptr(),
        &tensor_address,
        resource_holder->get_address_lock());

    hcclResult_t hccl_result =
        fn(tensor,
           tensor_address,
           *(comm.GetHcclHandle()),
           collective_stream,
           peerRank);
    HABANA_ASSERT(hcclSuccess == hccl_result, "P2P call returned error");

    recipe_counter.increase();
    auto _submit_events_task = [deviceCtxt,
                                collective_stream,
                                tensor_storage_ptr,
                                resource_holder,
                                &recipe_counter]() mutable {
      deviceCtxt->submit_events(
          collective_stream,
          tensor_storage_ptr,
          [resource_holder, &recipe_counter]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
          });
    };

    if (is_coalescing) {
      PT_DISTRIBUTED_DEBUG(
          "HCCL point_to_point call within group, postpone submit events after groupEnd");
      // Postponing submit events is needed because calls within group are
      // batched and run at groupEnd, due to that submit events before groupEnd
      // leads to problem with synchronization due to missing events.
      group_submit_events_tasks_queue.emplace_back(_submit_events_task);
    } else {
      _submit_events_task();
    }
  }
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    PointToPointFn fn,
    int peerRank) {
  PT_DISTRIBUTED_BEGIN;
  PT_DISTRIBUTED_DEBUG("ProcessGroupEagerHCCL::pointToPoint");

  const bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);

  // Update backend tensors if pipeline is enabled
  std::vector<at::Tensor> tensors_backend;
  if (pipeline_flag) {
    tensors_backend.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      tensors_backend.push_back(
          habana::eager::HbEagerTensorPool::get_backend_tensor(tensors[i]));

      // Set tensor pipeline metadata
      auto tensor_hb_tmeta{
          habana::get_tensor_extra_meta(tensors_backend.back())};
      tensor_hb_tmeta->set_tensor_pipelined();
    }
  }

  auto ctx = pipeline_flag
      ? std::make_unique<CollectiveContext>(tensors_backend, true)
      : std::make_unique<CollectiveContext>(tensors);

  if (pipeline_flag) {
    const auto tensors = ctx->tensors();
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [ctx = std::move(ctx),
         fn = std::move(fn),
         peerRank,
         coalescing_state = coalescing_state_,
         this]() mutable {
          PointToPoint_Execute_Task(
              *comm_,
              std::move(ctx),
              std::move(fn),
              peerRank,
              coalescing_state,
              group_submit_events_tasks_queue_);
        });
    // Restore the output tensors i.e. copy D2D in the main thread
    // So that such copies are also pipelined.
    restore_output_tensors(tensors, tensors_backend);
  } else {
    habana::eager::JoinPendingPipelineThreads();
    PointToPoint_Execute_Task(
        *comm_,
        std::move(ctx),
        std::move(fn),
        peerRank,
        coalescing_state_,
        group_submit_events_tasks_queue_);
  }

  auto work =
      c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(tensors, comm_);
  work->is_coalescing_fn_ = [this]() -> bool { return coalescing_state_; };
  return work;
}

void ProcessGroupEagerHCCL::initComms() {
  comm_->getDeviceCtxt();
}

void Collective_Execute_Task(
    habana::HcclCommunicator& comm,
    std::unique_ptr<CollectiveContext> ctx,
    CollectiveFn&& fn,
    [[maybe_unused]] bool is_allreduce,
    bool is_coalescing,
    std::vector<absl::AnyInvocable<void()>>& group_submit_events_tasks_queue) {
  PT_DISTRIBUTED_DEBUG("Collective_Execute_Task");
  auto deviceCtxt = comm.getDeviceCtxt();

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

    HABANA_ASSERT(
        input.get_device() == 0 && output.get_device() == 0,
        "All tensors are expected to be assigned to device with id 0");
    synStreamHandle collective_stream = comm.getCommStream();

    auto input_storage_ptr =
        (synapse_helpers::device_ptr)input.storage().data_ptr().get();
    auto output_storage_ptr =
        (synapse_helpers::device_ptr)output.storage().data_ptr().get();

    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    if (input_storage_ptr != output_storage_ptr) {
      deviceCtxt->prepare_stream(collective_stream, output_storage_ptr);
    }

    hcclResult_t hccl_result = hcclSuccess;
    auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

    auto resource_holder = std::make_shared<GenericResourceHolder>();
    if (!(common::IsRecordStreamEnabled() &&
          GET_ENV_FLAG_NEW(PT_HPU_USE_LAUNCH_RECORD_STREAM))) {
      resource_holder->add_tensor(input);
      resource_holder->add_tensor(output);
    } else {
      auto& device = habana::HPUDeviceContext::get_device();
      synapse_helpers::hpuStream_t hpu_stream;
      deviceCtxt->get_hpu_stream(collective_stream, &hpu_stream);

      void* in_data_ptr = input.data_ptr();
      void* out_data_ptr = output.data_ptr();
      device.get_device_memory().recordStream(in_data_ptr, hpu_stream);
      device.get_device_memory().recordStream(out_data_ptr, hpu_stream);
    }

    void* input_address;
    void* output_address;
    deviceCtxt->lock_address(
        {input.data_ptr(), output.data_ptr()},
        resource_holder->get_address_lock());
    input_address =
        reinterpret_cast<void*>(resource_holder->get_address_lock()->at(0));
    HABANA_ASSERT(input_address != nullptr, "input_address is null");
    output_address =
        reinterpret_cast<void*>(resource_holder->get_address_lock()->at(1));
    HABANA_ASSERT(output_address != nullptr, "output_address is null");

    hccl_result =
        fn(input,
           output,
           input_address,
           output_address,
           *(comm.GetHcclHandle()),
           collective_stream);
    HABANA_ASSERT(hcclSuccess == hccl_result, "Collective call returned error");

    recipe_counter.increase();
    towl::emitCollectiveLaunch("eager");

    auto _submit_events_task = [deviceCtxt,
                                collective_stream,
                                output_storage_ptr,
                                resource_holder,
                                &recipe_counter]() mutable {
      deviceCtxt->submit_events(
          collective_stream,
          output_storage_ptr,
          [resource_holder, &recipe_counter]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
            towl::emitCollectiveFinished("eager");
          });

      if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_SYNC)) {
        // each rank should wait `output_storage_ptr` mapping shared_event done
        deviceCtxt->wait_until_address_ready(output_storage_ptr);
      }
    };

    if (is_coalescing) {
      PT_DISTRIBUTED_DEBUG(
          "HCCL collective call within group, postpone submit events after groupEnd");
      // Postponing submit events is needed because calls within group are
      // batched and run at groupEnd, due to that submit events before groupEnd
      // leads to problem with synchronization due to missing events.
      group_submit_events_tasks_queue.emplace_back(_submit_events_task);
    } else {
      _submit_events_task();
    }
  }
}

void ProcessGroupEagerHCCL::groupStart() {
  auto _groupStart = [this]() {
    initComms();
    HABANA_ASSERT(
        hcclSuccess == hcclGroupStart(), "hcclGroupStart call returned error");
    group_submit_events_tasks_queue_.clear();
  };

  const bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);

  if (pipeline_flag) {
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        _groupStart);
  } else {
    habana::eager::JoinPendingPipelineThreads();
    _groupStart();
  }
}

void ProcessGroupEagerHCCL::groupEnd() {
  auto _groupEnd = [this]() {
    HABANA_ASSERT(
        hcclSuccess == hcclGroupEnd(), "hcclGroupEnd call returned error");
    PT_DISTRIBUTED_DEBUG(
        "Calling postponed ",
        group_submit_events_tasks_queue_.size(),
        " submit events tasks after groupEnd");

    // This can be optimized to submit one event with all addresses
    // TODO: [SW-223384]
    for (auto& submit_event_task : group_submit_events_tasks_queue_) {
      submit_event_task();
    }
  };

  const bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);

  if (pipeline_flag) {
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(_groupEnd);
  } else {
    habana::eager::JoinPendingPipelineThreads();
    _groupEnd();
  }
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    CollectiveFn fn,
    [[maybe_unused]] bool is_allreduce) {
  PT_DISTRIBUTED_BEGIN;
  PT_DISTRIBUTED_DEBUG("ProcessGroupEagerHCCL::collective");

  HABANA_ASSERT(inputs.size() == outputs.size());
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
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [ctx = std::move(ctx),
         fn = std::move(fn),
         is_allreduce,
         coalescing_state = coalescing_state_,
         this]() mutable {
          Collective_Execute_Task(
              *comm_,
              std::move(ctx),
              std::move(fn),
              is_allreduce,
              coalescing_state,
              group_submit_events_tasks_queue_);
        });
    // Restore the output tensors i.e. copy D2D in the main thread
    // So that such copies are also pipelined.
    restore_output_tensors(input_output_tensors, outputs_backend);
  } else {
    habana::eager::JoinPendingPipelineThreads();
    Collective_Execute_Task(
        *comm_,
        std::move(ctx),
        std::move(fn),
        is_allreduce,
        coalescing_state_,
        group_submit_events_tasks_queue_);
  }

  auto work =
      c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(outputs, comm_);
  work->is_coalescing_fn_ = [this]() -> bool { return coalescing_state_; };
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
  std::vector<at::Tensor> clone_tensors;
  clone_tensors.reserve(tensors.size());

  // Allocate memory for clone tensors and get backend tensors
  std::vector<std::pair<at::Tensor, at::Tensor>> tensors_backend;
  tensors_backend.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // Allocate memory for cloned tensors
    clone_tensors.push_back(
        at::empty_like(
            tensor,
            tensor.options().device(tensor.device()),
            c10::MemoryFormat::Contiguous));

    // Get backend tensors
    tensors_backend.emplace_back(
        habana::eager::HbEagerTensorPool::get_backend_tensor(tensor),
        habana::eager::HbEagerTensorPool::get_backend_tensor(
            clone_tensors.back()));

    // Set tensor pipeline metadata
    auto tensor_hb_tmeta{
        habana::get_tensor_extra_meta(tensors_backend.back().first)};
    tensor_hb_tmeta->set_tensor_pipelined();
    auto clone_tensor_hb_tmeta{
        habana::get_tensor_extra_meta(tensors_backend.back().second)};
    clone_tensor_hb_tmeta->set_tensor_pipelined();
  }

  auto pipeline_or_direct_send_permutes = [tensors_pair =
                                               std::move(tensors_backend)]() {
    for (auto&& tensor_pair : tensors_pair) {
      auto&& send_tensor = tensor_pair.first;
      auto&& clone_tensor = tensor_pair.second;
      synapse_helpers::layouts::MemoryPermutation permutation;
      std::tie(permutation, std::ignore) =
          habana_helpers::get_tensor_memory_permutation(send_tensor);
      PT_DISTRIBUTED_DEBUG("Send: permutation: ", VecToString(permutation));
      const bool is_permuted_or_non_contiguous =
          !permutation.empty() || !send_tensor.is_contiguous();

      /*
       * Get send tensor permutations (current op lowering stage).
       * Set send org tensor as a metadata to the clone tensor.
       * In the next op, i.e. copy send tensor to the clone tensor.
       * If (permutation or strided tensor)
       *   This copy clears the permutation/strides on the cloned tensor.
       * Else
       *   Copy op is discarded at its lowering stage.
       *
       * In the pipeline stage, a clone tensor is used, which contains
       * org send tensor in the metadata. The idea is to use the
       * clone tensor in case the org send tensor has permutation set.
       *
       * If there is no permutation/strides, then an org send tensor is used.
       * Further, this metadata can be used to discard the next D2D copy op
       * since clone tensor is not required and to avoid unnecessary copy
       *
       * This metadata is queried in the later pipeline stages to select
       * either clone tensor. or org send tensor for registering events on
       * collective stream/user streams.
       *
       * To avoid data race, this is the sequence of tensor metadata used.
       * Write in current op lowering and Read in next op lowering/execute.
       * Current Op Lowering: Set Send Org Metadata on the clone tensor
       * Next Op Copy D2D Lowering: Get MetaData or Tensor Permutation
       * Next Op P2P collective Op Execute: Get MetaData
       * Next Op Work Wait Execute: Get MetaData
       */

      auto clone_tensor_hb_tmeta{habana::get_tensor_extra_meta(clone_tensor)};
      clone_tensor_hb_tmeta->set_send_org_tensor_meta(
          is_permuted_or_non_contiguous, send_tensor);
    }
  };

  habana::eager::PipelineOrExecuteTask(
      std::move(pipeline_or_direct_send_permutes));

  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
  for (size_t i = 0; i < tensors.size(); i++) {
    // Copy D2D, if any permutation/strides It will be cleared
    // If No permutation/strides, This op will be discarded at its lowering.
    constexpr int num_outputs = 1;
    constexpr bool skip_lowering = true;
    habana::eager::EagerOp<at::Tensor&> hpu_op{
        "hpu::_copy_from",
        {tensors[i], clone_tensors[i]},
        {clone_tensors[i].sizes().vec()},
        num_outputs};
    hpu_op.set_eager_op_info(
        {habana::eager::eagerOpKind::InplaceOut,
         "hpu::_copy_from",
         num_outputs,
         skip_lowering});
    tensors[i] = hpu_op.call(const_cast<at::Tensor&>(clone_tensors[i]));
  }
}

// When recieving a tensor we make sure during send it's dense.
// So once we recive a tensor, we clear it's permutation info.
void ProcessGroupEagerHCCL::clearPermutesFromRecvTensors(
    std::vector<at::Tensor>& tensors) {
  // Get backend tensors
  std::vector<at::Tensor> tensors_backend;
  tensors_backend.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors_backend.push_back(
        habana::eager::HbEagerTensorPool::get_backend_tensor(tensors[i]));
    // Set tensor pipeline metadata
    auto tensor_hb_tmeta{habana::get_tensor_extra_meta(tensors_backend.back())};
    tensor_hb_tmeta->set_tensor_pipelined();
  }

  auto pipeline_or_direct_clear_permutes = [tensors =
                                                std::move(tensors_backend)]() {
    for (auto&& tensor : tensors) {
      auto s_meta{habana::get_storage_extra_meta(tensor)};
      if (s_meta) {
        auto t_meta{habana::get_tensor_extra_meta(tensor)};
        PT_DISTRIBUTED_DEBUG(
            "Receive: tensor: ",
            t_meta->get_id(),
            " clearing its permutation.");
        s_meta->set_memory_permutation({});
      }
    }
  };
  habana::eager::PipelineOrExecuteTask(
      std::move(pipeline_or_direct_clear_permutes));
}
// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

} // namespace c10d

namespace py = pybind11;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::object backend =
      py::module_::import("torch.distributed").attr("_Backend");
  intrusive_ptr_class_<::c10d::ProcessGroupEagerHCCL> processGroupHccl(
      module, "ProcessGroupHCCL", backend);

  processGroupHccl.def(
      py::init(&c10d::ProcessGroupHCCLRegistry<
               c10d::ProcessGroupEagerHCCL>::create));

  processGroupHccl.def(
      "_shutdown",
      [](const c10::intrusive_ptr<::c10d::ProcessGroupEagerHCCL>& self) {
        self->shutdown(std::nullopt);
      });
};
