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
#include "backend/helpers/collective_utils.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "habana_eager/eager_context.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_hccl_base.hpp"
#include "pytorch_helpers/habana_helpers/python_utils.h"

namespace c10d {

namespace {
class CollectiveContext {
 public:
  CollectiveContext(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs)
      : inputs_(inputs), outputs_(outputs) {
    TORCH_CHECK(inputs.size() == outputs.size());
    ensure_input_output_tensors_contiguity();
  }
  CollectiveContext(const CollectiveContext&) = delete;
  CollectiveContext& operator=(CollectiveContext&) = delete;

  ~CollectiveContext() {
    restore_output_tensors_if_needed();
  }

  CollectiveContext(std::vector<at::Tensor>& tensors)
      : CollectiveContext(tensors, tensors) {}

  std::vector<std::pair<at::Tensor, at::Tensor>>& tensors() {
    return in_out_tensors_contiguous_;
  }

 private:
  std::vector<std::pair<at::Tensor, at::Tensor>> in_out_tensors_contiguous_;
  std::vector<at::Tensor>& inputs_;
  std::vector<at::Tensor>& outputs_;

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
      at::Tensor input_contiguous = inputs_[i].contiguous();
      at::Tensor output_contiguous;
      if (inputs_[i].unsafeGetTensorImpl() ==
          outputs_[i].unsafeGetTensorImpl()) {
        // inplace collective: output == intput
        output_contiguous = input_contiguous;
      } else {
        // create tensor copy in case if provided output is not contiguous
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
        outputs_[i].copy_(in_out_tensors_contiguous_[i].second, true);
      }
    }
  }
};
} // namespace

ProcessGroupEagerHCCL::ProcessGroupEagerHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroupHcclBase(store, rank, size) {
  PT_EAGER_DEBUG(
      "Create ProcessGroupEagerHCCL, rank = ", rank, " size = ", size);
  always_support_int64_ = true;
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

void ProcessGroupEagerHCCL::destroy() {
  if (comm_) {
    // Consider adding additional suffix based on process group identifier
    // in case of multi process group scenarios
    std::string barrier_key = std::string("ProcessGroupEagerHCCL::destroy");
    auto worker_count = store_->add(barrier_key, 1);
    if (getRank() == 0) {
      while (worker_count != size_) {
        worker_count = store_->add(barrier_key, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    habana_helpers::AutoNoGIL gil_release;
    comm_->flush_stream();
    comm_.reset();
  }
}

ProcessGroupEagerHCCL::~ProcessGroupEagerHCCL() {
  PT_DISTRIBUTED_BEGIN;
  destroy();
  PT_DISTRIBUTED_END;
};

ProcessGroupEagerHCCL::WorkEager::WorkEager(
    const std::vector<at::Tensor>& outputs)
    : future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted(at::IValue(outputs));
}

ProcessGroupEagerHCCL::WorkEager::WorkEager()
    : future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {
  future_->markCompleted();
}

ProcessGroupEagerHCCL::WorkEager::~WorkEager() {}

bool ProcessGroupEagerHCCL::WorkEager::isCompleted() {
  return true;
}

bool ProcessGroupEagerHCCL::WorkEager::isSuccess() const {
  return true;
}

bool ProcessGroupEagerHCCL::WorkEager::wait(std::chrono::milliseconds timeout
                                            [[maybe_unused]]) {
  return true;
}

void ProcessGroupEagerHCCL::WorkEager::abort() {
  HABANA_ASSERT(false, __FUNCTION__, " not implemented");
}

void ProcessGroupEagerHCCL::WorkEager::synchronize() {}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupEagerHCCL::WorkEager::
    getFuture() {
  return future_;
};

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    PointToPointFn fn,
    int peerRank) {
  CollectiveContext collective_ctx(tensors);

  habana::eager::JoinPendingPipelineThreads();

  for (auto& input_output : collective_ctx.tensors()) {
    at::Tensor& tensor = input_output.first;
    TORCH_CHECK(
        tensor.get_device() == 0,
        "All tensors are expected to be assigned to device with id 0");
    auto deviceCtxt = comm_->getDeviceCtxt();
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

  auto work = c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(tensors);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    CollectiveFn fn,
    [[maybe_unused]] bool is_allreduce) {
  TORCH_CHECK(
      inputs.size() == outputs.size(),
      "Number of inputs has to be the same as num of outputs");

  CollectiveContext collective_ctx(inputs, outputs);

  habana::eager::JoinPendingPipelineThreads();

  for (auto& input_output : collective_ctx.tensors()) {
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
    auto deviceCtxt = comm_->getDeviceCtxt();
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

  auto work = c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(outputs);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::barrier(
    const BarrierOptions& opts [[maybe_unused]]) {
  PT_DISTRIBUTED_BEGIN;
  hostBarrier();

  if (comm_->GetHcclHandle() != nullptr) {
    synStreamHandle collective_stream = comm_->getCommStream();
    hcclBarrier(*comm_->GetHcclHandle(), collective_stream);
  }

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

class ProcessGroupEagerHCCLRegistry {
 public:
  using intrptr_t = c10::intrusive_ptr<::c10d::ProcessGroupEagerHCCL>;
  using weakptr_t = c10::weak_intrusive_ptr<::c10d::ProcessGroupEagerHCCL>;
  static ProcessGroupEagerHCCLRegistry& instance() {
    std::call_once(init_once_flag_, []() {
      instance_.reset(new ProcessGroupEagerHCCLRegistry());
      habana::HPURegistrar::get_hpu_registrar()
          .register_process_group_finalizer(habana::CallFinally([]() {
            PT_DISTRIBUTED_DEBUG("ProcessGroupEagerHCCLRegistry finalizer");
            instance_.reset();
          }));
    });

    return *instance_;
  }

  static intrptr_t create(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size) {
    intrptr_t r{};
    r = r.make(store, rank, size);
    ProcessGroupEagerHCCLRegistry::instance().insert(r);
    return r;
  }

  ~ProcessGroupEagerHCCLRegistry() {
    cleanup();
  }

 private:
  std::vector<weakptr_t> groups_;
  std::once_flag job_thread_once_flag_;
  std::mutex groups_mutex_;
  static std::once_flag init_once_flag_;
  static std::unique_ptr<ProcessGroupEagerHCCLRegistry> instance_;

  void cleanup() {
    PT_DISTRIBUTED_DEBUG("Clearing distributed process groups");
    size_t count{0};
    {
      std::unique_lock<std::mutex> lock{groups_mutex_};
      for (auto& wpg : groups_) {
        auto wp{wpg.lock()};
        if (wp) {
          wp->destroy();
          ++count;
        }
      }
    }
    PT_DISTRIBUTED_DEBUG(
        "Cleared ",
        count,
        " remaining distributed process groups out of ",
        groups_.size());
  }

  void insert(intrptr_t& new_pg) {
    std::unique_lock<std::mutex> lock{groups_mutex_};
    for (auto& wpg : groups_) {
      auto wp{wpg.lock()};
      if (!wp) {
        wp = new_pg;
        return;
      }
    }
    groups_.push_back(weakptr_t(new_pg));
  }
};

std::once_flag ProcessGroupEagerHCCLRegistry::init_once_flag_;
std::unique_ptr<ProcessGroupEagerHCCLRegistry>
    ProcessGroupEagerHCCLRegistry::instance_;

namespace py = pybind11;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  intrusive_ptr_class_<::c10d::ProcessGroupEagerHCCL, c10d::ProcessGroup>
      processGroupHccl(module, "ProcessGroupHCCL");

  processGroupHccl.def(py::init(&ProcessGroupEagerHCCLRegistry::create));

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
