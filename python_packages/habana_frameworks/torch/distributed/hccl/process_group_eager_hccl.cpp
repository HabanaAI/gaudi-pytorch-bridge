/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "process_group_eager_hccl.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include "backend/helpers/collective_utils.h"
#include "habana_eager/eager_context.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"

namespace c10d {

ProcessGroupEagerHCCL::ProcessGroupEagerHCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout)
    : ProcessGroup(rank, size), store_(store), barrier_cnt_(0) {
  PT_EAGER_DEBUG(
      "Create ProcessGroupEagerHCCL, rank = ", rank, " size = ", size);
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

ProcessGroupEagerHCCL::~ProcessGroupEagerHCCL() {
  PT_LAZY_DEBUG("Destroy ProcessGroupEagerHCCL");
  comm_.reset();
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

bool ProcessGroupEagerHCCL::WorkEager::wait(std::chrono::milliseconds timeout) {
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

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    bool is_allreduce) {
  auto work = c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>(outputs);

  for (size_t i = 0; i < inputs.size(); ++i) {
    at::Tensor& input = inputs[i];
    at::Tensor& output = outputs[i];

    if (input.numel() == 0) {
      // It is a W/A for SW-140597
      // When empty tensor is passed to collective op, its processing is
      // skipped.
      PT_DISTRIBUTED_DEBUG("Empty tensor, skipping collective");
      continue;
    }

    auto device = input.get_device();
    auto deviceCtxt = comm_->getDeviceCtxt(device);
    synStreamHandle collective_stream = comm_->getCommStream(device);

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

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    bool is_allreduce) {
  // Need to replace int by device work streams
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<int>&) {},
      [](std::vector<int>&) {},
      is_allreduce);
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();

  auto outputFlattened = habana_helpers::flatten_for_scatter_gather(
      outputTensors, inputTensors, size_);

  auto work = collective(
      inputTensors,
      outputFlattened,
      [&](at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        auto scalar_type = input.scalar_type();
        auto hccl_numel = input.numel();
        hcclDataType_t hccl_data_type;
        habana_helpers::getCountDatatype(
            scalar_type,
            input.element_size(),
            hccl_numel,
            hccl_data_type,
            true);
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] allgather with input_address=",
            send_buffer,
            " output_address=",
            recv_buffer,
            " numel=",
            input.numel(),
            " scalar_type=",
            input.scalar_type(),
            " element_size=",
            input.element_size(),
            " hccl_type=",
            hccl_data_type,
            " hccl_count=",
            hccl_numel);
        hcclResult_t hccl_result{hcclSuccess};
        hccl_result = hcclAllGather(
            send_buffer,
            recv_buffer,
            hccl_numel,
            hccl_data_type,
            hccl_comm,
            stream);
        return hccl_result;
      });
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    for (size_t j = 0; j < outputTensors[0].size(); ++j) {
      outputTensors[i][j].copy_(outputFlattened[i][j], true);
    }
  }
  PT_DISTRIBUTED_END;
  return work;
};

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();

  auto work = collective(
      tensors,
      tensors,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size = c10::elementSize(
            habana_helpers::getInternalDtype(input.scalar_type()));
        size_t chunk_size = habana_helpers::getHCCLSliceSize(
                                habana_helpers::collectiveAllReduce) /
            element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          const void* offseted_send_buffer = reinterpret_cast<const void*>(
              reinterpret_cast<const char*>(send_buffer) + data_offset);
          void* offseted_recv_buffer = reinterpret_cast<void*>(
              reinterpret_cast<char*>(recv_buffer) + data_offset);
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] allreduce with input_address :: ",
              offseted_send_buffer,
              " output_address :: ",
              offseted_recv_buffer,
              " elem_cnt :: ",
              num_elements_in_current_chunk,
              " data_type :: ",
              habana_helpers::getHCCLDataType(input.scalar_type()));

          hccl_result = hcclAllReduce(
              offseted_send_buffer,
              offseted_recv_buffer,
              num_elements_in_current_chunk,
              habana_helpers::getHCCLDataType(input.scalar_type()),
              habana_helpers::getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      },
      true /*is_allreduce*/);

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();
  size_t tensor_size = tensors.size();
  auto work = collective(
      tensors,
      tensors,
      [rootRank = opts.rootRank, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        const auto scalar_type = input.scalar_type();
        auto hccl_numel = input.numel();
        hcclDataType_t hccl_data_type;

        habana_helpers::getCountDatatype(
            scalar_type,
            input.element_size(),
            hccl_numel,
            hccl_data_type,
            true);

        size_t element_size = habana_helpers::getHCCLDataSize(hccl_data_type);
        size_t chunk_size_in_elems = habana_helpers::getHCCLSliceSize(
                                         habana_helpers::collectiveBroadcast) /
            element_size;

        size_t data_offset = 0;
        hcclResult_t hccl_result{hcclSuccess};

        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] broadcast with input_address=",
            send_buffer,
            " output_address=",
            recv_buffer,
            " numel=",
            input.numel(),
            " scalar_type=",
            input.scalar_type(),
            " element_size=",
            input.element_size(),
            " hccl_type=",
            hccl_data_type,
            " hccl_count=",
            hccl_numel);
        while (hccl_numel > 0) {
          size_t num_elements_in_current_chunk =
              (static_cast<size_t>(hccl_numel) > chunk_size_in_elems)
              ? chunk_size_in_elems
              : hccl_numel;

          hccl_result = hcclBroadcast(
              static_cast<const uint8_t*>(send_buffer) + data_offset,
              static_cast<uint8_t*>(recv_buffer) + data_offset,
              num_elements_in_current_chunk,
              hccl_data_type,
              rootRank,
              hccl_comm,
              stream);

          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          hccl_numel -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
  PT_DISTRIBUTED_END;
  return work;
}

void ProcessGroupEagerHCCL::hostBarrier() {
  PT_DISTRIBUTED_BEGIN;

  constexpr int64_t kSynchronizeBusyWaitMillis = 1;
  // Minumum three keys are required to avoid race condition
  constexpr int64_t kNumBarrierKeys = 3;

  auto hccl_rank = getRank();
  std::string barrier_key = std::string("HOST_BARRIER:");
  std::string storeKey = std::to_string(barrier_cnt_);
  storeKey += barrier_key;
  storeKey += std::to_string(size_);

  auto first_count = store_->add(storeKey, 1);
  TORCH_CHECK(first_count - 1 < size_, "Host barrier Key error");
  auto worker_count = store_->add(storeKey, 0);
  while (worker_count != size_) {
    worker_count = store_->add(storeKey, 0);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  if (hccl_rank == 0) {
    // Delete the previous key
    std::string storeKey_pre = std::to_string(
        barrier_cnt_ == 0 ? (kNumBarrierKeys - 1) : barrier_cnt_ - 1);
    storeKey_pre += barrier_key;
    storeKey_pre += std::to_string(size_);
    store_->deleteKey(storeKey_pre);
  }

  barrier_cnt_ = (barrier_cnt_ + 1) % kNumBarrierKeys;
  PT_DISTRIBUTED_END;
}

c10::intrusive_ptr<Work> ProcessGroupEagerHCCL::barrier(
    const BarrierOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  hostBarrier();

  auto comm = habana::HcclCommunicator::Get(comm_->GetId());
  std::vector<synStreamHandle> collective_streams = comm->getCommStreams();
  for (size_t i = 0; i < collective_streams.size(); i++) {
    hcclBarrier(*comm->GetHcclHandle(), collective_streams.at(i));
  }

  PT_DISTRIBUTED_END;
  return c10::make_intrusive<ProcessGroupEagerHCCL::WorkEager>();
};

} // namespace c10d

namespace py = pybind11;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  intrusive_ptr_class_<::c10d::ProcessGroupEagerHCCL, c10d::ProcessGroup>
      processGroupHccl(module, "ProcessGroupHCCL");

  processGroupHccl.def(py::init<
                       const c10::intrusive_ptr<c10d::Store>&,
                       int,
                       int,
                       std::chrono::milliseconds>());
};