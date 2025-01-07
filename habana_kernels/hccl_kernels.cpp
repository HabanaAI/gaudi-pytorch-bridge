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
#include "habana_kernels/hccl_kernels.h"
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch_ver/csrc/distributed/c10d/Types.hpp>
#include <torch_ver/csrc/distributed/c10d/Utils.hpp>
#include "backend/helpers/collective_utils.h"
#include "backend/helpers/create_tensor.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "common/utils.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"
#include "pytorch_helpers/habana_helpers/job_thread.h"

#include <hccl.h>
#include <hccl_types.h>

using RedOpType = c10d_ver::ReduceOp::RedOpType;

using namespace torch;
namespace habana {

namespace {

// TODO: SW-68572 move to hccl utils, reuse from ProcessGroupHCCL.cpp

std::map<at::ScalarType, hcclDataType_t> hcclDataType = {
    {at::kByte, hcclUint8},
    {at::kChar, hcclChar},
    {at::kDouble, hcclDouble},
    {at::kFloat, hcclFloat},
    {at::kHalf, hcclHalf},
    {at::kInt, hcclInt32},
    {at::kLong, hcclInt64},
    {at::kBFloat16, hcclBfloat16},
    {at::kBool, hcclUint8},
};

hcclDataType_t getHCCLDataType(at::ScalarType type) {
  // HCL doesn't have definition for fp8 types, use hcclUint8 instead
  // assume later function getCountDatatype() will set correct data type
  // TODO: removed this once HCL adds fp8 types
  if (at::kFloat8_e5m2 == type || at::kFloat8_e4m3fn == type) {
    return hcclUint8;
  }
  type = habana_helpers::getInternalDtype(type);
  auto it = hcclDataType.find(type);
  TORCH_CHECK(
      it != hcclDataType.end(),
      "Input tensor data type is not supported for HCCL process group: ",
      type);
  return it->second;
}

void getCountDatatype(
    c10::ScalarType scalar_type,
    int64_t& numel,
    hcclDataType_t& tensor_data_type) {
  switch (scalar_type) {
    case at::kBool:
      TORCH_CHECK(numel % 2 == 0, "Bool elements count not even")
      if (numel % 4 == 0) {
        numel = numel / 4;
        tensor_data_type = getHCCLDataType(at::kFloat);
      } else {
        numel = numel / 2;
        tensor_data_type = getHCCLDataType(at::kBFloat16);
      }
      break;
    case at::kChar:
    case at::kByte:
    case at::kFloat8_e5m2:
    case at::kFloat8_e4m3fn:
      numel = (numel * sizeof(char)) / sizeof(uint16_t);
      tensor_data_type = getHCCLDataType(at::kBFloat16);
      break;
    case at::kInt:
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kLong:
      if (common::IsInt64Supported()) {
        numel = (numel * 2);
      }
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kDouble:
      numel = (numel * sizeof(float)) / sizeof(float);
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kHalf:
      tensor_data_type = getHCCLDataType(at::kBFloat16);
      break;
    default:
      break;
  }
}

void adjustElementcount_int64(
    c10::ScalarType scalar_type,
    std::vector<size_t>& send_lengths,
    std::vector<size_t>& recv_lengths,
    size_t& ele_size) {
  if (common::IsInt64Supported() && scalar_type == at::kLong) {
    for (size_t i = 0; i < send_lengths.size(); i++) {
      send_lengths[i] = send_lengths[i] * 2;
      recv_lengths[i] = recv_lengths[i] * 2;
    }
  } else {
    if (scalar_type == at::kLong) {
      ele_size = ele_size / 2;
    }
  }
}

bool is_valid_reduction_dtype(hcclDataType_t data_type) {
  if (data_type == hcclBfloat16 || data_type == hcclFloat) {
    return true;
  }
  return false;
}

// HCCL op mapping
std::map<c10d::ReduceOp, hcclRedOp_t> hcclOp = {
    {c10d::ReduceOp::MIN, hcclMin},
    {c10d::ReduceOp::MAX, hcclMax},
    {c10d::ReduceOp::SUM, hcclSum},
    {c10d::ReduceOp::PRODUCT, hcclProd},
};

hcclRedOp_t getHCCLReduceOp(
    const c10d::ReduceOp& reduceOp,
    const at::ScalarType type) {
  if (type == at::kBool) {
    if (reduceOp == c10d::ReduceOp::SUM) {
      // bitwise or
      return hcclMax;
    } else if (reduceOp == c10d::ReduceOp::PRODUCT) {
      // bitwise and
      return hcclMin;
    } else if (reduceOp == c10d::ReduceOp::AVG) {
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with boolean inputs");
    }
  }
  try {
    return hcclOp.at(reduceOp);
  } catch (std::out_of_range& e) {
    TORCH_CHECK(false, "Unsupported ReduceOp for HCCL process group");
  }
}

class JobThreadLazyHCCL {
 public:
  static std::shared_ptr<habana_helpers::JobThread> getInstance() {
    static std::shared_ptr<habana_helpers::JobThread> job(
        new habana_helpers::JobThread);
    return job;
  }
};

constexpr int64_t kSynchronizeBusyWaitMillis = 1;

template <typename Fn>
void collective(
    std::vector<PtTensorInfoShared>& inputs,
    std::vector<PtTensorInfoShared>& outputs,
    std::vector<int64_t> devices,
    std::vector<int64_t> communicator_ids,
    bool async,
    synapse_helpers::event_done_callback done_cb,
    Fn fn) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    TORCH_CHECK(
        devices.at(i) == 0,
        "All tensors are expected to be assigned to device with id 0");
    if (inputs.at(i)->get_numel() == 0) {
      continue;
    }
    auto comm = HcclCommunicator::Get(communicator_ids.at(i));
    auto deviceCtxt = comm->getDeviceCtxt();
    synStreamHandle collective_stream = comm->getCommStream();

    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)inputs.at(i)->get_buffer_start();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)outputs.at(i)->get_buffer_start();

    std::vector<synapse_helpers::shared_event> event_lists = {};
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SFG)) {
      // Prepare stream and get event lists
      event_lists = deviceCtxt->prepare_stream_and_get_events(
          collective_stream, input_storage_ptr);
      if (input_storage_ptr != output_storage_ptr) {
        std::vector<synapse_helpers::shared_event> out_event_lists =
            deviceCtxt->prepare_stream_and_get_events(
                collective_stream, output_storage_ptr);
        event_lists.insert(
            event_lists.end(), out_event_lists.begin(), out_event_lists.end());
      }
    } else {
      // Prepare stream and send stream_wait_event
      deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
      if (input_storage_ptr != output_storage_ptr) {
        deviceCtxt->prepare_stream(collective_stream, output_storage_ptr);
      }
    }

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 input = std::make_shared<PtTensorInfo>(*inputs.at(i)),
                 output = std::make_shared<PtTensorInfo>(*outputs.at(i)),
                 comm = comm,
                 collective_stream = collective_stream,
                 async = async,
                 deviceCtxt = deviceCtxt,
                 event_lists = event_lists,
                 output_storage_ptr = output_storage_ptr,
                 done_cb = done_cb,
                 pr = pr]() mutable {
      PT_LAZY_DEBUG(
          "Collective call. input = ",
          *input,
          ", output = ",
          *output,
          ", comm_id = ",
          comm->GetId(),
          ", stream = ",
          collective_stream);

      if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SFG)) {
        auto& stream = deviceCtxt->get_stream_fromhandle(collective_stream);
        for (auto& event : event_lists) {
          event->stream_wait_event(stream);
        }
      }
      auto& recipe_counter = deviceCtxt->get_active_recipe_counter();

      struct ResourceHolder {
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
      };
      auto resource_holder = std::make_shared<ResourceHolder>();

      void* input_address;
      void* output_address;
      deviceCtxt->lock_address(
          {input->get_buffer(), output->get_buffer()},
          resource_holder->address_lock);
      input_address =
          reinterpret_cast<void*>(resource_holder->address_lock->at(0));
      HABANA_ASSERT(input_address != nullptr, "input_address is null");
      output_address =
          reinterpret_cast<void*>(resource_holder->address_lock->at(1));
      HABANA_ASSERT(output_address != nullptr, "output_address is null");

      hcclResult_t hccl_result =
          fn(input,
             output,
             input_address,
             output_address,
             std::move(comm),
             collective_stream);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
      recipe_counter.increase();
      deviceCtxt->submit_events(
          collective_stream,
          output_storage_ptr,
          [resource_holder, &recipe_counter, done_cb]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
            done_cb();
          });
      pr->set_value(hccl_result == hcclSuccess);

      if (!async) {
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(
            syn_result == synSuccess,
            "synStreamSynchronize for synchronized collective call failed");

        while (JobThreadLazyHCCL::getInstance()->jobCounter() != 0) {
          PT_LAZY_DEBUG(
              "[PYT-DIST] Waiting for lazy collectives jobs to complete");
          std::this_thread::sleep_for(
              std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
      }

      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
    } else {
      JobThreadLazyHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(output_storage_ptr, std::move(fut));
    }
  }
}

template <typename Fn>
void pointToPoint(
    std::vector<PtTensorInfoShared>& tensors,
    std::vector<at::Tensor>& pt_tensor,
    std::vector<int64_t> devices,
    std::vector<int64_t> communicator_ids,
    bool async,
    synapse_helpers::event_done_callback done_cb,
    Fn fn,
    int peerRank) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    TORCH_CHECK(
        devices.at(i) == 0,
        "All tensors are expected to be assigned to device with id 0");
    auto comm = HcclCommunicator::Get(communicator_ids.at(i));
    auto deviceCtxt = comm->getDeviceCtxt();
    synStreamHandle collective_stream = comm->getCommStream();

    synapse_helpers::device_ptr tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensors.at(i)->get_buffer_start();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 tensor = tensors.at(i),
                 pt_tensor = pt_tensor,
                 comm = comm,
                 collective_stream = collective_stream,
                 peerRank = peerRank,
                 async = async,
                 deviceCtxt = deviceCtxt,
                 done_cb = done_cb,
                 tensor_storage_ptr = tensor_storage_ptr,
                 pr = pr]() mutable {
      PT_LAZY_DEBUG(
          "pointToPoint call. input = ",
          *tensor,
          ", comm_id = ",
          comm->GetId(),
          ", stream = ",
          collective_stream,
          ", peerRank = ",
          peerRank);

      auto& recipe_counter = deviceCtxt->get_active_recipe_counter();
      recipe_counter.increase();

      // TBD: Need to store references to tensor
      struct ResourceHolder {
        std::vector<at::Tensor> pt_tensor;
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
      };
      auto resource_holder = std::make_shared<ResourceHolder>();

      void* tensor_address;
      deviceCtxt->lock_address(
          tensor->get_buffer(), &tensor_address, resource_holder->address_lock);
      resource_holder->pt_tensor = pt_tensor;

      auto hccl_result = fn(
          tensor, tensor_address, std::move(comm), collective_stream, peerRank);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
      deviceCtxt->submit_events(
          collective_stream,
          tensor_storage_ptr,
          [resource_holder = std::move(resource_holder),
           &recipe_counter,
           done_cb = done_cb]() mutable {
            resource_holder.reset();
            recipe_counter.decrease_and_notify();
            done_cb();
          });
      pr->set_value(hccl_result == hcclSuccess);

      if (!async) {
        synStatus syn_result = synSuccess;
        syn_result = synStreamSynchronize(collective_stream);
        TORCH_CHECK(
            syn_result == synSuccess,
            "synStreamSynchronize for synchronized collective call failed");
        while (JobThreadLazyHCCL::getInstance()->jobCounter() != 0) {
          PT_LAZY_DEBUG(
              "[PYT-DIST] Waiting for lazy collectives jobs to complete");
          std::this_thread::sleep_for(
              std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
      }
      return true;
    };

    if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_ASYNC_COLLECTIVE)) {
      func();
    } else {
      JobThreadLazyHCCL::getInstance()->addJob(std::move(func));
      deviceCtxt->submit_future(tensor_storage_ptr, std::move(fut));
    }
  }
}

} // anonymous namespace

void HcclBroadcastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");

  root_rank_ = inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclBroadcastOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, root_rank_);
  serialization::serialize(os, comm_id_);
}
void HcclBroadcastOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, root_rank_);
  serialization::deserialize(is, comm_id_);
}

void HcclBroadcastOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  collective(
      tensor_inputs,
      tensor_inputs,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_, root_rank = root_rank_](
          __attribute__((unused)) PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream) {
        auto tensor_data_type = getHCCLDataType(scalar_type);
        int64_t numel = input->get_numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        return hcclBroadcast(
            send_buffer,
            recv_buffer,
            numel,
            tensor_data_type,
            root_rank,
            *comm->GetHcclHandle(),
            stream);
      });
}

void HcclAllreduceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");

  static_assert(sizeof(RedOpType) <= sizeof(uint8_t));
  reduce_op_ = (uint8_t)inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclAllreduceOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, reduce_op_);
  serialization::serialize(os, comm_id_);
}
void HcclAllreduceOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, reduce_op_);
  serialization::deserialize(is, comm_id_);
}

void HcclAllreduceOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  HABANA_ASSERT(
      is_valid_reduction_dtype(getHCCLDataType(scalar_type_)),
      "HCCL supports only float or bfloat16 reduction");

  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  collective(
      tensor_inputs,
      tensor_inputs,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_, reduce_op = reduce_op_](
          __attribute__((unused)) PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream) {
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input->get_numel();
        size_t element_size =
            c10::elementSize(habana_helpers::getInternalDtype(scalar_type));
        size_t chunk_size = habana_helpers::getHCCLSliceSize(
                                habana_helpers::collectiveAllReduce, true) /
            element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          auto hccl_result = hcclAllReduce(
              (void*)((uint64_t)send_buffer + data_offset),
              (void*)((uint64_t)recv_buffer + data_offset),
              num_elements_in_current_chunk,
              getHCCLDataType(scalar_type),
              getHCCLReduceOp((RedOpType)reduce_op, scalar_type),
              *comm->GetHcclHandle(),
              stream);
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset += num_elements_in_current_chunk * element_size;
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
}

void HcclReduceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");

  dst_rank_ = inputs.at(1).toInt();
  static_assert(sizeof(RedOpType) <= sizeof(uint8_t));
  reduce_op_ = (uint8_t)inputs.at(2).toInt();
  comm_id_ = inputs.at(3).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclReduceOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, dst_rank_);
  serialization::serialize(os, reduce_op_);
  serialization::serialize(os, comm_id_);
}

void HcclReduceOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, dst_rank_);
  serialization::deserialize(is, reduce_op_);
  serialization::deserialize(is, comm_id_);
}

void HcclReduceOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  HABANA_ASSERT(
      is_valid_reduction_dtype(getHCCLDataType(scalar_type_)),
      "HCCL supports only float or bfloat16 reduction");

  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  collective(
      tensor_inputs,
      tensor_inputs,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_,
       reduce_op = reduce_op_,
       dst_rank = dst_rank_](
          __attribute__((unused)) PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream) {
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input->get_numel();
        size_t element_size =
            c10::elementSize(habana_helpers::getInternalDtype(scalar_type));
        size_t chunk_size = habana_helpers::getHCCLSliceSize(
                                habana_helpers::collectiveReduce, true) /
            element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          auto hccl_result = hcclReduce(
              (void*)((uint64_t)send_buffer + data_offset),
              (void*)((uint64_t)recv_buffer + data_offset),
              num_elements_in_current_chunk,
              getHCCLDataType(scalar_type),
              getHCCLReduceOp((RedOpType)reduce_op, scalar_type),
              dst_rank,
              *comm->GetHcclHandle(),
              stream);
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset += num_elements_in_current_chunk * element_size;
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
}

void HcclAllToAllOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg 2 needs to be of tensor type");
  TORCH_CHECK(inputs[2].isIntList(), "Input arg 3 needs to be of list type");
  TORCH_CHECK(inputs[3].isIntList(), "Input arg 4 needs to be of list type");

  auto outputTensor = inputs.at(4).toTensor();
  auto inputTensor = inputs.at(0).toTensor();

  outputSplitSizes = inputs.at(2).toIntVector();
  inputSplitSizes = inputs.at(3).toIntVector();

  comm_id_ = inputs.at(1).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_.at(1),
          graph,
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(outputTensor);
}

void HcclAllToAllOutOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, comm_id_);
  serialization::serialize(os, outputSplitSizes);
  serialization::serialize(os, inputSplitSizes);
}

void HcclAllToAllOutOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, comm_id_);
  serialization::deserialize(is, outputSplitSizes);
  serialization::deserialize(is, inputSplitSizes);
}

void HcclAllToAllOutOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  std::vector<PtTensorInfoShared> tensor_outputs = {inputs.at(4)};

  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    collective(
        tensor_inputs,
        tensor_outputs,
        {device_id_},
        {comm_id_},
        async,
        done_cb,
        [scalar_type = scalar_type_](
            PtTensorInfoShared& input,
            __attribute__((unused)) PtTensorInfoShared& output,
            const void* send_buffer,
            void* recv_buffer,
            std::shared_ptr<HcclCommunicator> comm,
            synStreamHandle stream) {
          int64_t count = input->get_numel();
          auto type = getHCCLDataType(scalar_type);
          getCountDatatype(scalar_type, count, type);
          hcclResult_t hccl_result{hcclSuccess};
          hccl_result = hcclAlltoAll(
              send_buffer,
              recv_buffer,
              count,
              type,
              *comm->GetHcclHandle(),
              stream);

          return hccl_result;
        });
  } else {
    collective(
        tensor_inputs,
        tensor_outputs,
        {device_id_},
        {comm_id_},
        async,
        done_cb,
        [scalar_type = scalar_type_,
         input_t = p_context_->pt_inputs_[0],
         output_t = p_context_->pt_outputs_[0],
         inputSplitSizes_ = inputSplitSizes,
         outputSplitSizes_ = inputSplitSizes](
            __attribute__((unused)) PtTensorInfoShared& input,
            __attribute__((unused)) PtTensorInfoShared& output,
            const void* send_buffer,
            void* recv_buffer,
            std::shared_ptr<HcclCommunicator> comm,
            synStreamHandle stream) {
          int numRanks = comm->GetSize();
          c10d::checkSplitSizes(inputSplitSizes_, input_t, numRanks);
          c10d::checkSplitSizes(outputSplitSizes_, output_t, numRanks);

          std::vector<size_t> send_lengths(numRanks);
          std::vector<size_t> recv_lengths(numRanks);
          std::vector<size_t> send_offsets(numRanks);
          std::vector<size_t> recv_offsets(numRanks);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes_, input_t, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes_, output_t, &recv_lengths, &recv_offsets);

          size_t ele_size = input_t.element_size();
          auto type = getHCCLDataType(scalar_type);
          int64_t count = input_t.numel();
          getCountDatatype(scalar_type, count, type);
          adjustElementcount_int64(
              scalar_type, send_lengths, recv_lengths, ele_size);
          hcclGroupStart();
          hcclResult_t hccl_result{hcclSuccess};
          for (const auto r : c10::irange(numRanks)) {
            if (send_lengths[r] != 0) {
              hccl_result = hcclSend(
                  reinterpret_cast<const unsigned char*>(send_buffer) +
                      send_offsets[r] * ele_size,
                  send_lengths[r],
                  type,
                  r,
                  *comm->GetHcclHandle(),
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }

            if (recv_lengths[r] != 0) {
              hccl_result = hcclRecv(
                  reinterpret_cast<unsigned char*>(recv_buffer) +
                      recv_offsets[r] * ele_size,
                  recv_lengths[r],
                  type,
                  r,
                  *comm->GetHcclHandle(),
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }
          }
          hcclGroupEnd();
          return hccl_result;
        });
  }
}
void HcclAllgatherOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.at(0).isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(
      inputs.at(1).isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(
      inputs.at(2).isTensor(), "Input arg 2 needs to be of tensor type");

  auto outputTensor = inputs.at(2).toTensor();
  auto inputTensor = inputs.at(0).toTensor();
  comm_id_ = inputs.at(1).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_.at(1),
          graph,
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(outputTensor);
}

void HcclAllgatherOutOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, comm_id_);
}

void HcclAllgatherOutOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, comm_id_);
}

void HcclAllgatherOutOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  std::vector<PtTensorInfoShared> tensor_outputs = {inputs.at(2)};
  collective(
      tensor_inputs,
      tensor_outputs,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_](
          PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream) {
        auto tensor_data_type = getHCCLDataType(scalar_type);
        int64_t numel = input->get_numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        hcclResult_t hccl_result = hcclAllGather(
            send_buffer,
            recv_buffer,
            numel,
            tensor_data_type,
            *comm->GetHcclHandle(),
            stream);
        return hccl_result;
      });
}

void HcclReduceScatterOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg 3 needs to be of tensor type");
  auto outputTensor = inputs.at(3).toTensor();
  static_assert(sizeof(RedOpType) <= sizeof(uint8_t));
  reduce_op_ = (uint8_t)inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(outputTensor);
}

void HcclReduceScatterOutOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, reduce_op_);
  serialization::serialize(os, comm_id_);
}

void HcclReduceScatterOutOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, reduce_op_);
  serialization::deserialize(is, comm_id_);
}

void HcclReduceScatterOutOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  std::vector<PtTensorInfoShared> tensor_outputs = {inputs.at(3)};
  collective(
      tensor_inputs,
      tensor_outputs,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_, reduce_op = reduce_op_](
          __attribute__((unused)) PtTensorInfoShared& input,
          PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream) {
        hcclResult_t hccl_result = hcclReduceScatter(
            send_buffer,
            recv_buffer,
            output->get_numel(),
            getHCCLDataType(scalar_type),
            getHCCLReduceOp((RedOpType)reduce_op, scalar_type),
            *comm->GetHcclHandle(),
            stream);
        return hccl_result;
      });
}

void HcclSendOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");
  TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 needs to be of scalar type");

  dst_rank_ = inputs.at(1).toInt();
  tag_ = inputs.at(2).toInt();
  comm_id_ = inputs.at(3).toInt();

  at::Tensor tensor = inputs[0].toTensor();
  // Tensor will be sent as part of lazy graph.
  // Don't allow Synapse to return it permuted as send/recv don't support
  // permuted tensors
  auto smeta{habana::get_storage_extra_meta(tensor)};
  if (smeta) {
    auto& syn_tensor = p_context_->syn_inputs_[0].ref();
    synTensorSetAllowPermutation(syn_tensor.get(), 0);
    syn_tensor.set_dont_allow_permute(true);
    smeta->set_dont_allow_permutation(true);
  }

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclSendOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, dst_rank_);
  serialization::serialize(os, tag_);
  serialization::serialize(os, comm_id_);
}

void HcclSendOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, dst_rank_);
  serialization::deserialize(is, tag_);
  serialization::deserialize(is, comm_id_);
}

void HcclSendOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};

  pointToPoint(
      tensor_inputs,
      p_context_->pt_inputs_,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_](
          PtTensorInfoShared& input,
          const void* send_buff,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream,
          int peerRank) {
        auto tensor_data_type = getHCCLDataType(scalar_type);
        int64_t numel = input->get_numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        return hcclSend(
            send_buff,
            numel,
            tensor_data_type,
            peerRank,
            *comm->GetHcclHandle(),
            stream);
      },
      dst_rank_);
}

void HcclRecvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg 0 needs to be of tensor type");
  TORCH_CHECK(inputs[1].isScalar(), "Input arg 1 needs to be of scalar type");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg 2 needs to be of scalar type");
  TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 needs to be of scalar type");

  src_rank_ = inputs.at(1).toInt();
  tag_ = inputs.at(2).toInt();
  comm_id_ = inputs.at(3).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclRecvOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, src_rank_);
  serialization::serialize(os, tag_);
  serialization::serialize(os, comm_id_);
}

void HcclRecvOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, src_rank_);
  serialization::deserialize(is, tag_);
  serialization::deserialize(is, comm_id_);
}

void HcclRecvOperator::RunCollective(
    const std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) const {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};

  pointToPoint(
      tensor_inputs,
      p_context_->pt_inputs_,
      {device_id_},
      {comm_id_},
      async,
      done_cb,
      [scalar_type = scalar_type_](
          PtTensorInfoShared& input,
          void* recv_buff,
          std::shared_ptr<HcclCommunicator> comm,
          synStreamHandle stream,
          int peerRank) {
        auto tensor_data_type = getHCCLDataType(scalar_type);
        int64_t numel = input->get_numel();
        getCountDatatype(scalar_type, numel, tensor_data_type);
        return hcclRecv(
            recv_buff,
            numel,
            tensor_data_type,
            peerRank,
            *comm->GetHcclHandle(),
            stream);
      },
      src_rank_);
}

} // namespace habana

static auto& HCCLKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hccl::broadcast_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclBroadcastOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::allreduce_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclAllreduceOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::reduce_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclReduceOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::alltoall_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclAllToAllOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::allgather_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclAllgatherOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::reduce_scatter_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclReduceScatterOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hccl::send_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::HcclSendOperator>(
                  device_id, node_type);
            })
        .add("hccl::recv_", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<habana::HcclRecvOperator>(
              device_id, node_type);
        });
