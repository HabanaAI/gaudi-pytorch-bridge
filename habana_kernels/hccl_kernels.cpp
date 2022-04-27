/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/hccl_kernels.h"
#include <ATen/ATen.h>
#undef UNUSED // Collision between pytorch_helpers/synapse_helpers/graph.h and
              // c10d::ReduceOp enum from c10d/Types.hpp
#include <c10d/Types.hpp>
#include "habana_helpers/logging.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"
#include "pytorch_helpers/habana_helpers/job_thread.h"
#include "pytorch_helpers/synapse_helpers/hccl_communicator.h"

#include <hccl.h>
#include <hccl_types.h>

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
};

at::ScalarType getInternalScalarType(at::ScalarType type) {
  type = (type == c10::ScalarType::Long) ? c10::ScalarType::Int : type;
  type = (type == c10::ScalarType::Double) ? c10::ScalarType::Float : type;
  return type;
}

hcclDataType_t getHCCLDataType(at::ScalarType type) {
  type = getInternalScalarType(type);
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
    case at::kChar:
    case at::kByte:
      numel = (numel * sizeof(char)) / sizeof(uint16_t);
      tensor_data_type = getHCCLDataType(at::kBFloat16);
      break;
    case at::kInt:
      tensor_data_type = getHCCLDataType(at::kFloat);
      break;
    case at::kLong:
      // there is implicit conversion from long to float
      numel = (numel * sizeof(float)) / sizeof(float);
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

bool is_valid_reduction_dtype(hcclDataType_t data_type) {
  if (data_type == hcclBfloat16 || data_type == hcclFloat) {
    return true;
  }
  return false;
}

size_t getHCCLSliceSizeMB() {
  static const size_t slice_size = GET_ENV_FLAG_NEW(PT_HCCL_SLICE_SIZE_MB);
  return slice_size * 1024 * 1024;
}

// HCCL op mapping
std::map<c10d::ReduceOp, hcclRedOp_t> hcclOp = {
    {c10d::ReduceOp::MIN, hcclMin},
    {c10d::ReduceOp::MAX, hcclMax},
    {c10d::ReduceOp::SUM, hcclSum},
    {c10d::ReduceOp::PRODUCT, hcclProd},
};

hcclRedOp_t getHCCLReduceOp(const c10d::ReduceOp& reduceOp) {
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
    auto comm = HcclCommunicator::Get(communicator_ids.at(i));
    auto deviceCtxt = comm->getDeviceCtxt(devices.at(i));
    synStreamHandle collective_stream = comm->getCommStream(devices.at(i));

    void* input_address;
    void* output_address;
    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)inputs.at(i)->get_buffer_start();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)outputs.at(i)->get_buffer_start();
    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    deviceCtxt->prepare_stream(collective_stream, output_storage_ptr);
    deviceCtxt->lock_address(inputs.at(i)->get_buffer(), &input_address);
    deviceCtxt->lock_address(outputs.at(i)->get_buffer(), &output_address);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 input = inputs.at(i),
                 output = outputs.at(i),
                 input_address = input_address,
                 output_address = output_address,
                 comm = comm,
                 collective_stream = collective_stream,
                 async = async,
                 deviceCtxt = deviceCtxt,
                 output_storage_ptr = output_storage_ptr,
                 done_cb = done_cb,
                 pr = pr]() mutable {
      PT_LAZY_DEBUG(
          "Collective call. input = ",
          input,
          ", output = ",
          output,
          ", input_address = ",
          input_address,
          ", output_address = ",
          output_address,
          ", comm_id = ",
          comm->GetId(),
          ", stream = ",
          collective_stream);
      hcclResult_t hccl_result =
          fn(input,
             output,
             input_address,
             output_address,
             std::move(comm),
             collective_stream);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
      deviceCtxt->submit_events(collective_stream, output_storage_ptr, done_cb);
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
    std::vector<int64_t> devices,
    std::vector<int64_t> communicator_ids,
    bool async,
    synapse_helpers::event_done_callback done_cb,
    Fn fn,
    int peerRank) {

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto comm = HcclCommunicator::Get(communicator_ids.at(i));
    auto deviceCtxt = comm->getDeviceCtxt(devices.at(i));
    synStreamHandle collective_stream = comm->getCommStream(devices.at(i));

    void* tensor_address;
    synapse_helpers::device_ptr tensor_storage_ptr =
        (synapse_helpers::device_ptr)tensors.at(i)->get_buffer_start();
    deviceCtxt->prepare_stream(collective_stream, tensor_storage_ptr);
    deviceCtxt->lock_address(tensors.at(i)->get_buffer(), &tensor_address);

    auto pr = std::make_shared<std::promise<bool>>();
    std::future<bool> fut = pr->get_future();
    auto func = [fn = fn,
                 tensor = tensors.at(i),
                 address = tensor_address,
                 comm = comm,
                 collective_stream = collective_stream,
                 peerRank = peerRank,
                 async = async,
                 deviceCtxt = deviceCtxt,
                 tensor_storage_ptr = tensor_storage_ptr,
                 done_cb = done_cb,
                 pr = pr]() mutable {
      PT_LAZY_DEBUG(
          "pointToPoint call. input = ",
          tensor,
          ", input_address = ",
          address,
          ", comm_id = ",
          comm->GetId(),
          ", stream = ",
          collective_stream,
          ", peerRank = ",
          peerRank);

      auto hccl_result =
          fn(tensor, address, std::move(comm), collective_stream, peerRank);
      TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
      deviceCtxt->submit_events(collective_stream, tensor_storage_ptr, done_cb);
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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

  static_assert(sizeof(c10d::ReduceOp) <= sizeof(uint8_t));
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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
            c10::elementSize(getInternalScalarType(scalar_type));
        size_t chunk_size = getHCCLSliceSizeMB() / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          auto hccl_result = hcclAllReduce(
              (void*)((uint64_t)send_buffer + data_offset),
              (void*)((uint64_t)recv_buffer + data_offset),
              num_elements_in_current_chunk,
              getHCCLDataType(scalar_type),
              getHCCLReduceOp((c10d::ReduceOp)reduce_op),
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
  static_assert(sizeof(c10d::ReduceOp) <= sizeof(uint8_t));
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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
            c10::elementSize(getInternalScalarType(scalar_type));
        size_t chunk_size = getHCCLSliceSizeMB() / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          auto hccl_result = hcclReduce(
              (void*)((uint64_t)send_buffer + data_offset),
              (void*)((uint64_t)recv_buffer + data_offset),
              num_elements_in_current_chunk,
              getHCCLDataType(scalar_type),
              getHCCLReduceOp((c10d::ReduceOp)reduce_op),
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
  TORCH_CHECK(inputs[2].isTensor(), "Input arg 2 needs to be of tensor type");

  auto outputTensor = inputs.at(2).toTensor();
  auto inputTensor = inputs.at(0).toTensor();
  comm_id_ = inputs.at(1).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclAllToAllOutOperator::Serialize(std::ostream& os) const {
  serialization::serialize(os, comm_id_);
}

void HcclAllToAllOutOperator::Deserialize(std::istream& is) {
  serialization::deserialize(is, comm_id_);
}

void HcclAllToAllOutOperator::RunCollective(
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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
        int numRanks = comm->GetSize();
        size_t count = input->get_numel() / numRanks;
        size_t rank_offset =
            count * c10::elementSize(getInternalScalarType(scalar_type));
        auto type = getHCCLDataType(scalar_type);
        hcclGroupStart();
        hcclResult_t hccl_result{hcclSuccess};
        for (auto r = 0; r < numRanks; r++) {
          hcclSend(
              reinterpret_cast<const unsigned char*>(send_buffer) +
                  r * rank_offset,
              count,
              type,
              r,
              *comm->GetHcclHandle(),
              stream);
          hcclRecv(
              reinterpret_cast<unsigned char*>(recv_buffer) + r * rank_offset,
              count,
              type,
              r,
              *comm->GetHcclHandle(),
              stream);
        }
        hcclGroupEnd();

        return hccl_result;
      });
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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
  static_assert(sizeof(c10d::ReduceOp) <= sizeof(uint8_t));
  reduce_op_ = (uint8_t)inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
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
            getHCCLReduceOp((c10d::ReduceOp)reduce_op),
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};

  pointToPoint(
      tensor_inputs,
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
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};

  pointToPoint(
      tensor_inputs,
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

static auto& KernelRegistry =
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
