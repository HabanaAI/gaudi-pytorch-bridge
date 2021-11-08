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
#include "pytorch_helpers/synapse_helpers/hccl_communicator.h"

#include <hccl.h>
#include <hccl_types.h>
#include <hcl_api.h>

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

template <typename Fn>
void collective(
    std::vector<PtTensorInfoShared>& inputs,
    std::vector<PtTensorInfoShared>& outputs,
    std::vector<int64_t> devices,
    std::vector<int64_t> communicator_ids,
    bool async,
    synapse_helpers::event_done_callback done_cb,
    Fn fn) {
  hcclResult_t hccl_result{hcclSuccess};

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto comm = HcclCommunicator::Get(communicator_ids.at(i));
    auto deviceCtxt = comm->getDeviceCtxt(devices.at(i));
    hcclStream_t collective_stream = comm->getCommStream(devices.at(i));

    void* input_address;
    void* output_address;
    synapse_helpers::device_ptr input_storage_ptr =
        (synapse_helpers::device_ptr)inputs.at(i)->get_buffer_start();
    synapse_helpers::device_ptr output_storage_ptr =
        (synapse_helpers::device_ptr)outputs.at(i)->get_buffer_start();
    deviceCtxt->prepare_stream(collective_stream, input_storage_ptr);
    deviceCtxt->lock_address(inputs.at(i)->get_buffer(), &input_address);
    deviceCtxt->lock_address(outputs.at(i)->get_buffer(), &output_address);
    hccl_result =
        fn(inputs.at(i),
           outputs.at(i),
           input_address,
           output_address,
           *(comm->GetHcclHandle()),
           collective_stream);
    TORCH_CHECK(hcclSuccess == hccl_result, "Collective call returned error");
    deviceCtxt->submit_events(collective_stream, output_storage_ptr, done_cb);
    if (!async) {
      synStatus syn_result = synSuccess;
      syn_result = synStreamSynchronize(collective_stream);
      TORCH_CHECK(
          syn_result == synSuccess,
          "synStreamSynchronize for synchronized collective call failed");
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

  auto inputTensor = inputs.at(0).toTensor();
  device_ = inputTensor.get_device();
  data_type_ = inputTensor.scalar_type();
  root_rank_ = inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclBroadcastOperator::RunCollective(
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  collective(
      tensor_inputs,
      tensor_inputs,
      {device_},
      {comm_id_},
      async,
      done_cb,
      [&](__attribute__((unused)) PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        PT_LAZY_DEBUG(
            "Calling hccl broadcast, send_buffer = ",
            send_buffer,
            " recv_buffer = ",
            recv_buffer,
            " num_elements = ",
            input->get_numel(),
            " data_type = ",
            data_type_,
            " root_rank = ",
            root_rank_,
            " comm_id = ",
            comm_id_);
        return hcclBroadcast(
            send_buffer,
            recv_buffer,
            input->get_numel(),
            getHCCLDataType(data_type_),
            root_rank_,
            hccl_comm,
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

  auto inputTensor = inputs.at(0).toTensor();
  device_ = inputTensor.get_device();
  data_type_ = inputTensor.scalar_type();
  reduce_op_ = inputs.at(1).toInt();
  comm_id_ = inputs.at(2).toInt();

  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
}

void HcclAllreduceOperator::RunCollective(
    std::vector<PtTensorInfoShared>& inputs,
    bool async,
    synapse_helpers::event_done_callback done_cb) {
  // TODO: SW-68569 verify input data type is supported, cast in allocate and
  // add synapse node if needed
  HABANA_ASSERT(is_valid_reduction_dtype(getHCCLDataType(data_type_)));

  std::vector<PtTensorInfoShared> tensor_inputs = {inputs.at(0)};
  collective(
      tensor_inputs,
      tensor_inputs,
      {device_},
      {comm_id_},
      async,
      done_cb,
      [&](__attribute__((unused)) PtTensorInfoShared& input,
          __attribute__((unused)) PtTensorInfoShared& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          hcclStream_t stream) {
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input->get_numel();
        size_t element_size =
            c10::elementSize(getInternalScalarType(data_type_));
        size_t chunk_size = getHCCLSliceSizeMB() / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;

          PT_LAZY_DEBUG(
              "Calling hccl allreduce, send_buffer = ",
              send_buffer,
              " recv_buffer = ",
              recv_buffer,
              " data_offset = ",
              data_offset,
              " num_elements = ",
              num_elements_in_current_chunk,
              " data_type = ",
              data_type_,
              " reduce_op = ",
              reduce_op_,
              " comm_id = ",
              comm_id_);

          auto hccl_result = hcclAllReduce(
              (void*)((uint64_t)send_buffer + data_offset),
              (void*)((uint64_t)recv_buffer + data_offset),
              num_elements_in_current_chunk,
              getHCCLDataType(data_type_),
              getHCCLReduceOp((c10d::ReduceOp)reduce_op_),
              hccl_comm,
              stream);
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset += num_elements_in_current_chunk * element_size;
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
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
            });
