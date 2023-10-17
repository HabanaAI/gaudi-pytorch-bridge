/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "process_group_hccl_base.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <unistd.h>
#include <future>
#include <map>
#include <sstream>

#include "backend/helpers/collective_utils.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/device_context.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/tensor_impl.h"
#include "pytorch_helpers/habana_helpers/job_thread.h"

using namespace synapse_helpers;
namespace c10d {

namespace {

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
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

bool resizeTensor(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList) {
  bool change = false;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    changed[i] = false;
    if ((at::kChar == btensor_type || at::kByte == btensor_type ||
         at::kBool == btensor_type) &&
        tensors[i].numel() % 2 != 0) {
      changed[i] = true;
      sizeList[i] = tensors[i].sizes().vec();
      strideList[i] = tensors[i].strides().vec();
      tensors[i] = tensors[i].resize_(tensors[i].numel() + 1);
      change = true;
    }
  }
  return change;
}

void restoreTensorsize(
    std::vector<at::Tensor>& tensors,
    std::unique_ptr<bool[]>& changed,
    std::vector<std::vector<int64_t>>& sizeList,
    std::vector<std::vector<int64_t>>& strideList,
    c10::intrusive_ptr<Work>& work) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto btensor_type = tensors[i].scalar_type();
    if (at::kChar == btensor_type || at::kByte == btensor_type ||
        at::kBool == btensor_type) {
      work->wait();
    }
    if (changed[i] == true) {
      tensors[i] = tensors[i].resize_(tensors[i].numel() - 1);
      tensors[i].unsafeGetTensorImpl()->set_sizes_and_strides(
          sizeList[i], strideList[i]);
    }
  }
}
bool is_valid_hccl_dtype(hcclDataType_t data_type) {
  if (data_type == hcclBfloat16 || data_type == hcclFloat) {
    return true;
  }
  return false;
}

} // namespace

// TBD: Store not used for now and config done from file
// Initial support added for multiple devices on a single node
// So using rank as the device id.  This will be enhanced further.
ProcessGroupHcclBase::ProcessGroupHcclBase(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size),
      always_support_int64_(false),
      store_(store),
      barrier_cnt_(0) {
  this->emulate_distributed_ = GET_ENV_FLAG_NEW(PT_HPU_EMULATE_DISTRIBUTED);
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
}

ProcessGroupHcclBase::~ProcessGroupHcclBase() {
  if (!this->emulate_distributed_ && comm_) {
    comm_.reset();
  }
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  auto work = collective(
      tensors,
      tensors,
      [rootRank = opts.rootRank, this](
          at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        hcclDataType_t hccl_data_type;
        const auto scalar_type = input.scalar_type();
        auto hccl_numel = input.numel();

        habana_helpers::getCountDatatype(
            scalar_type,
            input.element_size(),
            hccl_numel,
            hccl_data_type,
            always_support_int64_);

        size_t element_size = habana_helpers::getHCCLDataSize(hccl_data_type);
        size_t chunk_size_in_elems =
            getHCCLSliceSize(habana_helpers::collectiveBroadcast) /
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
          if (!this->emulate_distributed_) {
            hccl_result = hcclBroadcast(
                static_cast<const uint8_t*>(send_buffer) + data_offset,
                static_cast<uint8_t*>(recv_buffer) + data_offset,
                num_elements_in_current_chunk,
                hccl_data_type,
                rootRank,
                hccl_comm,
                stream);
          }

          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          hccl_numel -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<at::Tensor> allreduce_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto data_type = habana_helpers::getHCCLDataType(tensors[i].scalar_type());
    if (is_valid_hccl_dtype(data_type)) {
      allreduce_tensors.push_back(tensors[i]);
    } else {
      PT_DISTRIBUTED_DEBUG("[PYT-DIST] allreduce tensors converted to float");
      allreduce_tensors.push_back(tensors[i].to(c10::ScalarType::Float));
    }
  }

  auto work = collective(
      allreduce_tensors,
      allreduce_tensors,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size = c10::elementSize(
            habana_helpers::getInternalDtype(input.scalar_type()));
        size_t chunk_size =
            getHCCLSliceSize(habana_helpers::collectiveAllReduce) /
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

          if (!this->emulate_distributed_) {
            hccl_result = hcclAllReduce(
                offseted_send_buffer,
                offseted_recv_buffer,
                num_elements_in_current_chunk,
                habana_helpers::getHCCLDataType(input.scalar_type()),
                habana_helpers::getHCCLReduceOp(reduceOp),
                hccl_comm,
                stream);
          }
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      },
      true /*is_allreduce*/);

  for (size_t i = 0; i < tensors.size(); i++) {
    auto data_type = habana_helpers::getHCCLDataType(tensors[i].scalar_type());
    if (!is_valid_hccl_dtype(data_type)) {
      work->wait();
      tensors[i].copy_(allreduce_tensors[i].to(tensors[i].scalar_type()));
    }
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::allreduce_coalesced(
    std::vector<at::Tensor>& /*tensors*/,
    const AllreduceCoalescedOptions& /*opts*/) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with HCCL");
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  std::vector<at::Tensor> reduction_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto data_type = habana_helpers::getHCCLDataType(tensors[i].scalar_type());
    if (is_valid_hccl_dtype(data_type)) {
      reduction_tensors.push_back(tensors[i]);
    } else {
      PT_DISTRIBUTED_DEBUG("[PYT-DIST] reduction tensors converted to float");
      reduction_tensors.push_back(tensors[i].to(c10::ScalarType::Float));
    }
  }
  auto work = collective(
      reduction_tensors,
      reduction_tensors,
      [root = opts.rootRank * reduction_tensors.size() + opts.rootTensor,
       reduceOp = opts.reduceOp,
       this](
          at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] reduce with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(input.scalar_type()));
        hcclResult_t hccl_result{hcclSuccess};
        size_t num_elements = input.numel();
        size_t element_size = c10::elementSize(
            habana_helpers::getInternalDtype(input.scalar_type()));
        size_t chunk_size =
            getHCCLSliceSize(habana_helpers::collectiveReduce) / element_size;
        size_t data_offset = 0;
        while (num_elements > 0) {
          size_t num_elements_in_current_chunk =
              (num_elements > chunk_size) ? chunk_size : num_elements;
          if (!this->emulate_distributed_) {
            hccl_result = hcclReduce(
                reinterpret_cast<const void*>(
                    reinterpret_cast<const char*>(send_buffer) + data_offset),
                reinterpret_cast<void*>(
                    reinterpret_cast<char*>(recv_buffer) + data_offset),
                num_elements_in_current_chunk,
                habana_helpers::getHCCLDataType(input.scalar_type()),
                habana_helpers::getHCCLReduceOp(reduceOp),
                root,
                hccl_comm,
                stream);
          }
          TORCH_CHECK(
              hcclSuccess == hccl_result, "Collective call returned error");
          data_offset =
              data_offset + (num_elements_in_current_chunk * element_size);
          num_elements -= num_elements_in_current_chunk;
        }
        return hccl_result;
      });

  for (size_t i = 0; i < tensors.size(); i++) {
    auto data_type = habana_helpers::getHCCLDataType(tensors[i].scalar_type());
    if (!is_valid_hccl_dtype(data_type)) {
      work->wait();
      tensors[i].copy_(reduction_tensors[i].to(tensors[i].scalar_type()));
    }
  }

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const AllToAllOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  auto flattenedIn = newLikeFlat(inputTensors);
  auto flattenedOut = newLikeFlat(outputTensors);
  for (const auto i : c10::irange(inputTensors.size())) {
    flattenedIn[i].copy_(inputTensors.at(i));
  }
  std::vector<at::Tensor> inputTensorsFlat;
  std::vector<at::Tensor> outputTensorsFlat;
  inputTensorsFlat.push_back(flattenedIn);
  outputTensorsFlat.push_back(flattenedOut);
  auto work = collective(
      inputTensorsFlat,
      outputTensorsFlat,
      [&](at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        hcclDataType_t hccl_data_type;
        int64_t hccl_numel = input.numel();
        hcclResult_t hccl_result{hcclSuccess};
        const auto scalar_type = input.scalar_type();

        habana_helpers::getCountDatatype(
            scalar_type, input.element_size(), hccl_numel, hccl_data_type);

        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] alltoall with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            hccl_numel,
            " data_type :: ",
            hccl_data_type);

        hccl_result = hcclAlltoAll(
            send_buffer,
            recv_buffer,
            hccl_numel,
            hccl_data_type,
            hccl_comm,
            stream);
        return hccl_result;
      });

  work->wait();
  for (const auto i : c10::irange(outputTensors.size())) {
    outputTensors.at(i).copy_(
        flattenedOut[i].to(inputTensors.at(i).scalar_type()));
  }

  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    [[maybe_unused]] const AllToAllOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  c10::intrusive_ptr<Work> work;
  at::Tensor alltoall_out_tensors;
  at::Tensor alltoall_in_tensors;
  auto out_scalar_t = outputTensor.scalar_type();
  auto data_type = habana_helpers::getHCCLDataType(outputTensor.scalar_type());
  if (is_valid_hccl_dtype(data_type) || out_scalar_t == at::kInt ||
      out_scalar_t == at::kLong) {
    alltoall_out_tensors = outputTensor;
    alltoall_in_tensors = inputTensor;
  } else {
    PT_DISTRIBUTED_DEBUG("[PYT-DIST] alltoall tensors converted to float");
    alltoall_out_tensors = outputTensor.to(c10::ScalarType::Float);
    alltoall_in_tensors = inputTensor.to(c10::ScalarType::Float);
  }

  std::vector<at::Tensor> inputTensors;
  std::vector<at::Tensor> outputTensors;
  inputTensors.push_back(alltoall_in_tensors);
  outputTensors.push_back(alltoall_out_tensors);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    work = collective(
        inputTensors,
        outputTensors,
        [numRanks = getSize(), rank = getRank(), this](
            at::Tensor& input,
            [[maybe_unused]] at::Tensor& output,
            const void* send_buffer,
            void* recv_buffer,
            hcclComm_t& hccl_comm,
            synStreamHandle stream) {
          int64_t hccl_numel = input.numel();
          auto hccl_data_type =
              habana_helpers::getHCCLDataType(input.scalar_type());

          const auto scalar_type = input.scalar_type();
          habana_helpers::getCountDatatype(
              scalar_type, input.element_size(), hccl_numel, hccl_data_type);

          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] alltoall with input_address :: ",
              send_buffer,
              " output_address :: ",
              recv_buffer,
              " elem_cnt :: ",
              hccl_numel,
              " data_type :: ",
              hccl_data_type);
          hcclResult_t hccl_result{hcclSuccess};
          if (!this->emulate_distributed_) {
            hccl_result = hcclAlltoAll(
                send_buffer,
                recv_buffer,
                hccl_numel,
                hccl_data_type,
                hccl_comm,
                stream);
          }
          return hccl_result;
        });
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    work = collective(
        inputTensors,
        outputTensors,
        [numRanks = getSize(), inputSplitSizes, outputSplitSizes, this](
            at::Tensor& input,
            at::Tensor& output,
            const void* send_buffer,
            void* recv_buffer,
            hcclComm_t& hccl_comm,
            synStreamHandle stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          const auto scalar_type = input.scalar_type();

          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          int64_t hccl_numel = input.numel();
          auto hccl_data_type =
              habana_helpers::getHCCLDataType(input.scalar_type());
          PT_DISTRIBUTED_DEBUG(
              "[PYT-DIST] alltoall with input_address :: ",
              send_buffer,
              " output_address :: ",
              recv_buffer,
              " elem_cnt :: ",
              hccl_numel,
              " data_type :: ",
              hccl_data_type);
          size_t ele_size = input.element_size();
          habana_helpers::getCountDatatype(
              scalar_type, input.element_size(), hccl_numel, hccl_data_type);
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
                  hccl_data_type,
                  r,
                  hccl_comm,
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }

            if (recv_lengths[r] != 0) {
              hccl_result = hcclRecv(
                  reinterpret_cast<unsigned char*>(recv_buffer) +
                      recv_offsets[r] * ele_size,
                  recv_lengths[r],
                  hccl_data_type,
                  r,
                  hccl_comm,
                  stream);
              if (hccl_result != hcclSuccess)
                return hccl_result;
            }
          }
          hcclGroupEnd();
          return hccl_result;
        });
  }

  if (!is_valid_hccl_dtype(data_type) && out_scalar_t != at::kInt &&
      out_scalar_t != at::kLong) {
    work->wait();
    outputTensor.copy_(alltoall_out_tensors.to(outputTensor.scalar_type()));
  }
  PT_DISTRIBUTED_END;
  return work;
}

// _broadcast_oop adds an out-of-place broadcast
// One use-case is implementing a vector all_gather
// where unevenly sized inputs are gathered among participating ranks
c10::intrusive_ptr<Work> ProcessGroupHcclBase::_broadcast_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const BroadcastOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  auto out_tensor = outputTensors.back();
  auto in_tensor = inputTensors.back();
  if (out_tensor.numel() != in_tensor.numel()) {
    PT_DISTRIBUTED_FATAL(
        "Tensor input and output of _broadcast_oop must have the same number of elements");
  }
  auto work = collective(
      inputTensors,
      outputTensors,
      [rootRank = opts.rootRank, this](
          at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        hcclDataType_t hccl_data_type;
        const auto scalar_type = input.scalar_type();
        auto hccl_numel = input.numel();

        habana_helpers::getCountDatatype(
            scalar_type,
            input.element_size(),
            hccl_numel,
            hccl_data_type,
            always_support_int64_);

        size_t element_size = habana_helpers::getHCCLDataSize(hccl_data_type);
        size_t chunk_size_in_elems =
            getHCCLSliceSize(habana_helpers::collectiveBroadcast) /
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
          if (!this->emulate_distributed_) {
            hccl_result = hcclBroadcast(
                static_cast<const uint8_t*>(send_buffer) + data_offset,
                static_cast<uint8_t*>(recv_buffer) + data_offset,
                num_elements_in_current_chunk,
                hccl_data_type,
                rootRank,
                hccl_comm,
                stream);
          }

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

c10::intrusive_ptr<Work> ProcessGroupHcclBase::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  bool same_size = check_same_size(outputTensors.back());
  if (same_size) {
    bool change = false;
    size_t tensor_size = outputTensors[0].size();
    std::unique_ptr<std::unique_ptr<bool[]>[]> changed(
        new std::unique_ptr<bool[]>[tensor_size]());
    std::vector<std::vector<std::vector<int64_t>>> sizeList(tensor_size);
    std::vector<std::vector<std::vector<int64_t>>> strideList(tensor_size);
    for (size_t i = 0; i < outputTensors.size(); i++) {
      changed[i] = std::make_unique<bool[]>(outputTensors[i].size());
      sizeList[i].resize(outputTensors[i].size());
      strideList[i].resize(outputTensors[i].size());
      resizeTensor(outputTensors[i], changed[i], sizeList[i], strideList[i]);
    }
    std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
    std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
    std::vector<std::vector<int64_t>> in_strideList(tensor_size);
    change = resizeTensor(inputTensors, in_changed, in_sizeList, in_strideList);
    auto outputFlattened = habana_helpers::flatten_for_scatter_gather(
        outputTensors, inputTensors, size_);

    auto work = collective(
        inputTensors,
        outputFlattened,
        [&](at::Tensor& input,
            [[maybe_unused]] at::Tensor& output,
            const void* send_buffer,
            void* recv_buffer,
            hcclComm_t& hccl_comm,
            synStreamHandle stream) {
          auto scalar_type = input.scalar_type();
          auto hccl_data_type = habana_helpers::getHCCLDataType(scalar_type);
          auto hccl_numel = input.numel();
          habana_helpers::getCountDatatype(
              scalar_type,
              input.element_size(),
              hccl_numel,
              hccl_data_type,
              always_support_int64_);
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
          if (!this->emulate_distributed_) {
            hccl_result = hcclAllGather(
                send_buffer,
                recv_buffer,
                hccl_numel,
                hccl_data_type,
                hccl_comm,
                stream);
          }
          return hccl_result;
        });
    // Record even for outputFlattened on ncclStream
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      for (size_t j = 0; j < outputTensors[0].size(); ++j) {
        if (!this->emulate_distributed_) {
          outputTensors[i][j].copy_(outputFlattened[i][j], true);
        } else {
          outputTensors[i][j].copy_(inputTensors[i], true);
        }
      }
    }
    if (change) {
      PT_IRGRAPH_DEBUG("step marker due to ProcessGroupHcclBase::allgather");
      habana_lazy::HbLazyTensor::StepMarker();
    }
    for (size_t i = 0; i < outputTensors.size(); i++) {
      restoreTensorsize(
          outputTensors[i], changed[i], sizeList[i], strideList[i], work);
    }
    restoreTensorsize(
        inputTensors, in_changed, in_sizeList, in_strideList, work);

    PT_DISTRIBUTED_END;
    return work;
  } else {
    const auto num_devices = outputTensors.size();
    const auto num_reduces = outputTensors[0].size();
    auto rank = getRank();
    c10::intrusive_ptr<Work> work;
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        outputs_multi_dev[j] = outputTensors[j][i];
        inputs_multi_dev[j] = i == (rank * num_devices + j)
            ? inputTensors[j]
            : outputs_multi_dev[j];
      }
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      work = _broadcast_oop(outputs_multi_dev, inputs_multi_dev, broadcastOpts);
      if (i != num_reduces - 1) {
        work->wait();
      }
    }
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    [[maybe_unused]] const AllgatherOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  // size compatible with resize method api and with singularity of allgather
  // base
  auto tensor_size{1};
  std::unique_ptr<bool[]> in_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> in_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> in_strideList(tensor_size);

  std::unique_ptr<bool[]> out_changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> out_sizeList(tensor_size);
  std::vector<std::vector<int64_t>> out_strideList(tensor_size);

  bool change =
      resizeTensor(outputs, out_changed, out_sizeList, out_strideList);
  change |= resizeTensor(inputs, in_changed, in_sizeList, in_strideList);

  auto work = collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          [[maybe_unused]] at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] _allgather_base with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(input.scalar_type()));
        auto scalar_type = input.scalar_type();
        auto hccl_data_type = habana_helpers::getHCCLDataType(scalar_type);
        auto hccl_numel = input.numel();
        habana_helpers::getCountDatatype(
            scalar_type, input.element_size(), hccl_numel, hccl_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclAllGather(
              send_buffer,
              recv_buffer,
              hccl_numel,
              hccl_data_type,
              hccl_comm,
              stream);
        }
        return hccl_result;
      });

  if (change) {
    habana_lazy::HbLazyTensor::StepMarker();
  }

  restoreTensorsize(inputs, in_changed, in_sizeList, in_strideList, work);
  restoreTensorsize(outputs, out_changed, out_sizeList, out_strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::allgather_coalesced(
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& /* unused */,
    [[maybe_unused]] std::vector<at::Tensor>& /* unused */,
    [[maybe_unused]] const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHcclBase does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::gather(
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& outputTensors,
    [[maybe_unused]] std::vector<at::Tensor>& inputTensors,
    [[maybe_unused]] const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupHcclBase does not support gather");
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::scatter(
    [[maybe_unused]] std::vector<at::Tensor>& /*outputTensors*/,
    [[maybe_unused]] std::vector<std::vector<at::Tensor>>& /*inputTensors*/,
    [[maybe_unused]] const ScatterOptions& /*opts*/) {
  throw std::runtime_error("ProcessGroupHcclBase does not support scatter");
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  auto inputFlattened = habana_helpers::flatten_for_scatter_gather(
      inputTensors, outputTensors, size_);
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    for (size_t j = 0; j < inputTensors[0].size(); ++j) {
      inputFlattened[i][j].copy_(inputTensors[i][j], true);
    }
  }
  auto work = collective(
      inputFlattened,
      outputTensors,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        // Wait for event on input
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] reduce_scatter with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            output.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(input.scalar_type()));
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclReduceScatter(
              send_buffer,
              recv_buffer,
              output.numel(),
              habana_helpers::getHCCLDataType(input.scalar_type()),
              habana_helpers::getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
        }
        return hccl_result;
      });
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::_reduce_scatter_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const ReduceScatterOptions& opts) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (input_tensor.numel() != output_tensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size");
  }
  at::Tensor reduce_out_tensors;
  at::Tensor reduce_in_tensors;
  auto out_scalar_t = output_tensor.scalar_type();
  auto data_type = habana_helpers::getHCCLDataType(output_tensor.scalar_type());
  if (is_valid_hccl_dtype(data_type) || out_scalar_t == at::kInt ||
      out_scalar_t == at::kLong) {
    reduce_out_tensors = output_tensor;
    reduce_in_tensors = input_tensor;
  } else {
    PT_DISTRIBUTED_DEBUG("[PYT-DIST] alltoall tensors converted to float");
    reduce_out_tensors = output_tensor.to(c10::ScalarType::Float);
    reduce_in_tensors = input_tensor.to(c10::ScalarType::Float);
  }
  // just a wrapper to fit the collective interface
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputs;
  inputs.push_back(reduce_in_tensors);
  outputs.push_back(reduce_out_tensors);

  auto work = collective(
      inputs,
      outputs,
      [reduceOp = opts.reduceOp, this](
          at::Tensor& input,
          at::Tensor& output,
          const void* send_buffer,
          void* recv_buffer,
          hcclComm_t& hccl_comm,
          synStreamHandle stream) {
        // Wait for event on input
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] _reduce_scatter_base with input_address :: ",
            send_buffer,
            " output_address :: ",
            recv_buffer,
            " elem_cnt :: ",
            output.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(input.scalar_type()));
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclReduceScatter(
              send_buffer,
              recv_buffer,
              output.numel(),
              habana_helpers::getHCCLDataType(input.scalar_type()),
              habana_helpers::getHCCLReduceOp(reduceOp),
              hccl_comm,
              stream);
        }
        return hccl_result;
      });
  if (!is_valid_hccl_dtype(data_type) && out_scalar_t != at::kInt &&
      out_scalar_t != at::kLong) {
    work->wait();
    output_tensor.copy_(reduce_out_tensors.to(output_tensor.scalar_type()));
  }
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    [[maybe_unused]] int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  PT_IRGRAPH_DEBUG("step marker due to ProcessGroupHcclBase::send");
  habana_lazy::HbLazyTensor::StepMarker();
  permutedSendTensorsToDense(tensors);
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          void* send_buff,
          hcclComm_t& hccl_comm,
          synStreamHandle stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            send_buff,
            " elem_cnt :: ",
            input.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(input.scalar_type()));
        auto scalar_type = input.scalar_type();
        auto hccl_data_type = habana_helpers::getHCCLDataType(scalar_type);
        auto hccl_numel = input.numel();
        habana_helpers::getCountDatatype(
            scalar_type, input.element_size(), hccl_numel, hccl_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclSend(
              send_buff,
              hccl_numel,
              hccl_data_type,
              peerRank,
              hccl_comm,
              stream);
        }
        return hccl_result;
      },
      dstRank);
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    [[maybe_unused]] int tag) {
  PT_DISTRIBUTED_BEGIN;
  habana_lazy::NoAccThread no_acc_thread;
  size_t tensor_size = tensors.size();
  std::unique_ptr<bool[]> changed(new bool[tensor_size]);
  std::vector<std::vector<int64_t>> sizeList(tensor_size);
  std::vector<std::vector<int64_t>> strideList(tensor_size);
  resizeTensor(tensors, changed, sizeList, strideList);
  PT_IRGRAPH_DEBUG("step marker due to ProcessGroupHcclBase::recv");
  habana_lazy::HbLazyTensor::StepMarker();
  clearPermutesFromRecvTensors(tensors);
  auto work = pointToPoint(
      tensors,
      [&](at::Tensor& tensor,
          void* recv_buff,
          hcclComm_t& hccl_comm,
          synStreamHandle stream,
          int peerRank) {
        PT_DISTRIBUTED_DEBUG(
            "[PYT-DIST] send with input_address :: ",
            recv_buff,
            " elem_cnt :: ",
            tensor.numel(),
            " data_type :: ",
            habana_helpers::getHCCLDataType(tensor.scalar_type()));
        auto scalar_type = tensor.scalar_type();
        auto hccl_data_type = habana_helpers::getHCCLDataType(scalar_type);
        auto hccl_numel = tensor.numel();
        habana_helpers::getCountDatatype(
            scalar_type, tensor.element_size(), hccl_numel, hccl_data_type);
        hcclResult_t hccl_result{hcclSuccess};
        if (!this->emulate_distributed_) {
          hccl_result = hcclRecv(
              recv_buff,
              hccl_numel,
              hccl_data_type,
              peerRank,
              hccl_comm,
              stream);
        }
        return hccl_result;
      },
      srcRank);
  restoreTensorsize(tensors, changed, sizeList, strideList, work);
  PT_DISTRIBUTED_END;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHcclBase::recvAnysource(
    std::vector<at::Tensor>& /*tensors*/,
    int /*tag*/) {
  throw std::runtime_error("ProcessGroupHcclBase does not support recv");
}

void ProcessGroupHcclBase::hostBarrier() {
  PT_DISTRIBUTED_BEGIN;
  if (this->emulate_distributed_) {
    return;
  }

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

void ProcessGroupHcclBase::goodbyeHandshake() {
  // goodbyeHandshake is a host barrier which ensures that master rank (0) will
  // finish last. It is important due to the fact that store used to implement
  // this barrier is hosted on master rank, so once process group on master rank
  // is destructed the store will be destroyed as well. This is why process
  // group on master rank has to be destroyed at the end.
  PT_DISTRIBUTED_BEGIN
  if (this->emulate_distributed_) {
    return;
  }

  if (comm_) {
    std::ostringstream barrierKeyStream;
    barrierKeyStream << "GOODBYE_HS:" << comm_->GetId();
    std::string barrierKey = barrierKeyStream.str();
    auto worker_count = store_->add(barrierKey, 1);
    if (getRank() == 0) {
      while (worker_count != size_) {
        worker_count = store_->add(barrierKey, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }
}

} // namespace c10d
