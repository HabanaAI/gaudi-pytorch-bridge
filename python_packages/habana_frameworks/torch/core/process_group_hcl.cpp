/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "process_group_hcl.h"
#include <map>
#include "habana_lazy/hpu_lazy_tensors.h"

using namespace synapse_helpers;
namespace c10d {

namespace {

std::map<at::ScalarType, synDataType> hclDataType = {
    {at::kByte, syn_type_int8},
    {at::kChar, syn_type_int8},
    {at::kDouble, syn_type_na},
    {at::kFloat, syn_type_float},
    {at::kHalf, syn_type_bf16},
    {at::kInt, syn_type_int32},
    {at::kLong, syn_type_na},
    {at::kShort, syn_type_int16},
    {at::kBFloat16, syn_type_bf16},
};

// HCL op mapping
std::map<ReduceOp, HCL_Op> hclOp = {
    {ReduceOp::MIN, eHCLOpNone},
    {ReduceOp::MAX, eHCLOpNone},
    {ReduceOp::SUM, eHCLSum},
    {ReduceOp::PRODUCT, eHCLMul},
};

HCL_Op getHCLOpType(ReduceOp type) {
  try {
    return hclOp.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error(
        "Unsupported operation type for HCL process group");
  }
}

synDataType getHCLDataType(at::ScalarType type) {
  try {
    return hclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for HCL process group");
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
            "All tensor operands to scatter/gather must have the same size");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

namespace {
constexpr const char* const kRankExchangeStoreKey = "RANK_EXCHANGE_STORE_KEY";
constexpr int kByteOffset = 8;
} // namespace

template <typename T>
inline std::vector<T> toVec(int num, int numBytes) {
  std::vector<T> values;
  // Read off bytes from right to left, pushing them into
  // char array.
  for (int i = 0; i < numBytes; i++) {
    uint8_t x = (num >> (kByteOffset * i)) & 0xff;
    values.push_back(static_cast<T>(x));
  }
  return values;
}

// Converts from char vec (such as from store read) to int.
template <typename T>
inline int fromVec(const std::vector<T>& values) {
  int num = 0;
  // Set each byte at the correct location on num
  for (auto i = 0; i < values.size(); i++) {
    uint8_t x = static_cast<uint8_t>(values[i]);
    num |= (static_cast<int>(x) << (kByteOffset * i));
  }
  return num;
}

std::shared_ptr<hcl_communicator> ProcessGroupHCL::getComm(int deviceId) {
  if (hcl_communicator_.find(deviceId) == hcl_communicator_.end()) {
    char* config_json_path = std::getenv("HCL_CONFIG_PATH");
    auto global_comm =
        hcl_communicator::get_or_create_world(deviceId, config_json_path ?: "");
    auto world_size = global_comm->size();
    auto size = getSize();
    auto rank = getRank();
    if (world_size == size) {
      hcl_communicator_[deviceId] = global_comm;
    } else {
      auto hcl_rank = global_comm->my_hcl_rank();
      std::vector<int> valid_ranks;
      // HCL Subcomm requires the list of ranks participating in the
      // communicator Pytorch interface provides support to a store interface
      // with set get capabilities.  Temporarily using this till HCCL adds
      // support for sub comm groups.  Pytorch provides supprt for PrefixStore
      // which creates a separate store for each ProcessGroup call. Rank 0 waits
      // for information from all the other ranks about the HCL rank which are
      // participating in this communicator group. Once it receives all the
      // ranks it broadcasts the list of all ranks participating in the
      // communicator to all the other processes.
      if (rank == 0) {
        std::vector<uint8_t> rank_buff;
        valid_ranks.push_back(hcl_rank);
        for (auto i = 1; i < size; i++) {
          auto dataKey = kRankExchangeStoreKey + std::to_string(i);
          store_->wait({dataKey});
          std::vector<uint8_t> values = store_->get(dataKey);
          valid_ranks.push_back(fromVec(values));
        }
        sort(valid_ranks.begin(), valid_ranks.end());
        for (auto valid_rank : valid_ranks) {
          std::vector<uint8_t> values = toVec<uint8_t>(valid_rank, sizeof(int));
          rank_buff.insert(rank_buff.end(), values.begin(), values.end());
        }

        store_->set(kRankExchangeStoreKey + std::to_string(rank), rank_buff);
      } else {
        std::vector<uint8_t> rank_value = toVec<uint8_t>(hcl_rank, sizeof(int));
        store_->set(kRankExchangeStoreKey + std::to_string(rank), rank_value);
        auto rootKey = kRankExchangeStoreKey + std::to_string(0);
        store_->wait({rootKey});
        std::vector<uint8_t> values = store_->get(rootKey);
        for (auto i = 0; i < size; i++) {
          valid_ranks.push_back(fromVec(std::vector<uint8_t>(
              values.begin() + (i * sizeof(int)),
              values.begin() + ((i + 1) * sizeof(int)))));
        }
      }
      hcl_communicator_[deviceId] =
          global_comm->create_subcommunicator(valid_ranks);
    }
  }
  return hcl_communicator_.find(deviceId)->second;
}
// TBD: Store not used for now and config done from file
// Initial support added for multiple devices on a single node
// So using rank as the device id.  This will be enhanced further

ProcessGroupHCL::ProcessGroupHCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& opTimeout)
    : ProcessGroup(rank, size), stop_(false), store_(store) {}

ProcessGroupHCL::~ProcessGroupHCL() {
  destroy();
}

void ProcessGroupHCL::destroy() {}

void ProcessGroupHCL::abort() {
  destroy();
}

ProcessGroupHCL::WorkHCL::WorkHCL(
    const std::vector<at::Tensor>& outputs,
    const std::vector<int>& devices,
    std::vector<std::shared_ptr<hcl_communicator>>& hcl_comms)
    : outputs_(outputs),
      devices_(devices),
      hcl_comms_(hcl_comms),
      workStartTime_(std::chrono::steady_clock::now()) {}
ProcessGroupHCL::WorkHCL::~WorkHCL() {}

bool ProcessGroupHCL::WorkHCL::isCompleted() {
  return exception() || wait(); // check for the completion of work;
}

bool ProcessGroupHCL::WorkHCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }
  // Add support for query from device
  return true;
}

// Same as calling synchronize().
bool ProcessGroupHCL::WorkHCL::wait(
    std::chrono::milliseconds timeout /*=kNoTimeout*/) {
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupHCL::WorkHCL::synchronize() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    hcl_comms_[i]->synchronize_output(
        (synapse_helpers::device_ptr)outputs_[i].storage().data_ptr().get());
  }
}

void ProcessGroupHCL::WorkHCL::abort() {
  TORCH_CHECK(false, "ProcessGroupHCL::WorkHCL::abort not implemented.");
}

c10::intrusive_ptr<ProcessGroupHCL::WorkHCL> ProcessGroupHCL::initWork(
    std::vector<at::Tensor>& outputs,
    std::vector<int> devices,
    std::vector<std::shared_ptr<hcl_communicator>>& hcl_comms) {
  return c10::make_intrusive<ProcessGroupHCL::WorkHCL>(
      outputs, devices, hcl_comms);
}

// Get the list of devices from list of tensors
std::vector<int> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<int> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.get_device());
  }
  return res;
}

std::vector<std::shared_ptr<hcl_communicator>> ProcessGroupHCL::getCommList(
    const std::vector<int>& devices) {
  std::vector<std::shared_ptr<hcl_communicator>> comms(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    comms[i] = getComm(int(devices[i]));
  }
  return comms;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::hclcollective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post) {
  habana_lazy::HbLazyTensor::StepMarker();

  const auto devices = getDeviceList(inputs);
  auto comms = getCommList(devices);
  auto work = initWork(outputs, devices, comms);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (getHCLDataType(inputs[i].scalar_type()) != syn_type_na) {
      fn(inputs[i], outputs[i], *(comms[i]));
    } else {
      LOG(INFO) << "HCL called on unsupported data type\n";
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    // Update work
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::hclcollective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn) {
  // Need to replace int by device work streams
  return hclcollective(
      inputs, outputs, fn, [](std::vector<int>&) {}, [](std::vector<int>&) {});
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return hclcollective(
      tensors,
      tensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.broadcast(
            opts.rootRank,
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            input.numel(),
            getHCLDataType(input.scalar_type()));
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // Pre-processing
  std::vector<at::Tensor> tmp_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    synDataType dtype = getHCLDataType(tensors[i].scalar_type());
    if (!synapse_helpers::hcl_communicator::is_reduction_dtype_valid(dtype)) {
      tmp_tensors.push_back(tensors[i].to(c10::ScalarType::Float));
    } else {
      tmp_tensors.push_back(tensors[i]);
    }
  }

  auto work = hclcollective(
      tmp_tensors,
      tmp_tensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.allreduce(
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)output.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            (synapse_helpers::device_ptr)output.storage().data_ptr().get(),
            input.numel(),
            getHCLDataType(input.scalar_type()),
            getHCLOpType(opts.reduceOp));
      });

  // Post-processing
  for (size_t i = 0; i < tensors.size(); ++i) {
    synDataType dtype = getHCLDataType(tensors[i].scalar_type());
    if (!synapse_helpers::hcl_communicator::is_reduction_dtype_valid(dtype)) {
      tensors[i].copy_(tmp_tensors[i].to(tensors[i].scalar_type()));
    }
  }

  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with HCL");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  return hclcollective(
      tensors,
      tensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.reduce(
            opts.rootRank,
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)output.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            (synapse_helpers::device_ptr)output.storage().data_ptr().get(),
            input.numel(),
            getHCLDataType(input.scalar_type()),
            getHCLOpType(opts.reduceOp));
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  std::vector<at::Tensor> inputTensors;
  std::vector<at::Tensor> outputTensors;
  inputTensors.push_back(inputTensor);
  outputTensors.push_back(outputTensor);
  return hclcollective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.alltoall(
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)output.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            (synapse_helpers::device_ptr)output.storage().data_ptr().get(),
            input.numel(),
            getHCLDataType(input.scalar_type()));
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  auto outputFlattened =
      flatten_for_scatter_gather(outputTensors, inputTensors, size_);

  return hclcollective(
      inputTensors,
      outputFlattened,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        auto work = hcl_comm.allgather(
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)output.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            (synapse_helpers::device_ptr)output.storage().data_ptr().get(),
            input.numel(),
            getHCLDataType(input.scalar_type()));

        for (size_t i = 0; i < outputTensors.size(); ++i) {
          for (size_t j = 0; j < outputTensors[0].size(); ++j) {
            outputTensors[i][j].copy_(outputFlattened[i][j], true);
          }
        }
        return work;
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error(
      "allgather_base is currently not supported with HCL");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHCL does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupHCL does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupHCL does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupHCL does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return hclcollective(
      tensors,
      tensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.send(
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            input.numel() * sizeof(getHCLDataType(input.scalar_type())),
            dstRank,
            tag);
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return hclcollective(
      tensors,
      tensors,
      [&](at::Tensor& input, at::Tensor& output, hcl_communicator& hcl_comm) {
        return hcl_comm.receive(
            (synapse_helpers::device_ptr)input.data_ptr(),
            (synapse_helpers::device_ptr)input.storage().data_ptr().get(),
            input.numel() * sizeof(getHCLDataType(input.scalar_type())),
            srcRank,
            tag);
      });
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("ProcessGroupHCL does not support recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupHCL::barrier(
    const BarrierOptions& opts) {
  std::vector<std::shared_ptr<hcl_communicator>> comms;
  std::vector<int> res;
  std::vector<at::Tensor> outputs;

  for (size_t i = 0; i < hcl_communicator_.size(); i++) {
    hcl_communicator_[i]->barrier();
  }
  auto work = initWork(outputs, res, comms);

  return work;
}

} // namespace c10d
