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
#include "collective_utils.h"

#include <c10/core/ScalarType.h>
#include <torch_ver/csrc/distributed/c10d/Utils.hpp>
#include <map>
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/utils.h"

namespace habana_helpers {

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

// HCCL op mapping
const std::map<c10d::ReduceOp, hcclRedOp_t> hcclOp = {
    {c10d::ReduceOp::MIN, hcclMin},
    {c10d::ReduceOp::MAX, hcclMax},
    {c10d::ReduceOp::SUM, hcclSum},
    {c10d::ReduceOp::PRODUCT, hcclProd},
};

hcclRedOp_t getHCCLReduceOp(
    const c10d::ReduceOp reduceOp,
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

size_t getHCCLSliceSize(collectiveKind_t kind, bool lazy_collective) {
  if (habana::HPUDeviceContext::get_device().type() !=
      synDeviceType::synDeviceGaudi) {
    return INT64_MAX;
  }

  size_t slice_size = GET_ENV_FLAG_NEW(PT_HCCL_SLICE_SIZE_MB);
  if (lazy_collective || (slice_size != DEFAULT_HCCL_SLICE_SIZE_MB)) {
    // user has set slicing for tuning or its lazy collective.
    return slice_size * 1024 * 1024;
  }

  // hccl slicing is static for now and will get updated once SIMB is enabled
  switch (kind) {
    case collectiveAllReduce:
    case collectiveReduceScatter:
    case collectiveBroadcast:
      slice_size = 128;
      break;
    case collectiveReduce:
    case collectiveAllGather:
      slice_size = 16;
      break;
  }
  return slice_size * 1024 * 1024;
}

hcclDataType_t getHCCLDataType(at::ScalarType type) {
  // HCL doesn't have definition for fp8 types, use hcclUint8 instead
  // assume later function getCountDatatype() will set correct data type
  // TODO: removed this once HCL adds fp8 types
  if (at::kFloat8_e5m2 == type || at::kFloat8_e4m3fn == type) {
    return hcclUint8;
  }
  auto it = hcclDataType.find(type);
  TORCH_CHECK(
      it != hcclDataType.end(),
      "Input tensor data type is not supported for HCCL process group: ",
      type);
  return it->second;
}

size_t getHCCLDataSize(hcclDataType_t type) {
  const std::map<hcclDataType_t, size_t> type2size = {
      {hcclBfloat16, 2}, {hcclFloat, 4}};

  auto it = type2size.find(type);
  TORCH_CHECK(
      it != type2size.end(),
      "Getting size for given data type is not supported: ",
      type);
  return it->second;
}

void getCountDatatype(
    c10::ScalarType scalar_type,
    size_t element_size,
    int64_t& numel,
    hcclDataType_t& tensor_data_type,
    bool always_support_int64) {
  if ((scalar_type == at::kLong || scalar_type == at::kDouble) &&
      !common::IsInt64Supported() and !always_support_int64) {
    element_size = 4;
  }

  if (element_size == 4) {
    tensor_data_type = getHCCLDataType(at::kFloat);
    // numel doesn't change
  } else if (element_size == 2) {
    tensor_data_type = getHCCLDataType(at::kBFloat16);
    // numel doesn't change
  } else if ((numel * element_size) % 4 == 0) {
    tensor_data_type = getHCCLDataType(at::kFloat);
    numel = (numel * element_size) / 4;
  } else if ((numel * element_size) % 2 == 0) {
    tensor_data_type = getHCCLDataType(at::kBFloat16);
    numel = (numel * element_size) / 2;
  } else {
    TORCH_CHECK(
        false,
        "Provided tensor can't be represented as neither HCCL float32 nor HCCL bfloat16;",
        " numel: ",
        numel,
        ", element_size=",
        element_size,
        ", scalar_type: ",
        scalar_type);
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
    flattened[i] = c10d::newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace habana_helpers