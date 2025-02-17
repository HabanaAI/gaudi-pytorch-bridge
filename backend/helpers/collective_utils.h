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

#pragma once

#include <c10/core/ScalarType.h>
#include <hccl.h>
#include <hccl_types.h>
#include <torch_ver/csrc/distributed/c10d/Types.hpp>
#include <vector>

namespace habana_helpers {

typedef enum {
  collectiveAllReduce = 0,
  collectiveReduce = 1,
  collectiveAllGather = 2,
  collectiveReduceScatter = 3,
  collectiveBroadcast = 4,
} collectiveKind_t;

hcclRedOp_t getHCCLReduceOp(
    const c10d::ReduceOp reduceOp,
    const at::ScalarType type);

size_t getHCCLSliceSize(collectiveKind_t kind, bool lazy_collective = false);
size_t getHCCLDataSize(hcclDataType_t type);
void getCountDatatype(
    c10::ScalarType scalar_type,
    size_t element_size,
    int64_t& numel,
    hcclDataType_t& tensor_data_type,
    bool always_support_int64 = false);
hcclDataType_t getHCCLDataType(at::ScalarType type);
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size);

} // namespace habana_helpers
