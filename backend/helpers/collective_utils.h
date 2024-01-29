/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

hcclRedOp_t getHCCLReduceOp(const c10d::ReduceOp reduceOp);
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
