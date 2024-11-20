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
#include <torch/torch.h>

#define MAX_TPC_SUPPORTED_REPEAT_DIMS (5)
#define MAX_DIMS_FOR_ADVANCED_INDEXING (MAX_TPC_SUPPORTED_REPEAT_DIMS)

namespace habana {
std::vector<int64_t> ComputeOutputShapeWithAdvIndexing(
    std::vector<int64_t> input_shape,
    std::vector<bool> adv_index_dims,
    std::vector<std::vector<int64_t>> indexing_tensor_shapes);

std::vector<int64_t> indices_size(at::TensorList indices);

bool hasContiguousSubspace(c10::ArrayRef<c10::IValue> indices_ival);

int hasContiguousSubspace(std::vector<int64_t> implicit_indices_pos_vec);

std::tuple<std::vector<int64_t>, std::vector<at::Tensor>> transposeToFront(
    const at::Stack& stack);

std::vector<std::vector<int64_t>> calc_indexing_tensors_shapes(
    const at::Stack& stack);

std::vector<std::vector<int64_t>> calc_plain_indexing_tensors_shapes(
    const at::Stack& stack);

std::tuple<bool, int, std::vector<int64_t>, std::vector<at::Tensor>>
generate_advanced_indexing_indices_list(const at::Stack& stack);

bool check_for_adv_indexing(c10::ArrayRef<c10::IValue> indices_in_orig);
bool handle_bool_mask_indices(
    c10::ArrayRef<c10::IValue>& indices_in_orig,
    std::vector<c10::IValue>& indices_in_ivals_vec,
    std::vector<c10::optional<at::Tensor>>& bool_indices_vec);

std::vector<int64_t> ComputeIndexOperatorOutputShape(
    const at::Tensor& input,
    at::TensorList indices);


std::vector<int64_t> broadcast_size(at::TensorList indices);

std::vector<int64_t> CalcCatOutSize(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t* dim_inp);

} // namespace habana
