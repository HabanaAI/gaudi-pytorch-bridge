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
#include <ATen/Tensor.h>
#include "backend/synapse_helpers/layout_utils.h"
namespace lazy_to_backend {
bool is_const_tensor(const at::Tensor& tensor);
void* host_ptr_for_const_tensor(const at::Tensor& tensor);

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool>
get_memory_permutation(const at::Tensor& tensor);

bool is_lazy_inference_call_context();
} // namespace lazy_to_backend
