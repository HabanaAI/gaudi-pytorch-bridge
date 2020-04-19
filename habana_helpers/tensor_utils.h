/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace habana_helpers {
c10::ScalarType scalar_type(const c10::Scalar& s);

synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::TensorOptions& options,
    unsigned num_dimensions);

/*
@brief This function can be used to create an intermediate
       synapse_helper tensor of required shape (which is
       different from shape of input & output tensors)
*/
synapse_helpers::tensor create_tensor(
    const c10::IntArrayRef& shape,
    synGraphHandle graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype);

synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    const synGraphHandle graph,
    bool persistent,
    const c10::optional<c10::ScalarType> dtype = c10::nullopt);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*> tensors,
    synGraphHandle graph,
    const std::vector<bool> persistents,
    const std::vector<c10::optional<c10::ScalarType>> dtypes);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*> tensors,
    synGraphHandle graph,
    bool persistent);

synapse_helpers::tensor duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);

at::Tensor to_cpu(const at::Tensor& hpu_tensor);

// TODO: remove this function. Workaround for SW-9962
[[deprecated]] at::Tensor contiguous_tensor(const at::Tensor& tensor);

void copy_data_to_host(const at::Tensor& src, void* dst_ptr, uint32_t size);

void copy_data_to_device(void* src_ptr, const at::Tensor& dst, uint32_t size);

void copy_data_within_device(const at::Tensor& src, const at::Tensor& dst);

} // namespace habana_helpers
