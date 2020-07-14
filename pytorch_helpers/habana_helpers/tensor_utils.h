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
at::Tensor hpu_cast_tensor(const at::Tensor& Input, caffe2::TypeMeta type);

at::Tensor cast_tensor_to_integer(const at::Tensor& long_tensor);

c10::ScalarType scalar_type(const c10::Scalar& s);

synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::Tensor& self,
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
    const std::vector<at::Tensor>& tensors,
    synGraphHandle graph,
    const std::vector<bool> persistents,
    const std::vector<c10::optional<c10::ScalarType>> dtypes);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<at::Tensor>& tensors,
    synGraphHandle graph,
    bool persistent);

synapse_helpers::tensor duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);

std::vector<void*> extract_data_ptrs(const std::vector<at::Tensor>& vec);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::vector<std::string> names(
    const std::vector<synapse_helpers::tensor_or_ref>&);

std::vector<std::string> names(
    const std::deque<synapse_helpers::tensor_or_ref>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);

at::Tensor to_cpu(const at::Tensor& hpu_tensor);

void copy_data_to_host(const at::Tensor& src, void* dst_ptr, uint32_t size);

void copy_data_to_device(void* src_ptr, const at::Tensor& dst, uint32_t size);

void copy_data_within_device(const at::Tensor& src, const at::Tensor& dst);

void change_tensors_to_memory_format(
    std::vector<at::Tensor*> pt_outputs,
    std::vector<const at::Tensor*> pt_inputs,
    std::vector<const at::IntArrayRef*> pt_new_pos,
    c10::MemoryFormat memory_format);

c10::MemoryFormat get_memory_format(std::vector<const at::Tensor*> pt_inputs);

} // namespace habana_helpers
