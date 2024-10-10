/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <c10/util/ArrayRef.h>
#include <tuple>

#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "habana_helpers/pt_version_check.h"

class PtTensorInfo;
namespace habana_helpers {

synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);
c10::ScalarType synapse_to_pytorch_type(const synDataType type);
synDataType pytorch_to_synapse_type(const c10::Scalar& s);

c10::ScalarType scalar_type(const c10::Scalar& s);

/**
@brief This function can be used to create an intermediate
       synapse_helper tensor of required shape (which is
       different from shape of input & output tensors)
**/
synapse_helpers::tensor create_tensor(
    const c10::IntArrayRef& shape,
    const c10::IntArrayRef& stride,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    int devid,
    const c10::ScalarType dtype,
    const std::string& name = std::string());

synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    const c10::optional<c10::ScalarType> dtype = c10::nullopt,
    const std::string& name = std::string(),
    const std::string& inference_name = std::string());

synapse_helpers::tensor create_shape_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    synTensorType shape_tensor_type,
    const std::string& name = std::string(),
    void* host_ptr = nullptr);

synapse_helpers::tensor create_shape_tensor(
    const c10::IntArrayRef& input_shapes,
    synDeviceId syn_device,
    synapse_helpers::graph& graph,
    bool persistent,
    synTensorType shape_tensor_type,
    const std::string& name = std::string(),
    void* host_ptr = nullptr);

synapse_helpers::tensor create_const_tensor(
    const c10::IntArrayRef& shape,
    const c10::IntArrayRef& stride,
    synapse_helpers::graph& graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const std::string& name = std::string());

/**
@brief This function can be used to create an intermediate
       synapse_helper tensor of required shape and synDataType
       as ScalarType dosen't represent all synapse supported types
**/
synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    const synDataType dtype,
    const std::string& name = std::string(),
    const std::string& inference_name = std::string());

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<at::Tensor>& tensors,
    synapse_helpers::graph& graph,
    const std::vector<bool>& persistents,
    const std::vector<bool>& externals,
    const std::vector<c10::optional<c10::ScalarType>> dtypes);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<at::Tensor>& tensors,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external);

synapse_helpers::tensor duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor,
    synapse_helpers::graph& graph,
    bool external);

synapse_helpers::tensor duplicate_tensor_in_memory_section_with_size(
    const synapse_helpers::tensor& tensor,
    synapse_helpers::graph& graph,
    std::vector<int64_t>& sizes,
    std::vector<int64_t>& strides,
    const uint64_t offset,
    bool external,
    synapse_helpers::layouts::MemoryPermutation permutation = {});

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::vector<std::string> names(
    const std::vector<synapse_helpers::tensor_or_ref>&);

std::vector<std::string> names(
    const std::deque<synapse_helpers::tensor_or_ref>&);

bool is_const_tensor(const at::Tensor& tensor);

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool>
get_tensor_memory_permutation(const at::Tensor& tensor);

void set_tensor_memory_permutations(
    const at::Tensor& tensor,
    const synapse_helpers::layouts::MemoryPermutation& permutation);

void update_tensor_layout_and_permutation(
    const at::Tensor& pt_tensor,
    const PtTensorInfo& ti);
at::Tensor create_empty_tensor(const PtTensorInfo& ti);

} // namespace habana_helpers
