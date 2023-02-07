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

#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/script.h>

#include <synapse_common_types.h>
#include <synapse_helpers/device_types.h>
#include <synapse_helpers/graph.h>
#include <synapse_helpers/habana_tensor.h>

// set to 5 considering tensors upto 5d are most common case where we would like
// to use SmallVector to avoid heap allocation
constexpr uint32_t NUM_TENSOR_DIMS = 5;
using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using ValPtr = torch::jit::Value*;
using SmallSizeVec = c10::SmallVector<int64_t, NUM_TENSOR_DIMS>;

namespace habana_helpers {
std::string DebugString(const at::Tensor& t, bool print_data = false);
std::string DebugString(const IVal& a);
std::string DebugString(const IValPtrShared& a);
void PrintTensor(
    const at::Tensor& t,
    std::string tname,
    bool print_data = false);
} // namespace habana_helpers

#define PRINT_TENSOR(T) habana_helpers::PrintTensor(T, std::string(#T))
#define PRINT_TENSOR_WITH_DATA(T) \
  habana_helpers::PrintTensor(T, std::string(#T), true)

namespace habana_helpers {

int64_t tensor_numel(const at::Tensor& self);

std::vector<int64_t> infer_size(c10::IntArrayRef shape, int64_t numel);

at::Tensor hpu_cast_tensor(const at::Tensor& Input, caffe2::TypeMeta type);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::Tensor& self,
    unsigned num_dimensions);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<const at::Tensor*>& vec);

std::vector<void*> extract_data_ptrs(const std::vector<at::Tensor>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<at::Tensor>& vec);

std::string name_suffix_from_type(
    const c10::ScalarType pt_type,
    bool use_int64 = false);

at::Tensor to_cpu(const at::Tensor& hpu_tensor);

void copy_data_to_host(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_data_to_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_data_within_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_scalar_to_device(void* src_ptr, const at::Tensor& dst, uint32_t size);
void copy_scalars_to_device(
    const std::vector<std::pair<at::Tensor, at::Tensor>>& tensors_list);

at::Tensor GenerateAndCopyTensorToHPU(
    const at::Tensor& ref_tensor,
    const float value,
    bool is_persistent);

void change_tensors_to_memory_format(
    std::vector<at::Tensor*> pt_outputs,
    std::vector<const at::Tensor*> pt_inputs,
    std::vector<const at::IntArrayRef*> pt_new_pos,
    c10::MemoryFormat memory_format);

void change_tensor_strides(
    at::Tensor* pt_output,
    const at::Tensor* pt_input,
    const at::IntArrayRef* pt_new_pos);

c10::MemoryFormat get_memory_format(std::vector<const at::Tensor*> pt_inputs);

size_t hash_combine_scalars(
    size_t hash_code,
    at::ArrayRef<torch::jit::IValue> input_refs);

void recalc_strides(
    std::vector<int64_t>& self_strides,
    const std::vector<int64_t>& self_sizes);

bool is_supported_type(c10::ScalarType type);
bool is_shape_tensor(synTensorType shape_tensor);
std::vector<int64_t> calculate_strides(std::vector<int64_t> sizes);

} // namespace habana_helpers
