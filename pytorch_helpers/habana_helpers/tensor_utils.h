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

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>

#include <synapse_common_types.h>
#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "synapse_helpers/device_types.h"

namespace habana_helpers {
struct StorageLessWrapperTensorImpl : public c10::TensorImpl {
  explicit StorageLessWrapperTensorImpl(
      const at::Tensor& rep,
      at::optional<caffe2::TypeMeta> data_type = c10::nullopt)
      : TensorImpl(
            c10::DispatchKeySet(c10::DispatchKey::HABANATensorId),
            data_type.has_value() ? data_type.value() : rep.dtype(),
            rep.device()) {}

  void release_resources() override {}

  bool has_storage() const override {
    return false;
  }

  const at::Storage& storage() const override {
    TORCH_CHECK(0, "StorageLessWrapperTensorImpl tensors do not have storage");
  }
};

at::Tensor hpu_cast_tensor(const at::Tensor& Input, caffe2::TypeMeta type);

at::Tensor cast_tensor_to_integer(const at::Tensor& long_tensor);

at::Tensor cast_tensor_to_long(const at::Tensor& int_tensor);

c10::ScalarType scalar_type(const c10::Scalar& s);

synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::Tensor& self,
    unsigned num_dimensions);

bool alwaysAllocOnDevice();

at::Tensor nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options = {},
    at::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt,
    at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

at::Tensor nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options = {},
    at::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt,
    at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

at::Tensor createPTTensor(const at::Tensor& input, bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    c10::ScalarType data_type,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent);

/**
@brief This function can be used to create an intermediate
       synapse_helper tensor of required shape (which is
       different from shape of input & output tensors)
**/
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

/**
@brief This function can be used to create an intermediate
       synapse_helper tensor of required shape and synDataType
       as ScalarType dosen't represent all synapse supported types
**/
synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    const synGraphHandle graph,
    bool persistent,
    const synDataType dtype);

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

synapse_helpers::tensor duplicate_tensor_in_memory_section_with_size(
    const synapse_helpers::tensor& tensor,
    std::vector<int64_t>& sizes,
    const uint64_t offset);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<const at::Tensor*>& vec);

std::vector<void*> extract_data_ptrs(const std::vector<at::Tensor>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<at::Tensor>& vec);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::vector<std::string> names(
    const std::vector<synapse_helpers::tensor_or_ref>&);

std::vector<std::string> names(
    const std::deque<synapse_helpers::tensor_or_ref>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);

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

void copy_scalar_to_host(const at::Tensor& src, void* dst_ptr, uint32_t size);
void copy_scalar_to_device(void* src_ptr, const at::Tensor& dst, uint32_t size);

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
} // namespace habana_helpers
