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

#include <ATen/ATen.h>
#include <backend/synapse_helpers/device.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <fmt/format.h>
#include <synapse_common_types.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/habana_tensor.h"

// set to 5 considering tensors upto 5d are most common case where we would like
// to use SmallVector to avoid heap allocation
constexpr uint32_t NUM_TENSOR_DIMS = 5;
using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using VecOfIValPtrSh = std::vector<IValPtrShared>;
using ValPtr = torch::jit::Value*;
using CValPtr = const torch::jit::Value*;
using SmallSizeVec = c10::SmallVector<int64_t, NUM_TENSOR_DIMS>;
using CValuePtrToIValuePtrMap = std::unordered_map<CValPtr, IValPtrShared>;
using SynTensorOrRefList = std::vector<synapse_helpers::tensor_or_ref>;
using SharedSynTensorOrRefListPtr = std::shared_ptr<SynTensorOrRefList>;

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

void print_tensor_debug(const torch::Tensor& src);

inline bool IsCollective(const c10::Symbol& symbol) {
  static c10::Symbol hccl_namepsace =
      c10::Symbol::fromQualString("namespaces::hccl");
  return symbol.ns() == hccl_namepsace;
}

std::vector<int64_t> infer_size(c10::IntArrayRef shape, int64_t numel);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<const at::Tensor*>& vec);

std::vector<void*> extract_data_ptrs(const std::vector<at::Tensor>& vec);
std::vector<synapse_helpers::device_ptr> extract_storage_data_ptrs(
    const std::vector<at::Tensor>& vec);

void copy_data_to_host(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_data_to_host(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream);

void copy_data_to_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_data_to_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream,
    void* host_ptr = nullptr);

void copy_data_within_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking);

void copy_scalar_to_device(void* src_ptr, const at::Tensor& dst, uint64_t size);

void copy_scalars_to_device(
    const std::vector<std::pair<at::Tensor, at::Tensor>>& tensors_list);

size_t hash_combine_scalars(
    size_t hash_code,
    at::ArrayRef<torch::jit::IValue> input_refs);

void recalc_strides(
    std::vector<int64_t>& self_strides,
    const std::vector<int64_t>& self_sizes);

bool is_supported_type(c10::ScalarType type);

bool is_shape_tensor(synTensorType shape_tensor);

std::vector<int64_t> calculate_strides(std::vector<int64_t> sizes);

/**
 * Tokens that can be passed to PT_BACKEND_DEBUG_TENSOR to insert
 * stringified value of some internal tensor field without exposing
 * underlying class of the tensor.
 */
enum FormatTokens { Permutations = 1, Layout = 2, ImplPtr = 3, DataPtr = 4 };

namespace detail {
template <typename T, class Enable = void>
struct InternalFormatter final {};

template <typename T>
struct InternalFormatter<T> {
  static const T& format(const at::Tensor&, const T& t) {
    return t;
  }
};

template <>
struct InternalFormatter<FormatTokens> {
  static std::string format(const at::Tensor&, FormatTokens t);
};

} // namespace detail

template <typename... Args>
void debug_log_internal_tensor(
    const at::Tensor& tensor,
    const std::string_view format_string,
    Args... args) {
  PT_BRIDGE_DEBUG(fmt::format(
      format_string,
      habana_helpers::detail::InternalFormatter<Args>::format(
          tensor, args)...));
}
} // namespace habana_helpers

/**
 * Macro to eliminate explicit dependency of the debug logs in backend on
 * the GetHbInternalTensorImpl. This macro has printf semantics and all the
 * formatted args are passed as is to absl::StrFormat, except for the
 * FormatTokens that are converted into a string representation of the
 * requested tensor field.
 */
#define PT_BACKEND_DEBUG_TENSOR(tensor, format_string, args...)             \
  if (IS_MOD_DEBUG_ENABLED(PT_BRIDGE)) {                                    \
    habana_helpers::debug_log_internal_tensor(tensor, format_string, args); \
  }
