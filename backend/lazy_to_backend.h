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
#include <absl/strings/str_format.h>
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/logging.h"

namespace lazy_to_backend {
bool is_const_tensor(const at::Tensor& tensor);
void* host_ptr_for_const_tensor(const at::Tensor& tensor);

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool>
get_memory_permutation(const at::Tensor& tensor);
void set_memory_permutations(
    at::Tensor& tensor,
    synapse_helpers::layouts::MemoryPermutation permutation,
    const synRetrievedLaunchTensorInfoExt* info = nullptr);

void set_tensor_layout_format(at::Tensor& tensor, habana::LayoutFormat format);
habana::LayoutFormat get_tensor_layout_format(const at::Tensor& tensor);

bool is_lazy_inference_call_context();
bool is_shape_tensor(const at::Tensor& tensor);
at::Tensor create_empty_tensor(const PtTensorInfo& ti);
void set_host_ptr(const at::Tensor& tensor, void* host_ptr);
void* get_host_ptr(const at::Tensor& tensor);

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
    std::string_view format_string,
    Args... args) {
  PT_BRIDGE_DEBUG(absl::StrFormat(
      format_string,
      lazy_to_backend::detail::InternalFormatter<Args>::format(
          tensor, args)...));
}
} // namespace lazy_to_backend

/**
 * Macro to eliminate explicit dependency of the debug logs in backend on
 * the GetHbInternalTensorImpl. This macro has printf semantics and all the
 * formatted args are passed as is to absl::StrFormat, except for the
 * FormatTokens that are converted into a string representation of the
 * requested tensor field.
 */
#define PT_BACKEND_DEBUG_TENSOR(tensor, format_string, args...)              \
  if (IS_MOD_DEBUG_ENABLED(PtLogger::ModuleMask::BRIDGE)) {                  \
    lazy_to_backend::debug_log_internal_tensor(tensor, format_string, args); \
  }
