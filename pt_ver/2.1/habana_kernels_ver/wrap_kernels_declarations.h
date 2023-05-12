/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/ExpandUtils.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/script.h>

namespace hpu_wrap {

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride);
at::Tensor _unsafe_view(const at::Tensor& self, c10::SymIntArrayRef size);
at::Tensor empty(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> optional_memory_format);
at::Tensor empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);
at::Tensor slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<c10::SymInt> start,
    c10::optional<c10::SymInt> end,
    c10::SymInt step);
::std::vector<at::Tensor> split(
    const at::Tensor& self,
    c10::SymInt split_size,
    int64_t dim);
::std::vector<at::Tensor> split_with_sizes(
    const at::Tensor& self,
    c10::SymIntArrayRef split_sizes,
    int64_t dim);
at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe);
at::Tensor nonzero(const at::Tensor& self);
} // namespace hpu_wrap
