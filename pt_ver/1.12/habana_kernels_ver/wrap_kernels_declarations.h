/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>

namespace hpu_wrap {

at::Tensor _reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> optional_memory_format);
at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);
at::Tensor kl_div_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target);
at::Tensor& set_(
    at::Tensor& self,
    at::Storage source,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
::std::vector<at::Tensor> split(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim);
::std::vector<at::Tensor> split_with_sizes(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim);

} // namespace hpu_wrap
