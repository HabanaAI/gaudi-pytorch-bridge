/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> optional_memory_format);
at::Tensor empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);
at::Tensor slice(
    const at::Tensor& self,
    int64_t dim,
    ::std::optional<c10::SymInt> start,
    ::std::optional<c10::SymInt> end,
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
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe);
at::Tensor nonzero(const at::Tensor& self);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(
    const at::Tensor& self,
    bool sorted = true,
    bool return_inverse = false,
    bool return_counts = false);
::std::tuple<at::Tensor, at::Tensor> _unique(
    const at::Tensor& self,
    bool sorted = true,
    bool return_inverse = false);
at::Tensor batch_norm_elemt(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps);
at::Tensor batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batch_norm_backward_reduce(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g);
::std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_with_counts(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    const at::Tensor& counts);
at::Tensor instance_norm(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled);
at::Tensor softmax(
    const at::Tensor& self,
    int64_t dim,
    ::std::optional<at::ScalarType> dtype);
bool is_pinned(const at::Tensor& self, ::std::optional<c10::Device> device);
at::Tensor pin_memory(
    const at::Tensor& self,
    ::std::optional<c10::Device> device);
at::Tensor _pin_memory(
    const at::Tensor& self,
    ::std::optional<c10::Device> device);
at::Tensor repeat_interleave(
    const at::Tensor& self,
    ::std::optional<c10::SymInt> output_size);

} // namespace hpu_wrap
