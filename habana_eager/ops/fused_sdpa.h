/**
 * Copyright (c) 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <ATen/core/TensorBody.h>

namespace habana {
namespace eager {

at::Tensor fused_sdpa_autograd_wrap(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const ::std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    ::std::optional<double> scale,
    bool enable_gqa);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd_wrap(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const std::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const at::Tensor& fwd_out);

} // namespace eager
} // namespace habana
