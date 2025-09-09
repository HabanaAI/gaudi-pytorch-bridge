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

#include "common/warning_suppress.h"
SUPPRESS_W_PREFIX
SUPPRESS_WARRAY_BOUNDS_WSTRINGOP_OVERFLOW_P
#include <ATen/core/TensorBody.h>
SUPPRESS_W_SUFFIX

namespace habana::eager {

at::Tensor fused_sdpa_autograd_wrap(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const ::std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    ::std::optional<double> scale,
    bool enable_gqa);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_sdpa_recomp_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type,
    c10::SymIntArrayRef window_size = {-1, -1},
    const std::optional<at::Tensor>& sink = std::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_sdpa_recomp_bwd_wrap(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& m,
    const at::Tensor& linv,
    const std::optional<at::Tensor>& seed,
    const bool is_causal,
    const double p,
    const double scale,
    std::string_view softmax_mode,
    const at::Tensor& fwd_out);

} // namespace habana::eager
