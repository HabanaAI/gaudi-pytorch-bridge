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

#include <ATen/core/symbol.h>
#include <unordered_set>

namespace habana {

static const std::unordered_set<c10::Symbol> logical_ops{
    c10::Symbol::fromQualString("aten::reshape"),
    c10::Symbol::fromQualString("aten::view"),
    c10::Symbol::fromQualString("aten::t"),
    c10::Symbol::fromQualString("aten::transpose"),
    c10::Symbol::fromQualString("aten::squeeze"),
    c10::Symbol::fromQualString("aten::unsqueeze"),
    c10::Symbol::fromQualString("aten::permute"),
    c10::Symbol::fromQualString("aten::expand"),
    c10::Symbol::fromQualString("aten::slice"),
    c10::Symbol::fromQualString("aten::clone")};
static const c10::Symbol cast_to_fp8_symbol =
    c10::Symbol::fromQualString("hpu::cast_to_fp8_v2");
static const c10::Symbol cast_from_fp8_symbol =
    c10::Symbol::fromQualString("hpu::cast_from_fp8");
} // namespace habana
