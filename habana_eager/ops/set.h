/**
* Copyright (c) 2021-2024 Intel Corporation
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
at::Tensor& set_source_Storage_storage_offset(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride);
at::Tensor& set_source_Storage(at::Tensor& self, at::Storage source);
at::Tensor& set_source_Tensor(at::Tensor& self, const at::Tensor& source);
at::Tensor& set_(at::Tensor& self);
} // namespace eager
} // namespace habana
