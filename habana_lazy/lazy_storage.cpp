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

#include "lazy_storage.h"
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>

namespace habana_lazy {

HbLazyStorageImpl::HbLazyStorageImpl(const size_t size)
    : c10::StorageImpl(
          c10::StorageImpl::use_byte_size_t(),
          size,
          c10::DataPtr{nullptr, {c10::DeviceType::HPU, 0 /* device_id */}},
          nullptr,
          /*resizeable=*/false) {}

} // namespace habana_lazy
