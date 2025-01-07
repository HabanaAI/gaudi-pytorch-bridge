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
#include <ATen/Tensor.h>
#include <absl/strings/str_format.h>
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/logging.h"

namespace lazy_to_backend {

bool is_lazy_inference_call_context();

} // namespace lazy_to_backend
