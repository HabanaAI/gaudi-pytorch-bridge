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

#include "habana_helpers/logging.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/typeid.h>
#include <torch/csrc/jit/ir/ir.h>

CREATE_OSTREAM_FORMATTER(c10::ArrayRef<long int>);
CREATE_OSTREAM_FORMATTER(c10::Device);
CREATE_OSTREAM_FORMATTER(c10::IValue);
CREATE_OSTREAM_FORMATTER(caffe2::TypeMeta);
CREATE_OSTREAM_FORMATTER(torch::jit::Node);
CREATE_OSTREAM_FORMATTER(torch::jit::Graph);
CREATE_OSTREAM_FORMATTER(at::Tensor);

template <>
struct fmt::formatter<c10::SmallVector<long int, 5>> : ostream_formatter {};
