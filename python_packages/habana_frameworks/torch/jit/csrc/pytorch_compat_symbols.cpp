/**
 * Copyright (c) 2026 Intel Corporation
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

#include <torch/csrc/jit/python/pybind_utils.h>

#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#endif

#include "habana_helpers/pt_version_check.h"

namespace torch::jit {

std::optional<InferredType> detail::_tryToInferTypeImpl(py::handle input) {
#ifdef USE_DISTRIBUTED
  if (py::isinstance<c10d::ProcessGroup>(input)) {
    return InferredType(c10::CapsuleType::get());
  }
#endif
  return std::nullopt;
}

} // namespace torch::jit
