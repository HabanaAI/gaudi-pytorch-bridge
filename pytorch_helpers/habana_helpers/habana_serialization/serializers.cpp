/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "serializers.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/typeid.h>
#include <cstddef>
#include <cstring>
#include <ostream>
#include <string>

namespace serialization {

void serialize(std::ostream& os, const char* input) {
  // assumes that input is not nullptr
  size_t size = strlen(input) + 1;
  os.write(reinterpret_cast<char const*>(&size), sizeof(size));
  os.write(input, size);
}

void serialize(std::ostream& os, std::string const& input) {
  serialize(os, static_cast<int>(input.size()));
  os.write(input.data(), input.size());
}

// PT part
void serialize(std::ostream& os, c10::Device const& input) {
  c10::DeviceType dType = input.type();
  os.write(reinterpret_cast<char const*>(&dType), sizeof(c10::DeviceType));
  c10::DeviceIndex dIndex = input.index();
  os.write(reinterpret_cast<char const*>(&dIndex), sizeof(c10::DeviceIndex));
}

void serialize(std::ostream& os, caffe2::TypeMeta input) {
  c10::ScalarType scalarType =
      input.toScalarType(); // NOLINT(misc-include-cleaner)
  os.write(reinterpret_cast<char const*>(&scalarType), sizeof(c10::ScalarType));
}

void serialize(std::ostream& os, c10::TensorOptions const& input) {
  serialize(os, input.has_device());
  if (input.has_device()) {
    serialize(os, input.device());
  }

  serialize(os, input.has_dtype());
  if (input.has_dtype()) {
    serialize(os, input.dtype());
  }

  serialize(os, input.has_layout());
  if (input.has_layout()) {
    serialize(os, input.layout());
  }

  serialize(os, input.has_requires_grad());
  if (input.has_requires_grad()) {
    serialize(os, input.requires_grad());
  }

  serialize(os, input.has_pinned_memory());
  if (input.has_pinned_memory()) {
    serialize(os, input.pinned_memory());
  }

  serialize(os, input.memory_format_opt().has_value());
  if (input.memory_format_opt().has_value()) {
    serialize(os, input.memory_format_opt().value());
  }
}
} // namespace serialization
