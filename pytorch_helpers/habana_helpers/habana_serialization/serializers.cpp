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
#include "serializers.h"

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
  c10::ScalarType scalarType = input.toScalarType();
  os.write(reinterpret_cast<char const*>(&scalarType), sizeof(c10::ScalarType));
}

#define SERIALIZE_OPT(TYPE)                    \
  serialize(os, input.has_##TYPE());           \
  if (input.has_##TYPE()) {                    \
    serialize(os, input.TYPE##_opt().value()); \
  }

void serialize(std::ostream& os, c10::TensorOptions const& input) {
  SERIALIZE_OPT(device)
  SERIALIZE_OPT(dtype)
  SERIALIZE_OPT(layout)
  SERIALIZE_OPT(requires_grad)
  SERIALIZE_OPT(pinned_memory)
  SERIALIZE_OPT(memory_format)
}
} // namespace serialization
