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
#include "deserializers.h"

namespace serialization {

void deserialize(std::istream& is, char*& input) {
  size_t size = 0;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  input = new char[size];
  is.read(input, static_cast<std::streamsize>(size));
}

void deserialize(std::istream& is, std::string& output) {
  int size;
  deserialize(is, size);
  output.resize(static_cast<size_t>(size));
  is.read(&output[0], size);
}

// PT part
void deserialize_device(std::istream& is, c10::TensorOptions& input) {
  c10::DeviceType dType;
  is.read(reinterpret_cast<char*>(&dType), sizeof(c10::DeviceType));
  c10::DeviceIndex dIndex;
  is.read(reinterpret_cast<char*>(&dIndex), sizeof(c10::DeviceIndex));
  input = input.device(dType, dIndex);
}

void deserialize_dtype(std::istream& is, c10::TensorOptions& input) {
  c10::ScalarType scalarType;
  is.read(reinterpret_cast<char*>(&scalarType), sizeof(c10::ScalarType));
  input = input.dtype(scalarType);
}

void deserialize_layout(std::istream& is, c10::TensorOptions& input) {
  c10::Layout layout;
  is.read(reinterpret_cast<char*>(&layout), sizeof(c10::Layout));
  input = input.layout(layout);
}

void deserialize_requires_grad(std::istream& is, c10::TensorOptions& input) {
  bool requiresGrad;
  is.read(reinterpret_cast<char*>(&requiresGrad), sizeof(bool));
  input = input.requires_grad(requiresGrad);
}

void deserialize_memory_format(std::istream& is, c10::TensorOptions& input) {
  c10::MemoryFormat memoryFormat;
  is.read(reinterpret_cast<char*>(&memoryFormat), sizeof(c10::MemoryFormat));
  input = input.memory_format(memoryFormat);
}

void deserialize_pinned_memory(std::istream& is, c10::TensorOptions& input) {
  bool pinnedMemory;
  is.read(reinterpret_cast<char*>(&pinnedMemory), sizeof(bool));
  input = input.pinned_memory(pinnedMemory);
}

#define DESERIALIZE_OPT(TYPE)      \
  bool has_##TYPE = false;         \
  deserialize(is, has_##TYPE);     \
  if (has_##TYPE) {                \
    deserialize_##TYPE(is, input); \
  }

void deserialize(std::istream& is, c10::TensorOptions& input) {
  DESERIALIZE_OPT(device)
  DESERIALIZE_OPT(dtype)
  DESERIALIZE_OPT(layout)
  DESERIALIZE_OPT(requires_grad)
  DESERIALIZE_OPT(pinned_memory)
  DESERIALIZE_OPT(memory_format)
}
} // namespace serialization
