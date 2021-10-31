/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "deserializers.h"

namespace serialization {

void deserialize(std::istream& is, char*& input) {
  size_t size = 0;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  input = new char[size];
  is.read(input, size);
}

void deserialize(std::istream& is, std::string& output) {
  int size;
  deserialize(is, size);
  output.resize(size);
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
