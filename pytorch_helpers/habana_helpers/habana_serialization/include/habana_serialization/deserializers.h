/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <c10/core/TensorOptions.h>
#include <sstream>
#include <type_traits>
#include <vector>

namespace serialization {

// generic part
template <typename POD>
void deserialize(std::istream& is, POD& output) {
  // this only works on built in data types (PODs)
  static_assert(
      std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
      "Can only serialize POD types with this function");
  is.read(reinterpret_cast<char*>(&output), sizeof(output));
}

template <typename T>
void deserialize(std::istream& is, std::vector<T>& output) {
  int size;
  deserialize(is, size);
  for (int i = 0; i < size; ++i) {
    T elem;
    deserialize(is, elem);
    output.push_back(elem);
  }
}

void deserialize(std::istream& is, char*& input);

void deserialize(std::istream& is, std::string& output);

// PT part
void deserialize_device(std::istream& is, c10::TensorOptions& input);

void deserialize_dtype(std::istream& is, c10::TensorOptions& input);

void deserialize_layout(std::istream& is, c10::TensorOptions& input);

void deserialize_requires_grad(std::istream& is, c10::TensorOptions& input);

void deserialize_memory_format(std::istream& is, c10::TensorOptions& input);

void deserialize_pinned_memory(std::istream& is, c10::TensorOptions& input);

void deserialize(std::istream& is, c10::TensorOptions& input);

} // namespace serialization
