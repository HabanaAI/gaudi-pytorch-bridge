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

#include <c10/core/TensorOptions.h>
#include <map>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace serialization {

// generic part

void deserialize(std::istream& is, char*& input);

void deserialize(std::istream& is, std::string& output);

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

template <typename T>
void deserialize(std::istream& is, std::unordered_set<T>& output) {
  int size;
  deserialize(is, size);
  for (int i = 0; i < size; ++i) {
    T elem;
    deserialize(is, elem);
    output.insert(elem);
  }
}

template <typename T1, typename T2>
void deserialize(std::istream& is, std::vector<std::pair<T1, T2>>& output) {
  int size;
  deserialize(is, size);
  for (int i = 0; i < size; ++i) {
    T1 elem1;
    T2 elem2;
    deserialize(is, elem1);
    deserialize(is, elem2);
    output.push_back(std::make_pair(elem1, elem2));
  }
}

template <typename T1, typename T2>
void deserialize(std::istream& is, std::map<T1, T2>& output) {
  int size;
  deserialize(is, size);
  for (int i = 0; i < size; ++i) {
    T1 elem1;
    T2 elem2;
    deserialize(is, elem1);
    deserialize(is, elem2);
    output[elem1] = elem2;
  }
}

template <typename T1, typename T2>
void deserialize(std::istream& is, std::unordered_map<T1, T2>& output) {
  int size;
  deserialize(is, size);
  for (int i = 0; i < size; ++i) {
    T1 elem1;
    T2 elem2;
    deserialize(is, elem1);
    deserialize(is, elem2);
    output[elem1] = elem2;
  }
}

// PT part
void deserialize_device(std::istream& is, c10::TensorOptions& input);

void deserialize_dtype(std::istream& is, c10::TensorOptions& input);

void deserialize_layout(std::istream& is, c10::TensorOptions& input);

void deserialize_requires_grad(std::istream& is, c10::TensorOptions& input);

void deserialize_memory_format(std::istream& is, c10::TensorOptions& input);

void deserialize_pinned_memory(std::istream& is, c10::TensorOptions& input);

void deserialize(std::istream& is, c10::TensorOptions& input);

} // namespace serialization
