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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace serialization {

// generic part

void serialize(std::ostream& os, const char* input);
void serialize(std::ostream& os, std::string const& input);

template <typename POD>
void serialize(std::ostream& os, const POD& input) {
  // this only works on built in data types (PODs)
  static_assert(
      std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
      "Can only serialize POD types with this function");
  os.write(reinterpret_cast<char const*>(&input), sizeof(POD));
}

template <typename T>
void serialize(std::ostream& os, std::vector<T> const& input) {
  serialize(os, static_cast<int>(input.size()));
  for (auto const& elem : input) {
    serialize(os, elem);
  }
}

template <typename T>
void serialize(std::ostream& os, std::unordered_set<T> const& input) {
  serialize(os, static_cast<int>(input.size()));
  for (auto const& elem : input) {
    serialize(os, elem);
  }
}

template <typename T1, typename T2>
void serialize(std::ostream& os, std::vector<std::pair<T1, T2>> const& input) {
  serialize(os, static_cast<int>(input.size()));
  for (auto const& elem : input) {
    serialize(os, elem.first);
    serialize(os, elem.second);
  }
}

template <typename T1, typename T2>
void serialize(std::ostream& os, std::map<T1, T2> const& input) {
  serialize(os, static_cast<int>(input.size()));
  for (auto const& elem : input) {
    serialize(os, elem.first);
    serialize(os, elem.second);
  }
}

template <typename T1, typename T2>
void serialize(std::ostream& os, std::unordered_map<T1, T2> const& input) {
  serialize(os, static_cast<int>(input.size()));
  for (auto const& elem : input) {
    serialize(os, elem.first);
    serialize(os, elem.second);
  }
}

// PT part
void serialize(std::ostream& os, c10::Device const& input);

void serialize(std::ostream& os, caffe2::TypeMeta input);

void serialize(std::ostream& os, c10::TensorOptions const& input);

} // namespace serialization
