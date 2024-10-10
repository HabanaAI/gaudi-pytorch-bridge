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
