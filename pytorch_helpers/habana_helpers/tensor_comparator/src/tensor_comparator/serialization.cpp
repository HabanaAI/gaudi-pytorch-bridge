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

#include <nlohmann/json.hpp>
#include <serialization.hpp>
#include <ios>
#include "types.hpp"

#include <fstream>
#include <iostream>

using json = nlohmannV340::json;

namespace TensorComparison_pt {
void to_json(json& j, ComparisonResult r) {
  for (unsigned i{}; i < METHODS_MAX; ++i) {
    if (r[i].is_set()) {
      j[ComparisonMethodsNames[i]] = r[i].value();
    }
  }
}

void from_json(const json& j, Optional& r) {
  j.at("name").get_to(r);
}
namespace jsn {

json loadJson(const std::string& fileName) {
  std::ifstream ifs(fileName);
  if (!ifs.good()) {
    std::cout << "File " << fileName << " not found!" << std::endl;
    throw std::runtime_error("File not found");
  }

  json j;
  ifs >> j;

  return j;
}

ThresholdFileMap parse(const std::string& fileName) {
  return loadJson(fileName).get<ThresholdFileMap>();
}

template <typename T>
bool dump(T& t, const std::string& fileName) {
  std::fstream ofs(fileName, std::ios::out);
  if (!ofs.good()) {
    std::cout << "Tried to open file " << fileName
              << " for writing. something went wrong." << std::endl;
    throw std::runtime_error("File openning failed.");
  }

  json j(t);
  ofs << j.dump(4);
  return true;
}

template bool dump<ThresholdFileMap>(ThresholdFileMap&, const std::string&);
template bool dump<ResultMap>(ResultMap&, const std::string&);
template bool dump<ResultVec>(ResultVec&, const std::string&);
} // namespace jsn
namespace csv {
std::ostream& operator<<(std::ostream& stream, const Optional& entry) {
  if (entry.is_set()) {
    stream << entry.value();
  }
  stream << ",";

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ComparisonResult& result) {
  for (auto& e : result) {
    stream << e;
  }

  stream << result.GetComment();

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ResultMap& result) {
  stream << "Tensor name";
  for (auto& n : ComparisonMethodsNames) {
    stream << "," << n;
  }
  stream << ",comment";
  stream << "\n";

  for (auto& e : result) {
    stream << e.first << "," << e.second << "\n";
  }

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ResultVec& result) {
  stream << "Tensor name";
  for (auto& n : ComparisonMethodsNames) {
    stream << "," << n;
  }
  stream << ",comment";
  stream << "\n";

  for (auto& e : result) {
    stream << e.first << "," << e.second << "\n";
  }

  return stream;
}

bool dump(ResultMap& m, const std::string& fileName) {
  std::fstream ofs(fileName, std::ios::out);
  if (!ofs.good()) {
    std::cout << "Tried to open file " << fileName
              << " for writing. something went wrong." << std::endl;
    throw std::runtime_error("File openning failed.");
  }

  ofs << m;

  return true;
}

bool dump(ResultVec& m, const std::string& fileName) {
  std::fstream ofs(fileName, std::ios::out);
  if (!ofs.good()) {
    std::cout << "Tried to open file " << fileName
              << " for writing. something went wrong." << std::endl;
    throw std::runtime_error("File openning failed.");
  }

  ofs << m;

  return true;
}
} // namespace csv
} // namespace TensorComparison_pt