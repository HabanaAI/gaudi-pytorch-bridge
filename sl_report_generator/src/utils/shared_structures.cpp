/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "shared_structures.h"

namespace slrg {
std::ostream& operator<<(std::ostream& os, const InputType& type) {
  os << inputTypeToStrMap.at(type);
  return os;
}

std::string InputDescriptor::getAllowNoneStr() const {
  std::string allow_none_str =
      allow_none.has_value() ? (allow_none.value() ? "true" : "false") : "None";
  std::string ret = "  InputDescriptor::allow_none=";
  ret += allow_none_str + "\n";
  return ret;
}

std::string InputDescriptor::getAllowOnlyNoneStr() const {
  std::string allow_only_none_str = allow_only_none.has_value()
      ? (allow_only_none.value() ? "true" : "false")
      : "None";
  std::string ret = "  InputDescriptor::allow_only_none=";
  ret += allow_only_none_str + "\n";
  return ret;
}

std::string InputDescriptor::getArrayLengthStr() const {
  std::string array_length_str =
      (array_length.has_value() ? std::to_string(array_length.value())
                                : "None");
  std::string ret = "  InputDescriptor::array_length=";
  return ret + array_length_str + "\n";
}

std::string InputDescriptor::getDtypesStr() const {
  std::string ret = "  InputDescriptor::dtypes=";
  std::string dtypes_str = "None";
  if (dtypes.has_value()) {
    dtypes_str = "[";
    const auto& values = dtypes.value();
    for (auto dtype_it = std::begin(values); dtype_it != std::end(values);
         ++dtype_it) {
      dtypes_str += at::toString(*dtype_it);
      if (std::next(dtype_it) != std::end(values)) {
        dtypes_str += ", ";
      }
    }
    dtypes_str += "]";
  }
  return ret + dtypes_str + "\n";
}

std::string InputDescriptor::getIsArrayStr() const {
  std::string is_array_str = is_array ? "true" : "false";
  std::string ret = "  InputDescriptor::is_array=";
  return ret + is_array_str + "\n";
}

std::string InputDescriptor::getMatchPrecisionTypeStr() const {
  std::string match_precision_type_str = match_precision_type.has_value()
      ? (match_precision_type.value() ? "true" : "false")
      : "None";
  std::string ret = "  InputDescriptor::match_precision_type=";
  return ret + match_precision_type_str + "\n";
}

std::string InputDescriptor::getMatchRankStr() const {
  std::string match_rank_str =
      match_rank.has_value() ? (match_rank.value() ? "true" : "false") : "None";
  std::string ret = "  InputDescriptor::match_rank=";
  ret += match_rank_str + "\n";
  return ret;
}

std::string InputDescriptor::getRanksStr() const {
  std::string ret = "  InputDescriptor::ranks=";
  std::string ranks_str = "None";
  if (ranks.has_value()) {
    ranks_str = "[";
    const auto& values = ranks.value();
    for (auto rank_it = std::begin(values); rank_it != std::end(values);
         ++rank_it) {
      ranks_str += std::to_string(*rank_it);
      if (std::next(rank_it) != std::end(values)) {
        ranks_str += ", ";
      }
    }
    ranks_str += "]";
  }
  return ret + ranks_str + "\n";
}

std::string InputDescriptor::getIsOptionalStr() const {
  std::string ret = "  InputDescriptor::is_optional=";
  ret += (is_optional ? "true" : "false");
  ret += "\n";
  return ret;
}

std::string InputDescriptor::getValuesStr() const {
  std::string ret = "  InputDescriptor::values=";
  std::stringstream values_str;
  values_str << "None";
  if (values.has_value()) {
    values_str.str("");
    values_str << "[";
    const auto& vals = values.value();
    for (auto val_it = std::begin(vals); val_it != std::end(vals); ++val_it) {
      auto val = (*val_it);
      auto val_type_name = std::string(val.type().name());
      if (val_type_name == "f" || val_type_name == "float") {
        values_str << std::to_string(std::any_cast<float>(val)) << ":float";
      } else if (val_type_name == "d" || val_type_name == "double") {
        values_str << std::to_string(std::any_cast<double>(val)) << ":double";
      } else if (val_type_name == "l" || val_type_name == "long") {
        values_str << std::to_string(std::any_cast<int64_t>(val)) << ":long";
      } else if (val_type_name == "i" || val_type_name == "int") {
        values_str << std::to_string(std::any_cast<int>(val)) << ":int";
      } else if (val_type_name == "b" || val_type_name == "bool") {
        values_str << (std::any_cast<bool>(val) ? "true" : "false")
                   << std::string(":bool");
      } else if (val_type_name.find("string") != std::string::npos) {
        values_str << "'" << std::any_cast<std::string>(val) << "':string";
      } else if (val_type_name == "c" || val_type_name == "char") {
        values_str << std::to_string(std::any_cast<char>(val)) << ":char";
      } else if (val_type_name.find("ScalarType") != std::string::npos) {
        values_str << std::any_cast<c10::ScalarType>(val) << ":ScalarType";
      } else if (val_type_name.find("SymInt") != std::string::npos) {
        values_str << std::any_cast<c10::SymInt>(val) << ":SymInt";
      } else if (val_type_name.find("Layout") != std::string::npos) {
        values_str << std::any_cast<c10::Layout>(val) << ":Layout";
      } else if (val_type_name.find("Device") != std::string::npos) {
        values_str << std::any_cast<c10::Device>(val) << ":Device";
      } else if (val_type_name.find("MemoryFormat") != std::string::npos) {
        values_str << std::any_cast<c10::MemoryFormat>(val) << ":MemoryFormat";
      } else if (val_type_name.find("Storage") != std::string::npos) {
        values_str << std::any_cast<at::Storage>(val) << ":Stroage";
      } else {
        values_str << std::string("unknown:") << val_type_name;
      }
      if (std::next(val_it) != std::end(vals)) {
        values_str << ", ";
      }
    }
    values_str << "]";
  }
  return ret + values_str.str() + "\n";
}

std::string InputDescriptor::DebugString() const {
  std::stringstream ret;
  ret << "  InputDescriptor::name='" << name << "'\n";
  ret << "  InputDescriptor::type=" << type << "\n";
  ret << getIsOptionalStr();
  ret << getAllowOnlyNoneStr();
  ret << getAllowNoneStr();
  ret << getRanksStr();
  ret << getMatchRankStr();
  ret << getDtypesStr();
  ret << getMatchPrecisionTypeStr();
  ret << getValuesStr();
  ret << getIsArrayStr();
  ret << getArrayLengthStr();
  return ret.str();
}

std::ostream& operator<<(
    std::ostream& os,
    const InputDescriptor& input_descriptor) {
  os << input_descriptor.DebugString();
  return os;
}

bool& Report::operator[](const c10::ScalarType& precision_type) {
  auto it = support_map.find(precision_type);
  if (it == std::end(support_map)) {
    support_map[precision_type] = false;
  }
  return support_map[precision_type];
}

bool Report::operator[](const c10::ScalarType& precision_type) const {
  auto it = support_map.find(precision_type);
  if (it == std::end(support_map)) {
    return false;
  }
  return support_map.at(precision_type);
}

std::ostream& operator<<(std::ostream& os, const Report& report) {
  os << "\n{Size: " << report.support_map.size() << ", ";
  for (const auto& type : report.support_map)
    os << slrg::report_precision_types_string[type.first] << ":"
       << (type.second ? "true" : "false") << ", ";
  os << "Fp4: " << (report.fp4_support ? "true" : "false") << ", ";
  os << "Int4: " << (report.int4_support ? "true" : "false");
  os << "}\n";
  return os;
}
} // namespace slrg
