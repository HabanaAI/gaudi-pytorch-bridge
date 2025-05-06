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

#pragma once

#include <any>
#include <array>
#include <map>
#include <string>
#include <vector>

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/SymInt.h>

namespace slrg {
enum class InputType {
  PT_TENSOR,
  PT_SCALAR,
  NATIVE_FLOAT,
  NATIVE_INT,
  NATIVE_BOOL,
  NATIVE_STRING,
  DTYPE,
  SYM_INT,
  LAYOUT,
  DEVICE,
  MEMORY_FORMAT,
  GENERATOR,
  STORAGE
};

namespace {
[[maybe_unused]] std::array<at::ScalarType, 10> report_precision_types{
    at::ScalarType::Float,
    at::ScalarType::BFloat16,
    at::ScalarType::Half, // fp16
    at::ScalarType::Float8_e4m3fn, // fp8_143
    at::ScalarType::Float8_e5m2, // fp8_152
    at::ScalarType::Long,
    at::ScalarType::Int,
    at::ScalarType::Short, // int16
    at::ScalarType::Char, // int8
    at::ScalarType::Bool};

std::unordered_map<at::ScalarType, std::string> report_precision_types_string{
    {at::ScalarType::Float, "Float"},
    {at::ScalarType::BFloat16, "Bf16"},
    {at::ScalarType::Half, "Fp16"},
    {at::ScalarType::Float8_e4m3fn, "Fp8_143"},
    {at::ScalarType::Float8_e5m2, "Fp8_152"},
    {at::ScalarType::Long, "Long"},
    {at::ScalarType::Int, "Int"},
    {at::ScalarType::Short, "Int16"},
    {at::ScalarType::Char, "Int8"},
    {at::ScalarType::Bool, "Bool"}};

std::map<InputType, std::string> inputTypeToStrMap = {
    {InputType::PT_TENSOR, "PT_TENSOR"},
    {InputType::PT_SCALAR, "PT_SCALAR"},
    {InputType::NATIVE_FLOAT, "NATIVE_FLOAT"},
    {InputType::NATIVE_INT, "NATIVE_INT"},
    {InputType::NATIVE_BOOL, "NATIVE_BOOL"},
    {InputType::NATIVE_STRING, "NATIVE_STRING"},
    {InputType::DTYPE, "DTYPE"},
    {InputType::SYM_INT, "SYM_INT"},
    {InputType::LAYOUT, "LAYOUT"},
    {InputType::DEVICE, "DEVICE"},
    {InputType::MEMORY_FORMAT, "MEMORY_FORMAT"},
    {InputType::GENERATOR, "GENERATOR"},
    {InputType::STORAGE, "STORAGE"}};
} // namespace

std::ostream& operator<<(std::ostream& os, const InputType& type);

struct InputDescriptor {
  std::string name = "";
  InputType type = InputType::PT_TENSOR;
  bool is_optional = false;
  std::optional<bool> allow_only_none = std::nullopt;
  std::optional<bool> allow_none = std::nullopt;
  std::optional<std::vector<int64_t>> ranks = std::nullopt;
  std::optional<bool> match_rank = std::nullopt;
  std::optional<std::vector<at::ScalarType>> dtypes = std::nullopt;
  std::optional<bool> match_precision_type = std::nullopt;
  std::optional<std::vector<std::any>> values = std::nullopt;
  bool is_array = false;
  std::optional<int> array_length = std::nullopt;

  std::string DebugString() const;

 private:
  std::string getAllowNoneStr() const;
  std::string getAllowOnlyNoneStr() const;
  std::string getArrayLengthStr() const;
  std::string getDtypesStr() const;
  std::string getIsArrayStr() const;
  std::string getMatchPrecisionTypeStr() const;
  std::string getMatchRankStr() const;
  std::string getRanksStr() const;
  std::string getIsOptionalStr() const;
  std::string getValuesStr() const;
};

std::ostream& operator<<(
    std::ostream& os,
    const InputDescriptor& input_descriptor);

struct Report {
  Report(std::vector<c10::ScalarType> supported_types) {
    for (auto type : supported_types)
      support_map[type] = true;
  }
  Report(bool fp4_support, bool int4_support)
      : fp4_support(fp4_support), int4_support(int4_support) {}
  Report() = default;

  bool& operator[](const c10::ScalarType& precision_type);
  bool operator[](const c10::ScalarType& precision_type) const;

  bool operator==(const Report& rhs) const {
    return support_map == rhs.support_map;
  }

  friend std::ostream& operator<<(std::ostream&, const Report&);

  bool fp4_support = false;
  bool int4_support = false;

 private:
  std::unordered_map<c10::ScalarType, bool> support_map;
};

std::ostream& operator<<(std::ostream& os, const Report& report);

struct OperatorDescriptor {
  OperatorDescriptor(
      std::string op_name = "",
      std::string overload = "",
      std::string op_namespace = "")
      : op_name(op_name), overload(overload), op_namespace(op_namespace) {}
  std::string op_name;
  std::string overload;
  std::string op_namespace;
};

using FinalReport = std::unordered_map<
    std::string,
    std::vector<std::pair<OperatorDescriptor, Report>>>;

} // namespace slrg
