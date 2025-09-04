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

#include <vector>

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "utils/shared_structures.h"

namespace slrg {

class IStackGenerator {
 public:
  virtual ~IStackGenerator() = default;

  virtual std::vector<at::Stack> getStacks(at::ScalarType precision_type) = 0;

  virtual std::string getOpAndOverloadName() const = 0;

  virtual bool isBlacklistedPrecisionType(
      at::ScalarType precision_type) const = 0;

  virtual bool isWhitelistedPrecisionType(
      at::ScalarType precision_type) const = 0;

  virtual std::string DebugString() const = 0;
};

class StackGenerator : public IStackGenerator {
 public:
  StackGenerator(const std::vector<int64_t>& ranks = {1}) : ranks(ranks) {}

  StackGenerator(
      std::vector<InputDescriptor> inputs,
      std::vector<at::ScalarType> blacklisted_precision_types,
      std::vector<at::ScalarType> whitelisted_precision_types,
      const std::string& op_name,
      const std::string& op_and_overload_name = "",
      const std::vector<int64_t>& ranks = {1})
      : ranks(ranks),
        inputs(inputs),
        blacklisted_precision_types(blacklisted_precision_types),
        whitelisted_precision_types(whitelisted_precision_types),
        op_name(op_name) {
    this->op_and_overload_name =
        op_and_overload_name.empty() ? op_name : op_and_overload_name;
  }

  virtual ~StackGenerator() = default;

  std::vector<at::Stack> getStacks(
      at::ScalarType precision_type) final override;

  std::string getOpAndOverloadName() const final override {
    return op_and_overload_name;
  }

  bool isBlacklistedPrecisionType(
      at::ScalarType precision_type) const final override;
  bool isWhitelistedPrecisionType(
      at::ScalarType precision_type) const final override;

  std::string DebugString() const override;

 protected:
  std::vector<int64_t> ranks;
  std::vector<InputDescriptor> inputs;
  std::vector<at::ScalarType> blacklisted_precision_types;
  std::vector<at::ScalarType> whitelisted_precision_types;
  int default_array_length;
  int dim_size = 1;

  bool verbose = false;

  std::string op_name;
  std::string op_and_overload_name;

 private:
  void generateAllStacks();

  std::vector<at::Stack> generateStacks(
      const at::ScalarType precision_type,
      const int64_t rank) const;

  std::vector<c10::IValue> getInputVariants(
      const InputDescriptor& input_descriptor,
      const at::ScalarType precision_type,
      const int64_t rank) const;

  std::vector<c10::IValue> handle_inputs(
      const InputDescriptor& input,
      const at::ScalarType& precision_type,
      const std::int64_t rank) const;

  template <InputType T>
  std::vector<c10::IValue> generateIValues(
      const InputDescriptor& input_descriptor,
      const at::ScalarType precision_type,
      const int64_t rank) const;

  template <typename T>
  std::vector<c10::IValue> generateIValuesBasicTypes(
      const InputDescriptor& input_descriptor) const;

  bool stacks_generated = false;

  std::unordered_map<at::ScalarType, std::vector<at::Stack>> generated_stacks;
};

std::ostream& operator<<(
    std::ostream& os,
    const IStackGenerator& stack_generator);

class SchemaStackGenerator : public StackGenerator {
 public:
  SchemaStackGenerator(
      const std::string& schema,
      const std::string& op_name,
      const std::string& op_name_and_overload_name = "",
      const std::vector<int64_t>& ranks = {1},
      const int default_array_length = 1,
      const int dim_size = 1);
  virtual ~SchemaStackGenerator() = default;

 private:
  std::vector<InputDescriptor> generateInputs(const std::string& schema) const;

  std::vector<at::ScalarType> getBlacklistedPrecisionTypes() const;
  std::vector<at::ScalarType> getWhitelistedPrecisionTypes() const;
};

} // namespace slrg
