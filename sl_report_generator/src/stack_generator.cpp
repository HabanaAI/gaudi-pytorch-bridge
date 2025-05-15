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

#include <sstream>

#include "stack_generator.h"
#include "utils/shared_structures.h"
#include "utils/tensor_helpers.h"

namespace slrg {
namespace {
static std::unordered_map<std::string, InputType> input_type_map = {
    {"Tensor", InputType::PT_TENSOR},
    {"Scalar", InputType::PT_SCALAR},
    {"float", InputType::NATIVE_FLOAT},
    {"int", InputType::NATIVE_INT},
    {"bool", InputType::NATIVE_BOOL},
    {"str", InputType::NATIVE_STRING},
    {"ScalarType", InputType::DTYPE},
    {"SymInt", InputType::SYM_INT},
    {"Layout", InputType::LAYOUT},
    {"Device", InputType::DEVICE},
    {"MemoryFormat", InputType::MEMORY_FORMAT},
    {"Generator", InputType::GENERATOR},
    {"Storage", InputType::STORAGE}};

static std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::vector<std::any>>>
    default_values_specific_ops = {
        {"index_reduce",
         {{"reduce",
           {std::string("prod"),
            std::string("mean"),
            std::string("amax"),
            std::string("amin")}}}},
        {"scatter",
         {{"reduce",
           {std::string("sum"),
            std::string("prod"),
            std::string("mean"),
            std::string("amax"),
            std::string("amin")}}}},
        {"searchsorted",
         {{"right", {false}}, {"side", {std::string("left")}}}}};

static std::unordered_map<std::string, std::vector<std::any>>
    default_values_all_ops = {
        {"approximate", {std::string("none"), std::string("tanh")}},
        {"dim", {0}},
        {"dims", {0}},
        {"p", {0.5f}},
        {"pin_memory", {false}}};

static std::unordered_map<std::string, std::vector<at::ScalarType>>
    blacklisted_precision_types_op_map = {
        {"one_hot", // I32/I16 not supported in compile mode
         {at::ScalarType::Int, at::ScalarType::Short}},
        {"complex",
         {c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Long,
          c10::ScalarType::Int,
          c10::ScalarType::Short,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}}};

static std::unordered_map<std::string, std::vector<at::ScalarType>>
    whitelisted_precision_types_op_map = {};
} // namespace

std::vector<at::Stack> StackGenerator::getStacks(
    at::ScalarType precision_type) {
  if (isBlacklistedPrecisionType(precision_type))
    return {};

  generateAllStacks();
  return generated_stacks[precision_type];
}

bool StackGenerator::isBlacklistedPrecisionType(
    at::ScalarType precision_type) const {
  return std::any_of(
      std::begin(blacklisted_precision_types),
      std::end(blacklisted_precision_types),
      [&precision_type](at::ScalarType not_supported_type) {
        return not_supported_type == precision_type;
      });
};

bool StackGenerator::isWhitelistedPrecisionType(
    at::ScalarType precision_type) const {
  return std::any_of(
      std::begin(whitelisted_precision_types),
      std::end(whitelisted_precision_types),
      [&precision_type](at::ScalarType supported_type) {
        return supported_type == precision_type;
      });
};

void StackGenerator::generateAllStacks() {
  if (stacks_generated)
    return;

  for (const auto& report_precision_type : report_precision_types) {
    auto& stacks_for_precision_type = generated_stacks[report_precision_type];
    for (const auto& rank : ranks) {
      auto stacks_for_rank = generateStacks(report_precision_type, rank);
      stacks_for_precision_type.insert(
          std::end(stacks_for_precision_type),
          std::begin(stacks_for_rank),
          std::end(stacks_for_rank));
    }
  }

  stacks_generated = true;
}

std::vector<at::Stack> StackGenerator::generateStacks(
    const at::ScalarType precision_type,
    const int64_t rank) const {
  std::vector<at::Stack> generated_stacks{{}};
  for (const auto& input : inputs) {
    const auto input_variants = getInputVariants(input, precision_type, rank);
    const auto number_of_input_variants = input_variants.size();
    const auto number_of_stack_variants = generated_stacks.size();

    generated_stacks.reserve(
        number_of_stack_variants * number_of_input_variants);
    for (std::size_t i{0}; i < number_of_input_variants - 1; i++)
      generated_stacks.insert(
          std::end(generated_stacks),
          std::begin(generated_stacks),
          std::next(std::begin(generated_stacks), number_of_stack_variants));

    for (std::size_t i{0}; i < generated_stacks.size(); ++i)
      generated_stacks[i].push_back(
          input_variants[i / number_of_stack_variants]);
  }
  return generated_stacks;
}

template <InputType T>
std::vector<c10::IValue> StackGenerator::generateIValues(
    const InputDescriptor& input_descriptor,
    const at::ScalarType,
    const int64_t) const {
  std::stringstream error_message{};
  error_message << "StackGenerator::generateIValues not implemented for type '"
                << input_descriptor.type << "'.";
  throw std::logic_error(error_message.str());
}

namespace {
void validate_input_for_pt_tensor(
    const InputDescriptor& input,
    const std::string& op_and_overload_name) {
  if (not input.match_rank.has_value()) {
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op '" +
        op_and_overload_name +
        "': for tensors the 'match_rank' parameters must be specified");
  }

  if (not input.match_rank.value()) {
    if (not input.ranks.has_value()) {
      throw std::invalid_argument(
          "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op '" +
          op_and_overload_name +
          "': for tensors with 'match_rank'=false, the 'ranks' parameters must be specified");
    }
  }

  if (not input.match_precision_type.has_value()) {
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op '" +
        op_and_overload_name +
        "': for tensors the 'match_precision_type' parameter must be specified");
  }

  if (not input.match_precision_type.value()) {
    if (not input.dtypes.has_value()) {
      throw std::invalid_argument(
          "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op '" +
          op_and_overload_name +
          "': for tensors with 'match_precision_type'=false, the 'dtypes' parameter must be specified");
    }
  }
}
} // namespace

template <>
std::vector<c10::IValue> StackGenerator::generateIValues<InputType::PT_TENSOR>(
    const InputDescriptor& input_descriptor,
    const at::ScalarType precision_type,
    const int64_t rank) const {
  validate_input_for_pt_tensor(input_descriptor, op_and_overload_name);

  auto ranks = input_descriptor.match_rank.value()
      ? std::vector<int64_t>{rank}
      : input_descriptor.ranks.value();

  auto dtypes = input_descriptor.match_precision_type.value()
      ? std::vector<at::ScalarType>{precision_type}
      : input_descriptor.dtypes.value();

  std::vector<c10::IValue> values;
  for (const auto& dtype : dtypes) {
    for (const auto& tensor_rank : ranks) {
      auto t = c10::IValue(createTensor(tensor_rank, dtype));
      values.push_back(t);
    }
  }

  return values;
}

namespace {
void validate_input_for_pt_scalar(
    const InputDescriptor& input,
    std::string op_and_overload_name) {
  if (not input.values.has_value())
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op '" +
        op_and_overload_name + "': the 'values' parameter must be specified");

  if (input.values.value().empty())
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op '" +
        op_and_overload_name + "': the 'values' parameter cannot be empty");
}

c10::Scalar get_scalar_from_any(
    std::any value,
    std::string op_and_overload_name,
    std::string input_name) {
  c10::Scalar scalar;
  auto value_type_name = std::string(value.type().name());
  if (value_type_name == "f" || value_type_name == "float") {
    scalar = std::any_cast<float>(value);
  } else if (value_type_name == "d" || value_type_name == "double") {
    scalar = std::any_cast<double>(value);
  } else if (value_type_name == "l" || value_type_name == "long") {
    scalar = std::any_cast<int64_t>(value);
  } else if (value_type_name == "i" || value_type_name == "int") {
    scalar = std::any_cast<int>(value);
  } else if (value_type_name == "b" || value_type_name == "bool") {
    scalar = std::any_cast<bool>(value);
  } else if (value_type_name == "c" || value_type_name == "char") {
    scalar = std::any_cast<char>(value);
  } else {
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op '" +
        op_and_overload_name + "': unknown value type '" + value_type_name +
        "' for param '" + input_name + "'");
  }
  return scalar;
}
} // namespace

template <>
std::vector<c10::IValue> StackGenerator::generateIValues<InputType::PT_SCALAR>(
    const InputDescriptor& input_descriptor,
    const at::ScalarType,
    const int64_t) const {
  validate_input_for_pt_scalar(input_descriptor, op_and_overload_name);

  std::vector<c10::IValue> values;
  for (const auto& value : input_descriptor.values.value())
    values.emplace_back(get_scalar_from_any(
        value, op_and_overload_name, input_descriptor.name));

  return values;
}

namespace {
void validate_input_for_dtype(
    const InputDescriptor& input,
    std::string op_and_overload_name) {
  if (not input.values.has_value() || input.values.value().empty())
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::DTYPE> cannot process InputDescriptor for op '" +
        op_and_overload_name +
        "': for input with 'match_precision_type'=false, the 'values' parameter cannot be empty (param '" +
        input.name + "')");

  const auto& values = input.values.value();
  const auto not_all_scalars =
      std::any_of(std::begin(values), std::end(values), [](std::any value) {
        const auto value_type_name = std::string(value.type().name());
        return value_type_name.find("ScalarType") == std::string::npos;
      });

  if (not_all_scalars) {
    throw std::invalid_argument(
        "StackGenerator::generateIValues<InputType::DTYPE> cannot process InputDescriptor for op '" +
        op_and_overload_name +
        "': the 'values' parameter must contain only ScalarTypes (param '" +
        input.name + "')");
  }
}
} // namespace

template <>
std::vector<c10::IValue> StackGenerator::generateIValues<InputType::DTYPE>(
    const InputDescriptor& input_descriptor,
    const at::ScalarType precision_type,
    const int64_t) const {
  std::vector<c10::IValue> values;
  if (input_descriptor.match_precision_type.has_value() &&
      input_descriptor.match_precision_type.value()) {
    values = {c10::IValue(precision_type)};
  } else {
    validate_input_for_dtype(input_descriptor, op_and_overload_name);
    for (const auto& value : input_descriptor.values.value())
      values.emplace_back(std::any_cast<c10::ScalarType>(value));
  }
  return values;
}

template <typename T>
std::vector<c10::IValue> StackGenerator::generateIValuesBasicTypes(
    const InputDescriptor& input_descriptor) const {
  if (not input_descriptor.values.has_value()) {
    std::stringstream error_message{};
    error_message << "StackGenerator::generateIValuesBasicTypes<T, InputType::"
                  << input_descriptor.type
                  << "> cannot process InputDescriptor for op '"
                  << op_and_overload_name
                  << "': the 'values' parameter must be specified (param '"
                  << input_descriptor.name << "')";
    throw std::invalid_argument(error_message.str());
  }

  if (input_descriptor.values.value().empty()) {
    std::stringstream error_message{};
    error_message << "StackGenerator::generateIValuesBasicTypes<T, InputType::"
                  << input_descriptor.type
                  << "> cannot process InputDescriptor for op '"
                  << op_and_overload_name
                  << "': the 'values' parameter cannot be empty (param '"
                  << input_descriptor.name << "')";
    throw std::invalid_argument(error_message.str());
  }

  std::vector<c10::IValue> values;
  for (const auto& value : input_descriptor.values.value()) {
    try {
      T scalar = std::any_cast<T>(value);
      values.push_back(c10::IValue(scalar));
    } catch (const std::exception& e) {
      auto value_type_name = std::string(value.type().name());
      std::stringstream error_message{};
      error_message
          << "StackGenerator::generateIValuesBasicTypes<T, InputType::"
          << input_descriptor.type
          << "> cannot process InputDescriptor for op '" << op_and_overload_name
          << "': the 'values' parameter have values of type corresponding to InputType (param '"
          << input_descriptor.name << "'), got: '" << value_type_name << "'";
      throw std::invalid_argument(error_message.str());
    }
  }
  return values;
}

namespace {
auto is_only_none(const InputDescriptor& input) noexcept {
  return input.is_optional && input.allow_only_none.has_value() &&
      input.allow_only_none.value();
}

std::vector<c10::IValue> handle_optional_input(const InputDescriptor& input) {
  if (not input.is_optional)
    return {};

  if (not input.allow_none.has_value()) {
    throw std::invalid_argument(
        "StackGenerator cannot process InputDescriptor: for optional inputs at least one of the parameters 'allow_none' must be specified or 'allow_only_none' must be TRUE");
  }

  if (input.allow_none.value())
    return {c10::IValue()};

  return {};
}
} // namespace

std::vector<c10::IValue> StackGenerator::getInputVariants(
    const InputDescriptor& input_descriptor,
    const at::ScalarType precision_type,
    const int64_t rank) const {
  if (is_only_none(input_descriptor))
    return {c10::IValue()};
  else {
    auto variants = handle_optional_input(input_descriptor);
    const auto values = handle_inputs(input_descriptor, precision_type, rank);

    variants.insert(std::end(variants), std::begin(values), std::end(values));
    return variants;
  }
}

std::vector<c10::IValue> StackGenerator::handle_inputs(
    const InputDescriptor& input,
    const at::ScalarType& precision_type,
    const std::int64_t rank) const {
  std::vector<c10::IValue> values{};
  c10::TypePtr array_type = c10::AnyType::get();
  switch (input.type) {
    case InputType::PT_TENSOR: {
      values =
          generateIValues<InputType::PT_TENSOR>(input, precision_type, rank);
      array_type = c10::TensorType::get();
      break;
    }
    case InputType::PT_SCALAR: {
      values =
          generateIValues<InputType::PT_SCALAR>(input, precision_type, rank);
      break;
    }
    case InputType::NATIVE_FLOAT: {
      values = generateIValuesBasicTypes<float>(input);
      array_type = c10::FloatType::get();
      break;
    }
    case InputType::NATIVE_INT: {
      values = generateIValuesBasicTypes<int>(input);
      array_type = c10::IntType::get();
      break;
    }
    case InputType::NATIVE_BOOL: {
      values = generateIValuesBasicTypes<bool>(input);
      array_type = c10::BoolType::get();
      break;
    }
    case InputType::NATIVE_STRING: {
      values = generateIValuesBasicTypes<std::string>(input);
      array_type = c10::StringType::get();
      break;
    }
    case InputType::DTYPE: {
      values = generateIValues<InputType::DTYPE>(input, precision_type, rank);
      break;
    }
    case InputType::SYM_INT: {
      values = generateIValuesBasicTypes<c10::SymInt>(input);
      // An array of SymInts is treated by PT as an array of Ints
      array_type = c10::IntType::get();
      break;
    }
    case InputType::LAYOUT: {
      values = generateIValuesBasicTypes<c10::Layout>(input);
      array_type = c10::LayoutType::get();
      break;
    }
    case InputType::DEVICE: {
      values = generateIValuesBasicTypes<c10::Device>(input);
      break;
    }
    case InputType::MEMORY_FORMAT: {
      values = generateIValuesBasicTypes<c10::MemoryFormat>(input);
      break;
    }
    case InputType::GENERATOR: {
      std::stringstream error_message{};
      error_message
          << "StackGenerator::getInputVariants cannot process the input type: "
          << input.type << ". Generator should be replaced with 'Tensor seed'.";
      throw std::invalid_argument(error_message.str());
    }
    case InputType::STORAGE: {
      values.push_back(at::Storage(
          at::Storage::use_byte_size_t(),
          0,
          c10::GetAllocator(at::kCPU),
          true));
      break;
    }
    default: {
      std::stringstream error_message{};
      error_message
          << "StackGenerator::getInputVariants cannot process the input type: "
          << input.type;
      throw std::invalid_argument(error_message.str());
    }
  }
  if (input.is_array) {
    if (!input.array_length.has_value()) {
      std::stringstream error_message{};
      error_message
          << "StackGenerator::getInputVariants cannot process the input type: "
          << input.type
          << ". For parameter 'is_array'=true, the 'array_length' must be specified.";
      throw std::invalid_argument(error_message.str());
    }
    for (auto& value : values) {
      auto list = c10::List<c10::IValue>(array_type);
      for (int j = 0; j < input.array_length.value(); j++)
        list.push_back(value);

      value = c10::IValue(list);
    }
  }
  return values;
}

std::string StackGenerator::DebugString() const {
  std::stringstream ss;
  int size = inputs.size();
  for (int i = 0; i < size; i++) {
    ss << std::string("input[") << i << "]=\n" << inputs[i] << "\n\n";
  }
  return ss.str();
}

std::ostream& operator<<(
    std::ostream& os,
    const IStackGenerator& stack_generator) {
  os << stack_generator.DebugString();
  return os;
}

SchemaStackGenerator::SchemaStackGenerator(
    const std::string& schema,
    const std::string& op_name,
    const std::string& op_name_and_overload_name,
    const std::vector<int64_t>& ranks,
    const int default_array_length)
    : StackGenerator(ranks) {
  this->op_name = op_name;
  this->op_and_overload_name =
      op_name_and_overload_name.empty() ? op_name : op_name_and_overload_name;
  this->default_array_length = default_array_length;
  blacklisted_precision_types = getBlacklistedPrecisionTypes();
  whitelisted_precision_types = getWhitelistedPrecisionTypes();
  inputs = generateInputs(schema);
}

namespace {
std::vector<std::string> getInputsFromSchema(const std::string& input_schema) {
  if (input_schema.empty())
    return {};

  std::vector<std::string> params;

  auto schema = input_schema;

  std::size_t pos;
  while ((pos = schema.find(", ")) != std::string::npos) {
    params.push_back(schema.substr(0, pos));
    schema = schema.substr(pos + 2, schema.length());
  }

  params.push_back(schema);
  return params;
}

struct ParamTypeAndName {
  std::string type;
  std::string name;
};

ParamTypeAndName extractStrTypeAndParamName(const std::string& param) {
  size_t pos = param.find(" ");
  if (pos == std::string::npos) {
    return {param, ""};
  }
  return {param.substr(0, pos), param.substr(pos + 1, param.length())};
}

bool isArrayParam(const std::string& param) {
  return param.find("[") != std::string::npos;
}

bool isGeneratorParam(const std::string& param) {
  return param.find("Generator") != std::string::npos;
}

std::string eraseOptionalCharacters(const std::string& param) {
  auto result = param;

  size_t startPos = result.find("?");
  if (startPos != std::string::npos)
    result.erase(startPos, 1);

  return result;
}

int extractArrayLength(const std::string& param) {
  size_t startPos = param.find("[");
  size_t endPos = param.find("]");
  if (startPos == std::string::npos || endPos == std::string::npos ||
      (endPos - startPos) <= 1) {
    return 1;
  }

  auto size = param.substr(startPos + 1, endPos - startPos - 1);
  return std::stoi(size);
}

std::string eraseArrayCharacters(const std::string& param) {
  auto result = param;
  size_t startPos = result.find(
      "?["); // An array with optional values will always have specific value
  if (startPos == std::string::npos)
    startPos = result.find("[");

  size_t endPos = result.find("]");
  if (startPos != std::string::npos && endPos != std::string::npos)
    result.erase(startPos, endPos - startPos + 1);

  return result;
}

bool isOptionalParam(const std::string& param) {
  return param.find("?") != std::string::npos;
}

std::optional<std::vector<std::any>> getDefaultValue(
    std::string op_name,
    std::string param_name) {
  auto default_value_specific_op_it = default_values_specific_ops.find(op_name);
  if (default_value_specific_op_it != std::end(default_values_specific_ops)) {
    auto default_value_it =
        default_value_specific_op_it->second.find(param_name);
    if (default_value_it != std::end(default_value_specific_op_it->second)) {
      return default_value_it->second;
    }
  } else {
    auto default_value_it = default_values_all_ops.find(param_name);
    if (default_value_it != default_values_all_ops.end()) {
      return default_value_it->second;
    }
  }
  return std::nullopt;
}

void configureInputDescriptor(
    InputDescriptor* const input_desc,
    const bool default_value_found,
    const std::string& op_and_overload_name,
    const std::string& param_type,
    const std::string& param_name,
    const bool verbose) {
  auto& input_descriptor = *input_desc;
  switch (input_descriptor.type) {
    case InputType::PT_TENSOR: {
      input_descriptor.values = std::nullopt;
      if (param_name == "seed") {
        input_descriptor.match_rank = false;
        input_descriptor.ranks = {1};
        input_descriptor.match_precision_type = false;
        input_descriptor.dtypes = {at::ScalarType::Int};
      } else if (param_name == "index" || param_name == "indices") {
        input_descriptor.match_rank = true;
        input_descriptor.match_precision_type = false;
        input_descriptor.dtypes = {at::ScalarType::Long};
      } else {
        input_descriptor.match_rank = true;
        input_descriptor.match_precision_type = true;
      }
      break;
    }
    case InputType::PT_SCALAR: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{1};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << " (Scalar)'. Using generic default value: 1\n";
        }
      }
      break;
    }
    case InputType::NATIVE_FLOAT: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{1.0f};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << "' (float). Using generic default value: 1.0f\n";
        }
      }
      break;
    }
    case InputType::NATIVE_INT: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{1};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name << "' (int). Using default value: 1\n";
        }
      }
      break;
    }
    case InputType::NATIVE_BOOL: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{true, false};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << "' (bool). Using default values: {true, false}\n";
        }
      }
      break;
    }
    case InputType::NATIVE_STRING: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{std::string("")};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << "' (string). Using default values: empty string\n";
        }
      }
      break;
    }
    case InputType::DTYPE: {
      if (!default_value_found) {
        input_descriptor.match_precision_type = true;
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << "' (dtype). Using match_precision_type=true\n";
        }
      } else {
        input_descriptor.match_precision_type = false;
      }
      break;
    }
    case InputType::SYM_INT: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{c10::SymInt(1)};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name << "' (SymInt). Using default value: 1\n";
        }
      }
      break;
    }
    case InputType::LAYOUT: {
      if (!default_value_found) {
        input_descriptor.values = std::vector<std::any>{c10::Layout::Strided};
        if (verbose) {
          std::cout << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
                    << param_name
                    << "' (Layout). Using default value: Layout::Strided\n";
        }
      }
      break;
    }
    case InputType::DEVICE: {
      if (!default_value_found) {
        input_descriptor.values =
            std::vector<std::any>{c10::Device(c10::DeviceType::HPU)};
        if (verbose) {
          std::cout
              << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
              << param_name
              << "' (Device). Using default value: Device(DeviceType::HPU)\n";
        }
      }
      break;
    }
    case InputType::MEMORY_FORMAT: {
      if (!default_value_found) {
        input_descriptor.values =
            std::vector<std::any>{c10::MemoryFormat::Contiguous};
        if (verbose) {
          std::cout
              << "WARNING: Op '" + op_and_overload_name +
                  "' has no default value for '"
              << param_name
              << "' (MemoryFormat). Using default value: MemoryFormat::Contiguous\n";
        }
      }
      break;
    }
    case InputType::GENERATOR: {
      throw std::invalid_argument(
          "SchemaStackGenerator::generateInputs cannot process the input schema for '" +
          op_and_overload_name + "' op. Generator supports only 'None' value");
    }
    case InputType::STORAGE: {
      break;
    }
    default: {
      throw std::invalid_argument(
          "SchemaStackGenerator::generateInputs cannot process the input schema for '" +
          op_and_overload_name + "' op. Unknown input type: '" + param_type +
          "'");
    }
  }
}
} // namespace

std::vector<InputDescriptor> SchemaStackGenerator::generateInputs(
    const std::string& schema) const {
  const auto inputs = getInputsFromSchema(schema);

  std::vector<InputDescriptor> input_descriptors;
  for (const auto& input : inputs) {
    InputDescriptor input_descriptor;

    auto [param_type, param_name] = extractStrTypeAndParamName(input);

    if (isArrayParam(param_type)) {
      input_descriptor.is_array = true;
      input_descriptor.array_length =
          std::max(extractArrayLength(param_type), default_array_length);
      param_type = eraseArrayCharacters(param_type);
    }

    if (isGeneratorParam(param_type)) {
      param_type = "Tensor";
      param_name = "seed";
    }

    input_descriptor.name = param_name;
    input_descriptor.is_optional = isOptionalParam(param_type);

    if (input_descriptor.is_optional) {
      input_descriptor.allow_only_none = true;
      param_type = eraseOptionalCharacters(param_type);
    }

    auto input_type_it = input_type_map.find(param_type);
    if (input_type_it == input_type_map.end()) {
      throw std::invalid_argument(
          "SchemaStackGenerator::generateInputs cannot process the input schema for '" +
          op_and_overload_name + "' op. Unknown input type: '" + param_type +
          "' for param '" + param_type + " " + param_name + "'");
    } else {
      input_descriptor.type = input_type_it->second;
    }

    if (input_descriptor.type == InputType::NATIVE_STRING) {
      input_descriptor.is_optional = false;
    }

    if (not input_descriptor.is_optional) {
      input_descriptor.values = getDefaultValue(op_name, param_name);
      bool default_value_found = input_descriptor.values.has_value();

      configureInputDescriptor(
          &input_descriptor,
          default_value_found,
          op_and_overload_name,
          param_type,
          param_name,
          verbose);
    }
    input_descriptors.push_back(input_descriptor);
  }
  return input_descriptors;
}

std::vector<at::ScalarType> SchemaStackGenerator::getBlacklistedPrecisionTypes()
    const {
  auto it = blacklisted_precision_types_op_map.find(op_name);
  if (it != std::end(blacklisted_precision_types_op_map))
    return it->second;
  return {};
}

std::vector<at::ScalarType> SchemaStackGenerator::getWhitelistedPrecisionTypes()
    const {
  auto it = whitelisted_precision_types_op_map.find(op_name);
  if (it != std::end(whitelisted_precision_types_op_map))
    return it->second;
  return {};
}

} // namespace slrg
