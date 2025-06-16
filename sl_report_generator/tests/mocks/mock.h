/**
 * Copyright (c) 2025 Intel Corporation
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

#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace slrg::unit_tests {
using FunctionNameT = std::string;
using legal_types = std::variant<bool, std::string, std::vector<at::Stack>>;

class Mock {
 public:
  Mock(std::string class_name) : class_name(class_name){};
  virtual ~Mock() = default;

  std::uint64_t getFunctionCallCounter(FunctionNameT function_name) {
    return counters[function_name];
  }

  Mock& enable_function(FunctionNameT function_name) {
    enabled_functions[function_name] = true;
    return *this;
  }

  Mock& enable_functions(const std::vector<FunctionNameT>& function_names) {
    for (const auto& name : function_names)
      enabled_functions[name] = true;
    return *this;
  }

  Mock& register_outputs(
      FunctionNameT function_name,
      const std::vector<legal_types>& outputs) {
    auto& function_outputs = output_values[function_name];
    function_outputs.insert(
        std::begin(function_outputs), std::begin(outputs), std::end(outputs));
    return *this;
  }

  Mock& register_output(FunctionNameT function_name, legal_types output) {
    output_values[function_name].push_back(output);
    return *this;
  }

  Mock& enable_outputs_cycling(FunctionNameT function_name) {
    cycle_outputs[function_name] = true;
    return *this;
  }

  Mock& disable_outputs_cycling(FunctionNameT function_name) {
    cycle_outputs[function_name] = false;
    return *this;
  }

  template <typename T>
  T mock_function(FunctionNameT function_name) {
    increase_counter_verify_enablement(function_name);
    const auto output_val_it = output_values.find(function_name);
    if (output_val_it == std::end(output_values) ||
        output_val_it->second.empty()) {
      return {};
    }
    const auto output_vals = output_val_it->second;
    auto index = counters[function_name] - 1;
    if (cycle_outputs[function_name])
      index %= output_vals.size();
    return std::get<T>(output_vals.at(index));
  }

 protected:
  std::map<FunctionNameT, std::uint64_t> counters;
  std::map<FunctionNameT, bool> enabled_functions;
  std::map<FunctionNameT, bool> cycle_outputs;
  std::map<FunctionNameT, std::vector<legal_types>> output_values;

  const std::string class_name;

  void increase_counter_verify_enablement(FunctionNameT function_name) {
    counters[function_name]++;
    if (not enabled_functions[function_name])
      throw std::logic_error(
          "Unexpected " + class_name + "::" + function_name +
          " function call.");
  }
};
} // namespace slrg::unit_tests
