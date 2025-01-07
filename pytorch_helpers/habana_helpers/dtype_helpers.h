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

#include <ATen/Tensor.h>
#include <c10/core/DefaultDtype.h>
#include "backend/helpers/get_n_bytes.h"

namespace habana_helpers {

class DTypeHelper {
 public:
  enum class DtypePromoteVariant : uint8_t {
    kNone,
    kPromoteToCommon,
    kPromoteIntToFloat,
    kReduction
  };

  DTypeHelper& add_input(const c10::IValue* v);
  DTypeHelper& add_inputs(std::vector<const c10::IValue*>&& v);
  DTypeHelper& add_output(const c10::IValue* v);

  DTypeHelper& set_output_dtype(c10::ScalarType dtype);
  DTypeHelper& set_promote_to_common_type(bool type_promotion);
  DTypeHelper& set_promote_int_to_float(bool type_promotion);
  DTypeHelper& set_promote_int_to_long(bool type_promotion);
  DTypeHelper& set_safe_cast_to_output(bool safe_cast);

  void build();
  c10::ScalarType get_common_dtype(
      bool double_support = true,
      bool int64_support = true) const;
  c10::ScalarType get_result_dtype() const;

  static DTypeHelper binary_op_with_type_promotion(
      const std::vector<at::IValue>& stack,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static DTypeHelper binary_op_with_optional_int_to_float_promotion(
      const std::vector<at::IValue>& stack,
      bool int_to_float,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static DTypeHelper op_with_optional_dtype_promotion(
      const std::vector<at::IValue>& inputs,
      bool to_float,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static c10::ScalarType get_compute_dtype(
      const std::vector<at::IValue>& stack,
      c10::optional<at::Tensor> opt_output,
      DtypePromoteVariant promote_variant,
      bool safe_cast,
      c10::optional<c10::ScalarType> dtype = c10::nullopt,
      bool double_support = true,
      bool int64_support = true);

 private:
  bool promote_common_input_type_ = false;
  bool promote_int_to_float_ = false;
  bool promote_int_to_long_ = false;
  bool safe_cast_to_output_ = false;

  std::vector<const c10::IValue*> input_values_;
  std::vector<const c10::IValue*> output_values_;
  c10::ScalarType output_dtype_ = c10::ScalarType::Undefined;

  c10::ScalarType common_dtype_ = c10::ScalarType::Undefined;
  c10::ScalarType result_dtype_ = c10::ScalarType::Undefined;
};

} // namespace habana_helpers
