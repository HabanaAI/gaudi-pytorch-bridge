/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/DefaultDtype.h>
#include "backend/helpers/get_n_bytes.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"
#ifndef HAVE_FP8_SUPPORT
#define HAVE_FP8_SUPPORT \
  IS_PYTORCH_AT_LEAST(2, 1) || IS_PYTORCH_FORK_AT_LEAST(1, 1)
#endif

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

  static DTypeHelper unary_op_with_optional_int_to_float_promotion(
      const std::vector<at::IValue>& stack,
      bool int_to_float,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static DTypeHelper unary_op_with_optional_int_to_long_promotion(
      const std::vector<at::IValue>& stack,
      c10::optional<const at::IValue*> output,
      c10::optional<c10::ScalarType> dtype,
      bool promote_int_to_long);
  static DTypeHelper binary_op_with_type_promotion(
      const std::vector<at::IValue>& stack,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static DTypeHelper binary_op_with_optional_int_to_float_promotion(
      const std::vector<at::IValue>& stack,
      bool int_to_float,
      c10::optional<const at::IValue*> output,
      bool safe_cast);
  static DTypeHelper binary_op_with_int_to_float_promotion(
      const std::vector<at::IValue>& stack,
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
