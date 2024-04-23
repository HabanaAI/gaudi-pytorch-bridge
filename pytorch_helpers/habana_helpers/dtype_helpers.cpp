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

#include "dtype_helpers.h"
#include <ATen/native/TypeProperties.h>
#include "backend/synapse_helpers/env_flags.h"
#include "logging.h"

namespace habana_helpers {

DTypeHelper& DTypeHelper::add_input(const c10::IValue* v) {
  input_values_.push_back(v);
  return *this;
}

DTypeHelper& DTypeHelper::add_inputs(std::vector<const c10::IValue*>&& v) {
  if (input_values_.empty()) {
    input_values_ = std::move(v);
  } else {
    input_values_.reserve(input_values_.size() + v.size());
    std::move(v.begin(), v.end(), std::back_inserter(input_values_));
  }
  return *this;
}

DTypeHelper& DTypeHelper::add_output(const c10::IValue* v) {
  output_values_.push_back(v);
  return *this;
}

DTypeHelper& DTypeHelper::set_output_dtype(c10::ScalarType dtype) {
  output_dtype_ = dtype;
  return *this;
};

DTypeHelper& DTypeHelper::set_promote_to_common_type(bool type_promotion) {
  promote_common_input_type_ = type_promotion;
  return *this;
}

DTypeHelper& DTypeHelper::set_promote_int_to_float(bool type_promotion) {
  promote_int_to_float_ = type_promotion;
  return *this;
}

DTypeHelper& DTypeHelper::set_promote_int_to_long(bool type_promotion) {
  promote_int_to_long_ = type_promotion;
  return *this;
}

DTypeHelper& DTypeHelper::set_safe_cast_to_output(bool safe_cast) {
  safe_cast_to_output_ = safe_cast;
  return *this;
}

void DTypeHelper::build() {
  HABANA_ASSERT(!input_values_.empty());

  // Helper lambda to get dtype of value
  auto get_dtype = [](const c10::IValue* v) {
    if (v->isTensor()) {
      return v->toTensor().scalar_type();
    }

    return v->toScalar().type();
  };

  if (!promote_common_input_type_) {
    common_dtype_ = get_dtype(input_values_.at(0));
  }

  if (!output_values_.empty()) {
    for (auto& output : output_values_) {
      result_dtype_ = get_dtype(output);
      break;
    }
  }

  if (promote_common_input_type_) {
    at::native::ResultTypeState state = {};
    for (auto& input : input_values_) {
      if (input->isTensor()) {
        state = at::native::update_result_type_state(input->toTensor(), state);
      } else if (input->isScalar()) {
        state = at::native::update_result_type_state(input->toScalar(), state);
      } else {
        for (const at::Tensor& tensor : input->toTensorList()) {
          state = update_result_type_state(tensor, state);
        }
      }
    }
    common_dtype_ = at::native::result_type(state);
  }

  // Promotion of integer value to default floating point dtype.
  // This kind of promotion is expected for i.e. some binary operators like div
  // or unary operators like cosine.
  if (promote_int_to_float_ && c10::isIntegralType(common_dtype_, true)) {
    common_dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // Promotion of int32 value to int64.
  // This kind of promotion is expected for i.e. some unary operators like
  // cumsum.
  if (promote_int_to_long_ && c10::isIntegralType(common_dtype_, true)) {
    common_dtype_ = c10::ScalarType::Long;
  }

  // If output dtype and output tensor were specified, their dtypes must match
  if (output_dtype_ != c10::ScalarType::Undefined &&
      result_dtype_ != c10::ScalarType::Undefined) {
    HABANA_ASSERT(
        output_dtype_ == result_dtype_,
        "Expected out tensor to have dtype ",
        output_dtype_,
        ", but got ",
        result_dtype_,
        " instead");
  }

  // When safe cast check is enabled, the helper verifies if cast can be done
  // between computation and output dtype i.e. float to int conversion will not
  // be allowed.
  if (safe_cast_to_output_) {
    HABANA_ASSERT(
        c10::canCast(common_dtype_, result_dtype_),
        "result type ",
        common_dtype_,
        " can't be cast to the "
        "desired output type ",
        result_dtype_);
  }

  result_dtype_ = output_dtype_ == c10::ScalarType::Undefined ? result_dtype_
                                                              : output_dtype_;

  result_dtype_ = result_dtype_ == c10::ScalarType::Undefined ? common_dtype_
                                                              : result_dtype_;

  TORCH_CHECK(
      common_dtype_ != c10::ScalarType::Undefined,
      "Common data type cannot be determined");
  TORCH_CHECK(
      result_dtype_ != c10::ScalarType::Undefined,
      "Result data type cannot be determined");
}

c10::ScalarType DTypeHelper::get_common_dtype(
    bool double_support,
    bool int64_support) const {
  auto common_type = common_dtype_;
  if (!double_support) {
    common_type = common_type == c10::ScalarType::Double
        ? c10::ScalarType::Float
        : common_type;
  }

  if (!int64_support &&
      habana_helpers::is_downcast_to_int_needed(common_type)) {
    common_type = c10::ScalarType::Int;
  }
  return common_type;
}

c10::ScalarType DTypeHelper::get_result_dtype() const {
  return result_dtype_;
}

DTypeHelper DTypeHelper::op_with_optional_dtype_promotion(
    const std::vector<at::IValue>& inputs,
    bool to_float,
    c10::optional<const at::IValue*> output,
    bool safe_cast) {
  DTypeHelper dtype_helper;
  std::vector<const at::IValue*> input_tensors;
  input_tensors.reserve(inputs.size());
  for (const auto& val : inputs) {
    // Remove this check when we reuse FE dtype in BE
    if (val.isTensor() or val.isScalar() or val.isTensorList()) {
      input_tensors.emplace_back(&val);
    }
  }
  dtype_helper.add_inputs(std::move(input_tensors))
      .set_promote_to_common_type(true)
      .set_promote_int_to_float(to_float)
      .set_safe_cast_to_output(safe_cast);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

DTypeHelper DTypeHelper::unary_op_with_optional_int_to_float_promotion(
    const std::vector<at::IValue>& inputs,
    bool int_to_float,
    c10::optional<const at::IValue*> output,
    bool safe_cast) {
  DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0)})
      .set_promote_to_common_type(true)
      .set_promote_int_to_float(int_to_float)
      .set_safe_cast_to_output(safe_cast);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

DTypeHelper DTypeHelper::unary_op_with_optional_int_to_long_promotion(
    const std::vector<at::IValue>& inputs,
    c10::optional<const at::IValue*> output,
    c10::optional<c10::ScalarType> dtype,
    bool promote_int_to_long) {
  DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0)})
      .set_promote_int_to_long(promote_int_to_long);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }
  if (dtype.has_value()) {
    dtype_helper.set_output_dtype(dtype.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

DTypeHelper DTypeHelper::binary_op_with_type_promotion(
    const std::vector<at::IValue>& inputs,
    c10::optional<const at::IValue*> output,
    bool safe_cast) {
  DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0), &inputs.at(1)})
      .set_promote_to_common_type(true)
      .set_safe_cast_to_output(safe_cast);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

DTypeHelper DTypeHelper::binary_op_with_optional_int_to_float_promotion(
    const std::vector<at::IValue>& inputs,
    bool int_to_float,
    c10::optional<const at::IValue*> output,
    bool safe_cast) {
  DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0), &inputs.at(1)})
      .set_promote_to_common_type(true)
      .set_promote_int_to_float(int_to_float)
      .set_safe_cast_to_output(safe_cast);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

DTypeHelper DTypeHelper::binary_op_with_int_to_float_promotion(
    const std::vector<at::IValue>& inputs,
    c10::optional<const at::IValue*> output,
    bool safe_cast) {
  DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0), &inputs.at(1)})
      .set_promote_to_common_type(true)
      .set_promote_int_to_float(true)
      .set_safe_cast_to_output(safe_cast);
  if (output.has_value()) {
    dtype_helper.add_output(output.value());
  }

  dtype_helper.build();
  return dtype_helper;
}

c10::ScalarType DTypeHelper::get_compute_dtype(
    const std::vector<at::IValue>& stack,
    c10::optional<at::Tensor> opt_output,
    DtypePromoteVariant promote_variant,
    bool safe_cast,
    c10::optional<c10::ScalarType> dtype,
    bool double_support,
    bool int64_support) {
  std::vector<const at::IValue*> inputs;
  inputs.reserve(stack.size());
  for (const auto& val : stack) {
    // Remove this check when we reuse FE dtype in BE
    if (val.isTensor() or val.isScalar() or val.isTensorList()) {
      inputs.emplace_back(&val);
    }
  }

  bool promote_to_common_type =
      promote_variant == DtypePromoteVariant::kPromoteToCommon or
      promote_variant == DtypePromoteVariant::kPromoteIntToFloat;
  bool promote_int_to_float =
      promote_variant == DtypePromoteVariant::kPromoteIntToFloat;
  bool promote_int_to_long = promote_variant == DtypePromoteVariant::kReduction;

  DTypeHelper dtype_helper;
  dtype_helper.add_inputs(std::move(inputs))
      .set_promote_to_common_type(promote_to_common_type)
      .set_promote_int_to_float(promote_int_to_float)
      .set_promote_int_to_long(promote_int_to_long)
      .set_safe_cast_to_output(safe_cast);

  if (dtype.has_value()) {
    dtype_helper.set_output_dtype(*dtype);
  }

  at::IValue output;
  if (opt_output.has_value()) {
    output = *opt_output;
    dtype_helper.add_output(&output);
  }

  dtype_helper.build();
  return dtype.has_value()
      ? dtype_helper.get_result_dtype()
      : dtype_helper.get_common_dtype(double_support, int64_support);
}

} // namespace habana_helpers
