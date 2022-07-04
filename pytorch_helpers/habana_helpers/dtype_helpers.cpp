/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

DTypeHelper& DTypeHelper::set_fixed_output_dtype(c10::ScalarType dtype) {
  fixed_output_dtype_ = dtype;
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

  if (promote_common_input_type_) {
    at::native::ResultTypeState state = {};
    for (auto& input : input_values_) {
      if (input->isTensor()) {
        state = at::native::update_result_type_state(input->toTensor(), state);
      } else {
        state = at::native::update_result_type_state(input->toScalar(), state);
      }
    }
    common_dtype_ = at::native::result_type(state);
  }

  // Promotion of integer value to default floating point dtype.
  // This kind of promotion is expected for i.e. some binary operators i.e. div
  // or unary operators like cosine.
  if (promote_int_to_float_ && c10::isIntegralType(common_dtype_, true)) {
    common_dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  result_dtype_ = fixed_output_dtype_ == c10::ScalarType::Undefined
      ? common_dtype_
      : fixed_output_dtype_;

  HABANA_ASSERT(common_dtype_ != c10::ScalarType::Undefined);
}

c10::ScalarType DTypeHelper::get_common_dtype() const {
  return common_dtype_;
}

c10::ScalarType DTypeHelper::get_result_dtype() const {
  return result_dtype_;
}

} // namespace habana_helpers