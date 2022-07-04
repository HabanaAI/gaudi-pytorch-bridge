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

#include <ATen/Tensor.h>
#include <c10/core/DefaultDtype.h>

#pragma once

namespace habana_helpers {

class DTypeHelper {
 public:
  DTypeHelper& add_input(const c10::IValue* v);
  DTypeHelper& add_inputs(std::vector<const c10::IValue*>&& v);

  DTypeHelper& set_fixed_output_dtype(c10::ScalarType dtype);
  DTypeHelper& set_promote_to_common_type(bool type_promotion);
  DTypeHelper& set_promote_int_to_float(bool type_promotion);

  void build();
  c10::ScalarType get_common_dtype() const;
  c10::ScalarType get_result_dtype() const;

 private:
  bool promote_common_input_type_ = false;
  bool promote_int_to_float_ = false;

  std::vector<const c10::IValue*> input_values_;
  c10::ScalarType fixed_output_dtype_ = c10::ScalarType::Undefined;

  c10::ScalarType common_dtype_ = c10::ScalarType::Undefined;
  c10::ScalarType result_dtype_ = c10::ScalarType::Undefined;
};

} // namespace habana_helpers