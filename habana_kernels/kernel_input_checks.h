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

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <cxxabi.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/library.h>
#include "habana_kernels/fallback_helper.h"
#include "habana_kernels/op_support_level.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

class OpAttributeCheck {
 private:
  static OpAttributeCheck* instance;
  int arg_position;
  bool is_valid;
  OpAttributeCheck() {
    arg_position = 1;
    is_valid = true;
    populate_attribute_checks();
  }

 public:
  static std::unordered_map<
      std::string,
      std::unordered_map<int, std::vector<c10::IValue>>>
      ivalue_op_info;
  std::string op_name;
  std::string var_type;

  ~OpAttributeCheck() = default;

  static OpAttributeCheck* get_instance() {
    static OpAttributeCheck instance;
    instance.arg_position = 1;
    instance.is_valid = true;
    return &instance;
  }

  void populate_attribute_checks();

  void hpu_check_ivalues(std::string oper_name, torch::jit::Stack& inputs);

  bool get_status() {
    return is_valid;
  }
};

using TypeVector = std::vector<at::TypePtr>;

/**
 * Checks if `op` can be executed on HPU given provided arguments.  This
 * process might be a bit nuanced, however the most straightforward test is if
 * types of the arguments are supported. This check expects that type promotion
 * of arguments already happened. consider that whereas add(fp16, fp16) might
 * not be supported on some platforms, add(fp16, bf16) likely is.  This is
 * because since arguments have different types, they undergo type promotion to
 * fp32 which is broadly supported.
 */
OpSupportLevel hpu_check_inputs_impl(
    const std::string& op,
    const std::vector<at::Tensor>& tensors);
