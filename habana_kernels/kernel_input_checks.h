/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
