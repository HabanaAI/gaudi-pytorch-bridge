/*****************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <initializer_list>
#include <iostream>
#include <string>
#include <unordered_set>
#include "habana_helpers/logging.h"

class HabanaWhiteList {
 private:
  static std::unordered_set<std::string> HabanaWhiteListOps;

 public:
  static bool is_op_habana_whitelisted(torch::jit::Node* node);
  static void load_whitelisted_ops();
};

std::unordered_set<std::string> HabanaWhiteList::HabanaWhiteListOps = {};

bool HabanaWhiteList::is_op_habana_whitelisted(torch::jit::Node* node) {
  // This section of code is required for prim::Constant handling
  // Since we do not support any other nodes other than prim::Constant
  // We have to ensure that we return true only for prim::Constant node
  if (node->kind().is_prim()) {
    if ((torch::jit::prim::Constant == node->kind()) ||
        (torch::jit::prim::dtype == node->kind())) {
      return true;
    } else {
      return false;
    }
  }

  // Come here for all non-prim based nodes
  // which have a properly defined schema in aten
  // That can be obtained from the node
  else {
    if (node->kind().is_aten()) {
      auto schema = node->getOperator().schema();
      std::string schema_string = torch::jit::canonicalSchemaString(schema);
      if (HabanaWhiteListOps.find(schema_string) != HabanaWhiteListOps.end()) {
        return true;
      } else {
        return false;
      }
    }
  }

  return false;
}

void HabanaWhiteList::load_whitelisted_ops() {
  if (std::getenv("HABANA_GRAPH_FUSION_OPS_FILE")) {
    const char* wl_filename = std::getenv("HABANA_GRAPH_FUSION_OPS_FILE");
    std::string wl_file =
        (wl_filename == NULL) ? std::string() : std::string(wl_filename);

    if (!wl_file.empty()) {
      std::ifstream whiteListFile(wl_file);
      if (!whiteListFile.is_open()) {
        PT_BRIDGE_FATAL(" Unable to open whitelist File!");
        PT_BRIDGE_FATAL(wl_file);
      }
      std::string opname;
      while (whiteListFile) {
        getline(whiteListFile, opname);
        HabanaWhiteList::HabanaWhiteListOps.insert(opname);
      }
      whiteListFile.close();
    }
  } else {
    HabanaWhiteList::HabanaWhiteListOps = {
        "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
        "aten::abs(Tensor self) -> Tensor",
        "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
        "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor",
        "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride=[], int[] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor",
        "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
        "aten::cat(Tensor[] tensors, int dim) -> Tensor",
        "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[] output_mask) -> (Tensor,Tensor,Tensor)",
        "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
        "aten::div(Tensor self, Scalar other) -> Tensor",
        "aten::div(Tensor self, Tensor other) -> Tensor",
        "aten::embedding_bag_sum_bwd.out(Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, *, Tensor(a!) out) -> Tensor(a!)",
        "aten::embedding_bag_sum_fwd(Tensor input, Tensor indices_fwd, Tensor offsets_fwd, Tensor valid_count_fwd, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, Tensor grad_weight) -> Tensor",
        "aten::eq(Tensor self, Tensor other) -> Tensor",
        "aten::fill_(Tensor(a !) self, Scalar value)->Tensor(a !)",
        "aten::flatten(Tensor self, int start_dim, int end_dim) -> Tensor",
        "aten::gt(Tensor self, Tensor other) -> Tensor",
        "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor",
        "aten::max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)",
        "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode, Tensor indices) -> Tensor",
        "aten::mm(Tensor self, Tensor mat2) -> Tensor",
        "aten::mul(Tensor self, Scalar other) -> Tensor",
        "aten::mul(Tensor self, Tensor other) -> Tensor",
        "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
        "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[] output_mask) -> (Tensor, Tensor, Tensor)",
        "aten::neg(Tensor self) -> Tensor",
        "aten::permute(Tensor self, int[] dims) -> Tensor",
        "aten::relu(Tensor self) -> Tensor",
        "aten::reshape(Tensor self, int[] shape) -> Tensor",
        "aten::sigmoid(Tensor self) -> Tensor",
        "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor",
        "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::t(Tensor self) -> Tensor",
        "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
        "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor",
        "aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
        "aten::to(Tensor self, Device device, int dtype, bool non_blocking, bool copy, int? memory_format) -> Tensor",
        "aten::view(Tensor self, int[] size) -> Tensor"};
  }
}
