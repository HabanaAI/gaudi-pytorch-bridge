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

#include "fold_conv_batchnorm.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/script.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "weight_permute_graph.h"

#include <cmath>
#include <iterator>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pass_utils.h"
#include "recalculate_batchnorm_params.h"

using namespace torch;
using namespace torch::jit;

namespace habana_lazy {

bool computeUpdatedConvWeightAndBias(
    c10::IntArrayRef sizes,
    float* cw,
    float* cb,
    float* v,
    float* m,
    float* w,
    float* b,
    double bn_eps,
    bool cw_permution_in_hpu) {
  if ((cw == nullptr) || (cb == nullptr) || (v == nullptr) || (m == nullptr) ||
      (w == nullptr) || (b == nullptr)) {
    PT_LAZY_DEBUG("[computeUpdatedConvWeightAndBias] Null Ptr!");
    return false;
  }

  int kx = sizes.at(3);
  int ky = sizes.at(2);
  int ci = sizes.at(1);
  int co = sizes.at(0);

  // std::cout << "kx size: " << kx << std::endl << std::flush;
  // std::cout << "ky size: " << ky << std::endl << std::flush;
  // std::cout << "ci size: " << ci << std::endl << std::flush;
  // std::cout << "co size: " << co << std::endl << std::flush;

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto device_id = device.id();
  auto bytes = co * sizeof(float);

  void* host_ptr{nullptr};
  auto status = synHostMalloc(device_id, bytes * 2, 0, &host_ptr);
  HABANA_ASSERT(status == synStatus::synSuccess);
  double* s = (double*)host_ptr;
  for (auto i = 0; i < co; i++) {
    s[i] = ((double)w[i] / sqrt((double)v[i] + (double)bn_eps));
  }

  bool all_bias_zero = true;
  for (auto i = 0; i < co; i++) {
    if (cb[i] != 0) {
      all_bias_zero = false;
    }
  }

  PT_LAZY_DEBUG(
      "[computeUpdatedConvWeightAndBias] all_bias_zero: ", all_bias_zero);

  PT_LAZY_DEBUG("[computeUpdatedConvWeightAndBias] Calculate conv parameters");

  // Weight calculation
  // Ref: at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  if (cw_permution_in_hpu) {
    for (auto i = 0; i < co; i++) {
      auto t = s[i];
      for (auto a = 0; a < ci * ky * kx; a++) {
        // std::cout << "cw[" << a << "] = " << cw[a] << "------->";
        cw[a] = (float)((double)cw[a] * t);
        // std::cout << cw[a] << std::endl << std::flush;
      }
      cw += (ci * ky * kx);
    }
  } else {
    for (auto a = 0; a < (ky * kx); a++) {
      for (auto j = 0; j < ci; j++) {
        for (auto i = 0; i < co; i++) {
          auto t = s[i];
          // std::cout << "cw[" << i << "] = " << cw[i] << "------->";
          cw[i] = (float)((double)cw[i] * t);
          // std::cout << cw[i] << std::endl << std::flush;
        }
        cw += co;
      }
    }
  }

  // Bias calculation
  // Ref: at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w +
  // p.bn_b;
  for (auto i = 0; i < co; i++) {
    auto t = s[i];

    auto cb_old = cb[i];
    auto cb_new = (float)(((double)cb_old - (double)m[i]) * t + (double)b[i]);

    cb[i] = cb_new;
    b[i] = all_bias_zero ? cb_new : 0;
  }

  PT_LAZY_DEBUG(
      "[computeUpdatedConvWeightAndBias] Update remaining batch-norm parameters");
  for (auto i = 0; i < co; i++) {
    v[i] = 1.0;
    m[i] = 0;
    w[i] = 1.0;
  }

  //=========================================================================
  // Reference calculations:
  //=========================================================================
  // implementation taken from torch/nn/utils/fusion.py
  // at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  // const int64_t ndim = p.conv_w.dim();
  // at::DimVector sizes(ndim, 1);
  // sizes.at(0) = -1;

  // auto conv_w_dtype = p.conv_w.dtype();
  // auto conv_b_dtype = p.conv_b.dtype();

  // at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  // at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  // return std::make_tuple(new_w.to(conv_w_dtype), new_b.to(conv_b_dtype));
  //=========================================================================

  return true;
}

void CheckIfAutoCastNodePresent(
    std::shared_ptr<Graph>& graph,
    Node* conv,
    std::vector<Node*>& w_auto_cast,
    std::vector<Node*>& b_auto_cast) {
  for (auto node : graph->nodes()) {
    if ((strcmp(node->kind().toQualString(), "hpu::cast") == 0)) {
      auto cast_uses = node->output(0)->uses();
      for (auto cast_u : cast_uses) {
        if (cast_u.user == conv) {
          if (node->output(0) == conv->input(0)) {
            continue;
          } else if (node->output(0) == conv->input(1)) {
            PT_LAZY_DEBUG("[CheckIfAutoCastNodePresent] Conv weight auto_cast");
            w_auto_cast.emplace_back(node);
          } else if (node->output(0) == conv->input(2)) {
            PT_LAZY_DEBUG("[CheckIfAutoCastNodePresent] Conv bias auto_cast");
            b_auto_cast.emplace_back(node);
          }
        }
      }
    }
  }
}

/* Note:
   This function is akin to FoldFrozenConvBatchnorm() function in pytorch-fork
   with some tailoring to suit our need */

bool FuseConvBatchnorm(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs) {
  std::vector<Node*> nodes_for_deletion;
  std::vector<int32_t> indices_for_deletion;

  bool graph_modified = false;
  for (auto node : graph->nodes()) {
    auto node_name = node->kind().toQualString();
    PT_LAZY_DEBUG("Node Name: ", node_name);

    if ((strcmp(node_name, "hpu::native_batch_norm_inf") == 0) &&
        (node->inputs().at(0)->node()->kind() ==
         torch::jit::aten::convolution_overrideable)) {
      auto conv = node->inputs().at(0)->node();
      auto bn = node;

      std::vector<Node*> w_auto_cast;
      std::vector<Node*> b_auto_cast;
      CheckIfAutoCastNodePresent(graph, conv, w_auto_cast, b_auto_cast);
      auto auto_cast = (w_auto_cast.size() > 0) && (b_auto_cast.size() > 0);

      auto ib = auto_cast ? -1 : 2;
      auto nb = auto_cast ? b_auto_cast.at(0) : conv;

      auto conv_b_hb_tensor = GetBackEndTensorImpl(graph, stack, nb, ib);
      auto conv_b = GetDataInHostBuffer(graph, stack, nb, ib);
      if (!conv_b_hb_tensor || !conv_b) {
        PT_LAZY_DEBUG(
            "[FuseConvBatchnorm] Convolution without bias not yet supported");
        continue;
      }

      auto iw = auto_cast ? -1 : 1;
      auto nw = auto_cast ? w_auto_cast.at(0) : conv;

      auto conv_w_hb_tensor = GetBackEndTensorImpl(graph, stack, nw, iw);
      auto conv_w = GetDataInHostBuffer(graph, stack, nw, iw);

      auto conv_w_permutation = conv_w_hb_tensor->GetMemoryPermutation();
      PT_LAZY_DEBUG(
          "Conv weight permutation vector: ", VecToString(conv_w_permutation));

      // const auto& uses = conv->output()->uses();
      // if ((uses.size() > 1) ||
      //    (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && conv_w_hb_tensor &&
      //    !conv_w_hb_tensor->IsConstTensor()) ||
      //    (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && conv_b_hb_tensor &&
      //    !conv_b_hb_tensor->IsConstTensor())) {
      //     continue;
      // }

      int idx_bias = 1;
      auto bn_b = GetDataInHostBuffer(graph, stack, bn, idx_bias);
      redundant_inputs.emplace_back(bn->input(idx_bias));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_bias)->debugName());

      int idx_weight = 2;
      auto bn_w = GetDataInHostBuffer(graph, stack, bn, idx_weight);
      redundant_inputs.emplace_back(bn->input(idx_weight));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_weight)->debugName());

      int idx_running_mean = 3;
      auto bn_rm = GetDataInHostBuffer(graph, stack, bn, idx_running_mean);
      redundant_inputs.emplace_back(bn->input(idx_running_mean));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_running_mean)->debugName());

      int idx_running_var = 4;
      auto bn_rv = GetDataInHostBuffer(graph, stack, bn, idx_running_var);
      redundant_inputs.emplace_back(bn->input(idx_running_var));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_running_var)->debugName());

      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();

      auto status = computeUpdatedConvWeightAndBias(
          conv_w_hb_tensor->GetTensorSize(),
          (float*)conv_w,
          (float*)conv_b,
          (float*)bn_rv,
          (float*)bn_rm,
          (float*)bn_w,
          (float*)bn_b,
          bn_eps,
          conv_w_permutation.empty());
      if (!status) {
        PT_LAZY_DEBUG("[FuseConvBatchnorm] Compute unsuccessful!");
        continue;
      }

      PT_LAZY_DEBUG("[FuseConvBatchnorm] Update conv parameters");
      UpdateDataInDeviceMem(graph, stack, nw, iw, conv_w);
      UpdateDataInDeviceMem(graph, stack, nb, ib, conv_b);

      PT_LAZY_DEBUG("[FuseConvBatchnorm] Update batch-norm parameters");
      UpdateDataInDeviceMem(graph, stack, bn, idx_bias, bn_b);
      UpdateDataInDeviceMem(graph, stack, bn, idx_weight, bn_w);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_mean, bn_rm);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_var, bn_rv);

      bn->output()->replaceAllUsesWith(conv->output());
      nodes_for_deletion.emplace_back(bn);

      graph_modified = true;
    }
  }

  PT_LAZY_DEBUG("[FuseConvBatchnorm] Remove batch-norm nodes");
  for (auto node : nodes_for_deletion) {
    node->removeAllInputs();
    node->destroy();
  }

  PT_LAZY_DEBUG("[FuseConvBatchnorm] Exit");
  return graph_modified;
}

bool FoldConvBatchnorm(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs) {
  bool graph_modified = FuseConvBatchnorm(graph, stack, redundant_inputs);
  PT_LAZY_DEBUG("[FoldConvBatchnorm] graph_modified = ", graph_modified);
  if (graph_modified) {
    torch::jit::EliminateDeadCode(graph);
  }
  return graph_modified;
}

}; // namespace habana_lazy
