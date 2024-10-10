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
#include <cmath>
#include <iterator>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pass_utils.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "recalculate_batchnorm_params.h"

namespace {
bool computeUpdatedConvWeightAndBias(
    c10::IntArrayRef sizes,
    float* cw,
    float* cb,
    float* v,
    float* m,
    float* w,
    float* b,
    double bn_eps,
    bool cw_permutation_in_hpu) {
  // conv bias is optional
  if ((cw == nullptr) || (v == nullptr) || (m == nullptr) || (w == nullptr) ||
      (b == nullptr)) {
    PT_LAZY_DEBUG("[computeUpdatedConvWeightAndBias] Null Ptr!");
    return false;
  }

  int kx = sizes.at(3);
  int ky = sizes.at(2);
  int ci = sizes.at(1);
  int co = sizes.at(0);

  auto& device = habana::HPURegistrar::get_device();
  auto device_id = device.id();
  auto bytes = co * sizeof(float);

  void* host_ptr{nullptr};
  auto status = synHostMalloc(device_id, bytes * 2, 0, &host_ptr);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  double* s = (double*)host_ptr;
  for (auto i = 0; i < co; i++) {
    s[i] = ((double)w[i] / sqrt((double)v[i] + (double)bn_eps));
  }

  bool all_bias_zero = true;
  if (cb == nullptr) {
    void* cb_host_ptr{nullptr};
    auto status = synHostMalloc(device_id, bytes * 2, 0, &cb_host_ptr);
    HABANA_ASSERT(
        status == synStatus::synSuccess, Logger::synStatusToStr(status));
    cb = (float*)cb_host_ptr;
    for (auto i = 0; i < co; i++) {
      cb[i] = 0;
    }
  } else {
    for (auto i = 0; i < co; i++) {
      if (cb[i] != 0) {
        all_bias_zero = false;
        break;
      }
    }
  }

  PT_LAZY_DEBUG(
      "[computeUpdatedConvWeightAndBias] all_bias_zero: ", all_bias_zero);

  PT_LAZY_DEBUG("[computeUpdatedConvWeightAndBias] Calculate conv parameters");

  // Weight calculation
  // Ref: at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  if (cw_permutation_in_hpu) {
    for (auto i = 0; i < co; i++) {
      auto t = s[i];
      for (auto a = 0; a < ci * ky * kx; a++) {
        cw[a] = (float)((double)cw[a] * t);
      }
      cw += (ci * ky * kx);
    }
  } else {
    for (auto a = 0; a < (ky * kx); a++) {
      for (auto j = 0; j < ci; j++) {
        for (auto i = 0; i < co; i++) {
          auto t = s[i];
          cw[i] = (float)((double)cw[i] * t);
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
    torch::jit::Node* conv,
    std::vector<torch::jit::Node*>& w_auto_cast,
    std::vector<torch::jit::Node*>& b_auto_cast) {
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
  std::vector<torch::jit::Node*> nodes_for_deletion;
  std::vector<int32_t> indices_for_deletion;
  bool graph_modified = false;
  PtTensorInferenceData::get_instance().print_map();

  for (auto node : graph->nodes()) {
    auto node_name = node->kind().toQualString();
    PT_LAZY_DEBUG("Node Name: ", node_name);
    if ((strcmp(node_name, "hpu::native_batch_norm_inf") == 0) &&
        (node->inputs().at(0)->node()->kind() ==
         torch::jit::aten::convolution_overrideable)) {
      auto conv = node->inputs().at(0)->node();
      auto bn = node;

      std::vector<torch::jit::Node*> w_auto_cast;
      std::vector<torch::jit::Node*> b_auto_cast;
      CheckIfAutoCastNodePresent(graph, conv, w_auto_cast, b_auto_cast);
      auto w_auto_cast_en = (w_auto_cast.size() > 0);
      auto b_auto_cast_en = (b_auto_cast.size() > 0);

      auto ib = b_auto_cast_en ? -1 : 2;
      auto nb = b_auto_cast_en ? b_auto_cast.at(0) : conv;

      habana::TensorExtraMeta* conv_b_tmeta_ptr{nullptr};
      habana::StorageExtraMeta* conv_b_smeta_ptr{nullptr};
      std::tie(conv_b_tmeta_ptr, conv_b_smeta_ptr) =
          habana_lazy::GetBackEndTensorMeta(graph, stack, nb, ib);
      auto conv_b = habana_lazy::GetDataInHostBuffer(graph, stack, nb, ib);
      if (!conv_b_tmeta_ptr || !conv_b) {
        PT_LAZY_DEBUG("[FuseConvBatchnorm] Convolution bias not found");
      }

      auto iw = w_auto_cast_en ? -1 : 1;
      auto nw = w_auto_cast_en ? w_auto_cast.at(0) : conv;

      habana::TensorExtraMeta* conv_w_tmeta_ptr{nullptr};
      habana::StorageExtraMeta* conv_w_smeta_ptr{nullptr};
      std::tie(conv_w_tmeta_ptr, conv_w_smeta_ptr) =
          habana_lazy::GetBackEndTensorMeta(graph, stack, nw, iw);
      auto conv_w = habana_lazy::GetDataInHostBuffer(graph, stack, nw, iw);
      if (!conv_w_tmeta_ptr || !conv_w) {
        PT_LAZY_DEBUG(
            "[FuseConvBatchnorm] Convolution without weight not yet supported");
        continue;
      }

      auto conv_w_permutation = conv_w_smeta_ptr->get_memory_permutation();
      PT_LAZY_DEBUG(
          "Conv weight permutation vector: ", VecToString(conv_w_permutation));

      // const auto& uses = conv->output()->uses();
      // if ((uses.size() > 1) ||
      //    (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && conv_w_tmeta_ptr &&
      //    !conv_w_tmeta_ptr->is_const_tensor()) ||
      //    (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && conv_b_tmeta_ptr &&
      //    !conv_b_tmeta_ptr->is_const_tensor())) {
      //     continue;
      // }

      int idx_bias = 1;
      auto bn_b = habana_lazy::GetDataInHostBuffer(graph, stack, bn, idx_bias);
      if (!bn_b) {
        PT_LAZY_DEBUG("[FuseConvBatchnorm] BN without bias not yet supported");
        continue;
      }
      // If convolution doesn't have a bias, we will re-use bn bias as
      // convolution bias in the graph
      if (conv_b_tmeta_ptr && conv_b) {
        redundant_inputs.emplace_back(bn->input(idx_bias));
        PT_LAZY_DEBUG(
            "[FuseConvBatchnorm] redundant_input: ",
            bn->input(idx_bias)->debugName());
      }

      int idx_weight = 2;
      auto bn_w =
          habana_lazy::GetDataInHostBuffer(graph, stack, bn, idx_weight);
      if (!bn_w) {
        redundant_inputs.pop_back();
        continue;
      }
      redundant_inputs.emplace_back(bn->input(idx_weight));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_weight)->debugName());

      int idx_running_mean = 3;
      auto bn_rm =
          habana_lazy::GetDataInHostBuffer(graph, stack, bn, idx_running_mean);
      if (!bn_rm) {
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        continue;
      }
      redundant_inputs.emplace_back(bn->input(idx_running_mean));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_running_mean)->debugName());

      int idx_running_var = 4;
      auto bn_rv =
          habana_lazy::GetDataInHostBuffer(graph, stack, bn, idx_running_var);
      if (!bn_rv) {
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        continue;
      }
      redundant_inputs.emplace_back(bn->input(idx_running_var));
      PT_LAZY_DEBUG(
          "[FuseConvBatchnorm] redundant_input: ",
          bn->input(idx_running_var)->debugName());

      auto bn_eps =
          torch::jit::constant_as<double>(bn->namedInput("eps")).value();
      PT_LAZY_DEBUG("[FuseConvBatchnorm] bn_eps = ", bn_eps);

      auto status = computeUpdatedConvWeightAndBias(
          conv_w_tmeta_ptr->get_tensor_size(),
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
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        redundant_inputs.pop_back();
        continue;
      }

      PT_LAZY_DEBUG("[FuseConvBatchnorm] Update conv parameters");
      habana_lazy::UpdateDataInDeviceMem(graph, stack, nw, iw, conv_w);

      if (conv_b_tmeta_ptr) {
        habana_lazy::UpdateDataInDeviceMem(graph, stack, nb, ib, conv_b);
      }

      PT_LAZY_DEBUG("[FuseConvBatchnorm] Update batch-norm parameters");
      habana_lazy::UpdateDataInDeviceMem(graph, stack, bn, idx_bias, bn_b);
      habana_lazy::UpdateDataInDeviceMem(graph, stack, bn, idx_weight, bn_w);
      habana_lazy::UpdateDataInDeviceMem(
          graph, stack, bn, idx_running_mean, bn_rm);
      habana_lazy::UpdateDataInDeviceMem(
          graph, stack, bn, idx_running_var, bn_rv);

      bn->output()->replaceAllUsesWith(conv->output());

      if (!conv_b_tmeta_ptr) {
        PT_LAZY_DEBUG("[FuseConvBatchnorm] Use batchnorm bias as conv bias");
        if (w_auto_cast_en) {
          torch::jit::WithInsertPoint insert_point(conv);
          auto to_cast_type = graph->insertConstant(c10::ScalarType::BFloat16);
          auto bn_bias_cast_node =
              graph->create(nw->kind(), {bn->input(idx_bias), to_cast_type}, 1);
          bn_bias_cast_node->setScope(nw->scope());
          bn_bias_cast_node->copyAttributes(*nw);
          conv->replaceInput(ib, bn_bias_cast_node->output(0));
          // TODO - is this correct?
          stack.emplace_back(conv->input(2));
          graph->insertNode(bn_bias_cast_node);
        } else {
          conv->replaceInput(ib, bn->input(idx_bias));
        }
      }

      nodes_for_deletion.emplace_back(bn);

      graph_modified = true;
      PtTensorInferenceData::get_instance().update_map(
          bn->scope()->name().toUnqualString(),
          conv->scope()->name().toUnqualString());
    }
  }

  PT_LAZY_DEBUG("[FuseConvBatchnorm] Remove batch-norm nodes");
  for (auto node : nodes_for_deletion) {
    node->removeAllInputs();
    node->destroy();
  }

  PtTensorInferenceData::get_instance().print_map();
  PT_LAZY_DEBUG("[FuseConvBatchnorm] Exit");
  return graph_modified;
}
} // namespace

namespace habana_lazy {
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
