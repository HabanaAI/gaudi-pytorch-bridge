/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "recalculate_batchnorm_params.h"
#include <torch/script.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "weight_permute_graph.h"

#include <cmath>
#include <iterator>
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pass_utils.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace torch;
using namespace torch::jit;

namespace habana_lazy {

habana_lazy::HbInternalTensorImpl* GetBackEndTensorImpl(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx) {
  habana_lazy::HbInternalTensorImpl* impl{nullptr};

  if (idx != -1) {
    if (node->input(idx)->type() == NoneType::get()) {
      // std::cout << "[GetBackEndTensorImpl] [" << idx << "] NoneType" <<
      // std::endl << std::flush;
      return impl;
    }

    auto value = node->input(idx);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    if ((index >= 0) && (index < (int32_t)stack.size())) {
      if (stack[index].isTensor()) {
        auto tensor = stack[index].toTensor();
        if (tensor.has_storage()) {
          // std::cout << "[GetBackEndTensorImpl] [" << idx << "]" << std::endl
          // << std::flush;
          impl = habana_lazy::GetHbInternalTensorImpl(tensor);
          impl->SetTensorSize(tensor.sizes());
        }
      }
    }
  } else {
    auto value = node->input(0);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    if (stack[index].isTensor()) {
      auto tensor = stack[index].toTensor();
      if (tensor.has_storage()) {
        // std::cout << "[GetBackEndTensorImpl] [" << idx << "]" << std::endl <<
        // std::flush;
        impl = habana_lazy::GetHbInternalTensorImpl(tensor);
        impl->SetTensorSize(tensor.sizes());
      }
    }
  }

  return impl;
}

bool recomputeBatchnormParams(
    c10::IntArrayRef sizes,
    float* v,
    float* m,
    float* w,
    float* b,
    double bn_eps = 0.0) {
  if ((v == nullptr) || (m == nullptr) || (w == nullptr) || (b == nullptr)) {
    // std::cout << "[recomputeBatchnormParams] Null Ptr!" << std::endl <<
    // std::flush;
    return false;
  }

  // std::cout << "[recomputeBatchnormParams]" << std::endl << std::flush;

  int co = sizes.at(0);
  // std::cout << "co size: " << co << std::endl << std::flush;

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto device_id = device.id();
  auto bytes = co * sizeof(float);

  void* host_ptr{nullptr};
  auto status = synHostMalloc(device_id, bytes * 2, 0, &host_ptr);
  // std::cout << "[synHostMalloc - s] " << bytes * 2 << std::endl <<
  // std::flush; std::cout << "[synHostMalloc - s] " << status << std::endl <<
  // std::flush;
  HABANA_ASSERT(status == synStatus::synSuccess);
  double* s = (double*)host_ptr;
  for (auto i = 0; i < co; i++) {
    s[i] = ((double)1.0 / sqrt((double)v[i] + (double)bn_eps));
    // std::cout << "s[" << i << "] = " << s[i] << std::endl << std::flush;
  }

  // Weight calculation [G' = G/s = new Gamma]
  // std::cout << "[Weight calculation] " << std::endl << std::flush;
  for (auto i = 0; i < co; i++) {
    // std::cout << "w[" << i << "] = " << w[i] << "-->";
    auto t = s[i] * (double)w[i];
    w[i] = (float)t;
    // std::cout << w[i] << std::endl << std::flush;
  }

  // Bias calculation [B' = B - m.G' = B - m.G/s = new Beta]
  // std::cout << "[Bias calculation] " << std::endl << std::flush;
  for (auto i = 0; i < co; i++) {
    // std::cout << "b[" << i << "] = " << b[i] << "-->";
    b[i] = (float)((double)b[i] - ((double)m[i] * (double)w[i]));
    // std::cout << b[i] << std::endl << std::flush;
  }

  // Set running variance and running mean
  // std::cout << "[RV, RM setting] " << std::endl << std::flush;
  for (auto i = 0; i < co; i++) {
    v[i] = 1.0;
    m[i] = 0;
    // std::cout << "rv[" << i << "] = " << v[i] << ", " << "rm[" << i << "] = "
    // << m[i] << std::endl << std::flush;
  }

  return true;
}

void* GetDataInHostBuffer(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx) {
  void* host_ptr{nullptr};

  if (idx != -1) {
    if (node->input(idx)->type() == NoneType::get()) {
      // std::cout << "[GetDataInHostBuffer] [" << idx << "] NoneType" <<
      // std::endl << std::flush;
      return host_ptr;
    }

    auto value = node->input(idx);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    // std::cout << "[GetDataInHostBuffer] idx := " << idx << std::endl <<
    // std::flush; std::cout << "[GetDataInHostBuffer] getValuePosInStack := "
    // << index << std::endl << std::flush;

    if ((index >= 0) && (index < (int32_t)stack.size())) {
      if (stack[index].isTensor()) {
        // std::cout << "[GetDataInHostBuffer] [" << idx << "] isTensor" <<
        // std::endl << std::flush;
        auto tensor = stack[index].toTensor();
        if (tensor.has_storage()) {
          // std::cout << "[GetDataInHostBuffer] [" << idx << "] has_storage" <<
          // std::endl << std::flush;
          auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
          host_ptr = impl->get_host_ptr();
          if (host_ptr == nullptr) {
            auto& device = synapse_helpers::HPURegistrar::get_device();
            auto device_id = device.id();
            auto size_in_bytes = habana_lazy::GetNBytes(tensor);
            // std::cout << "[GetDataInHostBuffer] [" << idx << "]
            // size_in_bytes: " << size_in_bytes << std::endl << std::flush;
            auto status = synHostMalloc(device_id, size_in_bytes, 0, &host_ptr);
            HABANA_ASSERT(status == synStatus::synSuccess);
            std::atomic<bool> copyDone{false};
            auto syn_error = device.copy_data_to_host(
                reinterpret_cast<synapse_helpers::device_ptr>(
                    tensor.data_ptr()),
                (void*)host_ptr,
                reinterpret_cast<synapse_helpers::device_ptr>(
                    tensor.storage().data_ptr().get()),
                size_in_bytes,
                [&copyDone]() { copyDone = true; },
                true);
            TORCH_CHECK(syn_error.status == 0, syn_error.error);
            // wait for copy completion
            while (!copyDone) {
              std::this_thread::yield();
            }
          }
        }
      }
    }
  } else {
    auto value = node->input(0);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    if (stack[index].isTensor()) {
      // std::cout << "[GetDataInHostBuffer] [" << idx << "] isTensor" <<
      // std::endl << std::flush;
      auto tensor = stack[index].toTensor();
      if (tensor.has_storage()) {
        // std::cout << "[GetDataInHostBuffer] [" << idx << "] has_storage" <<
        // std::endl << std::flush;
        auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
        host_ptr = impl->get_host_ptr();
        if (host_ptr == nullptr) {
          auto& device = synapse_helpers::HPURegistrar::get_device();
          auto device_id = device.id();
          auto size_in_bytes = habana_lazy::GetNBytes(tensor);
          // std::cout << "[GetDataInHostBuffer] [" << idx << "] size_in_bytes:
          // " << size_in_bytes << std::endl << std::flush;
          auto status = synHostMalloc(device_id, size_in_bytes, 0, &host_ptr);
          HABANA_ASSERT(status == synStatus::synSuccess);
          std::atomic<bool> copyDone{false};
          auto syn_error = device.copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
              (void*)host_ptr,
              reinterpret_cast<synapse_helpers::device_ptr>(
                  tensor.storage().data_ptr().get()),
              size_in_bytes,
              [&copyDone]() { copyDone = true; },
              true);
          TORCH_CHECK(syn_error.status == 0, syn_error.error);
          // wait for copy completion
          while (!copyDone) {
            std::this_thread::yield();
          }
        }
      }
    }
  }

  return host_ptr;
}

void UpdateDataInDeviceMem(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx,
    void* host_ptr) {
  at::Tensor tensor;
  if (idx != -1) {
    auto value = node->input(idx);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    tensor = stack[index].toTensor();
  } else {
    auto value = node->input(0);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    tensor = stack[index].toTensor();
  }

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto size_in_bytes = habana_lazy::GetNBytes(tensor);
  // std::cout << "[UpdateDataInDeviceMem] [" << idx << "] size_in_bytes: " <<
  // size_in_bytes << std::endl << std::flush;

  WithInsertPoint guard(node);
  std::atomic<bool> copyDone{false};
  auto syn_error = device.copy_data_to_device(
      (void*)host_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
      reinterpret_cast<synapse_helpers::device_ptr>(
          tensor.storage().data_ptr().get()),
      size_in_bytes,
      [&copyDone]() { copyDone = true; },
      false,
      true);
  TORCH_CHECK(syn_error.status == 0, syn_error.error);
  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }

  if (idx != -1) {
    auto value = node->input(idx);
    value->setDebugName(value->debugName() + "_fused_bn");
  }
}

/* Note:
   Ref:https://jira.habana-labs.com/browse/SW-116081 */

void RecalculateBatchnormParams(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack) {
  for (auto node : graph->nodes()) {
    auto node_name = node->kind().toQualString();
    PT_BRIDGE_DEBUG("Node Name: ", node_name);
    // std::cout << "[RecalculateBatchnormParams] Node Name: " << node_name <<
    // std::endl << std::flush;

    if (strcmp(node_name, "hpu::native_batch_norm_inf") == 0) {
      // std::cout << "[RecalculateBatchnormParams] [Apply] " << std::endl <<
      // std::flush;
      PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Apply]");

      auto bn = node;
      int idx_bias = 1;

      auto bn_b_hb_tensor = GetBackEndTensorImpl(graph, stack, bn, idx_bias);
      auto bn_b = GetDataInHostBuffer(graph, stack, bn, idx_bias);
      if (bn_b) {
        // std::cout << "[RecalculateBatchnormParams] [KM-bn_b-1] " << bn_b <<
        // std::endl << std::flush; std::cout << "[RecalculateBatchnormParams]
        // [KM-bn_b-2] " << ((float*)bn_b)[0] << ", " << ((float*)bn_b)[1] <<
        // std::endl << std::flush;
      }

      int idx_weight = 2;
      auto bn_w = GetDataInHostBuffer(graph, stack, bn, idx_weight);
      if (bn_w) {
        // std::cout << "[RecalculateBatchnormParams] [KM-bn_w-1] " << bn_w <<
        // std::endl << std::flush; std::cout << "[RecalculateBatchnormParams]
        // [KM-bn_w-2] " << ((float*)bn_w)[0] << ", " << ((float*)bn_w)[1] <<
        // std::endl << std::flush;
      }

      int idx_running_mean = 3;
      auto bn_rm = GetDataInHostBuffer(graph, stack, bn, idx_running_mean);
      if (bn_rm) {
        // std::cout << "[RecalculateBatchnormParams] [KM-bn_rm-1] " << bn_rm <<
        // std::endl << std::flush; std::cout << "[RecalculateBatchnormParams]
        // [KM-bn_rm-2] " << ((float*)bn_rm)[0] << ", " << ((float*)bn_rm)[1] <<
        // std::endl << std::flush;
      }

      int idx_running_var = 4;
      auto bn_rv = GetDataInHostBuffer(graph, stack, bn, idx_running_var);
      if (bn_rv) {
        // std::cout << "[RecalculateBatchnormParams] [KM-bn_rv-1] " << bn_rv <<
        // std::endl << std::flush; std::cout << "[RecalculateBatchnormParams]
        // [KM-bn_rv-2] " << ((float*)bn_rv)[0] << ", " << ((float*)bn_rv)[1] <<
        // std::endl << std::flush;
      }

      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();
      // std::cout << "[RecalculateBatchnormParams] [recompute batchnorm Params]
      // with bn_eps = " << bn_eps << std::endl << std::flush;
      auto status = recomputeBatchnormParams(
          bn_b_hb_tensor->GetTensorSize(),
          (float*)bn_rv,
          (float*)bn_rm,
          (float*)bn_w,
          (float*)bn_b,
          bn_eps);
      if (!status) {
        // std::cout << "[RecalculateBatchnormParams] [recompute batchnorm
        // Params ERROR!] " << std::endl << std::flush;
        PT_LAZY_DEBUG(
            "[RecalculateBatchnormParams] [recompute batchnorm Params ERROR!]");
        continue;
      }

      // std::cout << "[RecalculateBatchnormParams] [Update batchnorm Params] "
      // << std::endl << std::flush;
      PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Update batchnorm Params]");
      UpdateDataInDeviceMem(graph, stack, bn, idx_bias, bn_b);
      UpdateDataInDeviceMem(graph, stack, bn, idx_weight, bn_w);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_mean, bn_rm);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_var, bn_rv);
    }
  }

  // std::cout << "[RecalculateBatchnormParams] [Exit] " << std::endl <<
  // std::flush;
  PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Exit]");
  return;
}
}; // namespace habana_lazy
