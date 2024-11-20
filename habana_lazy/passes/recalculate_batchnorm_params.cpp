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

#include "recalculate_batchnorm_params.h"
#include <torch/script.h>
#include <cmath>
#include <iterator>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/get_n_bytes.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pass_utils.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace {
size_t getValuePosInStack(
    std::shared_ptr<Graph>& graph,
    const torch::jit::Value* value) {
  auto graph_ins = graph->inputs();
  size_t idx = 0;
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return idx;
    }
    idx++;
  }
  return -1;
}
} // namespace

namespace habana_lazy {

::std::tuple<habana::TensorExtraMeta*, habana::StorageExtraMeta*>
GetBackEndTensorMeta(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    torch::jit::Node* node,
    const int idx) {
  habana::TensorExtraMeta* tmeta_ptr{nullptr};
  habana::StorageExtraMeta* smeta_ptr{nullptr};

  if (idx != -1) {
    if (node->input(idx)->type() == torch::jit::NoneType::get()) {
      return std::tie(tmeta_ptr, smeta_ptr);
    }

    auto value = node->input(idx);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    if ((index >= 0) && (index < (int32_t)stack.size())) {
      if (stack[index].isTensor()) {
        auto tensor = stack[index].toTensor();
        if (tensor.has_storage()) {
          tmeta_ptr = habana::get_tensor_extra_meta(tensor);
          smeta_ptr = habana::get_storage_extra_meta(tensor);
          tmeta_ptr->set_tensor_size(tensor.sizes());
        }
      }
    }
  } else {
    auto value = node->input(0);
    int32_t index = (int32_t)getValuePosInStack(graph, value);
    if (stack[index].isTensor()) {
      auto tensor = stack[index].toTensor();
      if (tensor.has_storage()) {
        tmeta_ptr = habana::get_tensor_extra_meta(tensor);
        smeta_ptr = habana::get_storage_extra_meta(tensor);
        tmeta_ptr->set_tensor_size(tensor.sizes());
      }
    }
  }

  return std::tie(tmeta_ptr, smeta_ptr);
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

  auto& device = habana::HPUDeviceContext::get_device();
  auto device_id = device.id();
  auto bytes = co * sizeof(float);

  void* host_ptr{nullptr};
  auto status = synHostMalloc(device_id, bytes * 2, 0, &host_ptr);
  // std::cout << "[synHostMalloc - s] " << bytes * 2 << std::endl <<
  // std::flush; std::cout << "[synHostMalloc - s] " << status << std::endl <<
  // std::flush;
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
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
    torch::jit::Node* node,
    const int idx) {
  void* host_ptr{nullptr};

  if (idx != -1) {
    if (node->input(idx)->type() == torch::jit::NoneType::get()) {
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
          auto tmeta{habana::get_tensor_extra_meta(tensor)};
          host_ptr = tmeta->get_host_ptr();
          if (host_ptr == nullptr) {
            auto& device = habana::HPUDeviceContext::get_device();
            auto device_id = device.id();
            auto size_in_bytes = habana_helpers::GetNBytes(tensor);
            // std::cout << "[GetDataInHostBuffer] [" << idx << "]
            // size_in_bytes: " << size_in_bytes << std::endl << std::flush;
            auto status = synHostMalloc(device_id, size_in_bytes, 0, &host_ptr);
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            std::atomic<bool> copyDone{false};
            habana::HPUDeviceContext::copy_data_to_host(
                reinterpret_cast<synapse_helpers::device_ptr>(
                    tensor.data_ptr()),
                (void*)host_ptr,
                reinterpret_cast<synapse_helpers::device_ptr>(
                    tensor.storage().data_ptr().get()),
                size_in_bytes,
                [&copyDone]() { copyDone = true; },
                true);
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
        auto tmeta{habana::get_tensor_extra_meta(tensor)};
        host_ptr = tmeta->get_host_ptr();
        if (host_ptr == nullptr) {
          auto& device = habana::HPUDeviceContext::get_device();
          auto device_id = device.id();
          auto size_in_bytes = habana_helpers::GetNBytes(tensor);
          // std::cout << "[GetDataInHostBuffer] [" << idx << "] size_in_bytes:
          // " << size_in_bytes << std::endl << std::flush;
          auto status = synHostMalloc(device_id, size_in_bytes, 0, &host_ptr);
          HABANA_ASSERT(
              status == synStatus::synSuccess, Logger::synStatusToStr(status));
          std::atomic<bool> copyDone{false};
          habana::HPUDeviceContext::copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
              (void*)host_ptr,
              reinterpret_cast<synapse_helpers::device_ptr>(
                  tensor.storage().data_ptr().get()),
              size_in_bytes,
              [&copyDone]() { copyDone = true; },
              true);
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
    torch::jit::Node* node,
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

  auto size_in_bytes = habana_helpers::GetNBytes(tensor);
  // std::cout << "[UpdateDataInDeviceMem] [" << idx << "] size_in_bytes: " <<
  // size_in_bytes << std::endl << std::flush;

  torch::jit::WithInsertPoint guard(node);
  std::atomic<bool> copyDone{false};
  habana::HPUDeviceContext::copy_data_to_device(
      (void*)host_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
      reinterpret_cast<synapse_helpers::device_ptr>(
          tensor.storage().data_ptr().get()),
      size_in_bytes,
      [&copyDone]() { copyDone = true; },
      false,
      true);
  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
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

    if (strcmp(node_name, "hpu::native_batch_norm_inf") == 0) {
      PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Apply]");

      auto bn = node;
      int idx_bias = 1;

      habana::TensorExtraMeta* bn_b_tmeta_ptr{nullptr};
      habana::StorageExtraMeta* bn_b_smeta_ptr{nullptr};
      std::tie(bn_b_tmeta_ptr, bn_b_smeta_ptr) =
          GetBackEndTensorMeta(graph, stack, bn, idx_bias);
      auto bn_b = GetDataInHostBuffer(graph, stack, bn, idx_bias);
      if (!bn_b_tmeta_ptr || !bn_b) {
        continue;
      }

      int idx_weight = 2;
      auto bn_w = GetDataInHostBuffer(graph, stack, bn, idx_weight);
      if (!bn_w) {
        continue;
      }

      int idx_running_mean = 3;
      auto bn_rm = GetDataInHostBuffer(graph, stack, bn, idx_running_mean);
      if (!bn_rm) {
        continue;
      }

      int idx_running_var = 4;
      auto bn_rv = GetDataInHostBuffer(graph, stack, bn, idx_running_var);
      if (!bn_rv) {
        continue;
      }

      auto bn_eps =
          torch::jit::constant_as<double>(bn->namedInput("eps")).value();
      PT_LAZY_DEBUG(
          "[RecalculateBatchnormParams] [recompute batchnorm Params] bn_eps = ",
          bn_eps);

      auto status = recomputeBatchnormParams(
          bn_b_tmeta_ptr->get_tensor_size(),
          (float*)bn_rv,
          (float*)bn_rm,
          (float*)bn_w,
          (float*)bn_b,
          bn_eps);

      if (!status) {
        PT_LAZY_DEBUG(
            "[RecalculateBatchnormParams] [recompute batchnorm Params ERROR!]");
        continue;
      }

      PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Update batchnorm Params]");
      UpdateDataInDeviceMem(graph, stack, bn, idx_bias, bn_b);
      UpdateDataInDeviceMem(graph, stack, bn, idx_weight, bn_w);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_mean, bn_rm);
      UpdateDataInDeviceMem(graph, stack, bn, idx_running_var, bn_rv);
    }
  }

  PT_LAZY_DEBUG("[RecalculateBatchnormParams] [Exit]");
  return;
}
}; // namespace habana_lazy
