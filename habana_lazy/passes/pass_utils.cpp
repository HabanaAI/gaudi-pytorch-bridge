/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "pass_utils.h"
using namespace torch::jit;
namespace habana_lazy {

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {0, 2, 3, 1};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {2, 3, 1, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {0, 3, 1, 2};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {1, 2, 3, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {3, 2, 0, 1};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {3, 0, 1, 2};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0,
        " InsertPermtue_graph: permute called for unsupported channel order");
  }

  return dims;
}

bool isRetunrOut(
    std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Value* value_out) {
  auto node_return = graph->return_node();
  for (auto value : node_return->inputs()) {
    if (value_out == value)
      return true;
  }
  return false;
}

bool is_4d_5d_value(const torch::jit::Value* value_in) {
  return (*value_in->type()->cast<TensorType>()->dim() == 4 ||
          *value_in->type()->cast<TensorType>()->dim() == 5)
      ? true
      : false;
}

bool WeightIdentificationPass::isTensor(const torch::jit::Value* value) {
  HABANA_ASSERT(value->node());
  return !(value->node()->kind() == c10::prim::Constant);
}

void WeightIdentificationPass::markInputs(const torch::jit::Value* in) {
  auto node = in->node();
  if (nullptr == node) {
    return;
  }
  std::string node_str = node->kind().toQualString();
  if (0 == kernelWeightIdx.count(node_str) &&
      !(strcmp(node->kind().toQualString(), "prim::ListConstruct") == 0) &&
      !(strcmp(node->kind().toQualString(), "prim::ListUnpack") == 0) &&
      !StridedKernels.count(node_str)) {
    for (auto& i : node->inputs()) {
      if (isTensor(i) && !weightTensors.count(i)) {
        weightTensors.insert(i);
        markInputs(i);
      }
    }
  }
}

void WeightIdentificationPass::markOutputs(const torch::jit::Value* in) {
  for (auto& use : in->uses()) {
    auto node = use.user;
    HABANA_ASSERT(node);

    // TODO: check if its binary/unary ops & then mark
    std::string node_str = node->kind().toQualString();
    if (0 == kernelWeightIdx.count(node_str)) {
      for (auto& out : node->outputs()) {
        if (!weightTensors.count(out)) {
          weightTensors.insert(out);
          markOutputs(out);
        }
      }
    }

    // markOutput of Outvariant kernels
    if (is_mark_out_varients) {
      auto it = kernelOutVariantIdx.find(node_str);
      if (kernelOutVariantIdx.end() != it) {
        auto outIdx = it->second;
        HABANA_ASSERT(outIdx < node->inputs().size());
        auto weightOut = node->inputs()[outIdx];
        if (!weightTensors.count(weightOut)) {
          weightTensors.insert(weightOut);
          markInputs(weightOut);
          markOutputs(in);
        }
      }
    }
  }
}

void WeightIdentificationPass::markWeights(const torch::jit::Value* in) {
  markInputs(in);
  markInOutputs(in);
}

void WeightIdentificationPass::markInOutputs(const torch::jit::Value* in) {
  for (auto& use : in->uses()) {
    auto node = use.user;
    HABANA_ASSERT(node);

    // TODO: check if its binary/unary ops & then mark
    std::string node_str = node->kind().toQualString();
    if (StridedKernels.count(node_str))
      continue;
    if (0 == kernelWeightIdx.count(node_str) &&
        !(strcmp(node->kind().toQualString(), "hpu::control_edge_other_") ==
          0) &&
        !(strcmp(node->kind().toQualString(), "prim::ListConstruct") == 0) &&
        !(strcmp(node->kind().toQualString(), "prim::ListUnpack") == 0)) {
      for (auto& in1 : node->inputs()) {
        if (isTensor(in1) && !weightTensors.count(in1)) {
          if (strcmp(node->kind().toQualString(), "prim::Return") != 0) {
            weightTensors.insert(in1);
            markInputs(in1);
          }
        }
      }

      for (auto& out : node->outputs()) {
        if (isTensor(out) && !weightTensors.count(out)) {
          weightTensors.insert(out);
          markInOutputs(out);
        }
      }
    }
  }
}

void WeightIdentificationPass::markWeightInTensors(
    std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto node : graph->nodes()) {
    std::string kernel = node->kind().toQualString();
    auto it = kernelWeightIdx.find(kernel);
    // mark kernel weight-inputs
    if (kernelWeightIdx.end() != it) {
      auto weightIdx = it->second;
      HABANA_ASSERT(weightIdx < node->inputs().size());
      auto weightIn = node->inputs()[weightIdx];
      weightTensors.insert(weightIn);
      markInputs(weightIn);
    }
  }
}

void WeightIdentificationPass::markWeightTensors(
    std::shared_ptr<Graph>& graph,
    bool mark_out_varients) {
  is_mark_out_varients = mark_out_varients;
  for (auto node : graph->nodes()) {
    std::string kernel = node->kind().toQualString();
    auto it = kernelWeightIdx.find(kernel);
    // mark kernel weight-inputs
    if (kernelWeightIdx.end() != it) {
      auto weightIdx = it->second;
      HABANA_ASSERT(weightIdx < node->inputs().size());
      auto weightIn = node->inputs()[weightIdx];
      weightTensors.insert(weightIn);
      markWeights(weightIn);
    }

    // mark kernel weight-outputs
    it = kernelOutWeightIdx.find(kernel);
    if (kernelOutWeightIdx.end() != it) {
      auto weightIdx = it->second;
      HABANA_ASSERT(weightIdx < node->outputs().size());
      auto weightOut = node->outputs()[weightIdx];
      weightTensors.insert(weightOut);
      markInOutputs(weightOut);
    }
  }
}
} // namespace habana_lazy
