/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once

#include <ATen/Tensor.h>
#include <absl/hash/hash.h>
#include <stdlib.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/ir/ir.h>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include "habana_kernels/habana_operator.h"
using namespace habana;

//For now its a simple map with PT tensor
//We can extend this structure later to map to add extra capabilities for debug etc.
typedef std::unordered_map<torch::jit::IValue*, synapse_helpers::tensor&>
    PTToSynapseTensorMap;

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(const torch::jit::Node* node, bool debug);
  void evaluate(torch::jit::Stack& stack);
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::string opname;
  bool debug_;
  at::ArrayRef<torch::jit::IValue> inputs;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<void*> input_buffers;
  std::vector<void*> output_buffers;
  //We keep a vector of kernels so that the context memory for each kernel is retained till graph execution
  //This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;
  // A map between the abstract value containers in graph and actual Ivalues in
  // stack
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue>
      value_to_ivalue;
  std::unordered_map<const torch::jit::Value*, habana::LayoutFormat>
      value_to_tensor_layout;
  //map between PT and synapse tensors
  PTToSynapseTensorMap pt_to_synapse_tensors;

  habana::LayoutFormat getTensorChannelOrder(torch::jit::Value* val);
  void preProcessInputs();
  void processInputs(
      synapse_helpers::graph& syn_graph,
      torch::jit::Node* node,
      const HabanaOperatorPtr &habana_kernel);
  void postProcessOutputs(synapse_helpers::graph& syn_graph);
  at::Tensor permuteTensor(
      synapse_helpers::graph& syn_graph,
      torch::jit::Value* value_in,
      const at::Tensor &input,
      habana::LayoutFormat permute_order);
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  void compile();
  void clear();
  bool isInGraphInputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Value* value);
  void CreateHabanaFusedOpKernel();
  bool CompileSynapseGraph(
      synapse_helpers::graph& synGraph,
      std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipeId);
  void GetSynapseInputs(
      const HabanaOperatorPtr &habana_op,
      synapse_helpers::graph& graph,
      torch::jit::Node* node);
  void GetSynapseOutputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node);
  bool isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat &supported_channel_order);
  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void handlePrimNodes(torch::jit::Node* node);
};
