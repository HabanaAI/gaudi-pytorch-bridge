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

// Adding the op strings to the key for recipe
// Later the drop the storage for the vector of strings
//   if possible pass the subgraph as argument
//   compute the hash directly from the subgraph within the constructor
struct RecipeArgumentSpec {
  RecipeArgumentSpec(bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::shared_ptr<torch::jit::Graph> irgraph)
  : cas(with_grad, input_refs), hash_code(cas.hashCode()), opstrs(std::string()){
    std::hash<std::string> str_hash;
    for (auto * node : irgraph->nodes()) {
      std::string s(node->kind().toQualString());
      // Adding delemeters for better readability
      opstrs.append("<" + s + ">");
    }
    hash_code = torch::hash_combine(hash_code, str_hash(opstrs));
  }

  bool operator==(const RecipeArgumentSpec& arg) const {
    bool ret = (cas == arg.cas && opstrs == arg.opstrs);
    return ret;
  }

  size_t hashCode() const {
    return hash_code;
  }

  friend std::ostream &operator<< (std::ostream &O, const RecipeArgumentSpec &v);

 private:
  torch::jit::CompleteArgumentSpec cas;
  size_t hash_code;
  std::string opstrs;
};

// Hash functor for RecipeArgumentSpec
struct RecipeArgumentSpecHash {
public:
  size_t operator()(const std::shared_ptr<RecipeArgumentSpec> & v) const {
    return v->hashCode();
  }
};

// Comparator for RecipeArgumentSpec
struct RecipeArgumentSpecEqual {
public:
  bool operator()(
    const std::shared_ptr<RecipeArgumentSpec> & v1,
    const std::shared_ptr<RecipeArgumentSpec> & v2) const {
    if (nullptr == v1 && nullptr == v2)
      return true;
    if (nullptr == v1)
      return false;
    if (nullptr == v2)
      return false;

    return (*v1) == (*v2);
  }
};

// Memory management is outside the scope of caching
// Input and output buffers need to be passed to the recipe
// The order of the inputs are according to the input stack
// The order of the outputs will match the order they appears within the subgraph
struct RecipeValueSpec {
  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  std::shared_ptr<std::vector<std::string>> syn_tensor_names;
  std::shared_ptr<std::vector<void *>> syn_tensor_buffers;
  // Is it possible to make a unique_ptr for aten_outputs
  std::shared_ptr<std::vector<torch::jit::IValue *>> aten_outputs;

  RecipeValueSpec(std::shared_ptr<synapse_helpers::graph::recipe_handle> r = nullptr)
  : recipe(r), syn_tensor_names(nullptr), syn_tensor_buffers(nullptr), aten_outputs(nullptr) {}

  void SelfCheck() {
    TORCH_CHECK(recipe != nullptr)
    TORCH_CHECK(syn_tensor_names != nullptr);
    TORCH_CHECK(syn_tensor_buffers != nullptr);
    TORCH_CHECK(syn_tensor_names->size() == syn_tensor_buffers->size());
    TORCH_CHECK(!aten_outputs->empty());
  }

  friend std::ostream &operator<< (std::ostream &O, const RecipeValueSpec &v);
};

struct RecipeCacheSimple {
  std::unordered_map<
    std::shared_ptr<RecipeArgumentSpec>,
    RecipeValueSpec,
    RecipeArgumentSpecHash,
    RecipeArgumentSpecEqual> map_;

  bool empty() { return (map_.size() == 0); }

  bool exists(std::shared_ptr<RecipeArgumentSpec> &key) {
    bool ret_flag { false };
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  RecipeValueSpec& get(std::shared_ptr<RecipeArgumentSpec> &key) {
    return map_[key];
  }

  void add(std::shared_ptr<RecipeArgumentSpec> &key, RecipeValueSpec &val) {
    map_.emplace(key, val);
  }

  friend std::ostream &operator<< (std::ostream &O, const RecipeCacheSimple &v);
};

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(const torch::jit::Node* node, bool debug);
  void evaluate(torch::jit::Stack& stack);
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::string opname_;
  bool debug_;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<void*> input_buffers;
  std::vector<void*> output_buffers;
  // We keep a vector of kernels so that the context memory
  // for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // A map between the abstract value containers in graph and actual Ivalues in
  // stack
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue *> value_to_ivalue;
  std::unordered_map<const torch::jit::Value*, habana::LayoutFormat> value_to_tensor_layout;
  //map between PT and synapse tensors
  PTToSynapseTensorMap pt_to_synapse_tensors;
  std::vector<synapse_helpers::tensor> meta_syn_tensors;

  // caching :: begin

  // TODO :
  // 1. Manage the newly created IValues
  // 2. Expose the enable_caching flag to python
  // 3. Switch to general logging from std::cout

  bool enable_caching = true;
  size_t num_inputs = 0;
  torch::jit::Stack *pt_stack = nullptr;

  at::ArrayRef<torch::jit::IValue> input_refs;
  size_t graph_id = 0;
  synapse_helpers::graph *syn_graph_ptr = nullptr;
  RecipeCacheSimple recipe_cache;

  // caching :: end

  habana::LayoutFormat getTensorChannelOrder(torch::jit::Value* val);
  void preProcessInputs();
  void processInputs(
      torch::jit::Node* node,
      const HabanaOperatorPtr &habana_kernel);
  void postProcessOutputs();
  at::Tensor permuteTensor(
      torch::jit::Value* value_in,
      const at::Tensor &input,
      habana::LayoutFormat permute_order);
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  void compile();
  void clear();
  bool isInGraphInputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Value* value);
  void CompileAndExecuteHabanaFusedOpKernel();
  bool CompileSynapseGraph(std::shared_ptr<synapse_helpers::graph::recipe_handle>& synh_recipe);
  void GetSynapseInputs(
      const HabanaOperatorPtr &habana_op,
      torch::jit::Node* node);
  void GetSynapseOutputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node);
  bool isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat &supported_channel_order);
  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void handlePrimNodes(torch::jit::Node* node);
  void LaunchRecipe(RecipeValueSpec &rv);
  bool IsCached(std::shared_ptr<RecipeArgumentSpec> &spec);
  void handleMetaOps(torch::jit::Node* node);
  void UpdateOutputs();
};
