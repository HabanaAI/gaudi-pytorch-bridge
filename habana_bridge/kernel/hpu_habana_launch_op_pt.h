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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>

#include <ATen/Tensor.h>
#include <absl/hash/hash.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_kernels/habana_operator.h"

using namespace habana;

struct TensorInfo;

//For now its a simple map with PT tensor
//We can extend this structure later to map to add extra capabilities for debug etc.
typedef torch::jit::IValue*  IValPtr;
typedef torch::jit::Value*   ValPtr;
typedef std::vector<size_t>  IdxVec;

typedef std::unordered_map<IValPtr, synapse_helpers::tensor&> PTToSynapseTensorMap;
typedef std::unordered_map<IValPtr, std::string>              IValPtrToSynTensorNameMap;
typedef std::unordered_map<IValPtr, unsigned>                 IValPtrToSynTensorSizeMap;
typedef std::unordered_map<IValPtr, TensorInfo>               IValPtrToTesorInfoMap;

// Adding the op strings to the key for recipe
// Later the drop the storage for the vector of strings
//   if possible pass the subgraph as argument
//   compute the hash directly from the subgraph within the constructor
struct RecipeArgumentSpec {
  RecipeArgumentSpec(bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph> &irgraph);

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

struct TensorInfo {
  TensorInfo (const IValPtr &ivp, const std::string &sn, const ValPtr &vp);

  friend std::ostream &operator<< (std::ostream &O, const TensorInfo &t);

  std::string   ir_name;
  std::string   syn_name;
  std::string   shape_str;
  void         *buffer = nullptr;
  unsigned      numel = 0;
  unsigned      size = 0;
};

// Memory management is outside the scope of caching
// Input and output buffers need to be passed to the recipe
// The order of the inputs are according to the input stack
// The order of the outputs will match the order they appears within the subgraph
struct RecipeValueSpec {
  RecipeValueSpec(std::shared_ptr<synapse_helpers::graph::recipe_handle> r = nullptr)
  : recipe(r),
    dtensorinfos(nullptr),
    aten_inputs(nullptr),
    aten_outputs(nullptr),
    pinput_indices(nullptr),
    htensor_wbuffers(nullptr)
  {
    count++;
    id = count;
  }

  void SelfCheck() {
    TORCH_CHECK(recipe != nullptr)
    TORCH_CHECK(dtensorinfos != nullptr);
    TORCH_CHECK(dtensorinfos->size() == num_tensors);
    TORCH_CHECK(!aten_inputs->empty());
    TORCH_CHECK(!aten_outputs->empty());
  }

  void print_hbuff(size_t buf_idx, std::ofstream &out, size_t iteration_count, int numel = -1);
  void d2h_dbuff(size_t buf_idx);

  friend std::ostream &operator<< (std::ostream &O, const RecipeValueSpec &v);

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;

  std::shared_ptr<std::vector<TensorInfo>>               dtensorinfos;

  std::shared_ptr<std::vector<IValPtr>>                  aten_inputs;
  std::shared_ptr<std::vector<IValPtr>>                  aten_outputs;

  std::shared_ptr<std::vector<std::vector<size_t>>>      pinput_indices;

  std::shared_ptr<std::vector<uint64_t>>                 htensor_wbuffers;

  size_t id {0};
  size_t iter_idx {0};
  size_t num_tensors {0};
  size_t num_inputs {0};

  static size_t count;
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
  ~HabanaLaunchOpPT();
  void evaluate(torch::jit::Stack& stack);
  void run(torch::jit::Stack& stack);

 private:
  static size_t                       instance_count_;

  std::shared_ptr<torch::jit::Graph>  subgraph_;
  std::string                         opname_;
  std::string                         id_str;
  size_t                              ref_count_ = 0;
  bool                                debug_;

  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;
  // A map between the abstract value containers in graph and actual Ivalues in stack
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue *> value_to_ivalue;
  std::unordered_map<const torch::jit::Value*, habana::LayoutFormat> value_to_tensor_layout;
  //map between PT and synapse tensors
  PTToSynapseTensorMap pt_to_synapse_tensors;
  std::vector<synapse_helpers::tensor> meta_syn_tensors;

  // TensorInfos for launcing the recipe
  std::vector<TensorInfo>          input_tensorinfos;
  std::vector<TensorInfo>          pinput_tensorinfos;
  std::vector<TensorInfo>          output_tensorinfos;
  synapse_helpers::graph          *syn_graph_ptr = nullptr;

  // caching :: begin

  // TODO :
  // 1. Manage the newly created IValues
  // 2. Expose the enable_caching_ flag to python
  // 3. Switch to general logging from std::cout

  size_t                           num_inputs = 0;
  size_t                           num_tensor_inputs = 0;
  at::ArrayRef<torch::jit::IValue> input_refs;
  torch::jit::Stack               *pt_stack = nullptr;

  IValPtrToTesorInfoMap            input_tensorinfo_map;

  RecipeCacheSimple                recipe_cache;

  std::unordered_map<IValPtr, IdxVec> input_to_pinput_indices;

  // caching :: end

  bool                             enable_caching_ = getenv("HABANA_PGM_ENABLE_CACHE") ? true : false;
  int                              tensor_dump_numel_;
  bool                             enable_tensor_dump_;

  std::string                      tdmp_dir_name_;
  std::string                      tdmp_file_name_pre_;
  std::string                      tdmp_file_name_;
  size_t                           iteration_count_ = 0;

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
  void handleMetaOps(torch::jit::Node* node);

  void LaunchRecipe(RecipeValueSpec &rv);
  void UpdateOutputs();
  template <class T>
  void clearMember(T& m_container);

  bool IsCached(std::shared_ptr<RecipeArgumentSpec> &spec);
  void ReorderInputs(RecipeValueSpec &rv);

  void DumpTensors_pre(RecipeValueSpec &rv);
  void DumpTensors(RecipeValueSpec &rv);
};
