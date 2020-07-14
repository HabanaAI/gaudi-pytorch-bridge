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

#include <sys/stat.h>
#include <sys/types.h>
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
#include <absl/types/variant.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_kernels/habana_operator.h"
#include "synapse_helpers/graph.h"

using namespace habana;

struct TensorInfo;

using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;

using IValPtr = torch::jit::IValue*;
using ValPtr = torch::jit::Value*;
using CValPtr = const torch::jit::Value*;
using IdxVec = std::vector<size_t>;

using PTToSynapseTensorMap =
    std::unordered_map<IValPtr, synapse_helpers::tensor&>;
using IValPtrToSynTensorNameMap = std::unordered_map<IValPtr, std::string>;
using IValPtrToSynTensorSizeMap = std::unordered_map<IValPtr, unsigned>;
using IValPtrToTesorInfoMap = std::unordered_map<IValPtr, TensorInfo>;

using tensor_or_ref = synapse_helpers::tensor_or_ref;
using SynTensorOrRefList = std::vector<tensor_or_ref>;
using SharedSynTensorOrRefListPtr = std::shared_ptr<SynTensorOrRefList>;

using IValPtrSharedToTesorInfoMap =
    std::unordered_map<IValPtrShared, TensorInfo>;

struct TensorInfo {
  TensorInfo(const IValPtrShared& ivp, const std::string& sn, const ValPtr& vp);
  TensorInfo(
      const at::Tensor& pt_tensor,
      const std::string& sn,
      const std::string& irn);

  friend std::ostream& operator<<(std::ostream& O, const TensorInfo& t);

  std::string ir_name;
  std::string syn_name;
  std::string shape_str;
  void* buffer{nullptr};
  unsigned numel{0};
  unsigned size{0};

  // Will hold the index of parent tensor info for aliases
  bool is_duplicate{false};
  size_t parent_index{ULONG_MAX};
};

// Adding the op strings to the key for recipe
// Later the drop the storage for the vector of strings
//   if possible pass the subgraph as argument
//   compute the hash directly from the subgraph within the constructor
struct RecipeArgumentSpec {
  RecipeArgumentSpec(
      bool with_grad,
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph);

  bool operator==(const RecipeArgumentSpec& arg) const {
    bool ret = (cas == arg.cas && opstrs == arg.opstrs);
    return ret;
  }

  size_t hashCode() const {
    return hash_code;
  }

  friend std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v);

 private:
  torch::jit::CompleteArgumentSpec cas;
  size_t hash_code;
  std::string opstrs;
};

// Hash functor for RecipeArgumentSpec
struct RecipeArgumentSpecHash {
 public:
  size_t operator()(const std::shared_ptr<RecipeArgumentSpec>& v) const {
    return v->hashCode();
  }
};

// Comparator for RecipeArgumentSpec
struct RecipeArgumentSpecEqual {
 public:
  bool operator()(
      const std::shared_ptr<RecipeArgumentSpec>& v1,
      const std::shared_ptr<RecipeArgumentSpec>& v2) const {
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
// The order of the outputs will match the order they appears within the
// subgraph
struct RecipeValueSpec {
  RecipeValueSpec(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> r = nullptr)
      : recipe(r),
        dtensorinfos(nullptr),
        aten_outputs(nullptr),
        htensor_wbuffers(nullptr) {
    count++;
    id = count;
  }

  void SelfCheck() {
    TORCH_CHECK(recipe != nullptr)
    TORCH_CHECK(dtensorinfos != nullptr);
    TORCH_CHECK(dtensorinfos->size() == num_tensors);
    TORCH_CHECK(!aten_outputs->empty());
  }

  void print_hbuff(
      size_t buf_idx,
      std::ofstream& out,
      size_t iteration_count,
      int numel = -1);
  void d2h_dbuff(size_t buf_idx);

  friend std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v);

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  std::shared_ptr<std::vector<TensorInfo>> dtensorinfos;
  std::shared_ptr<std::vector<IValPtrShared>> aten_outputs;
  std::vector<at::Tensor> aten_intermediates;
  std::shared_ptr<std::vector<uint64_t>> htensor_wbuffers;

  size_t id{0};
  size_t iter_idx{0};
  size_t num_tensors{0};

  size_t num_inputs{0};
  size_t num_duplicates{0};
  size_t num_interims{0};
  size_t num_outputs{0};

  static size_t count;
};

struct RecipeCacheSimple {
  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      RecipeValueSpec,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;

  bool empty() {
    return (map_.size() == 0);
  }

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  RecipeValueSpec& get(std::shared_ptr<RecipeArgumentSpec>& key) {
    return map_[key];
  }

  void add(std::shared_ptr<RecipeArgumentSpec>& key, RecipeValueSpec& val) {
    map_.emplace(key, val);
  }

  friend std::ostream& operator<<(std::ostream& O, const RecipeCacheSimple& v);
};

struct habanaTensorLayoutInfo
{
  habana::LayoutFormat layout;
  habana::LayoutFormat layout_at_graph_entry;
};

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(const torch::jit::Node* node, bool debug);
  ~HabanaLaunchOpPT();
  void evaluate(torch::jit::Stack& stack);
  void run(torch::jit::Stack& stack);

 private:
  static size_t instance_count_;

  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::string opname_;
  std::string id_str;
  size_t ref_count_ = 0;
  bool debug_;

  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // A map between the abstract value containers in graph and actual Ivalues in
  // stack
  std::unordered_map<CValPtr, habanaTensorLayoutInfo> value_to_tensor_layout;

  // A map for value to persistent flag
  std::unordered_map<CValPtr, bool> value_to_persistent_flag;

  // map between PT and synapse tensors
  std::vector<synapse_helpers::tensor> meta_syn_tensors;

  std::vector<IValPtrShared> pt_stack_sh;
  std::unordered_map<CValPtr, IValPtrShared> value_to_ivalue;
  std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>
      pt_to_synapse_tensors;


  // TIVs for launcing the recipe
  // tiv : absl::variant<TensorInfo, std::vector<TensorInfo>> objects
  std::unordered_map<IValPtrShared,
      absl::variant<TensorInfo, std::vector<TensorInfo>>> input_tiv_map;
  std::vector<absl::variant<TensorInfo, std::vector<TensorInfo>>> input_tivs;
  std::vector<absl::variant<TensorInfo, std::vector<TensorInfo>>> duplicate_tivs;

  // Temp additions to enable BatchNorm..tensors created that are not in graph
  // We get this to enable correct patching
  // Right now our patching is tightly coupled to graph nodes
  // BN is exception case, we can review our patching design for this
  std::vector<TensorInfo> interim_tensorinfos;

  std::vector<TensorInfo> output_tensorinfos;
  synapse_helpers::graph* syn_graph_ptr = nullptr;

  std::vector<at::Tensor> aten_intermediates;

  // caching :: begin

  size_t num_inputs = 0;
  // The inputs holding data usually are of type tensor and tensorList.
  // The following member keeps track of total number of tensor and tensorList inputs
  size_t num_tensor_inputs = 0;

  bool use_persistent_tensors;
  at::ArrayRef<torch::jit::IValue> input_refs;
  torch::jit::Stack* pt_stack = nullptr;

  RecipeCacheSimple recipe_cache;

  // caching :: end

  bool enable_caching_;
  int tensor_dump_numel_;
  bool enable_tensor_dump_;

  std::string tdmp_dir_name_;
  std::string tdmp_file_name_pre_;
  std::string tdmp_file_name_;
  size_t iteration_count_ = 0;

  habana::LayoutFormat getTensorChannelOrder(torch::jit::Value* val);
  void preProcessInputs();
  void processInputs(
      torch::jit::Node* node,
      const HabanaOperatorPtr& habana_kernel);
  void postProcessOutputs();
  at::Tensor permuteTensor(
      torch::jit::Value* value_in,
      const at::Tensor& input,
      habana::LayoutFormat permute_order);
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  void compile();
  void clear();
  bool isInGraphInputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Node* node, size_t index);
  std::vector<bool> nodeOutputPersistence(torch::jit::Node* node);
  void CompileAndExecuteHabanaFusedOpKernel();
  bool CompileSynapseGraph(
      std::shared_ptr<synapse_helpers::graph::recipe_handle>& synh_recipe);
  void GetSynapseInputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  void GetSynapseOutputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  bool isChannelOrderSupported(
      torch::jit::Value* val,
      const habana::LayoutFormat& supported_channel_order);
  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void handlePrimNodes(torch::jit::Node* node);
  void handleMetaOps(torch::jit::Node* node);

  void PrintATenTensors(RecipeValueSpec& rv);
  void LaunchRecipe(
      RecipeValueSpec& rv,
      at::ArrayRef<torch::jit::IValue> input_refs);
  void UpdateOutputs();
  void UpdateOutputs(RecipeValueSpec& rv);
  template <class T>
  void clearMember(T& m_container);

  bool IsCached(std::shared_ptr<RecipeArgumentSpec>& spec);

  void OrderInputs(RecipeValueSpec& rv);
  void FlattenAndLinkInputTIVs(RecipeValueSpec& rv);

  void ReorderInputs(RecipeValueSpec& rv);

  void DumpTensors_pre(RecipeValueSpec& rv);
  void DumpTensors(RecipeValueSpec& rv);
  void create_duplicate_syn_tensor(
      at::Tensor* tensor,
      torch::jit::Value* value_in,
      bool persistence = true);
};
