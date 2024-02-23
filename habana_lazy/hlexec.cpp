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
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/peephole.h>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/program/create_executor.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/lazy_arg_spec.h"
#include "ops/constant.h"
#include "ops/convolution.h"
#include "passes/fold_conv_batchnorm.h"
#include "passes/fuse_bn_relu_residual_add.h"
#include "passes/fuse_mm_transpose.h"
#include "passes/recalculate_batchnorm_params.h"
#include "passes/replace_inplace_ops.h"
#include "passes/replace_views_with_reshapes.h"
#include "passes/transform_graph.h"
#include "pytorch_helpers/visualize/visualize.h"

namespace habana_lazy {
namespace exec {

namespace {

/*
 * Launcher represents underlying execution mechanism used by HlExec class.
 */
struct Launcher {
  virtual ~Launcher() = default;

  virtual void Run(torch::jit::Stack& stack, bool dry_run = false) = 0;
};

/*
 * Launcher for HabanaLaunchOpPT
 */
struct HabanaLaunchOpLauncher : Launcher {
  HabanaLaunchOpLauncher(
      const std::shared_ptr<habana::OptimizedJITGraphAndMetaData>& graph_meta,
      const std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend>& info)
      : habana_launch_op_(graph_meta) {
    if (info) {
      habana_launch_op_.set_lazy_front_end_info(info);
    }
  }

  void Run(torch::jit::Stack& stack, bool dry_run) override {
    return habana_launch_op_.run(stack, {}, {}, dry_run);
  }

  habana::HabanaLaunchOpPT habana_launch_op_;
};

/*
 * Launcher for clustered programs
 */
struct ClusteredProgramLauncher : Launcher {
  ClusteredProgramLauncher(
      std::size_t graph_hash,
      const std::shared_ptr<habana::OptimizedJITGraphAndMetaData>& graph_meta) {
    executor_ = habana::program::CreateExecutor(graph_hash, graph_meta);
    TORCH_CHECK(executor_ != nullptr);
  }

  void Run(torch::jit::Stack& stack, bool /*dry_run*/) override {
    executor_->Run(stack);
  }

  std::unique_ptr<habana::program::Executor> executor_;
};

/*
 * Creates launcher.
 *
 * When PT_HPU_CLUSTERED_PROGRAM is set then launcher for clustered programs
 * is created, ohterwise launcher for HabanaLaunchOpPT is used.
 */
std::unique_ptr<Launcher> CreateLauncher(
    std::size_t graph_hash,
    const std::shared_ptr<habana::OptimizedJITGraphAndMetaData>& graph_meta,
    const std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend>& info) {
  static bool is_clustered_program_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_CLUSTERED_PROGRAM);

  if (is_clustered_program_enabled) {
    return std::make_unique<ClusteredProgramLauncher>(graph_hash, graph_meta);
  }
  return std::make_unique<HabanaLaunchOpLauncher>(graph_meta, info);
}

} // namespace

OptPassCfg* OptPassCfg::p_instance_ = nullptr;

std::unordered_map<size_t, size_t> HlExec::s_graphIndexMap;
size_t HlExec::s_graphIndex;

HlExec::HlExec() {
  mp_g_ = std::make_shared<Graph>();
  m_g_hash_ = 0;
}

HlExec::HlExec(ScopePtr scope) {
  mp_g_ = std::make_shared<Graph>(scope);
  m_g_hash_ = 0;
}

void HlExec::Launch(
    torch::jit::Stack& stack,
    const c10::hpu::HPUStream& stream,
    bool dry_run) {
  PT_LAZY_TRACE;
  auto& device = habana::HPURegistrar::get_device();
  auto context = get_device_lazy_execution_context(device.id());
  // TODO : remove this env variable use
  // This is temporarily done to deactivate code in synapse helpers for lazy
  // mode kernel registration We will move to using shape utilities instead and
  // not do env variable based check anymore
  // We have short-circuited certain utilities in synapse helpers, we need to
  // remove that code

  if (context->getCapturing()) {
    // save the graph for perf mode
    context->saveGraph(mp_g_);

    // save the hash for perf mode
    context->saveHash(m_g_hash_);

    // save the graph key for perf mode
    context->saveGraphKey(mp_g_and_meta_data_->get_cached_graph_key());

    // save the graph key for perf mode
    context->saveOpStrs(mp_g_and_meta_data_->get_cached_opstrs());
  }

  std::string opName = getHabanaLazyGraphName();
  if (lazyInfo) {
    opName = lazyInfo->get_lazy_op_name();
  }

  auto graphIndex =
      GetGraphIndex(m_g_hash_, torch::jit::last(stack, mp_g_->inputs().size()));
  bool isDynamic = habana_helpers::GetRefineDynamicShapeStatus();
  mp_g_and_meta_data_->SetGraphIndex(graphIndex);
  mp_g_and_meta_data_->SetOpName(opName);
  mp_g_and_meta_data_->SetHPUStream(stream);
  mp_g_and_meta_data_->SetDynamicGraph(isDynamic);

  auto launcher = CreateLauncher(m_g_hash_, mp_g_and_meta_data_, lazyInfo);
  try {
    launcher->Run(stack, dry_run);
  } catch (const std::exception& e) {
    PT_BRIDGE_DEBUG("HabanaLaunchOpPT Run returned exception....\n", e.what());
    get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLAZY);
    throw;
  }
}

/*
 * Find duplicate in stack
 */
void HlExec::FindDuplicateInStack(
    const ir::PostOrderData& po_data,
    torch::jit::Stack& stack,
    std::vector<size_t>& parent_vec,
    std::vector<bool>& is_duplicate_vec) {
  size_t num_inputs = po_data.inputs.size();

  // Assumption : stack[i] is the corresponding input of po_data.inputs[i]
  TORCH_CHECK(
      stack.size() == num_inputs,
      " stack_size ",
      stack.size(),
      " != num_inputs ",
      num_inputs);

  std::unordered_map<uint64_t, size_t> input_addr_map;
  size_t num_duplicate_inputs = 0;
  size_t stack_size = stack.size();

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    TORCH_CHECK(input.isTensor());
    if (!input.toTensor().has_storage()) {
      return;
    }
  }

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    TORCH_CHECK(input.isTensor());
    auto input_addr = (uint64_t)(input.toTensor().data_ptr());

    // input_addr == 0 not considered for duplicate removal since this address
    // is used for ZST tensors. 2 different ZST tensors can both have addr = 0
    // and removing one of them results in cycles in synapse graph in some cases
    if (input_addr_map.count(input_addr) != 0 && input_addr != 0) {
      auto pidx = input_addr_map.at(input_addr);
      auto parent_tensor = stack[pidx].toTensor();
      auto input_tensor = input.toTensor();
      // Check for shape and stride match
      if (input_tensor.sizes() == parent_tensor.sizes() &&
          input_tensor.strides() == parent_tensor.strides()) {
        is_duplicate_vec[i] = true;
        parent_vec[i] = pidx;
        num_duplicate_inputs++;

        PT_LAZY_DEBUG(
            "Duplicate input address ",
            input_addr,
            " found for value %",
            po_data.inputs[i].ToString(),
            " current duplicate count ",
            num_duplicate_inputs);
      } else {
        PT_LAZY_DEBUG(
            "Same input address ",
            input_addr,
            " with different shape/stride found for value %",
            po_data.inputs[i].ToString(),
            " and value%",
            po_data.inputs[pidx].ToString());
      }
    } else {
      input_addr_map[input_addr] = i;
    }
  }
}

/*
 * Prune duplicate stack inputs
 */
void HlExec::PruneDuplicateStackInputs(
    torch::jit::Stack& stack,
    std::vector<bool>& is_duplicate_vec) {
  for (int64_t j = (int64_t)is_duplicate_vec.size() - 1; j >= 0; j--) {
    if (is_duplicate_vec[j]) {
      PT_LAZY_DEBUG("Deleting ", j, "th entry from the stack");
      stack.erase(stack.begin() + j);
    }
  }
}

/*
 * Prune duplicate graph inputs
 */
void HlExec::PruneDuplicateGraphInputs(
    std::vector<size_t>& parent_vec,
    std::vector<bool>& is_duplicate_vec) {
  PT_LAZY_TRACE;

  auto jit_ir_graph_inputs = mp_g_->inputs();
  bool is_pruned{false};
  for (size_t i = 0; i < jit_ir_graph_inputs.size(); i++) {
    if (is_duplicate_vec[i]) {
      size_t parent_idx = parent_vec[i];
      TORCH_CHECK(
          parent_idx != ULONG_MAX && parent_idx < i,
          " invalid parent index ",
          parent_idx,
          " found for input index ",
          i);
      auto vptr = jit_ir_graph_inputs[parent_idx];
      PT_LAZY_DEBUG(
          "Replacing %",
          jit_ir_graph_inputs[i]->debugName(),
          " with %",
          vptr->debugName());
      jit_ir_graph_inputs[i]->replaceAllUsesWith(vptr);
    }
  }

  for (int64_t j = (int64_t)is_duplicate_vec.size() - 1; j >= 0; j--) {
    if (is_duplicate_vec[j]) {
      is_pruned = true;
      PT_LAZY_DEBUG(
          "Deleting ",
          j,
          "th input %",
          mp_g_->inputs().at(j)->debugName(),
          "of the graph");
      mp_g_->eraseInput(j);
    }
  }

  if (is_pruned) {
    PT_LAZY_DEBUG(
        "After pruning duplicates, JIT IR Graph ====\n",
        mp_g_->toString(),
        "JIT IR Graph ----\n");
  }
}

/**
 * This method prunes the redundant inputs from the input stack in lazy cache
 * hit case
 */
void HlExec::deleteRedundantInputsFromInputStack(torch::jit::Stack& stack) {
  PT_LAZY_TRACE;

  std::vector<size_t> indices_for_deletion;

  for (size_t i = 0; i < stack.size(); i++) {
    auto tensor = stack[i].toTensor();
    auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
    if (impl->isRedundant()) {
      indices_for_deletion.push_back(i);
    }
  }

  if (!indices_for_deletion.empty()) {
    std::sort(indices_for_deletion.rbegin(), indices_for_deletion.rend());
    for (size_t i = 0; i < indices_for_deletion.size(); i++) {
      size_t idx = indices_for_deletion.at(i);
      stack.erase(stack.cbegin() + idx);
    }
  }
}

/**
 * This method prunes the redundant inputs from the JIT IR Graph and the stack
 */
void HlExec::SearchAndDeleteRedundantInputs(
    ir::PostOrderData& po_data,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs) {
  PT_LAZY_TRACE;

  std::vector<size_t> indices_for_deletion;
  std::vector<size_t> po_data_input_indices_for_deletion;
  auto jit_ir_graph_inputs = mp_g_->inputs();

  for (auto r_value_in : redundant_inputs) {
    size_t idx = 0;
    for (auto value_in : jit_ir_graph_inputs) {
      if (r_value_in->unique() == value_in->unique()) {
        indices_for_deletion.emplace_back(idx);
      }
      idx++;
    }
  }

  for (auto r_value_in : redundant_inputs) {
    size_t idx = 0;
    for (auto value_in : po_data.inputs) {
      std::string str1 = r_value_in->debugName();
      std::string str2 = value_in.ToString();
      if (str1.compare(str2) == 0) {
        // std::cout << "~~~ po_data_input_indices_for_deletion ~~~\n"
        //           << r_value_in->debugName()
        //           << ", "
        //           << value_in.ToString()
        //           << std::endl
        //           << std::flush;
        po_data_input_indices_for_deletion.emplace_back(idx);
      }
      idx++;
    }
  }

  if (!indices_for_deletion.empty()) {
    std::sort(indices_for_deletion.rbegin(), indices_for_deletion.rend());
    for (auto i : indices_for_deletion) {
      mp_g_->eraseInput(i);

      auto tensor = stack[i].toTensor();
      auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
      impl->setRedundant();

      stack.erase(stack.cbegin() + i);
    }
  }

  if (!po_data_input_indices_for_deletion.empty()) {
    std::sort(
        po_data_input_indices_for_deletion.rbegin(),
        po_data_input_indices_for_deletion.rend());
    for (auto i : po_data_input_indices_for_deletion) {
      po_data.inputs.erase(po_data.inputs.cbegin() + i);
    }
  }

  auto& device = habana::HPURegistrar::get_device();
  auto context = get_device_lazy_execution_context(device.id());
  // Save po_data input and output to context for perf mode
  if (context->getCapturing() &&
      context->updateInputsRequired(po_data_input_indices_for_deletion)) {
    context->updateInputs(po_data.inputs);
  }
}

/*
 * Get the JIT graph from cache, or create it.
 *
 * Given a hash_code derived from LazyArgumentSpec for a
 * post order graph and inputs, this cache can be looked up
 * for finding an optimize JIT graph.
 *
 * On a cache miss, the caller is expected to create the optimized
 * JIT graph and add to cache.
 *
 * Cache Lookup
 * ============
 * auto las = LazyArgumentSpec(true, post_order_graph, input_tensors);
 * auto jit_graph_and_meta_data =
 * habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(las.hashCode());
 *
 * Cache hit
 * =========
 * if (jit_graph_and_metat_data != nullptr) lower_jit_graph(...)
 *
 * Cache miss handling
 * ===================
 * // Create a JIT graph from the post order graph
 * auto jit_graph = Create(post_order_graph, input_tensors);
 * // Create a LazyArgumentSpec
 * auto las = LazyArgumentSpec(true, post_order_graph, input_tensors);
 * // Compute meta data for JIT graph and store in cache along with JIT graph
 * auto jit_graph_and_meta_data =
 * std::make_shared<habana::OptimizedJITGraphAndMetaData>(jit_graph,
 * input_refs); habana::JitGraphCache::GetJitCache().Add(las.hashCode,
 * jit_graph_and_meta_data); lower_jit_graph(...)
 *
 */
void HlExec::GetOrCreate(ir::PostOrderData& po_data, torch::jit::Stack& stack) {
  PT_LAZY_TRACE;

  size_t num_inputs = po_data.inputs.size();

  auto orig_stack = stack;
  std::vector<size_t> parent_vec(num_inputs, ULONG_MAX);
  std::vector<bool> is_duplicate_vec(num_inputs, false);
  FindDuplicateInStack(po_data, stack, parent_vec, is_duplicate_vec);
  PruneDuplicateStackInputs(stack, is_duplicate_vec);
  CreateNodeBcastMap(po_data.post_order);

  uint64_t unique_cntr = 0;

  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VALIDATE_GRAPH_RUNNING_HASH) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
    m_g_hash_ = m_fwd_graph_hash_;
  } else {
    m_g_hash_ = habana_lazy::LazyArgumentSpec(
                    true,
                    stack,
                    po_data.post_order_nodes_hash,
                    po_data.inputs,
                    po_data.value_input_nodes_map,
                    po_data.outputs,
                    parent_vec,
                    node_bcast_map_)
                    .hashCode();
    unique_cntr = get_habana_lazy_executor().getGraphindexCntr(m_g_hash_);
    m_g_hash_ = at::hash_combine(m_g_hash_, unique_cntr);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VALIDATE_GRAPH_RUNNING_HASH)) {
    HABANA_ASSERT(m_g_hash_ == m_fwd_graph_hash_);
  }

  auto ConstructJITGraph{
      // Create a JIT graph from the post order graph
      // Optimization is done during Create() itself
      [&]() -> void {
        std::vector<torch::jit::Value*> redundant_inputs;
        mp_g_ = std::make_shared<Graph>();
        Create(
            po_data.post_order,
            po_data.inputs,
            po_data.outputs,
            orig_stack,
            redundant_inputs);

        PruneDuplicateGraphInputs(parent_vec, is_duplicate_vec);
        if (!redundant_inputs.empty()) {
          SearchAndDeleteRedundantInputs(po_data, stack, redundant_inputs);
        }

        at::ArrayRef<torch::jit::IValue> input_refs =
            torch::jit::last(stack, mp_g_->inputs().size());
        const bool isDynamic = habana_helpers::GetRefineDynamicShapeStatus();
        mp_g_and_meta_data_ =
            std::make_shared<habana::OptimizedJITGraphAndMetaData>(
                mp_g_, input_refs, unique_cntr, node_bcast_map_, "", isDynamic);
        mp_g_and_meta_data_->set_fwd_graph_builder_stack_map(
            m_fwd_graph_stack_map_);
        IdentifyAndSetGraphNodes(po_data.post_order);
      }};

  if (std::getenv("PT_HPU_LAZY_CACHE_DISABLE")) {
    PT_LAZY_DEBUG(
        "JIT Cache disabled :: key ",
        m_g_hash_,
        ", graph_index ",
        GetGraphIndex(
            m_g_hash_, torch::jit::last(stack, mp_g_->inputs().size())),
        ", bcast_map = ",
        node_bcast_map_.size());
    ConstructJITGraph();
    return;
  }

  size_t optimized_lazy_eager_key = 0;
  if (lazyInfo) {
    optimized_lazy_eager_key = lazyInfo->get_optimized_lazy_eager_key();
  }

  // To not read the normal cache for optimized eager
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2 || !optimized_lazy_eager_key) {
    mp_g_and_meta_data_ =
        habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
            m_g_hash_);
  }

  // Cache miss
  // ==========
  if (mp_g_and_meta_data_ == nullptr) {
    ConstructJITGraph();
    // To not write the normal cache for optimized eager
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2 || !optimized_lazy_eager_key) {
      PT_LAZY_DEBUG(
          "JIT Cache miss :: key ",
          m_g_hash_,
          ", graph_index ",
          GetGraphIndex(
              m_g_hash_, torch::jit::last(stack, mp_g_->inputs().size())),
          ", bcast_map = ",
          node_bcast_map_.size());
      PT_IRGRAPH_DEBUG("JIT Cache miss");
      // Cache miss handling
      // ===================
      habana::JitGraphCache::GetJitCache().Add(m_g_hash_, mp_g_and_meta_data_);
    }
  } else {
    PT_LAZY_DEBUG(
        "JIT Cache hit :: key ",
        m_g_hash_,
        ", graph_index ",
        GetGraphIndex(
            m_g_hash_, torch::jit::last(stack, mp_g_->inputs().size())),
        ", bcast_map = ",
        node_bcast_map_.size());
    PT_IRGRAPH_DEBUG("JIT Cache hit");
    mp_g_ = mp_g_and_meta_data_->get_cached_graph();
    HABANA_ASSERT(mp_g_ != nullptr);

    if (mp_g_->inputs().size() != stack.size()) {
      deleteRedundantInputsFromInputStack(stack);
    }

    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      mp_g_and_meta_data_->SetDynamicGraph(true);
      at::ArrayRef<torch::jit::IValue> input_refs =
          torch::jit::last(stack, mp_g_->inputs().size());
      mp_g_and_meta_data_->ComputeGraphHashCode(mp_g_, input_refs);
    }
    visualize::DumpCachedGraph(mp_g_, m_g_hash_);
  }

  if (optimized_lazy_eager_key != 0) {
    bool IsOptimizedLazyEagerCached =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().IsCached(
            optimized_lazy_eager_key);
    if (IsOptimizedLazyEagerCached == false) {
      habana::OptimizedJitGraphCache::GetOptimizedJitCache().Add(
          optimized_lazy_eager_key, mp_g_and_meta_data_);
      // To Do - To incorporate the Graph index change
      PT_LAZY_DEBUG(
          "Optimized Path JIT Cache miss :: key ", optimized_lazy_eager_key);
    }
  }
}

void HlExec::LoadDSCheckpoint(const std::string& path) {
  std::ifstream ds_checkpoint(std::string(path), std::ifstream::binary);
  HABANA_ASSERT(ds_checkpoint, "Failed to open ds_checkpoint file (load)");
  habana::DynamicBucketInfoMap::load_ds_checkpoint(ds_checkpoint);
  Deserialize(ds_checkpoint);
}

void HlExec::SaveDSCheckpoint(const std::string& path) {
  std::ofstream ds_checkpoint(path, std::ofstream::binary);
  HABANA_ASSERT(ds_checkpoint, "Failed to open ds_checkpoint file (save)");
  habana_lazy::HbLazyTensor::StepMarkerFinish();
  habana::DynamicBucketInfoMap::save_ds_checkpoint(ds_checkpoint);
  Serialize(ds_checkpoint);
}

void HlExec::Serialize(std::ostream& os) {
  using namespace serialization;
  serialize(os, s_graphIndexMap);
  serialize(os, s_graphIndex);
}

void HlExec::Deserialize(std::istream& is) {
  using namespace serialization;
  deserialize(is, s_graphIndexMap);
  deserialize(is, s_graphIndex);
}

size_t HlExec::GetGraphIndex(
    size_t hash,
    at::ArrayRef<torch::jit::IValue> input_refs) {
  if (GET_ENV_FLAG_NEW(PT_HPU_VISUALIZE_GRAPH_INDEX)) {
    return visualize::GetGraphIndex(hash);
  }

  size_t sym_hash_code = habana::ComputeSymSizeHashCode(input_refs);
  hash = at::hash_combine(hash, sym_hash_code);
  auto perm_hash_code = habana::ComputePermutationHashCode(input_refs);
  hash = at::hash_combine(hash, perm_hash_code);

  static std::mutex s_mutex;
  std::lock_guard<std::mutex> guard(s_mutex);
  size_t graphIndex = hash;
  if (s_graphIndexMap.count(hash) == 0) {
    s_graphIndexMap[hash] = s_graphIndex;
    graphIndex = s_graphIndex;
    s_graphIndex++;
  } else {
    graphIndex = s_graphIndexMap[hash];
  }
  return graphIndex;
}

void HlExec::CreateNodeBcastMap(const ir::NodePtrList& nodes) {
  for (const auto& node : nodes) {
    if (c10::Symbol::fromQualString("prim::constant") != node->op() &&
        (std::string(node->op().toQualString()).find("hpu::input") ==
         std::string::npos)) {
      node_bcast_map_.insert(
          node_bcast_map_.end(),
          node->get_broadcast_details().begin(),
          node->get_broadcast_details().end());
    }
  }
}

void HlExec::IdentifyAndSetGraphNodes(const ir::NodePtrList& nodes) {
  for (const auto& node : nodes) {
    if ((std::string(node->op().toQualString()).find("hpu::habanaOptimizer") !=
         std::string::npos) ||
        (std::string(node->op().toQualString()).find("hpu::optimizer") !=
         std::string::npos)) {
      mp_g_and_meta_data_->set_is_eager_compiler_supported(false);
      mp_g_and_meta_data_->set_is_shape_agnostic_supported(false);
      break;
    }
  }
}

/*
 * Creates the Graph
 */
void HlExec::Create(
    const ir::NodePtrList& nodes,
    const ir::ValueList& inputs,
    const ir::ValueList& outputs,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs) {
  PT_LAZY_TRACE;
  LazyOutputToJitValueMap ir_map;

  for (const auto& inp : inputs) {
    auto t = mp_g_->addInput(inp.ToString());
    HABANA_ASSERT(!inp.m_data_ptr.expired());
    std::shared_ptr<Data> d = inp.m_data_ptr.lock();
    t->setType(c10::TensorType::createContiguous(
        *(d->logical_element_type), d->device, d->sizes));
    t->setDebugName(inp.ToString());
    ir_map[ir::Output(inp)] = t;
  }

  for (const auto& node : nodes) {
    // Is it a scalar node?
    if (c10::Symbol::fromQualString("prim::constant") == node->op()) {
      // add constant
      auto scalar_node = dynamic_cast<ir::ScalarConstant*>(node.get());
      auto scalar_const = scalar_node->getIValue();
      // TBD: Should we create a constant node, or should it be
      // a 1-element tensor as input?
      // Keeping it as a constant node
      // allows for optimized graph (no DMA required, some optimizations
      // like avoiding multiply with 1 can be removed).
      // Keeping it as a variable (1-elem input) allows to be able to
      // reuse the same graph when the scalar values change.
      auto c = mp_g_->insertConstant(scalar_const);
      ir_map[node->GetOutput(0)] = c;
    } else if (
        std::string(node->op().toQualString()).find("hpu::input") !=
        std::string::npos) {
      // Its a tensor, should already be there in the value maps
      HABANA_ASSERT(ir_map.find(node->GetOutput(0)) != ir_map.end());
    } else {
      std::vector<JitValue*> args_vector;
      auto node_input_vals = node->GetInputs();
      std::transform(
          node_input_vals.begin(),
          node_input_vals.end(),
          std::back_inserter(args_vector),
          [&](const HabanaLazyValue& inp) -> JitValue* {
            auto it = ir_map.find(ir::Output(inp));
            HABANA_ASSERT(it != ir_map.end());
            return it->second;
          });

      // Total inputs to a node is size of meta data + size of inputs
      // Allocate vector with nulllptr with inputs_size
      std::vector<JitValue*> node_inputs(
          args_vector.size() + node->GetMetaData().size(), nullptr);

      // Iterate thru each of the metadata and create constant node and
      // assign this to correct index in the input array
      std::for_each(
          node->GetMetaData().cbegin(),
          node->GetMetaData().cend(),
          [&](const auto& meta_data) {
            HABANA_ASSERT(node_inputs[meta_data.first] == nullptr);
            node_inputs[meta_data.first] =
                mp_g_->insertConstant(meta_data.second);
          });

      // Now we will fill the inputs in the array whereever its null
      size_t j = 0;
      std::for_each(node_inputs.begin(), node_inputs.end(), [&](auto& node) {
        if (nullptr == node) {
          node = args_vector[j++];
        }
      });
      HABANA_ASSERT(j == args_vector.size());
      std::shared_ptr<torch::jit::WithCurrentScope> scope_context;
      auto scope_name = (node->GetScope() ? *node->GetScope() : "");
      if (AccThread::IsAccThreadEnabled() ? !node->GetModuleName().empty()
                                          : node->GetScope() != NULL) {
        scope_context = std::make_shared<torch::jit::WithCurrentScope>(
            *mp_g_,
            c10::make_intrusive<torch::jit::Scope>(
                torch::jit::ScopePtr(),
                c10::Symbol::fromQualString("debug::" + scope_name)));
      }
      at::ArrayRef<JitValue*> args(node_inputs);
      auto jit_node = mp_g_->create(node->op(), args, node->GetNumOutputs());
      // if (AccThread::IsAccThreadEnabled()) {
      //   jit_node->setScope(c10::make_intrusive<torch::jit::Scope>(
      //       torch::jit::ScopePtr(),
      //       c10::Symbol::fromQualString("debug::" + scope_name)));
      // }

      jit_node->i_(
          torch::jit::attr::deterministic,
          node->getDeterministic() ||
              at::globalContext().deterministicAlgorithms());

      mp_g_->insertNode(jit_node);

      if (c10::Symbol::fromQualString("prim::ListConstruct") == node->op() ||
          node->is_output_tensor_list()) {
        auto* list_node = dynamic_cast<ir::ListConstruct*>(node.get());
        if (list_node && list_node->isOptional()) {
          jit_node->output()->setType(torch::jit::ListType::create(
              torch::jit::OptionalType::ofTensor()));
        } else {
          jit_node->output()->setType(torch::jit::ListType::ofTensors());
        }
      } else {
        for (size_t idx = 0; idx < jit_node->outputs().size(); idx++) {
          if (jit_node->output(idx)->type()->kind() ==
              c10::TypeKind::TensorType) {
            auto irout_val = node->GetOutput(idx);
            auto jit_value_out = jit_node->output(idx);
            jit_value_out->setType(c10::TensorType::createContiguous(
                *(irout_val.get_scalar_type()),
                *(irout_val.get_device()),
                *(irout_val.get_sizes())));
            jit_value_out->setDebugName(irout_val.ToString());
          }
        }
      }
      auto jit_outputs = jit_node->outputs();
      int i = 0;
      for (const auto jit_output : jit_outputs) {
        ir_map[node->GetOutput(i++)] = jit_output;
      }
    }
  }

  for (const auto& output : outputs) {
    auto out = ir::Output(output);
    mp_g_->registerOutput(ir_map.at(out));
  }

  // Optimize the graph based on the passes enabled
  Optimize(stack, redundant_inputs);
}

void HlExec::Optimize(
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs) {
  PT_LAZY_TRACE;
  visualize::DumpPreGraph(mp_g_, m_g_hash_);

  if (OptPassCfg::GetInstance()->IsEnabledFuseTMM()) {
    fuse_mm_transpose(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "fuse_mm_transpose");
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseBnRelu()) {
    fuse_bn_relu(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "fuse_bn_relu");
  }

  if (OptPassCfg::GetInstance()->IsEnabledReplaceInplaceOps()) {
    replace_inplace_ops(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "replace_inplace_ops");
    OptPassCfg::GetInstance()->SetDeadCodeElimination(true);
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseTMM() ||
      OptPassCfg::GetInstance()->IsEnabledDeadCodeElimination() ||
      OptPassCfg::GetInstance()->IsEnabledFuseBnRelu()) {
    torch::jit::EliminateDeadCode(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "eliminate_dead_code");
  }

  if (OptPassCfg::GetInstance()->IsEnabledCSEElimination()) {
    torch::jit::EliminateCommonSubexpression(mp_g_);
    visualize::DumpOptimizedGraph(
        mp_g_, m_g_hash_, "eliminate_common_subexpression");
  }

  if (OptPassCfg::GetInstance()->IsEnabledConstPooling()) {
    torch::jit::ConstantPooling(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "constant_pooling");
  }

  if (OptPassCfg::GetInstance()->IsEnabledPeepholeOpt()) {
    torch::jit::PeepholeOptimize(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "peephole_optimize");
  }

  if (OptPassCfg::GetInstance()->IsEnabledSubgraphRewrite()) {
    transform_graph(mp_g_);
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "transform_graph");
  }

  if (OptPassCfg::GetInstance()->IsEnabledReplaceViews()) {
    replace_views_with_reshapes(mp_g_);
    visualize::DumpOptimizedGraph(
        mp_g_, m_g_hash_, "replace_views_with_reshapes");
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseConvBn()) {
    PT_LAZY_DEBUG("[Inference] FoldConvBatchnorm called!");
    FoldConvBatchnorm(mp_g_, stack, redundant_inputs);
    PT_LAZY_DEBUG("[Inference] FoldConvBatchnorm applied!");
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "fold_conv_bn");
  }

  if (!redundant_inputs.empty()) {
    PT_LAZY_DEBUG(
        "[HlExec::Optimize] redundant_inputs size: ", redundant_inputs.size());
  }

  if (OptPassCfg::GetInstance()->IsEnabledBnParamRecalc()) {
    PT_LAZY_DEBUG("[Inference] RecalculateBatchnormParams called!");
    RecalculateBatchnormParams(mp_g_, stack);
    PT_LAZY_DEBUG("[Inference] RecalculateBatchnormParams applied!");
    visualize::DumpOptimizedGraph(mp_g_, m_g_hash_, "recalculate_bn_params");
  }

  visualize::DumpPostGraph(mp_g_, m_g_hash_);
}

} // namespace exec
} // namespace habana_lazy
