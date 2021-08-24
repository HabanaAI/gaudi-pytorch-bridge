/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "hlexec.h"
#include "hpu_lazy_cache.h"
#include "ops/constant.h"
#include "ops/convolution.h"
#include "passes/fuse_bn_relu_residual_add.h"
#include "passes/fuse_mm_transpose.h"
#include "passes/permute_graph.h"
#include "passes/replace_inplace_ops.h"
#include "passes/transform_graph.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"
#include "synapse_helpers/device.h"

namespace habana_lazy {
namespace exec {
OptPassCfg* OptPassCfg::p_instance_ = nullptr;

HlExec::HlExec() {
  mp_g_ = std::make_shared<Graph>();
}

HlExec::HlExec(ScopePtr scope) {
  mp_g_ = std::make_shared<Graph>(scope);
}

void HlExec::Launch(torch::jit::Stack& stack) {
  PT_LAZY_TRACE;
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto context = habana_lazy_executor.getDeviceExecutionContext(device.id());
  // TODO : remove this env variable use
  // This is temporarily done to deactivate code in synapse helpers for lazy
  // mode kernel registration We will move to using shape utilities instead and
  // not do env variable based check anymore
  // We have short-circuited certain utilities in synapse helpers, we need to
  // remove that code
  setenv("PT_HPU_LAZY_LOWERING", "1", 1);
  context->setExecutionMode(kLOWERING);

  // save the graph for perf mode
  context->saveGraph(mp_g_);

  habana::HabanaLaunchOpPT launch{mp_g_, false};
  launch.run(stack);

  context->setExecutionMode(kLAZY);
  context->MarkTensorsExecuted();
  unsetenv("PT_HPU_LAZY_LOWERING");
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
    HABANA_ASSERT(input.isTensor());
    if (!input.toTensor().has_storage()) {
      return;
    }
  }

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    HABANA_ASSERT(input.isTensor());
    auto input_addr = (uint64_t)(input.toTensor().data_ptr());

    if (input_addr_map.count(input_addr) != 0) {
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

  PT_LAZY_DEBUG(
      "Initial JIT IR Graph ====\n", mp_g_->toString(), "JIT IR Graph ----\n");

  auto jit_ir_graph_inputs = mp_g_->inputs();
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
      PT_LAZY_DEBUG(
          "Deleting ",
          j,
          "th input %",
          mp_g_->inputs().at(j)->debugName(),
          "of the graph");
      mp_g_->eraseInput(j);
    }
  }

  PT_LAZY_DEBUG(
      "After pruning duplicates, JIT IR Graph ====\n",
      mp_g_->toString(),
      "JIT IR Graph ----\n");
}

/*
 * Get the JIT graph fron cache, or create it
 */
void HlExec::GetOrCreate(
    const ir::PostOrderData& po_data,
    torch::jit::Stack& stack) {
  PT_LAZY_TRACE;

  size_t num_inputs = po_data.inputs.size();

  auto orig_stack = stack;
  std::vector<size_t> parent_vec(num_inputs, ULONG_MAX);
  std::vector<bool> is_duplicate_vec(num_inputs, false);
  FindDuplicateInStack(po_data, stack, parent_vec, is_duplicate_vec);
  PruneDuplicateStackInputs(stack, is_duplicate_vec);

  if (std::getenv("PT_HPU_LAZY_CACHE_DISABLE")) {
    mp_g_ = std::make_shared<Graph>();
    Create(po_data.post_order, po_data.inputs, po_data.outputs, stack);
    PruneDuplicateGraphInputs(parent_vec, is_duplicate_vec);
    return;
  }
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      stack,
      po_data.post_order_nodes_hash,
      po_data.inputs,
      po_data.value_input_nodes_map,
      po_data.outputs,
      parent_vec);
  mp_g_ = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las.hashCode());

  // Cache miss
  // ==========
  if (mp_g_ == nullptr) {
    PT_LAZY_DEBUG("JIT Cache miss");
    mp_g_ = std::make_shared<Graph>();
    // Cache miss handling
    // ===================
    // Create a JIT graph from the post order graph
    // Optimization is done during Create() itself
    Create(po_data.post_order, po_data.inputs, po_data.outputs, orig_stack);
    PruneDuplicateGraphInputs(parent_vec, is_duplicate_vec);
    LazyGraphCache::GetLazyCache().Add(las.hashCode(), mp_g_);
  } else {
    PT_LAZY_DEBUG("JIT Cache hit");
  }
}

/*
 * Creates the Graph
 */
void HlExec::Create(
    const ir::NodePtrList nodes,
    const ir::ValueList inputs,
    const ir::ValueList outputs,
    torch::jit::Stack& stack) {
  PT_LAZY_TRACE;
  LazyOutputToJitValueMap ir_map;

  for (auto inp : inputs) {
    auto t = mp_g_->addInput(inp.ToString());
    HABANA_ASSERT(!inp.m_data_ptr.expired());
    std::shared_ptr<Data> d = inp.m_data_ptr.lock();
    t->setType(c10::TensorType::create(
        d->logical_element_type, d->device, d->sizes.size(), false));
    ir_map[ir::Output(inp)] = t;
  }

  for (auto node : nodes) {
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
    } else if (node->ToString().find("hpu::input") != std::string::npos) {
      // Its a tensor, should already be there in the value maps
      HABANA_ASSERT(ir_map.find(node->GetOutput(0)) != ir_map.end());
    } else {
      std::vector<JitValue*> args_vector;
      auto node_input_vals = node->GetInputs();
      std::transform(
          node_input_vals.begin(),
          node_input_vals.end(),
          std::back_inserter(args_vector),
          [&](HabanaLazyValue inp) -> JitValue* {
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

      at::ArrayRef<JitValue*> args(node_inputs);
      auto jit_node = mp_g_->create(node->op(), args, node->GetNumOutputs());
      mp_g_->insertNode(jit_node);

      if (c10::Symbol::fromQualString("prim::ListConstruct") == node->op() ||
          node->is_output_tensor_list()) {
        jit_node->output()->setType(torch::jit::ListType::ofTensors());
      } else {
        for (size_t idx = 0; idx < jit_node->outputs().size(); idx++) {
          if (jit_node->output(idx)->type()->kind() ==
              c10::TypeKind::TensorType) {
            auto irout_val = node->GetOutput(idx);
            auto jit_value_out = jit_node->output(idx);
            jit_value_out->setType(c10::TensorType::create(
                irout_val.get_scalar_type(),
                irout_val.get_device(),
                irout_val.get_dims(),
                false));
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

  for (auto output : outputs) {
    auto out = ir::Output(output);
    mp_g_->registerOutput(ir_map.at(out));
  }

  // Optimize the graph based on the passes enabled
  Optimize(stack);
}

void HlExec::Optimize(torch::jit::Stack& stack) {
  PT_LAZY_TRACE;
  // Permute Pass to insert permute nodes should be run before any other JIT
  // optimization pass. Reason for this is because Permute pass relies on extra
  // information (e.g. dims) for each tensor added at JIT graph graph creation
  // time to decide on permute node insertion. If any other pass runs before
  // permute pass and inserts a new node (e.g. inplace replacement pass removes
  // inplace node and adds corresponding out-of-place node), then this new node
  // will not have required extra information for permute pass to work properly.
  if (OptPassCfg::GetInstance()->IsEnabledPermutePass()) {
    InsertPermute_graph(mp_g_, stack);
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseTMM()) {
    fuse_mm_transpose(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseBnRelu()) {
    fuse_bn_relu(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledReplaceInplaceOps()) {
    replace_inplace_ops(mp_g_);
    OptPassCfg::GetInstance()->SetDeadCodeElimination(true);
  }

  if (OptPassCfg::GetInstance()->IsEnabledFuseTMM() ||
      OptPassCfg::GetInstance()->IsEnabledDeadCodeElimination() ||
      OptPassCfg::GetInstance()->IsEnabledFuseBnRelu()) {
    torch::jit::EliminateDeadCode(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledCSEElimination()) {
    torch::jit::EliminateCommonSubexpression(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledConstPooling()) {
    torch::jit::ConstantPooling(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledPeepholeOpt()) {
    torch::jit::PeepholeOptimize(mp_g_);
  }

  if (OptPassCfg::GetInstance()->IsEnabledSubgraphRewrite()) {
    transform_graph(mp_g_);
  }
}

} // namespace exec
} // namespace habana_lazy
