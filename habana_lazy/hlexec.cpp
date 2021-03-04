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

  HabanaLaunchOpPT launch{mp_g_, false};
  launch.run(stack);

  context->setExecutionMode(kLAZY);
  context->MarkTensorsExecuted();
  unsetenv("PT_HPU_LAZY_LOWERING");
}

/*
 * Get the JIT graph fron cache, or create it
 */
void HlExec::GetOrCreate(
    const ir::PostOrderData& po_data,
    torch::jit::Stack& stack) {
  PT_LAZY_TRACE;
  if (std::getenv("PT_HPU_LAZY_CACHE_DISABLE")) {
    mp_g_ = std::make_shared<Graph>();
    Create(po_data.post_order, po_data.inputs, po_data.outputs);
    return;
  }
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      po_data.post_order,
      stack,
      po_data.post_order_nodes_hash,
      po_data.inputs,
      po_data.value_input_nodes_map,
      po_data.outputs.size());
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
    Create(po_data.post_order, po_data.inputs, po_data.outputs);
    // Create a lazyArgumentSpec
    las = habana_lazy::LazyArgumentSpec(
        true,
        po_data.post_order,
        stack,
        po_data.post_order_nodes_hash,
        po_data.inputs,
        po_data.value_input_nodes_map,
        po_data.outputs.size());
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
    const ir::ValueList outputs) {
  PT_LAZY_TRACE;
  LazyOutputToJitValueMap ir_map;

  for (auto inp : inputs) {
    auto t = mp_g_->addInput(inp.ToString());
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

      if (c10::Symbol::fromQualString("prim::ListConstruct") == node->op()) {
        jit_node->output()->setType(torch::jit::ListType::ofTensors());
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
    mp_g_->registerOutput(ir_map[out]);
  }

  // Optimize the graph based on the passes enabled
  Optimize();
}

void HlExec::Optimize() {
  PT_LAZY_TRACE;
  if (OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization) {
    fuse_mm_transpose(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_fuse_bn_relu_optimization) {
    fuse_bn_relu(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization ||
      OptPassCfg::GetInstance()->enable_eliminate_dead_code ||
      OptPassCfg::GetInstance()->enable_fuse_bn_relu_optimization) {
    torch::jit::EliminateDeadCode(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_eliminate_common_subexpression) {
    torch::jit::EliminateCommonSubexpression(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_constant_pooling) {
    torch::jit::ConstantPooling(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_peephole_optimization) {
    torch::jit::PeepholeOptimize(mp_g_);
  }

  if (OptPassCfg::GetInstance()->enable_subgraph_rewrite) {
    transform_graph(mp_g_);
  }
}

} // namespace exec
} // namespace habana_lazy