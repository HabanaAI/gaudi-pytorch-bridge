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
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <torch/csrc/api/include/torch/version.h>

#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUCheck.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/passes/hpu_habana_persistence_marker_pass.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/jitgraph_utils.h"
#include "habana_helpers/misc_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch::jit;
using namespace jitgraph_utils;
using namespace habana;

void PersistenceMarkerPass::set_persistence_input(
    torch::jit::Node* node,
    int inputId) {
  auto val = node->input(inputId);

  if (val->type()->kind() == c10::TypeKind::TensorType) {
    valptr_to_persistent_map_[val] = true;
  } else if (val->type()->kind() == c10::TypeKind::ListType) {
    // This case is needed for fused clip norm
    // fused clip norm has List(as_strided(grads) ->fused_norm. Since fused norm
    // is an inplace op,  we need to mark as_strided output as persistent move
    // one level up and set persistence for all the list inputs
    auto list_in_vals = val->node()->inputs();

    for (auto in_val : list_in_vals) {
      if (in_val->type()->kind() == c10::TypeKind::TensorType) {
        valptr_to_persistent_map_[in_val] = true;
      }
    }
  }
}

void PersistenceMarkerPass::set_persistence_output(
    torch::jit::Node* node,
    int outputId) {
  auto val = node->output(outputId);

  if (val->type()->kind() == c10::TypeKind::TensorType) {
    valptr_to_persistent_map_[val] = true;
  }
}

void PersistenceMarkerPass::MarkPersistenceNodes(
    torch::jit::graph_node_list graph_nodes) {
  for (auto* node : graph_nodes) {
    if (node->kind().is_prim()) {
      continue;
    }

    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
        0,
        node->schema().operator_name(),
        habana_launch_op_ptr_->getNodeScalarType(node));
    if (HabanaKernel == nullptr)
      continue;

    if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
      // Set the deterministic val
      auto one = torch::jit::attr::alpha;
      PT_BRIDGE_DEBUG("Deterministic value in BuildGraph: ", node->i(one));
      HabanaKernel->setDeterministic(node->i(one));
    }

    // override the persistence logic if any kernel sets it as persistent
    // We assume that first index for output will be the persistent.
    // We used to assume it also for input but it caused difficult to debug bugs
    // when assumption was incorrect. So we no longer assume anything but
    // take right persistent input id.
    // GC doesnt recommend using workspace tensors for intermediate inplace ops.
    // Inplace -> out of place replacement pass will remove  intermediate
    // inplace ops anyway Remaining inplace ops at graph outputs will be set
    // with persistent i/o
    int inputId = 0;
    if (HabanaLaunchOpPT::isControlEdge(node) ||
        habana_helpers::IsCollective(node->kind()) ||
        // must the be last condition as it can change inputId
        ((inputId = inplaceInputId(node)) >= 0)) {
      set_persistence_input(node, inputId);
      set_persistence_output(node, 0);
    }
  } // for (auto* node : graph_nodes)
} // function end

void PersistenceMarkerPass::set_external_input(torch::jit::Node* node) {
  for (auto& val : node->inputs()) {
    if (val->type()->kind() == c10::TypeKind::TensorType) {
      MarkProducerExternal(val);
    } else if (val->type()->kind() == c10::TypeKind::ListType) {
      auto list_in_vals = val->node()->inputs();
      for (auto in_val : list_in_vals) {
        if (in_val->type()->kind() == c10::TypeKind::TensorType) {
          MarkProducerExternal(val);
        }
      }
    }
  }
}

void PersistenceMarkerPass::MarkProducerExternal(torch::jit::Value* val) {
  while (HabanaLaunchOpPT::isControlEdge(val->node())) {
    val = val->node()->inputs().at(0);
  }
  if (isInGraphInputs(val) != -1) {
    PT_LAZY_DEBUG(
        "Not adding ",
        val->debugName(),
        " to extenal map since it is an input to the graph")
  } else {
    PT_LAZY_DEBUG("Adding ", val->debugName(), " to extenal map")
    valptr_to_external_map_[val] = true;
  }
}

void PersistenceMarkerPass::ExternalMarkingPass(
    torch::jit::graph_node_list graph_nodes) {
  for (auto* node : graph_nodes) {
    if (node->kind().is_prim()) {
      continue;
    }

    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
        0,
        node->schema().operator_name(),
        habana_launch_op_ptr_->getNodeScalarType(node));
    if (HabanaKernel == nullptr)
      continue;

    // collective inputs must be set external in order to
    // trigger before graph execution ends
    if (habana_helpers::IsCollective(node->kind())) {
      set_external_input(node);
    }

  } // for (auto* node : graph_nodes)
} // function end

void PersistenceMarkerPass::RunMetaDataAdjustmentPasses(
    torch::jit::graph_node_list graph_nodes) {
  // This pass marks tensors persistent if they are nt persistent from graph
  // but are made persistent due to synapse limitations
  MarkPersistenceNodes(graph_nodes);

  // This pass marks tensors external if they are used as input tensors for
  // collective ops. Used for Signal From Graph to signal the tensor data is
  // ready prior to recipe completion
  ExternalMarkingPass(graph_nodes);
}

std::unique_ptr<PersistenceMarkerPassData> PersistenceMarkerPass::VisitGraph(
    const std::shared_ptr<torch::jit::Graph> graph) {
  TORCH_CHECK(NULL != habana_launch_op_ptr_);
  TORCH_CHECK(NULL != graph.get());
  RunMetaDataAdjustmentPasses(graph->nodes());
  return std::make_unique<PersistenceMarkerPassData>(
      valptr_to_persistent_map_, valptr_to_external_map_);
}
