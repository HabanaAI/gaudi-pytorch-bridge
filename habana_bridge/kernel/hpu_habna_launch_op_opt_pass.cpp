/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch::jit;
using namespace habana;

void HabanaLaunchOpPT::set_persistence_input(torch::jit::Node* node) {
  auto val = node->input(0);

  if (val->type()->kind() == c10::TypeKind::TensorType) {
    valptr_to_persistent_map[val] = true;
  } else if (val->type()->kind() == c10::TypeKind::ListType) {
    // This case is needed for fused clip norm
    // fused clip norm has List(as_strided(grads) ->fused_norm. Since fused norm
    // is an inplace op,  we need to mark as_strided output as persistent move
    // one level up and set persistence for all the list inputs
    auto list_in_vals = val->node()->inputs();

    for (auto in_val : list_in_vals) {
      if (in_val->type()->kind() == c10::TypeKind::TensorType) {
        valptr_to_persistent_map[in_val] = true;
      }
    }
  }
}

void HabanaLaunchOpPT::set_persistence_output(torch::jit::Node* node) {
  auto val = node->output(0);

  if (val->type()->kind() == c10::TypeKind::TensorType) {
    valptr_to_persistent_map[val] = true;
  }
}

void HabanaLaunchOpPT::persistenceMarkingPass(
    torch::jit::graph_node_list graph_nodes) {
  for (auto* node : graph_nodes) {
    if (node->kind().is_prim()) {
      continue;
    }

    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
        0, node->schema().operator_name(), getNodeScalarType(node));
    if (HabanaKernel == nullptr)
      continue;

    // override the persistence logic if any kernel sets it as persistent
    // We assume that first index for input and output will be the persistent
    // GC doesnt recommend using workspace tensors for intermediate inplace ops.
    // Inplace -> out of place replacement pass will remove  intermediate
    // inplace ops anyway Remaining inplace ops at graph outputs will be set
    // with persistent i/o
    if (isControlEdge(node) || isInplace(node) ||
        habana_lazy::IsCollective(node->kind())) {
      set_persistence_input(node);
      set_persistence_output(node);
    }
  } // for (auto* node : graph_nodes)
} // function end

void HabanaLaunchOpPT::set_external_input(torch::jit::Node* node) {
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

void HabanaLaunchOpPT::MarkProducerExternal(torch::jit::Value* val) {
  while (isControlEdge(val->node())) {
    val = val->node()->inputs().at(0);
  }
  if (isInGraphInputs(val) != -1) {
    PT_LAZY_DEBUG(
        "Not adding ",
        val->debugName(),
        " to extenal map since it is an input to the graph")
  } else {
    PT_LAZY_DEBUG("Adding ", val->debugName(), " to extenal map")
    valptr_to_external_map[val] = true;
  }
}

void HabanaLaunchOpPT::externalMarkingPass(
    torch::jit::graph_node_list graph_nodes) {
  for (auto* node : graph_nodes) {
    if (node->kind().is_prim()) {
      continue;
    }

    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
        0, node->schema().operator_name(), getNodeScalarType(node));
    if (HabanaKernel == nullptr)
      continue;

    // collective inputs must be set external in order to
    // trigger before graph execution ends
    if (habana_lazy::IsCollective(node->kind())) {
      set_external_input(node);
    }

  } // for (auto* node : graph_nodes)
} // function end

void HabanaLaunchOpPT::runMetaDataAdjustmentPasses(
    torch::jit::graph_node_list graph_nodes) {
  // This pass marks tensors persistent if they are nt persistent from graph
  // but are made persistent due to synapse limitations
  persistenceMarkingPass(graph_nodes);

  // This pass marks tensors external if they are used as input tensors for
  // collective ops. Used for Signal From Graph to signal the tensor data is
  // ready prior to recipe completion
  // TODO: SW-80913 enable for gaudi 2
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto device_type = device.type();
  if (device_type == synDeviceGaudi || device_type == synDeviceGaudiM) {
    externalMarkingPass(graph_nodes);
  }
}
