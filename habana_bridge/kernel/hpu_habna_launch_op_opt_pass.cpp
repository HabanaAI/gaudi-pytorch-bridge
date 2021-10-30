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
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch::jit;
using namespace habana;

void HabanaLaunchOpPT::markLayoutForOriginNodes(torch::jit::Value* val) {
  auto node = val->node();
  if (node->kind().is_prim()) {
    return;
  }

  auto node_ins = node->inputs();

  // Get kernel context
  habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
      0, node->schema().operator_name(), getNodeScalarType(node));
  // If we dont get a valid kernel, it means the node may be pointing to a
  // subgraph or something we dont need to propagate in such cases, return
  if (HabanaKernel == nullptr) {
    return;
  }
  // Get the metadata for all inputs, used for preprocessing inputs
  auto& habana_kernel_meta_data = HabanaKernel->GetKernelMetaData();
  // We only need to mark for cases where the output layout is derived from
  // inputs else its kernel would have already taken care of layouts if its
  // already aware
  // TODO: need to check logic for kernels with multiple outputs
  if (habana_kernel_meta_data.output_layout.size() == 0 ||
      habana_kernel_meta_data.output_layout[0] != habana::LayoutFormat::ANY)
    return;
  int in_meta_size = habana_kernel_meta_data.input_layout.size();
  int tensor_idx = 0;
  for (const auto value_in : node_ins) {
    if (value_in->type()->kind() == c10::TypeKind::TensorType) {
      if (tensor_idx >= in_meta_size ||
          habana_kernel_meta_data.input_layout[tensor_idx] ==
              habana::LayoutFormat::ANY) {
        if (value_to_tensor_layout[value_in].layout !=
            habana::LayoutFormat::HWCK) {
          value_to_tensor_layout[value_in].layout = habana::LayoutFormat::HWCK;
          markLayoutForOriginNodes(value_in);
        }
      }
      tensor_idx++;
    }
  }
}

// This is a pass which  marks all the HWCK tensors wherever it
// can get info via the kenrel metadata
// From there it backtracks the nodes that generated this input for it and marks
// their inputs too if required
void HabanaLaunchOpPT::weightLayoutMarkingPass(
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
    // Get the metadata for all inputs, used for preprocessing inputs
    auto& habana_kernel_meta_data = HabanaKernel->GetKernelMetaData();
    auto node_ins = node->inputs();
    size_t tensor_idx = 0;
    for (const auto value_in : node_ins) {
      if (value_in->type()->kind() == c10::TypeKind::TensorType) {
        // If we find a kernel depicting HWCK usage, we mark the tensor
        // Also, we go back to the origins of this tensor and mark them too
        if (tensor_idx < habana_kernel_meta_data.input_layout.size() &&
            habana_kernel_meta_data.input_layout.at(tensor_idx) ==
                habana::LayoutFormat::HWCK) {
          value_to_tensor_layout[value_in].layout = habana::LayoutFormat::HWCK;
          markLayoutForOriginNodes(value_in);
        }
        tensor_idx++;
      }
    }
  }
}

void HabanaLaunchOpPT::set_persistence_input(torch::jit::Node* node) {
  auto val = node->input(0);

  if (val->type()->kind() == c10::TypeKind::TensorType) {
    valptr_to_persistent_map[val] = true;
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
    if (isControlEdge(node) || isInplace(node)) {
      set_persistence_input(node);
      set_persistence_output(node);
    }
  } // for (auto* node : graph_nodes)
} // function end

void HabanaLaunchOpPT::runMetaDataAdjustmentPasses(
    torch::jit::graph_node_list graph_nodes) {
  // This is a pass to gather meta data the tensors attached to the graph as
  // inputs. Right now all weight tensors are permuted in script to HWCK and
  // are "invisible to Pytorch" as it doesnt support this format and cannot be
  // represented in Aten tensor
  // So weight tensors are assumed to be in HWCK even if we see them as "NCHW"
  // or "NHWC" at at::tensor level This is ok for cases where kernels can tell
  // us if an input tensor is weight type, we can mark the correct layout and
  // use. But for cases where first use is in generic elementwise ops like
  // mul, we cannot know its a weight tensor for that op and may introduce
  // unwanted permutes. This pass enables us to go through the whole graph and
  // mark all tensors before we start lowering, so that such cases can be
  // avoided
  weightLayoutMarkingPass(graph_nodes);

  // This pass marks tensors persistent if they are nt persistent from graph
  // but are made persistent due to synapse limitations
  persistenceMarkingPass(graph_nodes);
}
