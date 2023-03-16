/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_eager/eager_exec.h"
#include <absl/strings/str_join.h>
#include <c10/util/hash.h>
#include <torch/csrc/jit/ir/ir.h>
#include <limits>
#include <memory>
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUStream.h"
#include "habana_eager/eager_view.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {
namespace eager {

torch::jit::Stack EagerExec::launch() {
  PT_EAGER_TRACE;
  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};
  synEventHandle event_handle{};
  synapse_helpers::hpuStream_t event_stream{0};
  bool event_flag{0};
  // auto& device = synapse_helpers::HPURegistrar::get_device();
  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size - out is 1, so size(inputs)
  stack.reserve(m_inputs.size());

  for (const auto& in : m_inputs) {
    stack.emplace_back(in);
  }
  auto orig_stack = stack;
  UniqueIdxVec parent_vec{find_duplicate_in_stack(stack)};
  PT_EAGER_DEBUG("Eager Op unique input vector ", parent_vec.to_string());

  prune_duplicate_stack_inputs(stack, parent_vec);

  auto& cache{OptimizedJitGraphCache::GetOptimizedJitCache()};
  size_t key{calculate_operator_key(parent_vec)};
  auto graph_and_meta{cache.GetOptimizedJITGraphAndMetaData(key)};
  if (graph_and_meta) {
    PT_EAGER_DEBUG("Eager Op JIT graph cache HIT for key ", key);
  } else {
    PT_EAGER_DEBUG("Eager Op JIT graph cache miss for key ", key);
    auto graph{create_eager_graph()};
    prune_duplicate_graph_inputs(parent_vec, graph);
    at::ArrayRef<torch::jit::IValue> input_refs =
        torch::jit::last(stack, graph->inputs().size());
    graph_and_meta = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
        graph,
        input_refs,
        0ull /*unique_cntr*/,
        std::vector<bool>{} /*node_bcast_map_*/);
    /*  auto graphIndex =
          GetGraphIndex(m_g_hash_, torch::jit::last(stack,
       mp_g_->inputs().size()));*/
    static int graphIndex{0};
    ++graphIndex;

    graph_and_meta->SetGraphIndex(graphIndex);
    graph_and_meta->SetOpName(m_symbol.toQualString());
    graph_and_meta->SetHPUStream(stream);
    graph_and_meta->SetEventHandle(event_handle);
    graph_and_meta->SetEventRecordStream(event_stream);
    graph_and_meta->SetEventFlag(event_flag);
    cache.Add(key, graph_and_meta);
  }

  try {
    habana::HabanaLaunchOpPT habana_launch_op_{graph_and_meta};
    habana_launch_op_.run(stack);
    return stack;
  } catch (const std::exception& e) {
    PT_EAGER_DEBUG("HabanaLaunchOpPT Run returned exception....\n", e.what());
    throw;
  }
}

std::shared_ptr<torch::jit::Graph> EagerExec::create_eager_graph() {
  PT_EAGER_TRACE;
  using JitValue = torch::jit::Value;
  auto graph = std::make_shared<torch::jit::Graph>();

  std::vector<JitValue*> args_vector;

  for (const auto& inp : m_inputs) {
    auto t = graph->addInput(inp.toString());
    t->setType(c10::TensorType::createContiguous(
        inp.scalar_type(), inp.device(), inp.sizes()));
    // TODO do we need debug names?
    // t->setDebugName(inp.toString());
    args_vector.push_back(t);
  }

  // Total inputs to a node is size of meta data + size of inputs
  // Allocate vector with nulllptr with inputs_size
  std::vector<JitValue*> node_inputs(
      args_vector.size() + m_metadata.size(), nullptr);

  // Iterate thru each of the metadata and create constant node and
  // assign this to correct index in the input array
  std::for_each(
      m_metadata.cbegin(), m_metadata.cend(), [&](const auto& meta_data) {
        node_inputs[meta_data.first] = graph->insertConstant(meta_data.second);
      });

  // Now we will fill the inputs in the array whereever its null
  size_t j = 0;
  std::for_each(node_inputs.begin(), node_inputs.end(), [&](auto& node) {
    if (nullptr == node) {
      node = args_vector[j++];
    }
  });
  HABANA_ASSERT(j == args_vector.size()); // make sure all the inputs were used

  // TODO Do we need scopes for single-node JIT graphs?
  //   std::shared_ptr<torch::jit::WithCurrentScope> scope_context;
  //       auto scope_name = node->GetModuleName().empty()
  //           ? (node->GetScope() ? *node->GetScope() : "")
  //           : node->GetModuleName();
  //       if (AccThread::IsAccThreadEnabled() ?
  //       !node->GetModuleName().empty()
  //                                           : node->GetScope() != NULL) {
  //         scope_context = std::make_shared<torch::jit::WithCurrentScope>(
  //             *mp_g_,
  //             c10::make_intrusive<torch::jit::Scope>(
  //                 torch::jit::ScopePtr(),
  //                 c10::Symbol::fromQualString("debug::" + scope_name)));
  //       }

  at::ArrayRef<JitValue*> args(node_inputs);
  auto jit_node = graph->create(m_symbol, args, m_outputs.size());
  // TODO scope
  //   if (AccThread::IsAccThreadEnabled()) {
  //     jit_node->setScope(c10::make_intrusive<torch::jit::Scope>(
  //         torch::jit::ScopePtr(),
  //         c10::Symbol::fromQualString("debug::" + scope_name)));
  //   }
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto one = torch::jit::attr::alpha;
    /*Need to set this node if the deterministic mode is ON*/
    auto& device = synapse_helpers::HPURegistrar::get_device();
    jit_node->i_(one, device.getDeterministic());
    PT_BRIDGE_DEBUG(
        "Deterministic val during Jit Node creation: ", jit_node->i(one));
  }

  graph->insertNode(jit_node);

  // TODO Do we need special handling for prim::ListConstruct?
  //   if (c10::Symbol::fromQualString("prim::ListConstruct") == node->op() ||
  //       node->is_output_tensor_list()) {
  //     auto* list_node = dynamic_cast<ir::ListConstruct*>(node.get());
  //     if (list_node && list_node->isOptional()) {
  //       jit_node->output()->setType(
  //           torch::jit::ListType::create(torch::jit::OptionalType::ofTensor()));
  //     } else {
  //       jit_node->output()->setType(torch::jit::ListType::ofTensors());
  //     }
  //   } else {
  for (size_t idx = 0; idx < jit_node->outputs().size(); idx++) {
    auto jit_value_out = jit_node->output(idx);
    if (jit_node->output(idx)->type()->kind() == c10::TypeKind::TensorType) {
      const auto& out_val = m_outputs.at(idx);

      jit_value_out->setType(c10::TensorType::createContiguous(
          out_val.scalar_type, out_val.device, out_val.sizes));
      // TODO do we need debug names?
      // jit_value_out->setDebugName(irout_val.ToString());
    }
    graph->registerOutput(jit_value_out);
  }

  post_process_eager_graph(graph);

  return graph;
  // TODO This is part of Create/ConstructJITGraph in HLExec. Do we need it?
  // Optimize(stack);
  // PruneDuplicateGraphInputs(parent_vec, is_duplicate_vec);
}

size_t EagerExec::calculate_operator_key(const UniqueIdxVec& parent_vec) {
  PT_EAGER_TRACE;
  size_t optimized_key = static_cast<uint32_t>(m_symbol);
  optimized_key = at::hash_combine(optimized_key, m_outputs.size());
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    optimized_key = at::hash_combine(optimized_key, device.getDeterministic());
  }
  std::unordered_set<size_t> input_hash_values;
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    optimized_key = at::hash_combine(optimized_key, parent_vec[i]);
    optimized_key = at::hash_combine(optimized_key, i);
    const at::IValue& input = m_inputs[i];
    // Create stack based on input tensors / tensor lists.
    // Metadata and scalars are part of key calculation, so we skip them.
    if (m_metadata.find(i) != m_metadata.end()) {
      if (input.isList()) {
        for (auto& v : input.toListRef()) {
          optimized_key = at::hash_combine(optimized_key, at::IValue::hash(v));
        }
      } else {
        optimized_key =
            at::hash_combine(optimized_key, at::IValue::hash(input));
      }
      continue;
    } else if (input.isScalar()) {
      optimized_key =
          at::hash_combine(optimized_key, at::IValue::hash(input.toScalar()));
      continue;
    }

    if (input.isTensor()) {
      const at::Tensor& t = input.toTensor();
      if (t.defined()) {
        if (t.device().type() != c10::DeviceType::HPU) {
          // non HPU tensors to be handled later
          optimized_key = 0;
          break;
        }
        // Calculate hash based on unique tensor inputs.
        size_t input_hash_val = at::IValue::hash(input);
        if (input_hash_values.count(input_hash_val)) {
          continue;
        }
        input_hash_values.emplace(input_hash_val);
        update_key_for_tensor(t, optimized_key);
        if (optimized_key == 0) {
          break;
        }
      } else {
        optimized_key = at::hash_combine(
            optimized_key, at::IValue::hash(torch::jit::IValue()));
      }
    } else if (input.isTensorList()) {
      // Not handled so returning null key
      optimized_key = 0;
      break;
    }
  }

  return optimized_key;
}

void EagerExec::update_key_for_tensor(const at::Tensor& t, size_t& key) {
  key = at::hash_combine(key, static_cast<size_t>(t.scalar_type()));
  for (auto s : t.strides())
    key = at::hash_combine(key, s);
  key = at::hash_combine(key, static_cast<size_t>(t.storage_offset()));
  key = at::hash_combine(key, static_cast<size_t>(t.suggest_memory_format()));
  key = at::hash_combine(key, static_cast<size_t>(t.layout()));
  key = at::hash_combine(key, t.dim());
}

UniqueIdxVec EagerExec::find_duplicate_in_stack(torch::jit::Stack& stack) {
  size_t num_inputs = m_inputs.size();
  UniqueIdxVec parent_vec{num_inputs};

  // Assumption : stack[i] is the corresponding input of po_data.inputs[i]
  TORCH_CHECK(
      stack.size() == num_inputs,
      " stack_size ",
      stack.size(),
      " != num_inputs ",
      num_inputs);

  size_t stack_size = stack.size();
  std::unordered_map<uint64_t, size_t> input_addr_map;
  input_addr_map.reserve(stack_size);
  size_t num_duplicate_inputs = 0;

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    TORCH_CHECK(input.isTensor());
    if (!input.toTensor().has_storage()) {
      return parent_vec;
    }
  }

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    TORCH_CHECK(input.isTensor());
    auto input_addr = (uint64_t)(input.toTensor().data_ptr());

    if (input_addr == 0) {
      // input_addr == 0 not considered for duplicate removal since this address
      // is used for ZST tensors. 2 different ZST tensors can both have addr = 0
      // and removing one of them results in cycles in synapse graph in some
      // cases
      input_addr_map[input_addr] = i;
      continue;
    }
    if (input_addr_map.find(input_addr) == input_addr_map.end()) {
      // unique input
      input_addr_map[input_addr] = i;
      continue;
    }
    auto pidx = input_addr_map.at(input_addr);
    auto parent_tensor = stack[pidx].toTensor();
    auto input_tensor = input.toTensor();
    // Check for shape and stride match
    if (input_tensor.sizes() == parent_tensor.sizes() &&
        input_tensor.strides() == parent_tensor.strides()) {
      parent_vec[i] = pidx;
      num_duplicate_inputs++;

      PT_EAGER_DEBUG(
          "Duplicate input address ",
          input_addr,
          " found for value %",
          m_inputs[i].toString(),
          " current duplicate count ",
          num_duplicate_inputs);
    } else {
      PT_EAGER_DEBUG(
          "Same input address ",
          input_addr,
          " with different shape/stride found for value %",
          m_inputs[i].toString(),
          " and value%",
          m_inputs[pidx].toString());
    }
  }
  return parent_vec;
}

/*
 * Prune duplicate stack inputs
 */
void EagerExec::prune_duplicate_stack_inputs(
    torch::jit::Stack& stack,
    const UniqueIdxVec& parent_vec) {
  for (int64_t j = (int64_t)parent_vec.size() - 1; j >= 0; j--) {
    if (parent_vec.is_duplicate(j)) {
      PT_EAGER_DEBUG("Deleting ", j, "th entry from the stack");
      stack.erase(stack.begin() + j);
    }
  }
}

void EagerExec::prune_duplicate_graph_inputs(
    const UniqueIdxVec& parent_vec,
    std::shared_ptr<torch::jit::Graph>& graph) {
  PT_EAGER_TRACE;

  auto jit_ir_graph_inputs = graph->inputs();
  bool is_pruned{false};
  for (size_t i = 0; i < jit_ir_graph_inputs.size(); i++) {
    if (parent_vec.is_duplicate(i)) {
      size_t parent_idx = parent_vec[i];
      TORCH_CHECK(
          parent_idx != ULONG_MAX && parent_idx < i,
          " invalid parent index ",
          parent_idx,
          " found for input index ",
          i);
      auto vptr = jit_ir_graph_inputs[parent_idx];
      PT_EAGER_DEBUG(
          "Replacing %",
          jit_ir_graph_inputs[i]->debugName(),
          " with %",
          vptr->debugName());
      jit_ir_graph_inputs[i]->replaceAllUsesWith(vptr);
    }
  }

  for (int64_t j = (int64_t)parent_vec.size() - 1; j >= 0; j--) {
    if (parent_vec.is_duplicate(j)) {
      is_pruned = true;
      PT_EAGER_DEBUG(
          "Deleting ",
          j,
          "th input %",
          graph->inputs().at(j)->debugName(),
          "of the graph");
      graph->eraseInput(j);
    }
  }

  if (is_pruned) {
    PT_EAGER_DEBUG(
        "After pruning duplicates, JIT IR Graph ====\n",
        graph->toString(),
        "JIT IR Graph ----\n");
  }
}

std::string UniqueIdxVec::to_string() const {
  struct Formatter {
    void operator()(std::string* out, size_t i) const {
      out->append((i == UNIQUE_ID) ? "U" : std::to_string(i));
    }
  };
  return absl::StrCat("{", absl::StrJoin(idx_, ",", Formatter()), "}");
}

void EagerExec::post_process_eager_graph(std::shared_ptr<JitGraph>& graph) {
  PT_EAGER_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING)) {
    PT_BRIDGE_DEBUG("[Eager] Apply I/O View Handling pass.");
    HandleInputOutputViews(graph, m_inputs, m_eager_op_meta_data);
  }
}

} // namespace eager
} // namespace habana
