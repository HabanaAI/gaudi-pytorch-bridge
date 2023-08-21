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
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/eager_view.h"
#include "habana_eager/ops/eager_op.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {
namespace eager {

namespace {
bool is_metadata_candidate(const at::IValue& input) {
  return input.isBool() || input.isDevice() || input.isIntList() ||
      input.isDoubleList() || input.isBoolList() || input.isString() ||
      input.isNone() ||
      (input.isList() && !input.toList().elementType()->cast<at::TensorType>());
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

enum class ProcessList { asTensor, asList };

template <ProcessList process_list = ProcessList::asList, class T>
void traversing_inputs(const std::vector<at::IValue>& inputs, T&& visitor) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    const at::IValue& input = inputs[i];
    if (is_metadata_candidate(input)) {
      visitor(input);
    } else if (input.isScalar()) {
      visitor(input.toScalar());
    } else if (input.isTensor()) {
      const at::Tensor& t = input.toTensor();
      if (t.defined()) {
        visitor(t);
      } else {
        visitor(torch::jit::IValue());
      }
    } else if (input.isList()) {
      const auto& list = input.toListRef();
      for (const auto& li : list) {
        HABANA_ASSERT(
            li.isTensor(),
            "Got unhandled list item type: ",
            li.tagKind(),
            " at index ",
            i,
            ".");
        if constexpr (process_list == ProcessList::asTensor)
          visitor(li.toTensor());
      }
      if constexpr (process_list == ProcessList::asList)
        visitor(list);
    } else {
      PT_BRIDGE_FATAL("Got unhandled type: ", input.tagKind(), " at index ", i);
      HABANA_ASSERT(0);
    }
  }
}
} // namespace

size_t OutputSpecsOrTensors::size() {
  return std::visit(
      overloaded{
          [](std::vector<OutputSpec>& specs) { return specs.size(); },
          [](std::vector<at::Tensor>& tensors) { return tensors.size(); }},
      m_outputs);
}

c10::TensorTypePtr OutputSpecsOrTensors::get_tensor_type(size_t indx) {
  return std::visit(
      overloaded{
          [indx](std::vector<OutputSpec>& specs) {
            const auto& out_val = specs.at(indx);
            return c10::TensorType::createContiguous(
                out_val.scalar_type, out_val.device, out_val.sizes);
          },
          [indx](std::vector<at::Tensor>& tensors) {
            const auto& out_val = tensors.at(indx);
            return c10::TensorType::createContiguous(
                out_val.scalar_type(), out_val.device(), out_val.sizes());
          }},
      m_outputs);
}

std::optional<std::vector<at::Tensor>> OutputSpecsOrTensors::get_tensors() {
  return std::holds_alternative<std::vector<at::Tensor>>(m_outputs)
      ? std::optional<std::vector<at::Tensor>>{std::get<
            std::vector<at::Tensor>>(m_outputs)}
      : std::nullopt;
}

std::vector<std::vector<int64_t>> OutputSpecsOrTensors::get_shapes() {
  std::vector<std::vector<int64_t>> shapes;
  std::visit(
      overloaded{
          [&](std::vector<OutputSpec>& specs) {
            std::transform(
                specs.begin(),
                specs.end(),
                std::back_inserter(shapes),
                [](OutputSpec& spec) -> std::vector<int64_t> {
                  return spec.sizes;
                });
          },
          [&](std::vector<at::Tensor>& tensors) {
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(shapes),
                [](at::Tensor& tensor) -> std::vector<int64_t> {
                  return tensor.sizes().vec();
                });
          }},
      m_outputs);
  return shapes;
}

std::vector<at::IValue> convert_inputs_to_backend_tensors(
    std::vector<at::IValue>& inputs) {
  std::vector<at::IValue> stack;
  stack.reserve(inputs.size());

  traversing_inputs<ProcessList::asList>(
      inputs,
      overloaded{
          // metadata
          [&stack](const torch::jit::IValue& v) { stack.push_back(v); },
          // scalars
          [&stack](const at::Scalar& s) { stack.push_back(s); },
          // tensors
          [&stack](const at::Tensor& t) {
            if (t.device().type() == c10::DeviceType::HPU) {
              stack.push_back(
                  HbEagerTensorPool::getInstance().get_backend_tensor(t));
              return;
            }

            if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
              stack.push_back(t);
              return;
            }

            HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
          },
          [&stack](const c10::ArrayRef<torch::jit::IValue>& list) {
            c10::List<at::Tensor> l;
            l.reserve(list.size());
            for (auto& v : list) {
              HABANA_ASSERT(v.isTensor())
              auto& t = v.toTensor();
              HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
              l.push_back(
                  HbEagerTensorPool::getInstance().get_backend_tensor(t));
            }

            stack.push_back(l);
          }});
  return stack;
}

std::vector<at::IValue> convert_cpu_wrapped_numbers(
    const std::vector<at::IValue>& inputs) {
  auto& scalar_cache = HPURegistrar::get_device().GetScalarCache();
  auto stack = inputs;
  for (size_t i = 0; i < stack.size(); i++) {
    auto& value = stack[i];
    if (!value.isTensor()) {
      continue;
    }

    auto t = value.toTensor();
    if (!t.defined()) {
      continue;
    }

    if (t.device().type() == c10::DeviceType::HPU) {
      continue;
    }

    HABANA_ASSERT(
        (t.unsafeGetTensorImpl()->is_wrapped_number()),
        "Unexpected CPU tensor");

    stack[i] = scalar_cache.GetTensor(t.item());
  }

  // Copy wrapped number tensors to HPU
  scalar_cache.CopyScalarsToDevice();
  return stack;
}

torch::jit::Stack EagerExec::launch() {
  PT_EAGER_TRACE_WITH_NAME(m_graph_name);
  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};
  synEventHandle event_handle{};
  synapse_helpers::hpuStream_t event_stream{0};
  bool event_flag{0};
  // auto& device = HPURegistrar::get_device();

  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size - out is 1, so size(inputs)
  auto stack = convert_cpu_wrapped_numbers(m_inputs);
  auto orig_inputs = stack;
  stack = prepare_input_stack(stack);

  UniqueIdxVec parent_vec{find_duplicate_in_stack(stack)};
  PT_EAGER_DEBUG("Eager Op unique input vector ", parent_vec.to_string());

  prune_duplicate_stack_inputs(stack, parent_vec);

  mark_maybe_grad_view();

  auto& cache{OptimizedJitGraphCache::GetOptimizedJitCache()};
  size_t key{calculate_operator_key(parent_vec, orig_inputs)};
  auto graph_and_meta{cache.GetOptimizedJITGraphAndMetaData(key)};
  if (graph_and_meta) {
    PT_EAGER_DEBUG("Eager Op JIT graph cache HIT for key ", key);
    for (const auto& val : stack) {
      if (!val.isTensor()) {
        continue;
      }

      // If an input tensor is not contiguous view handling JIT IR pass
      // would have modified the input tensor to base tensor. Need to
      // perform this operation for the cache hit case as well
      auto in = val.toTensor();
      auto input_smeta{habana::get_storage_extra_meta(in)};

      if (habana::is_view_lowering(in) || !in.is_contiguous()) {
        // modify the backend tensor of the view as the base
        auto impl = in.unsafeGetTensorImpl();
        impl->set_sizes_contiguous(habana::get_base_tensor_size(in));
        impl->set_storage_offset(0);
        PT_EAGER_DEBUG(
            "Eager op: Input tensor converted to base for the cache hit case");
      }
    }
  } else {
    PT_EAGER_DEBUG("Eager Op JIT graph cache miss for key ", key);
    auto graph{create_eager_graph(orig_inputs)};
    auto eager_compiler_supported =
        is_eager_compiler_supported_for_graph(graph);
    post_process_eager_graph(graph, eager_compiler_supported);
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
    graph_and_meta->SetOpName(m_graph_name);
    graph_and_meta->SetHPUStream(stream);
    graph_and_meta->SetFrontendType(habana_helpers::HabanaFrontendTypes::EAGER);
    graph_and_meta->set_is_eager_compiler_supported(eager_compiler_supported);
    graph_and_meta->set_is_shape_agnostic_supported(eager_compiler_supported);
    cache.Add(key, graph_and_meta);
  }
  graph_and_meta->set_output_shapes(m_outputs.get_shapes());

  try {
    habana::HabanaLaunchOpPT habana_launch_op_{graph_and_meta};
    habana_launch_op_.run(stack, m_outputs.get_tensors());
    return stack;
  } catch (const std::exception& e) {
    PT_EAGER_DEBUG("HabanaLaunchOpPT Run returned exception....\n", e.what());
    throw;
  }
}

std::shared_ptr<torch::jit::Graph> EagerExec::create_eager_graph(
    torch::jit::Stack& stack) {
  PT_EAGER_TRACE;
  using JitValue = torch::jit::Value;
  auto graph = std::make_shared<torch::jit::Graph>();
  std::vector<JitValue*> node_inputs;

  size_t idx = 0;
  traversing_inputs(
      stack,
      overloaded{
          // metadata
          [&node_inputs, &graph, &idx](const torch::jit::IValue& c) {
            node_inputs.push_back(graph->insertConstant(c));
          },
          // scalar inputs
          [&node_inputs, &graph, &idx](const at::Scalar& c) {
            auto s = graph->addInput("s" + std::to_string(++idx));
            auto scalarType = c.type();
            if (isFloatingType(scalarType)) {
              s->setType(c10::FloatType::get());
            } else if (isIntegralType(scalarType, false)) {
              s->setType(c10::IntType::get());
            } else if (isIntegralType(scalarType, true)) {
              s->setType(c10::BoolType::get());
            } else {
              HABANA_ASSERT(0, "Unknown scalar type");
            }
            node_inputs.push_back(s);
          },
          // tensor inputs
          [&node_inputs, &graph](const at::Tensor& tensor) {
            auto t = graph->addInput(tensor.toString());
            t->setType(c10::TensorType::createContiguous(
                tensor.scalar_type(), tensor.device(), tensor.sizes()));
            node_inputs.push_back(t);
          },
          // list tensors input
          [&node_inputs,
           &graph](const c10::ArrayRef<torch::jit::IValue>& list) {
            std::vector<JitValue*> list_inp_args;
            for (int i = 0; i < list.size(); ++i) {
              auto tensor = list[i].toTensor();
              auto t = graph->addInput(tensor.toString());
              t->setType(c10::TensorType::createContiguous(
                  tensor.scalar_type(), tensor.device(), tensor.sizes()));
              list_inp_args.push_back(t);
            }
            auto jit_node = graph->create(
                c10::Symbol::fromQualString("prim::ListConstruct"),
                list_inp_args,
                1);
            // Do we need to handle Optional ?
            jit_node->output()->setType(torch::jit::ListType::ofTensors());
            graph->insertNode(jit_node);
            node_inputs.push_back(jit_node->output(0));
          }});

  auto jit_node = graph->create(m_symbol, node_inputs, m_outputs.size());

  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto one = torch::jit::attr::alpha;
    /*Need to set this node if the deterministic mode is ON*/
    auto& gconfig = HPURegistrar::get_hpu_global_config();
    jit_node->i_(one, gconfig.getDeterministic());
    PT_BRIDGE_DEBUG(
        "Deterministic val during Jit Node creation: ", jit_node->i(one));
  }
  graph->insertNode(jit_node);

  for (size_t idx = 0; idx < jit_node->outputs().size(); idx++) {
    auto jit_value_out = jit_node->output(idx);
    if (jit_node->output(idx)->type()->kind() == c10::TypeKind::TensorType) {
      jit_value_out->setType(m_outputs.get_tensor_type(idx));
      // TODO do we need debug names?
      // jit_value_out->setDebugName(irout_val.ToString());
    }
    graph->registerOutput(jit_value_out);
  }

  return graph;
}

size_t EagerExec::calculate_operator_key(
    const UniqueIdxVec& parent_vec,
    torch::jit::Stack& stack) {
  PT_EAGER_TRACE;
  size_t optimized_key = static_cast<uint32_t>(m_symbol);
  optimized_key = at::hash_combine(optimized_key, m_outputs.size());
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto& gconfig = HPURegistrar::get_hpu_global_config();
    optimized_key = at::hash_combine(optimized_key, gconfig.getDeterministic());
  }
  for (size_t i = 0; i < parent_vec.size(); ++i)
    optimized_key = at::hash_combine(optimized_key, parent_vec[i]);

  std::unordered_set<size_t> input_hash_values;
  int inp_index = 0;
  traversing_inputs<ProcessList::asTensor>(
      stack,
      overloaded{
          [this, &optimized_key, &inp_index](const torch::jit::IValue& input) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);
            if (input.isList()) {
              for (auto& v : input.toListRef()) {
                optimized_key =
                    at::hash_combine(optimized_key, at::IValue::hash(v));
              }
            } else {
              // at::IValue::hash of None is zero, same as for zero scalar,
              // in order to distinguish None and Zero scalar we ignore None
              if (!input.isNone()) {
                optimized_key =
                    at::hash_combine(optimized_key, at::IValue::hash(input));
              }
            }
          },
          [this, &optimized_key, &inp_index](const at::Scalar& input) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);

            // TODO: remove from hash
            optimized_key =
                at::hash_combine(optimized_key, at::IValue::hash(input));

            optimized_key =
                at::hash_combine(optimized_key, at::IValue::hash(input.type()));
          },
          [this, &optimized_key, &input_hash_values, &inp_index](
              const at::Tensor& tensor) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);
            size_t input_hash_val = c10::get_hash(tensor.unsafeGetTensorImpl());
            if (input_hash_values.count(input_hash_val) == 0) {
              input_hash_values.emplace(input_hash_val);
              update_key_for_tensor(tensor, optimized_key);
            }
          }});
  return optimized_key;
}

void EagerExec::update_key_for_tensor(const at::Tensor& t, size_t& key) {
  key = at::hash_combine(key, static_cast<size_t>(t.scalar_type()));

  // hash view attribute
  auto input_smeta{habana::get_storage_extra_meta(t)};
  auto input_tmeta{habana::get_tensor_extra_meta(t)};
  key = at::hash_combine(key, static_cast<size_t>(habana::is_view_lowering(t)));
  key = at::hash_combine(key, static_cast<size_t>(t.is_contiguous()));
  key =
      at::hash_combine(key, static_cast<size_t>(input_tmeta->is_view_tensor()));

  // for views - base tensor size used in JIT IR pass varies w.r.t. permutation
  // for views as well as non views - we need to incorporate permute information
  // of inputs in the key so that no need to set and get the permute information
  // from bridge to synapse during the cache hit. during cache miss case bridge
  // needs to set the permute information for the inputs while need to read the
  // permute information of the outputs.
  if (input_smeta) {
    for (auto s : input_smeta->get_memory_permutation()) {
      key = at::hash_combine(key, s);
    }
  }

  if (habana::is_view_lowering(t) || !t.is_contiguous()) {
    auto base_smeta{habana::get_storage_base_meta(t)};
    for (auto s : base_smeta->get_memory_permutation()) {
      key = at::hash_combine(key, s);
    }

    // TODO: remove the below code block once the node params are patched.
    for (auto s : t.strides())
      key = at::hash_combine(key, s);
    // two different sized tensors can have same strides. ex: [2, 4, 1], and
    // [2, 1, 4]
    for (auto s : t.sizes())
      key = at::hash_combine(key, s);

    key = at::hash_combine(key, static_cast<size_t>(t.storage_offset()));
  }

  key = at::hash_combine(key, static_cast<size_t>(t.suggest_memory_format()));
  key = at::hash_combine(key, static_cast<size_t>(t.layout()));
  key = at::hash_combine(key, t.dim());
}

UniqueIdxVec EagerExec::find_duplicate_in_stack(torch::jit::Stack& stack) {
  size_t stack_size = stack.size();
  UniqueIdxVec parent_vec{stack_size};

  std::unordered_map<uint64_t, size_t> input_addr_map;
  input_addr_map.reserve(stack_size);
  size_t num_duplicate_inputs = 0;

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    if (!input.isTensor()) {
      continue;
    }

    TORCH_CHECK(input.isTensor());
    if (!input.toTensor().has_storage()) {
      return parent_vec;
    }
  }

  for (size_t i = 0; i < stack_size; i++) {
    auto& input = stack[i];
    if (!input.isTensor()) {
      continue;
    }

    TORCH_CHECK(input.isTensor());
    auto input_addr = (uint64_t)(input.toTensor().data_ptr());

    if (input_addr == 0) {
      // input_addr == 0 not considered for duplicate removal since this
      // address is used for ZST tensors. 2 different ZST tensors can both
      // have addr = 0 and removing one of them results in cycles in synapse
      // graph in some cases
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
          input_tensor.toString(),
          " current duplicate count ",
          num_duplicate_inputs);
    } else {
      PT_EAGER_DEBUG(
          "Same input address ",
          input_addr,
          " with different shape/stride found for value %",
          input_tensor.toString(),
          " and value%",
          stack[pidx].toTensor().toString());
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

torch::jit::Stack EagerExec::prepare_input_stack(
    const torch::jit::Stack& inputs) {
  torch::jit::Stack stack;
  stack.reserve(stack.size());
  traversing_inputs<ProcessList::asTensor>(
      inputs,
      overloaded{// metadata
                 [&stack](const torch::jit::IValue& v) {},
                 // scalars
                 [&stack](const at::Scalar& s) { stack.push_back(s); },
                 // tensors
                 [&stack](const at::Tensor& t) { stack.push_back(t); }});

  return stack;
}

std::string UniqueIdxVec::to_string() const {
  struct Formatter {
    void operator()(std::string* out, size_t i) const {
      out->append((i == UNIQUE_ID) ? "U" : std::to_string(i));
    }
  };
  return absl::StrCat("{", absl::StrJoin(idx_, ",", Formatter()), "}");
}

void EagerExec::set_eager_op_info(EagerOpMetaData&& eager_op_meta_data) {
  PT_EAGER_TRACE;

  m_eager_op_meta_data = eager_op_meta_data;
}

void EagerExec::post_process_eager_graph(
    std::shared_ptr<JitGraph>& graph,
    bool eager_compiler_supported) {
  PT_EAGER_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING)) {
    PT_EAGER_DEBUG("Apply I/O View Handling pass.");
    HandleInputOutputViews(
        graph, m_inputs, m_eager_op_meta_data, eager_compiler_supported);
  }
}

bool EagerExec::is_eager_compiler_supported_for_graph(
    std::shared_ptr<JitGraph>& graph) {
  if (habana::HPURegistrar::get_device().type() == synDeviceGaudi) {
    return false;
  }
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_COMPILER)) {
    return false;
  }
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    if (std::string((*it)->kind().toQualString()).find("hpu::optimizer") !=
        std::string::npos) {
      return false;
    }
  }
  return true;
}

/*
Enabling permutations on view outputs is risky. The below code performs pattern
matching to enable it conditionally for grad views on a all reduce bucket. Fork
reference: pytorch-fork/torch/csrc/distributed/c10d/reducer.cpp Pattern: The
tensor marked should be a out tensor belonging to mul.out kernel variant and is
a contiguous view on a 1D buffer
*/
void EagerExec::mark_maybe_grad_view() {
  if (!GET_ENV_FLAG_NEW(PT_HPU_EAGER_ENABLE_GRADIENT_VIEW_LAYOUT_OPT))
    return;
  if (std::string(m_symbol.toQualString()) != "aten::mul")
    return;
  if (m_eager_op_meta_data.op_kind_ != InplaceOut)
    return;
  if (!m_inputs.back().isTensor())
    return;
  auto& t = m_inputs.back().toTensor();
  if (!t.is_contiguous())
    return;
  auto tmeta{habana::get_tensor_extra_meta(t)};
  if (!tmeta->is_view_tensor())
    return;
  if (habana::get_base_tensor_size(t).size() != 1)
    return;
  // setting this flag will allow permutations on the view output
  tmeta->set_maybe_grad_view();
  PT_EAGER_DEBUG(
      "Marked grad view. size: ", t.sizes(), " offset ", t.storage_offset());
}

} // namespace eager
} // namespace habana
