/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/scalar_cache.h"
#include "habana_eager/eager_view.h"
#include "habana_eager/ops/eager_op.h"
#include "passes/handle_views_insert_permute.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/visualize/visualize.h"

namespace habana {
namespace eager {

namespace {

using namespace std::literals;

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
void traversing_ivalues(const std::vector<at::IValue>& ivalues, T&& visitor) {
  for (size_t i = 0; i < ivalues.size(); ++i) {
    const at::IValue& ivalue = ivalues[i];
    if (is_metadata_candidate(ivalue)) {
      visitor(ivalue);
    } else if (ivalue.isScalar()) {
      visitor(ivalue.toScalar());
    } else if (ivalue.isTensor()) {
      const at::Tensor& t = ivalue.toTensor();
      if (t.defined()) {
        visitor(t);
      } else {
        visitor(torch::jit::IValue());
      }
    } else if (ivalue.isList()) {
      const auto& list = ivalue.toListRef();
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
    } else if (ivalue.isTuple()) {
      const auto& tuple = ivalue.toTupleRef();
      if (tuple.size() == 0) {
        continue;
      }
      PT_BRIDGE_FATAL("Tuple not supportd at index ", i);
      HABANA_ASSERT(0);
    } else {
      PT_BRIDGE_FATAL(
          "Got unhandled type: ", ivalue.tagKind(), " at index ", i);
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

std::variant<std::vector<OutputSpec>, std::vector<at::Tensor>>&
OutputSpecsOrTensors::get_outputs() {
  return m_outputs;
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

std::vector<at::IValue> convert_ivalues_to_backend_tensors(
    std::vector<at::IValue>& ivalues,
    std::optional<at::Symbol> symbol) {
  std::vector<at::IValue> stack;
  stack.reserve(ivalues.size());
  traversing_ivalues<ProcessList::asList>(
      ivalues,
      overloaded{
          // metadata
          [&stack](const torch::jit::IValue& v) { stack.push_back(v); },
          // scalars
          [&stack](const at::Scalar& s) { stack.push_back(s); },
          // tensors
          [&stack, &symbol](const at::Tensor& t) {
            if (t.device().type() == c10::DeviceType::HPU) {
              if (habana::get_tensor_extra_meta(t)->is_shape_tensor()) {
                stack.push_back(t);
              } else {
                stack.push_back(HbEagerTensorPool::get_backend_tensor(t));
              }
              return;
            }

            if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
              stack.push_back(t);
              return;
            }

            // In eager flow the symbol has value
            if (symbol.has_value()) {
              std::string qualstring(symbol.value().toQualString());
              std::string maskedFillPrefix = "aten::masked_fill";
              /* The 3rd input for the masked fill, if placed on the CPU, should
               * be converted to Scalar. This is an exception for the
               * masked_fill operation. Pytorch accepts the 3rd input on the CPU
               * when the operation is performed on cuda/xpu */
              if (stack.size() == 2 &&
                  std::equal(
                      std::begin(maskedFillPrefix),
                      std::end(maskedFillPrefix),
                      std::begin(qualstring))) {
                stack.push_back(t.item());
                return;
              }
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
              l.push_back(HbEagerTensorPool::get_backend_tensor(t));
            }

            stack.push_back(l);
          }});
  return stack;
}

std::vector<at::IValue> convert_cpu_wrapped_numbers(
    const std::vector<at::IValue>& inputs) {
  auto& scalar_cache = HPURegistrar::get_device().get_scalar_cache();
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

void process_node_params(
    const CValPtrMap& jit_val_map,
    const std::vector<at::IValue>& inputs,
    CValPtrtoIValueMap& jit_val_to_ivalue_map) {
  for (const auto& v : jit_val_map) {
    const auto& val = v.first;
    const auto& param_type = std::get<0>(v.second);
    const auto& input_idx = std::get<1>(v.second);

    const auto& input = inputs.at(input_idx);
    if (input.isTensorList() || input.isTensor()) {
      HABANA_ASSERT(
          param_type != NodeParamType::METADATA,
          "Expected view node param but got type: ",
          static_cast<int>(param_type));

      c10::TensorImpl* impl;
      if (input.isTensorList()) {
        const auto& tensor_vec = input.toTensorVector();
        const auto& tensor_idx = std::get<2>(v.second);
        impl = tensor_vec.at(tensor_idx).unsafeGetTensorImpl();
      } else { // isTensor
        impl = input.toTensor().unsafeGetTensorImpl();
      }

      torch::jit::IValue iVal;
      switch (param_type) {
        case NodeParamType::VIEW_SIZES:
          iVal = torch::jit::IValue(impl->sizes());
          break;

        case NodeParamType::VIEW_STRIDES:
          iVal = torch::jit::IValue(impl->strides());
          break;

        case NodeParamType::VIEW_OFFSET:
          iVal = torch::jit::IValue(impl->storage_offset());
          break;

        default:
          HABANA_ASSERT(
              0, "Unsupported param type: ", static_cast<int>(param_type));
      }
      jit_val_to_ivalue_map[val] = iVal;
    } else if (is_metadata_candidate(input)) {
      HABANA_ASSERT(
          param_type == NodeParamType::METADATA,
          "Invalid node param type: ",
          static_cast<int>(param_type));

      jit_val_to_ivalue_map[val] = input;
    } else {
      HABANA_ASSERT(0, "Unsupported input ivalue type ", input.tagKind());
    }
  }
}

bool EagerExec::check_and_skip_lowering() {
  // Check if eager op is skip lowering candidate
  if (!m_eager_op_meta_data.skip_lowering_) {
    return false;
  }

  // Check condition: if input tensor metadata send org tensor is available
  // It means it does not have permutation and lowering is not required
  for (const auto& input : m_inputs) {
    if (!input.isTensor()) {
      continue;
    }
    auto tensor = input.toTensor();
    auto tensor_tmeta{habana::get_tensor_extra_meta(tensor)};
    if (tensor_tmeta->get_send_org_tensor()) {
      return true;
    }
  }
  return false;
}

void EagerExec::launch() {
  PT_EAGER_TRACE_WITH_NAME(m_graph_name);
  if (check_and_skip_lowering()) {
    PT_EAGER_DEBUG("Eager Op :", m_symbol.toQualString(), " Skip lowering ! ");
    return;
  }
  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};

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
    // Get node params w.r.t orig_inputs if available
    const CValPtrMap& jit_val_map = graph_and_meta->get_param_jit_val_map();
    CValPtrtoIValueMap jit_val_to_ivalue_map;
    process_node_params(jit_val_map, orig_inputs, jit_val_to_ivalue_map);

    // Set param agnostic flag if node params are available for the view ops
    // or ops which uses either scalars or tensor shapes as node params
    const bool param_agnsotic_flag = jit_val_to_ivalue_map.size() ||
        NodeParamAgnosticOpList::isNodeParamAgnosticOp(m_symbol);
    graph_and_meta->set_is_param_agnostic_supported(param_agnsotic_flag);
    graph_and_meta->set_param_jit_val_to_ivalue_map(jit_val_to_ivalue_map);

    if (!graph_and_meta->get_new_strided_insert_output_shape().empty()) {
      auto& temp_outputs = m_outputs.get_outputs();
      if (auto* vec_output_specs =
              std::get_if<std::vector<OutputSpec>>(&temp_outputs)) {
        if (!vec_output_specs->empty()) {
          vec_output_specs->at(0).sizes =
              habana::get_base_tensor_size(stack[1].toTensor());
        }
      } else if (
          auto* vec_tensors =
              std::get_if<std::vector<at::Tensor>>(&temp_outputs)) {
        if (!vec_tensors->empty()) {
          vec_tensors->at(0).sizes() =
              habana::get_base_tensor_size(stack[1].toTensor());
        }
      }
    }

  } else {
    PT_EAGER_DEBUG("Eager Op JIT graph cache miss for key ", key);
    auto dump_graphs =
        std::string(GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE)) == "all" ||
        std::string(GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE)) == "eager";
    CValPtrMap jit_val_map; // map for capturing node params jit values
    auto graph{create_eager_graph(orig_inputs, jit_val_map)};
    if (dump_graphs)
      visualize::DumpEagerOrCompileGraph(
          graph,
          m_graph_name + "_" + std::to_string(key) + "_eager_preprocess");

    auto eager_compiler_supported =
        is_eager_compiler_supported_for_graph(graph);
    post_process_eager_graph(graph, jit_val_map);
    prune_duplicate_graph_inputs(parent_vec, graph);
    if (dump_graphs)
      visualize::DumpEagerOrCompileGraph(
          graph,
          m_graph_name + "_" + std::to_string(key) + "_eager_postprocess");

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
    graph_and_meta->set_is_pipeline_supported(m_is_pipeline_supported);
    graph_and_meta->set_param_jit_val_map(jit_val_map);
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_JIT_CACHE)) {
      cache.Add(key, graph_and_meta);
    }
  }

  for (const auto& val : stack) {
    if (!val.isTensor()) {
      continue;
    }

    // If an input tensor is not contiguous view handling JIT IR pass
    // would have modified the input tensor to base tensor. Need to
    // perform this operation for the cache hit case as well
    auto in = val.toTensor();
    [[maybe_unused]] auto input_smeta{habana::get_storage_extra_meta(in)};

    if (habana::is_view_lowering(in) || !in.is_contiguous()) {
      // modify the backend tensor of the view as the base
      auto impl = in.unsafeGetTensorImpl();
      impl->set_sizes_contiguous(habana::get_base_tensor_size(in));
      impl->set_storage_offset(0);
      PT_EAGER_DEBUG("Eager op: Input tensor converted to base");
    }

    // check if skip tensor permutation flag is set
    auto in_tmeta{habana::get_tensor_extra_meta(in)};
    if (in_tmeta->is_send_org_tensor_permuted()) {
      PT_EAGER_DEBUG(
          "Eager op:", m_symbol.toQualString(), " Skip tensor permutation");
      graph_and_meta->set_skip_tensor_permutation();
    }
  }

  try {
    auto& temp_outputs = m_outputs.get_outputs();
    if (!m_eager_op_meta_data.new_strided_insert_output_shape_.empty()) {
      graph_and_meta->set_new_strided_insert_output_shape(
          m_eager_op_meta_data.new_strided_insert_output_shape_);

      if (auto* vec_output_specs =
              std::get_if<std::vector<OutputSpec>>(&temp_outputs)) {
        if (!vec_output_specs->empty()) {
          vec_output_specs->at(0).sizes =
              m_eager_op_meta_data.new_strided_insert_output_shape_;
        }
      } else if (
          auto* vec_tensors =
              std::get_if<std::vector<at::Tensor>>(&temp_outputs)) {
        if (!vec_tensors->empty()) {
          vec_tensors->at(0).sizes() =
              m_eager_op_meta_data.new_strided_insert_output_shape_;
        }
      }
    }
    auto habana_launch_op =
        std::make_unique<habana::HabanaLaunchOpPT>(graph_and_meta);
    habana_launch_op->set_input_stack(stack);
    habana_launch_op->set_require_h2d_st(
        m_eager_op_meta_data.require_h2d_, m_eager_op_meta_data.require_st_);
    HabanaLaunchOpPipeline::LoweringTask(
        std::move(habana_launch_op),
        habana_launch_op->get_input_stack(),
        m_outputs.get_tensors(),
        m_outputs.get_shapes());
  } catch (const std::exception& e) {
    PT_EAGER_DEBUG("HabanaLaunchOpPT Run returned exception....\n", e.what());
    throw;
  }
}

std::shared_ptr<torch::jit::Graph> EagerExec::create_eager_graph(
    torch::jit::Stack& stack,
    CValPtrMap& jit_val_map) {
  PT_EAGER_TRACE;
  using JitValue = torch::jit::Value;
  auto graph = std::make_shared<torch::jit::Graph>();
  std::vector<JitValue*> node_inputs;
  size_t idx = 0;
  const bool add_val_flag =
      NodeParamAgnosticOpList::isNodeParamAgnosticOp(m_symbol);
  traversing_ivalues(
      stack,
      overloaded{
          // metadata
          [&node_inputs, &graph, &add_val_flag, &jit_val_map](
              const torch::jit::IValue& c) {
            node_inputs.push_back(graph->insertConstant(c));
            if (add_val_flag) {
              int idx = node_inputs.size() - 1;
              jit_val_map[node_inputs.back()] =
                  std::make_tuple(NodeParamType::METADATA, idx, idx);
            }
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
            for (size_t i = 0; i < list.size(); ++i) {
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

  /*Need to set this node if the deterministic mode is ON*/
  jit_node->i_(
      torch::jit::attr::deterministic,
      HPURegistrar::get_hpu_global_config().getDeterministic() ||
          at::globalContext().deterministicAlgorithms());
  PT_BRIDGE_DEBUG(
      "Deterministic val during Jit Node creation: ",
      jit_node->i(torch::jit::attr::deterministic));

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

  optimized_key = at::hash_combine(
      optimized_key,
      HPURegistrar::get_hpu_global_config().getDeterministic() ||
          at::globalContext().deterministicAlgorithms());

  for (size_t i = 0; i < parent_vec.size(); ++i)
    optimized_key = at::hash_combine(optimized_key, parent_vec[i]);

  std::unordered_set<size_t> input_hash_values;
  std::vector<uint64_t> storage_base_addresses;
  int inp_index = 0;
  const bool skip_ivalue_hash_flag =
      NodeParamAgnosticOpList::isNodeParamAgnosticOp(m_symbol);
  const bool skip_scalar_hash_flag = skip_ivalue_hash_flag &&
      !NodeParamAgnosticOpList::IsScalarNotPatchableOp(m_symbol);
  traversing_ivalues<ProcessList::asTensor>(
      stack,
      overloaded{
          [&optimized_key, &inp_index, &skip_ivalue_hash_flag](
              const torch::jit::IValue& input) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);
            if (!skip_ivalue_hash_flag) {
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
            }
          },
          [&optimized_key, &inp_index, &skip_scalar_hash_flag](
              const at::Scalar& input) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);
            if (!skip_scalar_hash_flag) {
              optimized_key =
                  at::hash_combine(optimized_key, at::IValue::hash(input));

              optimized_key = at::hash_combine(
                  optimized_key, at::IValue::hash(input.type()));
            }
          },
          [&optimized_key,
           &input_hash_values,
           &inp_index,
           &storage_base_addresses](const at::Tensor& tensor) {
            optimized_key = at::hash_combine(optimized_key, inp_index++);
            size_t input_hash_val = c10::get_hash(tensor.unsafeGetTensorImpl());
            if (input_hash_values.count(input_hash_val) == 0) {
              input_hash_values.emplace(input_hash_val);

              // hash memory section id if valid storage present
              if (tensor.has_storage()) {
                uint64_t base_address = reinterpret_cast<uint64_t>(
                    tensor.storage().data_ptr().get());
                if (base_address) {
                  // find the base address in the storage_base_addresses vector
                  // whose index is analogous to section id i.e. unique memory
                  // section
                  std::vector<uint64_t>::iterator it = std::find(
                      storage_base_addresses.begin(),
                      storage_base_addresses.end(),
                      base_address);

                  int section_id;
                  if (it != storage_base_addresses.end()) {
                    // tensor base address is the view
                    // resuse the old section id i.e. index of the vector
                    section_id =
                        std::distance(storage_base_addresses.begin(), it);
                  } else {
                    // tensor base address is the unique address
                    // assign the new section i.e. add it to the vector
                    section_id = storage_base_addresses.size();
                    storage_base_addresses.push_back(base_address);
                  }

                  // hash section id
                  optimized_key = at::hash_combine(optimized_key, section_id);
                }
              }

              update_key_for_tensor(tensor, optimized_key);
            }
          }});
  return optimized_key;
}

// ZST tensor has rank '1' with shape '0' but no allocation
static inline bool is_zst(const at::Tensor& t) {
  return (t.dim() == 1 && t.sizes()[0] == 0);
}

// define tensor type ZST for hashing the value
// other tensor types can be added here, if required
enum class TensorType { ZST_TENSOR = 1 };

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
    if (base_smeta) {
      for (auto s : base_smeta->get_memory_permutation()) {
        key = at::hash_combine(key, s);
      }
    }

    if (is_eager_caching_supported()) {
      // hash view params
      for (auto s : t.strides())
        key = at::hash_combine(key, s);
      // two different sized tensors can have same strides
      // ex: [2, 4, 1], and [2, 1, 4]
      for (auto s : t.sizes())
        key = at::hash_combine(key, s);

      key = at::hash_combine(key, static_cast<size_t>(t.storage_offset()));
    }
  }

  // exp_bias is a parameter of the synTensor,
  // so it has to participate in hash calculation
  if (auto exp_bias_opt = input_tmeta->get_exp_bias()) {
    key = at::hash_combine(key, *exp_bias_opt);
  }

  key = at::hash_combine(key, static_cast<size_t>(t.suggest_memory_format()));
  key = at::hash_combine(key, static_cast<size_t>(t.layout()));
  key = at::hash_combine(key, t.dim());

  /*
   * hash if zst tensor is true
   * To Do: Check if there is a need to hash a non-ZST tensor
   *        Hashing key calculations should not be condition-based
   *        and should consider the properties of the object.
   */
  if (is_zst(t)) {
    key = at::hash_combine(key, static_cast<size_t>(TensorType::ZST_TENSOR));
  }
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
  traversing_ivalues<ProcessList::asTensor>(
      inputs,
      overloaded{// metadata
                 [](const torch::jit::IValue&) {},
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
    CValPtrMap& jit_val_map) {
  PT_EAGER_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING)) {
    PT_EAGER_DEBUG("Replace copy with SI pass.");
    HandleOutputInsert(*graph, m_inputs, m_eager_op_meta_data, jit_val_map);
    PT_EAGER_DEBUG("Apply I/O View Handling pass.");
    HandleInputOutputViews(*graph, m_inputs, m_eager_op_meta_data, jit_val_map);
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

  auto eager_compiler_unsupported_op_prefixes =
      habana::OptimizedJitGraphCache::GetOptimizedJitCache()
          .get_eager_compiler_unsupported_op_prefixes();

  for (const auto& node : graph->nodes()) {
    std::string_view opname = node->kind().toQualString();
    for (const auto& prefix : eager_compiler_unsupported_op_prefixes) {
      if (opname.find(prefix) != std::string::npos) {
        return false;
      }
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
