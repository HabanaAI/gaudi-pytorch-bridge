/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "habana_eager/graph_exec.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_exec_passes.h"
#include "habana_eager/graph_storage.h"
#include "habana_eager/graph_weight_permute.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/thread_pool/thread_pool.h"

#include "habana_eager/eager_view.h"
#include "pytorch_helpers/habana_helpers/h2d_scales.h"
#include "pytorch_helpers/habana_helpers/towl.h"
#include "pytorch_helpers/visualize/visualize.h"

#include <chrono>

namespace habana::graph {

void PrintRangeInfos(std::vector<habana_helpers::RangeInfo>& range_infos) {
  PT_DYNAMIC_SHAPE_DEBUG("RangeInfos:");
  for (auto& info : range_infos) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Index=",
        info.index,
        " expr=",
        info.expr,
        " stride=",
        info.expr_strides,
        " min_shape=",
        info.min_shape,
        " max_shape=",
        info.max_shape);
  }
}

void PatchDynamicTensors(LaunchDynamicShapes& launch_shapes) {
  size_t num_tensors = launch_shapes.ds_tensors.size();
  PT_DYNAMIC_SHAPE_DEBUG("Num DS tensors to be patched = ", num_tensors);

  // Figure out the total H2D size required bu this graph
  size_t h2d_memory_required = 0;
  for (size_t i = 0; i < num_tensors; i++) {
    auto tensor = launch_shapes.ds_tensors[i];
    auto tmeta{habana::get_tensor_extra_meta(tensor)};
    if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR) {
      h2d_memory_required += 2 * tmeta->get_host_total_elem();
    }
  }

  // Allocate the total H2D required in single chunk
  void* alloc_pointer{nullptr};
  if (h2d_memory_required > 0) {
    auto& device = HPUDeviceContext::get_device();
    device.get_host_memory().uncached_malloc(
        &alloc_pointer, h2d_memory_required);
  }
  void* h2d_pointer{alloc_pointer};

  for (size_t i = 0; i < num_tensors; i++) {
    auto tensor = launch_shapes.ds_tensors[i];
    std::vector<int64_t> patch_data = launch_shapes.patch_values[i];
    auto tmeta{habana::get_tensor_extra_meta(tensor)};
    if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR) {
      // Set the host and compile pointer from allocated chunk and
      // increment the h2d_pointer to point to end of current H2D
      tmeta->set_alloc_ptr(alloc_pointer);
      tmeta->set_host_ptr(h2d_pointer);
      char* ptr =
          static_cast<char*>(h2d_pointer) + tmeta->get_host_total_elem();
      tmeta->set_compile_host_ptr(ptr);
      h2d_pointer = static_cast<char*>(ptr + tmeta->get_host_total_elem());
      habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
      if (h2d_dt_type == habana::HostDataType::INT32_T) {
        std::vector<int32_t> h2d_data(patch_data.begin(), patch_data.end());
        UpdateH2DTensorData<int32_t>(tensor, h2d_data);
      } else if (h2d_dt_type == habana::HostDataType::UINT32_T) {
        std::vector<uint32_t> h2d_data(patch_data.begin(), patch_data.end());
        UpdateH2DTensorData<uint32_t>(tensor, h2d_data);
      } else if (h2d_dt_type == habana::HostDataType::UINT64_T) {
        std::vector<uint64_t> h2d_data(patch_data.begin(), patch_data.end());
        UpdateH2DTensorData<uint64_t>(tensor, h2d_data);
      }
    } else if (tmeta->get_tensor_type() == SHAPE_TENSOR) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(patch_data);
      // If the tmeta contains data, it means it we need to patch
      // it with actual strides as well in tmeta. But Actual stride
      // information is present in previous H2D tensor(while filling tensors
      // in stridedView and StridedInsert we have made sure that offset tensor
      // follows H2D), get the H2D tensor and fill the strides value from it.
      if (tmeta->get_shape_struct().has_shape_tensor_data()) {
        auto tensor_H2D = launch_shapes.ds_tensors[i - 1];
        auto tmeta_H2D{habana::get_tensor_extra_meta(tensor_H2D)};
        HABANA_ASSERT(
            tmeta_H2D->get_tensor_type() == HOST_TO_DEVICE_TENSOR,
            "Invalid tensor used for updating actual stride value");
        std::vector<int64_t> updated_h2d_data =
            launch_shapes.patch_values[i - 1];
        auto num_strides = updated_h2d_data[0];
        std::vector<int64_t> actual_strides;
        for (int64_t i = 2; i < (2 + num_strides); ++i) {
          actual_strides.push_back(updated_h2d_data[i]);
        }
        // Since strides here are reversed, make it unreverse
        std::reverse(actual_strides.begin(), actual_strides.end());
        tmeta->get_shape_struct().set_strides_tensor_shape(actual_strides);
      }
    }
  }
}

void ProcessRangeInfos(
    InputSymbolIndexMap in_symbol_idx_map,
    std::vector<habana_helpers::RangeInfo>& range_infos,
    bool has_random) {
  // Index -1 in RangeInfo means this is backend added tensor
  // can be ST or H2D, evaluate min-max range from range_infos.expr
  // and in_symbol_idx_map symbols
  InputSymbolMap in_symbol_value_map;
  // Min Evaluation
  SymExprFactory::getInstance().clear_expr_cache();
  std::for_each(
      in_symbol_idx_map.begin(),
      in_symbol_idx_map.end(),
      [&](const std::pair<std::string, int64_t>& p) {
        int64_t scalar_index = p.second;
        // This is added to correct the scalar index of the original stack.
        // Random ops support adds additional 2 inputs to the stack at index
        // 0 and 1.
        if (has_random) {
          scalar_index = scalar_index + 2;
        }
        auto value =
            static_cast<double>(range_infos[scalar_index].min_shape[0]);
        auto value_sh = std::make_shared<double>(value);
        in_symbol_value_map[p.first] = value_sh;
      });
  for (auto& info : range_infos) {
    if (info.index < 0 && info.expr.size() > 2) {
      SymExprFactory& expr_factory = SymExprFactory::getInstance();
      auto size_expr =
          std::make_shared<SizeExpression>(info.expr, in_symbol_value_map);
      std::vector<int64_t> concrete_size =
          expr_factory.evaluate_symsize(size_expr);
      info.min_shape = concrete_size;
    }
  }

  // Max Evaluation
  SymExprFactory::getInstance().clear_expr_cache();
  std::for_each(
      in_symbol_idx_map.begin(),
      in_symbol_idx_map.end(),
      [&](const std::pair<std::string, int64_t>& p) {
        int64_t scalar_index = p.second;
        // This is added to correct the scalar index of the original stack.
        // Random ops support adds additional 2 inputs to the stack at index
        // 0 and 1.
        if (has_random) {
          scalar_index = scalar_index + 2;
        }
        auto value =
            static_cast<double>(range_infos[scalar_index].max_shape[0]);
        auto value_sh = std::make_shared<double>(value);
        in_symbol_value_map[p.first] = value_sh;
      });
  for (auto& info : range_infos) {
    if (info.index < 0 && info.expr.size() > 2) {
      SymExprFactory& expr_factory = SymExprFactory::getInstance();
      auto size_expr =
          std::make_shared<SizeExpression>(info.expr, in_symbol_value_map);
      std::vector<int64_t> concrete_size =
          expr_factory.evaluate_symsize(size_expr);
      info.max_shape = concrete_size;
    }
  }
}

void GraphExec::LaunchRecipeTask(
    GraphExec* gexec,
    torch::jit::Stack&& inputs,
    std::vector<at::Tensor>&& outputs,
    LaunchDynamicShapes launch_shapes,
    InputSymbolMap&& in_symbol_value_map) {
  PT_EAGER_TRACE_WITH_NAME(gexec->m_graph_name);
  PatchDynamicTensors(launch_shapes);
  gexec->LaunchRecipe(std::move(inputs), outputs, in_symbol_value_map);
}

GraphExec::GraphExec(
    size_t recipe_id,
    std::shared_ptr<torch::jit::Graph> graph,
    const std::string& parent_graph_name,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference,
    bool has_preallocated_outputs,
    bool has_randoms,
    InputSymbolIndexMap in_symbol_idx_map,
    std::vector<habana_helpers::RangeInfo>& range_infos,
    std::vector<int64_t>& const_indexes,
    bool mark_dynamic,
    const std::vector<bool>& is_reusable)
    : m_graph_index(recipe_id),
      m_graph(graph),
      m_dynamic(dynamic),
      m_inference(inference),
      m_has_preallocated_outputs(has_preallocated_outputs),
      m_has_randoms(has_randoms),
      m_in_symbol_idx_map(in_symbol_idx_map),
      m_range_infos(range_infos),
      m_const_indexes(const_indexes),
      m_mark_dynamic(mark_dynamic && dynamic) {
  PT_EAGER_TRACE;

  m_graph_name = parent_graph_name;
  if (m_graph_name.find("_fx") != std::string::npos)
    m_graph_name = m_graph_name.replace(m_graph_name.find("fx"), 2, "jit") +
        "_" + std::to_string(recipe_id);

  m_is_pipeline_supported = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE);

  // Temporailly record the original jit graph input to reusable info map
  std::unordered_map<torch::jit::Value*, bool> input_reusable_pairs;
  if (!is_reusable.empty()) {
    size_t jit_graph_inputs_size = m_graph->inputs().size();
    HABANA_ASSERT(jit_graph_inputs_size == is_reusable.size());
    for (size_t i = 0; i < jit_graph_inputs_size; ++i) {
      auto input = m_graph->inputs().at(i);
      bool reusable = is_reusable[i];
      input_reusable_pairs.emplace(input, reusable);
    }
  }

  UpdateSeedTensors(example_inputs);

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
    // we need sync because graph pass HandleInputViews uses permutation info
    HPUDeviceContext::join_lowering_thread();
  }
  RunGraphPasses(example_inputs);

  LogRecipeInfo(example_inputs);

  torch::jit::Stack in_stack = example_inputs;
  PT_DYNAMIC_SHAPE_DEBUG("Is Dynamic Graph = ", IsDynamicGraph());
  if (IsDynamicGraph()) {
    ProcessDynamicGraph(example_inputs);
    if (m_static_fallback) {
      PT_DYNAMIC_SHAPE_WARN(
          "Number of tensor dims exceeds the limit, falling back to static!");
      m_mark_dynamic = false;
    }
    in_stack = ProcessDynamicStack(example_inputs, true);
    m_sym_expr_hash = habana::ComputeNodeSymOutputHashCode(m_graph);

    // Check if any of the symbols where replaced with concrete values.
    // If then, make the m_sym_expr_hash invalid.
    bool invalid_symbols = HasInvalidDynamicSymbols();
    if (invalid_symbols) {
      m_sym_expr_hash = ULONG_MAX;
      PT_DYNAMIC_SHAPE_DEBUG(
          "Graph input symbols are invalid, symbol replacement happend!!!");
    }
    if (m_mark_dynamic) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "mark_dynamic flow is enabled for user min max ranges");
      PrintRangeInfos(m_range_infos);
      ProcessRangeInfos(m_in_symbol_idx_map, m_range_infos, m_has_randoms);
      // Removing the inputs from list which are removed from stack inputs
      auto list_begin = m_range_infos.begin();
      for (auto idx : m_dgraph_meta->remove_input_indexes) {
        m_range_infos.erase(list_begin + idx);
      }
      PrintRangeInfos(m_range_infos);
      HABANA_ASSERT(in_stack.size() == m_range_infos.size());
    }
  }

  if (habana_helpers::is_h2d_scales_enabled()) {
    // HandleH2dScales must run after dynamic passes, because it needs valid
    // indices of the original stack.
    [[maybe_unused]] static bool scales_created =
        HPUDeviceContext::h2d_scales_cache().CreateH2dScales();
    pass::HandleH2dScales(m_graph, example_inputs, m_h2d_scales_idx_names);
  }

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(in_stack, m_graph->inputs().size());

  std::string jit_graph_name;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_JIT_GRAPH_NAME_HASH)) {
    jit_graph_name = m_graph_name;
  }

  // The jit graph inputs number may be changed, we need to re-construct the
  // reusable info
  if (!input_reusable_pairs.empty()) {
    for (size_t i = 0; i < m_graph->inputs().size(); ++i) {
      auto input = m_graph->inputs().at(i);
      bool reusable = input_reusable_pairs.count(input)
          ? input_reusable_pairs.at(input)
          : false;
      m_is_reusable.emplace_back(reusable);
    }
  }

  bool is_dynamic_compile = IsDynamicGraph() && !m_static_fallback;
  m_graph_and_meta = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
      m_graph,
      input_refs,
      0ULL /*unique_cntr*/,
      std::vector<bool>{} /*node_bcast_map_*/,
      jit_graph_name,
      is_dynamic_compile,
      m_input_new_base_sizes,
      habana_helpers::HabanaFrontendTypes::COMPILE,
      m_is_reusable);

  m_graph_and_meta->SetGraphIndex(m_graph_index);
  m_graph_and_meta->SetFrontendType(
      habana_helpers::HabanaFrontendTypes::COMPILE);
  m_graph_and_meta->SetOpName(m_graph_name);
  m_graph_and_meta->set_is_eager_compiler_supported(false);
  m_graph_and_meta->set_is_pipeline_supported(m_is_pipeline_supported);
  m_graph_and_meta->set_sym_expr_hash(m_sym_expr_hash);
  m_graph_and_meta->SetUserMarkDynamic(m_mark_dynamic);
  m_graph_and_meta->SetUserRangesDynamic(m_range_infos);
};

bool GraphExec::IsDynamicGraph() {
  return m_dynamic;
}

void GraphExec::ProcessDynamicGraph(torch::jit::Stack& example_inputs) {
  m_dgraph_meta = std::make_shared<DynamicGraphMetaData>();
  pass::HandleDynamicOps(
      m_graph,
      example_inputs,
      m_dgraph_meta,
      &m_input_new_base_sizes,
      &m_range_infos);
  m_static_fallback = m_dgraph_meta->static_fallback;
  pass::HandlePostDynamic(m_dgraph_meta, m_input_new_base_sizes);
  PT_EAGER_DEBUG(
      "Jit for ", m_graph_name, " after processing dynamicity\n", *m_graph);
}

std::vector<at::IValue> GraphExec::ProcessDynamicStack(
    torch::jit::Stack& orig_stack,
    bool is_first_launch) {
  PT_EAGER_TRACE;
  torch::jit::Stack new_stack;
  LaunchDynamicShapes launch_shapes;
  new_stack.reserve(
      orig_stack.size() + m_dgraph_meta->ds_input_patching_list.size());
  new_stack.insert(new_stack.end(), orig_stack.begin(), orig_stack.end());
  pass::HandleDynamicInputPatching(
      new_stack, m_dgraph_meta, launch_shapes, is_first_launch);
  HABANA_ASSERT(
      m_graph->inputs().size() == new_stack.size(),
      "Graph inputs size not patching with stack size!!");
  if (!is_first_launch && !m_dgraph_meta->negative_size_nodes.empty())
    pass::ResolveNegativeSTSizes(
        m_graph, new_stack, m_dgraph_meta, launch_shapes);
  if (!is_first_launch)
    m_ds_patch_data.launch_shapes.push(launch_shapes);
  return new_stack;
}

// Patching H2D scale tensors created in HandleH2dScales pass
// with values from CPU tensors.
void GraphExec::PatchScaleH2dTensors(torch::jit::Stack& orig_stack) {
  PT_EAGER_TRACE;
  // Patching hw-aligned scales with preallocated H2D tensors.
  const auto& h2d_scales_cache = HPUDeviceContext::h2d_scales_cache();
  std::vector<size_t> non_hw_scales_indices;

  bool cast_trivial_scales_optimization_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_H2D_TRIVIAL_SCALES_MODE) > 0;
  bool gemm_trivial_scales_optimization_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_H2D_TRIVIAL_SCALES_MODE) > 1;

  for (const auto& [scale_indices, node_name] : m_h2d_scales_idx_names) {
    if ((node_name == "hpu::cast_to_fp8_v2" or
         node_name == "hpu::cast_from_fp8") and
        cast_trivial_scales_optimization_enabled) {
      const auto idx = scale_indices[0];
      const auto& cpu_scale = orig_stack[idx].toTensor();
      HABANA_ASSERT(cpu_scale.is_cpu(), "Expected scale as a CPU tensor.");

      if (cpu_scale.item().toDouble() == 1.0) {
        orig_stack[idx] = std::nullopt;
        PT_BRIDGE_DEBUG(
            "CPU scale of op ",
            node_name,
            " was set to None, because its value is 1.0");
        continue;
      }
    } else if (
        node_name == "hpu::fp8_gemm_v2" and
        gemm_trivial_scales_optimization_enabled) {
      const auto idx_a = scale_indices[0];
      const auto idx_b = scale_indices[1];
      const auto& cpu_scale_a = orig_stack[idx_a].toTensor();
      const auto& cpu_scale_b = orig_stack[idx_b].toTensor();
      HABANA_ASSERT(cpu_scale_a.is_cpu(), "Expected scale_a as a CPU tensor.");
      HABANA_ASSERT(cpu_scale_b.is_cpu(), "Expected scale_b as a CPU tensor.");

      if (cpu_scale_a.item().toDouble() == 1 / cpu_scale_b.item().toDouble()) {
        orig_stack[idx_a] = std::nullopt;
        orig_stack[idx_b] = std::nullopt;
        PT_BRIDGE_DEBUG(
            "CPU scales of op ",
            node_name,
            " were set to None, because the're reciprocals");
        continue;
      }
    }
    for (const auto idx : scale_indices) {
      const auto& cpu_scale = orig_stack[idx].toTensor();
      HABANA_ASSERT(cpu_scale.is_cpu(), "Expected scale as a CPU tensor.");

      const auto scale_value = cpu_scale.item().toDouble();

      const auto maybe_h2d_scale =
          h2d_scales_cache.TryGetH2dScale(scale_value, cpu_scale.scalar_type());

      std::string_view caching_message;
      if (maybe_h2d_scale.has_value()) {
        // Update the original stack with the H2D tensor.
        orig_stack[idx] = torch::jit::IValue(maybe_h2d_scale.value());
        caching_message = "from cache ";
      } else {
        non_hw_scales_indices.push_back(idx);
      }

      PT_BRIDGE_DEBUG(
          "CPU scale of op ",
          node_name,
          " was patched ",
          caching_message,
          "into H2D tensor with value=",
          scale_value);
    }
  }

  // Patching non-hw-aligned scales with newly created H2D tensors.
  // Calculate total H2D size required for scale tensors. Scale tensor
  // always contain one float or bfloat16 value, so total size depends directly
  // on the number of CPU scale tensors and their dtypes.
  const size_t float_count = std::count_if(
      non_hw_scales_indices.begin(),
      non_hw_scales_indices.end(),
      [&orig_stack](const auto non_hw_idx) {
        return orig_stack[non_hw_idx].toTensor().scalar_type() ==
            at::ScalarType::Float;
      });
  const size_t bfloat_count = non_hw_scales_indices.size() - float_count;

  static constexpr size_t bfloat_size = sizeof(at::BFloat16);
  static constexpr size_t float_size = sizeof(float);
  size_t h2d_memory_required =
      4 * (float_count * float_size + bfloat_count * bfloat_size);

  // Allocate the total H2D required in single chunk.
  void* alloc_pointer{nullptr};
  if (h2d_memory_required > 0) {
    auto& device = HPUDeviceContext::get_device();
    device.get_host_memory().uncached_malloc(
        &alloc_pointer, h2d_memory_required);
  }
  void* h2d_pointer{alloc_pointer};

  for (const auto non_hw_idx : non_hw_scales_indices) {
    const auto& cpu_scale = orig_stack[non_hw_idx].toTensor();
    orig_stack[non_hw_idx] =
        torch::jit::IValue(h2d_scales_cache.CreateH2dTensorScale(
            cpu_scale.data_ptr(),
            cpu_scale.scalar_type(),
            alloc_pointer,
            h2d_pointer));
  }
}

std::string GraphExec::LogRecipeInfo(torch::jit::Stack& example_inputs) {
  PT_EAGER_INFO(
      "Jit for ",
      m_graph_name,
      " dynamic: ",
      m_dynamic,
      " inference: ",
      m_inference,
      ":\n",
      *m_graph);

  for (size_t input_idx = 0; input_idx < m_graph->inputs().size();
       input_idx++) {
    if (example_inputs[input_idx].isTensor()) {
      torch::Tensor tensor{example_inputs[input_idx].toTensor()};
      if (tensor.device().type() == c10::DeviceType::CPU) {
        // CPU scale tensors will be processed later.
        continue;
      }
      synapse_helpers::layouts::MemoryPermutation m_perm;
      std::tie(m_perm, std::ignore) =
          habana_helpers::get_tensor_memory_permutation(tensor);
      PT_EAGER_INFO(
          m_graph->inputs().at(input_idx)->debugName(),
          ": ",
          habana_helpers::DebugString(example_inputs[input_idx]),
          " Perm: ",
          VecToString(m_perm));
    }
  }

  return "";
}

void GraphExec::RunPass(
    std::function<bool()> pass,
    bool dump_graphs,
    const std::string& pass_name,
    int& pass_ordinal) {
  auto start = std::chrono::high_resolution_clock::now();
  auto graph_changed = pass();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  towl::emitTimeDurationJit(pass_name, static_cast<float>(duration.count()));
  if (graph_changed && dump_graphs)
    visualize::DumpEagerOrCompileGraph(
        m_graph,
        m_graph_name + "-" + std::to_string(pass_ordinal++) + "-" + pass_name);
}

void GraphExec::RunGraphPasses(torch::jit::Stack& example_inputs) {
  PT_EAGER_TRACE;
  PT_EAGER_DEBUG("Jit for ", m_graph_name, " before passes\n", *m_graph);
  auto dump_graphs =
      std::string(GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE)) == "all" ||
      std::string(GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE)) == "compile";
  int pass_ordinal = 0; // Makes sure that dumped graph files alphabetical order
                        // corresponds to execution order
  if (dump_graphs)
    visualize::DumpEagerOrCompileGraph(
        m_graph,
        m_graph_name + "-" + std::to_string(pass_ordinal++) + "-before_passes");
  RunPass(
      [this, &example_inputs]() {
        return pass::MarkParamsAsConst(
            this->m_graph, example_inputs, m_const_indexes);
      },
      dump_graphs,
      "MarkParamsAsConst",
      pass_ordinal);
  RunPass(
      [this, &example_inputs]() {
        return pass::HandleInputViews(
            this->m_graph,
            example_inputs,
            this->m_input_new_base_sizes,
            this->m_range_infos);
      },
      dump_graphs,
      "HandleInputViews",
      pass_ordinal);
  RunPass(
      [this]() { return pass::ReplaceGetItemWithListUnpack(this->m_graph); },
      dump_graphs,
      "ReplaceGetItemWithListUnpack",
      pass_ordinal);
  RunPass(
      [this]() { return pass::HandleTupleOnOutput(this->m_graph); },
      dump_graphs,
      "HandleTupleOnOutput",
      pass_ordinal);
  RunPass(
      [this]() { return pass::AddDeterministicAttribute(this->m_graph); },
      dump_graphs,
      "AddDeterministicAttribute",
      pass_ordinal);
  RunPass(
      [this]() { return pass::RemoveDetachOp(this->m_graph); },
      dump_graphs,
      "RemoveDetachOp",
      pass_ordinal);

  if (m_has_preallocated_outputs) {
    RunPass(
        [this]() {
          return pass::GetOutputsOrderInGraph(
              this->m_graph, this->m_outputs_order);
        },
        dump_graphs,
        "GetOutputsOrderInGraph",
        pass_ordinal);
  } else {
    RunPass(
        [this]() { return pass::RemoveDummyOutput(this->m_graph); },
        dump_graphs,
        "RemoveDummyOutput",
        pass_ordinal);
  }
}

void GraphExec::PopulateSymbolValueMap(
    torch::jit::Stack& stack,
    InputSymbolMap& symbol_value_map) {
  std::for_each(
      m_in_symbol_idx_map.begin(),
      m_in_symbol_idx_map.end(),
      [&](const std::pair<std::string, int64_t>& p) {
        int64_t scalar_index = p.second;
        // This is added to correct the scalar index of the original stack.
        // Random ops support adds additional 2 inputs to the stack at index
        // 0 and 1.
        if (m_has_randoms) {
          scalar_index = scalar_index + 2;
        }
        HABANA_ASSERT(
            stack[scalar_index].isScalar(),
            "Wrong symbol index received!!!",
            scalar_index);
        auto value =
            static_cast<double>(stack[scalar_index].toScalar().toLong());
        auto value_sh = std::make_shared<double>(value);
        symbol_value_map.emplace(p.first, value_sh);
      });
}

torch::jit::Stack GraphExec::launch(
    torch::jit::Stack& stack,
    std::vector<at::Tensor>& outputs) {
  PT_EAGER_TRACE_WITH_NAME(m_graph_name);

  UpdateSeedTensors(stack);

  InputSymbolMap in_symbol_value_map;
  if (IsDynamicGraph()) {
    PT_EAGER_INFO("Launch dynamic recipe. is_first_launch: ", is_first_launch);
    torch::jit::Stack original_stack = stack;
    is_first_launch = false;
    stack = ProcessDynamicStack(original_stack, is_first_launch);

    // [TODO] Disable hybrid sif until SW-153320
    // Ticket appears to be done however when bellow line is removed tests fail

    habana_helpers::SetHybridSIFTorchCompile(false);

    PT_EAGER_INFO("Dynamic graph Info:", LogRecipeInfo(stack));
    if (GET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF)) {
      PopulateSymbolValueMap(original_stack, in_symbol_value_map);
    }
  }

  PatchScaleH2dTensors(stack);

  torch::jit::Stack backend_inputs =
      habana::eager::convert_ivalues_to_backend_tensors(stack);

  std::vector<at::Tensor> backend_outputs;
  backend_outputs.reserve(outputs.size());
  for (auto& tensor : outputs) {
    backend_outputs.push_back(
        habana::eager::HbEagerTensorPool::getInstance().get_backend_tensor(
            tensor));
  }

  m_graph_and_meta->set_is_pipeline_supported(m_is_pipeline_supported);
  LaunchDynamicShapes launch_shapes;
  if (!m_ds_patch_data.launch_shapes.empty()) {
    launch_shapes = m_ds_patch_data.launch_shapes.front();
    m_ds_patch_data.launch_shapes.pop();
  }
  if (m_is_pipeline_supported) {
    // Check if condition needed specific to dynamic
    habana::eager::ScheduleWorkAndUpdateLoweringThreadHandle(
        LaunchRecipeTask,
        this,
        std::move(backend_inputs),
        std::move(backend_outputs),
        std::move(launch_shapes),
        std::move(in_symbol_value_map));
    return {};
  }
  std::optional<std::vector<at::Tensor>> maybe_backend_outputs;
  if (!backend_outputs.empty()) {
    maybe_backend_outputs = backend_outputs;
  }
  habana::eager::JoinPendingPipelineThreads();
  PatchDynamicTensors(launch_shapes);

  torch::jit::Stack ret_stack = LaunchRecipe(
      std::move(backend_inputs), maybe_backend_outputs, in_symbol_value_map);
  return habana::eager::convert_ivalues_to_backend_tensors(ret_stack);
}

void GraphExec::ResetSeed() {
  m_reset_seed = true;
}

torch::jit::Stack GraphExec::LaunchRecipe(
    torch::jit::Stack stack,
    std::optional<std::vector<at::Tensor>> maybe_outputs,
    InputSymbolMap in_symbol_value_map) {
  // Important - this function is meant to be run on lowering thread.
  PT_EAGER_TRACE;
  if (maybe_outputs.has_value() && !maybe_outputs.value().empty()) {
    std::vector<at::Tensor>& outputs{maybe_outputs.value()};
    std::vector<at::Tensor> reordered_outputs;
    HABANA_ASSERT(m_outputs_order.size() == outputs.size());
    reordered_outputs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      reordered_outputs.push_back(outputs[m_outputs_order[i]]);
    }
    maybe_outputs = reordered_outputs;
  }

  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(stack, m_graph->inputs().size());

  for (auto& input_base_sizes_pair : m_input_new_base_sizes) {
    int64_t input_idx{input_base_sizes_pair.first};
    std::vector<int64_t> base_sizes{input_base_sizes_pair.second};
    HABANA_ASSERT(input_refs.at(input_idx).isTensor());
    torch::Tensor input_tensor{input_refs.at(input_idx).toTensor()};

    auto base_sizes_to_set = habana::get_base_tensor_size(input_tensor);
    auto* impl = input_tensor.unsafeGetTensorImpl();
    impl->set_storage_offset(0);
    impl->set_sizes_contiguous(base_sizes_to_set);
    input_base_sizes_pair.second = base_sizes_to_set;
  }

  bool enable_optim_output_sif =
      (m_graph_and_meta->GetDynamicGraph() &&
       GET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF) &&
       m_graph_and_meta->get_sym_expr_hash() != ULONG_MAX);
  m_graph_and_meta->set_enable_optim_output_sif(enable_optim_output_sif);

  auto graph_symint_hash = habana::ComputeSymSizeHashCode(input_refs);
  m_graph_and_meta->set_graph_symint_hash(graph_symint_hash);
  auto graph_key_with_perm = at::hash_combine(
      m_graph_and_meta->get_cached_graph_key(), graph_symint_hash);
  auto graph_perm_hash = habana::ComputePermutationHashCode(input_refs);
  m_graph_and_meta->set_graph_perm_hash(graph_perm_hash);
  m_graph_and_meta->set_shapeless_with_dims_hash(at::hash_combine(
      m_graph_and_meta->get_shapeless_with_dims_hash(), graph_perm_hash));
  graph_key_with_perm = at::hash_combine(graph_key_with_perm, graph_perm_hash);
  m_graph_and_meta->set_graph_key_with_perm(graph_key_with_perm);

  if (enable_optim_output_sif) {
    m_graph_and_meta->set_maybe_static_recipe(true);

    if (m_initial_graph_key_with_perm == SIZE_MAX)
      m_initial_graph_key_with_perm = graph_key_with_perm;

    m_curr_symval_hash =
        habana_helpers::CalculateSymbolValuesHash(in_symbol_value_map);
    m_graph_and_meta->set_curr_symval_hash(m_curr_symval_hash);

    if (m_initial_symval_hash == SIZE_MAX) {
      m_initial_symval_hash = m_curr_symval_hash;
    } else if (
        (m_initial_symval_hash != m_curr_symval_hash) &&
        (graph_key_with_perm == m_initial_graph_key_with_perm)) {
      // If current symbol values differ from those in initial run,
      // then a dynamic recipe will get compiled.
      // But if graph_key_with_perm changes from intial run,
      // then a new static recipe will get compiled.
      m_graph_and_meta->set_maybe_static_recipe(false);
    }
  }

  m_graph_and_meta->SetHPUStream(stream);
  try {
    if (m_is_pipeline_supported) {
      auto habana_launch_op =
          std::make_unique<habana::HabanaLaunchOpPT>(m_graph_and_meta);
      habana_launch_op->set_input_stack(stack);
      habana_launch_op->set_symbol_values(in_symbol_value_map);
      HabanaLaunchOpPipeline::LoweringTask(
          std::move(habana_launch_op),
          habana_launch_op->get_input_stack(),
          maybe_outputs);
      return {};
    } else {
      habana::HabanaLaunchOpPT habana_launch_op(m_graph_and_meta);
      habana_launch_op.set_input_stack(stack);
      habana_launch_op.set_symbol_values(in_symbol_value_map);
      habana_launch_op.run(
          habana_launch_op.get_input_stack(), nullptr, maybe_outputs);
      return habana_launch_op.get_input_stack();
    }
  } catch (const std::exception& e) {
    PT_EAGER_FATAL("HabanaLaunchOpPT Run returned exception....\n", e.what());
  }
}

void GraphExec::UpdateSeedTensors(torch::jit::Stack& stack) {
  PT_EAGER_TRACE;

  if (m_has_randoms) {
    if (m_reset_seed) {
      m_seed_tensors.seed =
          torch::randint(std::numeric_limits<int32_t>::max(), {}, torch::kInt)
              .to("hpu");
      m_seed_tensors.counter = torch::tensor(0, {torch::kInt}).to("hpu");
      m_reset_seed = false;
    }

    stack[0] = *m_seed_tensors.seed;
    stack[1] = *m_seed_tensors.counter;
  }
}

bool GraphExec::HasInvalidDynamicSymbols() {
  bool invalid_symbol = false;

  for (auto it = m_in_symbol_idx_map.begin(); it != m_in_symbol_idx_map.end();
       ++it) {
    if (std::isdigit(it->first[0])) {
      size_t pos = 0;
      std::stod(it->first, &pos);
      // invalid symbol if it's completely numeric
      if (pos == it->first.size()) {
        PT_DYNAMIC_SHAPE_DEBUG("key:", it->first, ", value:", it->second);
        invalid_symbol = true;
        break;
      }
    }
  }

  return invalid_symbol;
}

} // namespace habana::graph
