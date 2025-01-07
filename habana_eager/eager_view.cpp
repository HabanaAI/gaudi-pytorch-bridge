/**
* Copyright (c) 2021-2024 Intel Corporation
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

#include "habana_eager/eager_view.h"
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace habana {
namespace eager {

namespace {

using namespace std::literals;

/* In general, inplace ops  read from input tensor and then write to the same
tensor.
The below list of ops ignore the values in the input tensor and overwrite the
contents*/
std::unordered_set<std::string_view> underscored_ops_reported_as_non_inplace = {
    "aten::zero_"sv,
    "aten::_foreach_zero_"sv,
    "aten::fill_"sv,
    "hpu::bernoulli_"sv,
    "hpu::uniform_"sv,
    "hpu::random_"sv,
    "hpu::normal_"sv,
    "hpu::geometric_"sv,
    "hpu::log_normal_"sv,
    "hpu::exponential_"sv};

/* below ops modify the o/p dtype in their out of place variant or
 * convert out variant to regular one that may result in dtype promotion
 * thereby requiring cast node*/
std::unordered_set<std::string_view> ops_needing_cast = {
    "aten::eq"sv,          "aten::ne"sv,
    "aten::ge"sv,          "aten::le"sv,
    "aten::gt"sv,          "aten::lt"sv,
    "aten::logical_and"sv, "aten::logical_or"sv,
    "aten::logical_xor"sv, "aten::logical_not"sv,
    "aten::add"sv,         "aten::sub"sv,
    "aten::mul"sv,         "aten::div"sv,
    "aten::remainder"sv,   "aten::floor_divide_"sv,
    "aten::clamp"sv,       "aten::clamp_max"sv,
    "aten::clamp_min"sv,   "aten::xlogy"sv,
};

bool check_if_op_doesnt_use_input(const JitNode* node) {
  return (
      underscored_ops_reported_as_non_inplace.find(
          node->kind().toQualString()) !=
      underscored_ops_reported_as_non_inplace.end());
}

void insert_cast_node(
    JitGraph& graph,
    JitNode* node,
    JitNode* insert_after_node,
    c10::ScalarType dtype) {
  torch::jit::WithInsertPoint insert_point(node);
  auto value_in = insert_after_node->output(0);
  auto op_copy = c10::Symbol::fromQualString("aten::_to_copy");
  auto dst_dtype = graph.insertConstant(dtype);
  auto dummy_args = graph.insertConstant(torch::jit::IValue());
  auto non_blocking = graph.insertConstant(false);
  auto copy_node = graph.create(
      op_copy,
      {value_in,
       dst_dtype,
       dummy_args,
       dummy_args,
       dummy_args,
       non_blocking,
       dummy_args},
      1);
  graph.insertNode(copy_node);
  set_deterministic(copy_node);

  insert_after_node->output(0)->replaceAllUsesAfterNodeWith(
      copy_node, copy_node->output(0));
}

JitNode* insert_strided_view_node(
    JitGraph& graph,
    JitNode* node,
    at::Tensor input,
    JitValue* jitval_in,
    const bool consumer_op_doesnt_use_input) {
  ViewParam p;
  p.setParam(input);
  torch::jit::WithInsertPoint insert_point(node);
  auto op_strided_view = c10::Symbol::fromQualString("aten::as_strided");
  auto value_sizes = graph.insertConstant(torch::jit::IValue(p.getViewSizes()));
  auto value_strides =
      graph.insertConstant(torch::jit::IValue(p.getViewStrides()));
  auto value_offset =
      graph.insertConstant(torch::jit::IValue(p.getViewOffset()));
  auto jit_node = graph.create(
      op_strided_view,
      {jitval_in, value_sizes, value_strides, value_offset},
      1);
  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), p.getViewSizes()));
  auto sizes = habana::get_base_tensor_size(input);
  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), sizes));
  PT_EAGER_DEBUG(
      "update graph view input's sizes: ",
      input.sizes(),
      " to base sizes: ",
      sizes);

  set_deterministic(jit_node);
  if (consumer_op_doesnt_use_input) {
    set_as_strided_meta(jit_node);
  }
  graph.insertNode(jit_node);

  jitval_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

bool check_inplace_op(const EagerOpMetaData& eager_op_meta_data) {
  return (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::Inplace);
}

bool is_view(const at::Tensor& t) {
  return (habana::is_view_lowering(t) || (!t.is_contiguous()));
}

bool is_schema_incompatible_between_hpu_aten(const std::string_view op_name) {
  /* Below ops have an incompatible signature between hpu and aten.
  The conversion from old kind to OutOfPlace should take place within the HPU */
  static std::unordered_set<std::string_view> ops_replace_within_hpu = {
      "hpu::index"sv, "hpu::rrelu_with_noise"sv, "hpu::rrelu_with_noise_"sv};
  return ops_replace_within_hpu.find(op_name) != ops_replace_within_hpu.end();
}

static JitNode* replace_with_out_of_place_op(
    JitGraph& graph,
    JitNode* node,
    const EagerOpMetaData& eager_op_meta_data) {
  torch::jit::WithInsertPoint insert_point(node);
  const auto old_kind = node->kind().toQualString();
  // Make sure to use consistent operator name as aten but in hpu namespace
  auto aten_op_symbol =
      c10::Symbol::fromQualString(eager_op_meta_data.op_name_);
  std::string hpu_op_name =
      "hpu::" + std::string(aten_op_symbol.toUnqualString());
  const std::string& new_kind =
      is_schema_incompatible_between_hpu_aten(old_kind)
      ? hpu_op_name
      : eager_op_meta_data.op_name_;

  PT_EAGER_DEBUG(
      "[replace_with_out_of_place_op] Op Name: ",
      old_kind,
      " is replaced by: ",
      new_kind);

  auto new_node = graph.create(c10::Symbol::fromQualString(new_kind));
  new_node->addInput(node->input(0));
  auto num_inputs = node->inputs().size() - eager_op_meta_data.num_out_tensors_;
  for (size_t i = 1; i < num_inputs; ++i) {
    new_node->addInput(node->input(i));
  }
  new_node->setScope(node->scope());
  new_node->copyAttributes(*node);
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    if (i > 0) {
      new_node->addOutput();
    }
    new_node->output(i)->copyMetadata(node->output(i));
  }
  graph.insertNode(new_node);
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    node->output(i)->replaceAllUsesWith(new_node->output(i));
  }
  node->destroy();

  return new_node;
}

JitNode* insert_strided_insert_node(
    JitGraph& graph,
    JitNode* node,
    JitValue* input_jitval,
    const at::Tensor& input_tensor,
    size_t node_output_idx) {
  ViewParam p{};
  p.setParam(input_tensor);

  auto op_strided_insert = c10::Symbol::fromQualString("hpu::strided_insert");
  auto value_strides =
      graph.insertConstant(torch::jit::IValue(p.getViewStrides()));
  auto value_offset =
      graph.insertConstant(torch::jit::IValue(p.getViewOffset()));

  auto jit_node = graph.create(
      op_strided_insert,
      {input_jitval,
       node->output(node_output_idx),
       value_strides,
       value_offset},
      1);

  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(),
      input_tensor.device(),
      {p.getTotalElements()}));

  jit_node->input(1)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(), input_tensor.device(), p.getViewSizes()));

  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(),
      input_tensor.device(),
      {p.getTotalElements()}));

  set_deterministic(jit_node);
  graph.insertNode(jit_node);

  node->output(node_output_idx)
      ->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

JitNode* replace_copy_with_strided_insert(
    JitGraph& graph,
    JitNode* node,
    const at::Tensor& input_tensor,
    EagerOpMetaData& eager_op_meta_data,
    std::vector<at::IValue>& inputs) {
  ViewParam p{};
  p.setParam(input_tensor);
  torch::jit::WithInsertPoint insert_point(node);
  auto op_strided_insert = c10::Symbol::fromQualString("hpu::strided_insert");
  auto value_strides =
      graph.insertConstant(torch::jit::IValue(p.getViewStrides()));
  auto value_offset =
      graph.insertConstant(torch::jit::IValue(p.getViewOffset()));
  auto jit_node = graph.create(
      op_strided_insert,
      {node->input(1), node->input(0), value_strides, value_offset},
      1);
  inputs.push_back(value_strides);
  inputs.push_back(value_offset);

  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(),
      input_tensor.device(),
      habana::get_base_tensor_size(input_tensor)));

  jit_node->input(1)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(), input_tensor.device(), p.getViewSizes()));
  auto new_output_size = habana::get_base_tensor_size(input_tensor);
  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      input_tensor.scalar_type(), input_tensor.device(), new_output_size));
  eager_op_meta_data.new_strided_insert_output_shape_ = new_output_size;
  set_deterministic(jit_node);
  graph.insertNode(jit_node);
  node->output(0)->replaceAllUsesWith(jit_node->output(0));
  node->destroy();

  return jit_node;
}

struct HandleInputOutputViewState {
  unsigned strided_view_nodes_count = 0;
  unsigned strided_insert_nodes_count = 0;
  unsigned inplace_ordinary_tensors = 0;
  unsigned cumulative_output_idx = 0;
};

void HandleInputOutputView(
    JitGraph& graph,
    EagerOpMetaData& eager_op_meta_data,
    JitNode* node_consuming_input,
    JitNode* node_producing_output,
    JitValue* input_jitval,
    at::IValue input_ival,
    size_t input_idx_in_node_ci,
    size_t input_idx_in_node_po,
    size_t first_out_id,
    bool is_inplace_op,
    bool op_doesnt_use_input,
    HandleInputOutputViewState& state,
    CValPtrMap& jit_val_map) {
  HABANA_ASSERT(
      input_ival.isTensor(),
      "Expected tensor, when parsing idx: ",
      input_idx_in_node_po,
      ", ",
      input_idx_in_node_ci);
  const auto& t = input_ival.toTensor();

  if (t.device().type() != c10::DeviceType::HPU) {
    return;
  }

  bool idx_is_out = input_idx_in_node_po >= first_out_id;

  std::optional<size_t> node_output_idx;
  bool inplace_input = false;
  if (idx_is_out ||
      (inplace_input =
           (is_inplace_op &&
            eager_op_meta_data.out_indices_.count(input_idx_in_node_po)))) {
    node_output_idx = state.cumulative_output_idx++;
  }

  if (!is_view(t)) {
    if (inplace_input) {
      // When any non-view inplace input is present we have to resign from
      // replacement inplace node with non-inplace node as such tensor
      // will not be updated.
      // See usage of this counter later.
      // Views are different as strided_insert node can modify the input
      // even it is not the output in IR.
      ++state.inplace_ordinary_tensors;
    }
    return;
  }

  if (!idx_is_out || (eager_op_meta_data.num_out_tensors_ > 1)) {
    auto sv_node = insert_strided_view_node(
        graph, node_consuming_input, t, input_jitval, op_doesnt_use_input);
    ++state.strided_view_nodes_count;

    // update node params jit value map
    jit_val_map[sv_node->input(1)] = std::make_tuple(
        NodeParamType::VIEW_SIZES, input_idx_in_node_po, input_idx_in_node_ci);
    jit_val_map[sv_node->input(2)] = std::make_tuple(
        NodeParamType::VIEW_STRIDES,
        input_idx_in_node_po,
        input_idx_in_node_ci);
    jit_val_map[sv_node->input(3)] = std::make_tuple(
        NodeParamType::VIEW_OFFSET, input_idx_in_node_po, input_idx_in_node_ci);
  }

  if (node_output_idx) {
    auto si_node = insert_strided_insert_node(
        graph, node_producing_output, input_jitval, t, *node_output_idx);
    ++state.strided_insert_nodes_count;

    // update node params jit value map
    jit_val_map[si_node->input(2)] = std::make_tuple(
        NodeParamType::VIEW_STRIDES,
        input_idx_in_node_po,
        input_idx_in_node_ci);
    jit_val_map[si_node->input(3)] = std::make_tuple(
        NodeParamType::VIEW_OFFSET, input_idx_in_node_po, input_idx_in_node_ci);

    if (ops_needing_cast.find(node_producing_output->kind().toQualString()) !=
        ops_needing_cast.end()) {
      insert_cast_node(graph, si_node, node_producing_output, t.scalar_type());
    }
  }
}

class PtEagerGraphDebug {
 public:
  PtEagerGraphDebug(const JitGraph& graph) : m_graph(graph) {}

  void before(const char* label) {
    m_label = label;
    print("[Before]");
  }

  void after(bool status = true) {
    if (status) {
      print("[After]");
    } else {
      PT_EAGER_DEBUG("\n", m_label, " not required.=====================\n");
    }
    m_label = m_bad_label;
  }

 private:
  void print(const char* mode) {
    PT_EAGER_DEBUG(
        "\n",
        m_label,
        ":=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        mode,
        '\n',
        m_graph.toString(),
        "JIT_IR_Graph_END\n");
  }

  const JitGraph& m_graph;
  static constexpr const char* m_bad_label = "BAD_LABEL";
  const char* m_label = m_bad_label;
};

void AssertNumInputs(
    const char* l1,
    size_t ni1,
    const char* l2,
    size_t ni2,
    std::optional<size_t> idx = {}) {
  HABANA_ASSERT(
      ni1 == ni2,
      "Number of inputs inconsistent between ",
      l1,
      " (",
      ni1,
      ") and ",
      l2,
      " (",
      ni2,
      ")",
      idx ? " at index " : "",
      idx ? std::to_string(*idx) : std::string{});
}

} // namespace

void set_as_strided_meta(JitNode* node) {
  auto meta = torch::jit::attr::arg1;
  node->i_(meta, 1);
  PT_EAGER_DEBUG("as_strided node set for meta attribute");
}

void set_deterministic(JitNode* node) {
  node->i_(
      torch::jit::attr::deterministic,
      HPUGlobalConfig::get().getDeterministic() ||
          at::globalContext().deterministicAlgorithms());
  PT_EAGER_DEBUG(
      "Deterministic val during Jit Node creation: ",
      node->i(torch::jit::attr::deterministic));
}

void HandleOutputInsert(
    JitGraph& graph,
    std::vector<at::IValue>& inputs,
    EagerOpMetaData& eager_op_meta_data,
    CValPtrMap& jit_val_map) {
  PT_EAGER_TRACE;

  PT_EAGER_DEBUG(
      "[HandleOutputInsert] Eager Op Info = ", eager_op_meta_data.to_string());
  if (!(eager_op_meta_data.op_name_ ==
        std::string("hpu::_copy_from_strided_insert"))) {
    PT_EAGER_DEBUG(
        "[HandleOutputInsert] Node replacement with SI not required.");
    return;
  }

  JitNode* node{nullptr};
  for (auto it = graph.nodes().begin(); it != graph.nodes().end(); ++it) {
    switch (it->kind()) {
      case at::prim::Constant:
      case at::prim::ListConstruct:
        break;

      default:
        TORCH_CHECK(
            node == nullptr,
            "Expecting exactly one non auxiliary node, but already found ",
            node->kind().toQualString(),
            " and ",
            it->kind().toQualString());
        node = *it;
    }
  }

  PT_EAGER_DEBUG("[HandleOutputInsert] Op Name: ", node->kind().toQualString());

  PtEagerGraphDebug pt_eager_graph_debug(graph);
  pt_eager_graph_debug.before("Copy node replacement with SI:");

  auto num_inputs = node->inputs().size();
  auto inputs_list_size = inputs.size();

  AssertNumInputs("node", num_inputs, "inputs list", inputs_list_size);
  std::vector<at::IValue> temp_inputs = inputs;
  int idx = 1;
  JitNode* si_node{nullptr};
  auto ival = inputs[idx];
  if (ival.isTensor()) {
    const auto& t = ival.toTensor();

    if (is_view(t)) {
      si_node = replace_copy_with_strided_insert(
          graph, node, t, eager_op_meta_data, inputs);

      jit_val_map[si_node->input(2)] =
          std::make_tuple(NodeParamType::VIEW_STRIDES, idx, idx);

      jit_val_map[si_node->input(3)] =
          std::make_tuple(NodeParamType::VIEW_OFFSET, idx, idx);
    }
  }
  pt_eager_graph_debug.after("Copy node replacement with SI:");

  idx = 0;
  ival = temp_inputs[idx];
  if (ival.isTensor()) {
    const auto& t = ival.toTensor();
    if (is_view(t)) {
      auto jitval = si_node->inputs()[1];
      bool op_doesnt_use_input = check_if_op_doesnt_use_input(si_node);
      pt_eager_graph_debug.before("Copy node replacement with SV:");
      auto sv_node = insert_strided_view_node(
          graph, si_node, t, jitval, op_doesnt_use_input);
      pt_eager_graph_debug.after("Copy node replacement with SV:");

      // update node params jit value map
      jit_val_map[sv_node->input(1)] =
          std::make_tuple(NodeParamType::VIEW_SIZES, idx, idx);
      jit_val_map[sv_node->input(2)] =
          std::make_tuple(NodeParamType::VIEW_STRIDES, idx, idx);
      jit_val_map[sv_node->input(3)] =
          std::make_tuple(NodeParamType::VIEW_OFFSET, idx, idx);
    }
  }
}

void HandleInputOutputViews(
    JitGraph& graph,
    const c10::ArrayRef<at::IValue> inputs,
    EagerOpMetaData& eager_op_meta_data,
    CValPtrMap& jit_val_map) {
  PT_EAGER_TRACE;

  PT_EAGER_DEBUG(
      "[HandleInputOutputViews] Eager Op Info = ",
      eager_op_meta_data.to_string());
  if (eager_op_meta_data.op_name_ ==
      std::string("hpu::_copy_from_strided_insert")) {
    return;
  }

  JitNode* node{nullptr};
  for (auto it = graph.nodes().begin(); it != graph.nodes().end(); ++it) {
    switch (it->kind()) {
      case at::prim::Constant:
      case at::prim::ListConstruct:
        break;

      default:
        TORCH_CHECK(
            node == nullptr,
            "Expecting exactly one non auxiliary node, but already found ",
            node->kind().toQualString(),
            " and ",
            it->kind().toQualString());
        node = *it;
    }
  }

  PT_EAGER_DEBUG(
      "[HandleInputOutputViews] Op Name: ", node->kind().toQualString());

  PtEagerGraphDebug pt_eager_graph_debug(graph);
  pt_eager_graph_debug.before("SV/SI node insertion:");

  bool is_inplace_op = check_inplace_op(eager_op_meta_data);
  bool op_doesnt_use_input = check_if_op_doesnt_use_input(node);

  auto num_inputs = node->inputs().size();
  auto inputs_list_size = inputs.size();
  auto first_out_idx = num_inputs - eager_op_meta_data.num_out_tensors_;

  AssertNumInputs("node", num_inputs, "inputs list", inputs_list_size);

  HandleInputOutputViewState io_view_state{};

  for (size_t idx = 0; idx < num_inputs; ++idx) {
    auto jitval = node->inputs()[idx];
    auto ival = inputs[idx];
    if (ival.isTensor()) {
      HandleInputOutputView(
          graph,
          eager_op_meta_data,
          node,
          node,
          jitval,
          ival,
          idx,
          idx,
          first_out_idx,
          is_inplace_op,
          op_doesnt_use_input,
          io_view_state,
          jit_val_map);
    } else if (ival.isTensorList()) {
      JitNode* list_node = node->input(idx)->node();
      const auto& tensor_vec = ival.toTensorVector();
      const auto& jitvals = list_node->inputs();

      AssertNumInputs(
          "list node",
          jitvals.size(),
          "input tensor list",
          tensor_vec.size(),
          idx);

      for (size_t i = 0; i < tensor_vec.size(); ++i) {
        HandleInputOutputView(
            graph,
            eager_op_meta_data,
            list_node,
            node,
            jitvals[i],
            tensor_vec[i],
            i,
            idx,
            first_out_idx,
            is_inplace_op,
            op_doesnt_use_input,
            io_view_state,
            jit_val_map);
      }
    }
  }

  pt_eager_graph_debug.after(
      io_view_state.strided_view_nodes_count ||
      io_view_state.strided_insert_nodes_count);

  if (io_view_state.strided_insert_nodes_count &&
      !io_view_state.inplace_ordinary_tensors) {
    pt_eager_graph_debug.before("Node replacement");

    // These kernels completely ignore the data in input tensor and hence the
    // input tensor can be reused by updating inplace. Further it also avoids
    // implementing out of place variants
    bool replaced = false;
    if (!underscored_ops_reported_as_non_inplace.count(
            node->kind().toQualString())) {
      if (eager_op_meta_data.num_out_tensors_ <= 1) {
        replace_with_out_of_place_op(graph, node, eager_op_meta_data);
        replaced = true;
      }
    }

    pt_eager_graph_debug.after(replaced);
  }
}

} // namespace eager
} // namespace habana
