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

// #include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <queue>
#include <sstream>

#include "habana_eager/graph_exec.h"

#include "habana_eager/eager_view.h"
#include "habana_helpers/logging_pt.h"

#include "pytorch_helpers/visualize/visualize.h"

namespace habana {
namespace graph {
namespace pass {

struct HandleInputViewsPass {
  explicit HandleInputViewsPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run(
      torch::jit::Stack& example_inputs,
      std::vector<habana_helpers::RangeInfo>& range_infos) {
    bool changed{processInputs(m_graph->inputs(), example_inputs, range_infos)};
    return changed;
  }

  std::map<int64_t, std::vector<int64_t>> get_base_sizes_to_set_during_launch() {
    return m_input_base_sizes_to_set;
  }

 private:
  bool processInputs(
      at::ArrayRef<torch::jit::Value*> inputs,
      torch::jit::Stack& example_inputs,
      std::vector<habana_helpers::RangeInfo>& range_infos) {
    bool changed{false};
    for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      torch::jit::Value* input{inputs.at(input_idx)};
      if (!example_inputs[input_idx].isTensor()) {
        continue;
      }
      torch::Tensor input_tensor{example_inputs[input_idx].toTensor()};
      [[maybe_unused]] auto storage_meta{
          habana::get_storage_extra_meta(input_tensor)};

      if (habana::is_view_lowering(input_tensor) ||
          !input_tensor.is_contiguous()) {
        auto& uses = input->uses();
        auto& first_use = uses[0];
        torch::jit::Node* first_user = first_use.user;
        auto& last_use = uses.back();
        torch::jit::Node* last_user = last_use.user;

        static const std::array<c10::Symbol, 4> view_ops_symbols{
            c10::Symbol::fromQualString("aten::as_strided"),
            c10::Symbol::fromQualString("aten::slice_scatter"),
            c10::Symbol::fromQualString("aten::select_scatter"),
            c10::Symbol::fromQualString("aten::as_strided_scatter")};

        if (std::find(
                view_ops_symbols.begin(),
                view_ops_symbols.end(),
                first_user->kind()) != view_ops_symbols.end()) {
          // Moving for next input as view for this one are already handled in
          // graph
          continue;
        }
        habana::eager::ViewParam view_params;
        view_params.setParam(input_tensor);

        bool needs_strided_insert = false;
        if ((uses.size() > 1) && (last_use.offset == 0)) {
          std::string_view node_name = last_user->kind().toQualString();
          if (node_name.back() == '_') {
            needs_strided_insert = true;
          }
        }

        m_input_base_sizes_to_set[input_idx] = std::vector<int64_t>();
        auto output_size = "[" + range_infos[input_idx].expr + "]";
        insert_strided_view_node(
            input_tensor,
            first_user,
            input,
            view_params,
            m_input_base_sizes_to_set.at(input_idx),
            output_size);

        // @TODO : Check if we can fill min and max shapes in form of symbols.
        // Till we find a way, since the strided tensor is replaced by base
        // tensor in the stack, we have to do the same in range_info DS. The
        // problem being we dont have a way currently to fetch shapes of base
        // tensor in form of symbolic so that min max can be inferred.
        std::stringstream ss;
        ss << "[";
        for (auto value : m_input_base_sizes_to_set.at(input_idx)) {
          ss << value << ", ";
        }
        std::string result = ss.str();
        if (!result.empty()) {
          result.erase(result.size() - 2);
        }
        result += "]";
        range_infos[input_idx].expr = result;
        PT_DYNAMIC_SHAPE_DEBUG(
            "HandleViewPass filling RangeInfo at index ",
            range_infos[input_idx].index,
            " with static min and max = ",
            result);
        // Change the index to -1 so that min and max range be processed
        range_infos[input_idx].index = -1;

        changed |= true;

        if (needs_strided_insert) {
          insert_strided_insert_node(
              input_tensor,
              input,
              last_user->output(0),
              view_params,
              output_size);

          replace_with_out_of_place_op(last_user);
        }
      }
    }

    return changed;
  }

  void insert_strided_view_node(
      at::Tensor input_tensor,
      torch::jit::Node* node,
      torch::jit::Value* value_in,
      const habana::eager::ViewParam& p,
      std::vector<int64_t>& base_sizes_to_set,
      std::string output_shape) {
    PT_EAGER_TRACE;
    torch::jit::WithInsertPoint insert_point(node);

    auto op_strided_view = c10::Symbol::fromQualString("aten::as_strided");
    auto value_sizes =
        m_graph->insertConstant(torch::jit::IValue(p.getViewSizes()));
    auto value_strides =
        m_graph->insertConstant(torch::jit::IValue(p.getViewStrides()));
    auto value_offset =
        m_graph->insertConstant(torch::jit::IValue(p.getViewOffset()));

    auto jit_node = m_graph->create(
        op_strided_view,
        {value_in, value_sizes, value_strides, value_offset},
        1);

    jit_node->output(0)->setType(c10::TensorType::createContiguous(
        input_tensor.scalar_type(), input_tensor.device(), p.getViewSizes()));

    base_sizes_to_set = habana::get_base_tensor_size(input_tensor);

    jit_node->input(0)->setType(c10::TensorType::createContiguous(
        input_tensor.scalar_type(),
        input_tensor.device(),
        input_tensor.sizes()));

    m_graph->insertNode(jit_node);

    if (GET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF)) {
      auto symbol_outputshape = c10::Symbol::attr("output_shapes");
      jit_node->s_(symbol_outputshape, output_shape);
      PT_EAGER_DEBUG("HandleViewPass filling output shape = ", output_shape);
    }

    value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));
  }

  void insert_strided_insert_node(
      at::Tensor input_tensor,
      torch::jit::Value* value_in,
      torch::jit::Value* value_out,
      const habana::eager::ViewParam& p,
      std::string output_shape) {
    PT_EAGER_TRACE;

    auto op_strided_insert =
        c10::Symbol::fromQualString("hpu::strided_insert_");
    auto value_strides =
        m_graph->insertConstant(torch::jit::IValue(p.getViewStrides()));
    auto value_offset =
        m_graph->insertConstant(torch::jit::IValue(p.getViewOffset()));

    auto jit_node = m_graph->create(
        op_strided_insert,
        {value_in, value_out, value_strides, value_offset},
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

    m_graph->insertNode(jit_node);

    if (GET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF)) {
      auto symbol_outputshape = c10::Symbol::attr("output_shapes");
      jit_node->s_(symbol_outputshape, output_shape);
      PT_EAGER_DEBUG("HandleViewPass filling output shape = ", output_shape);
    }

    value_out->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));
  }

  void replace_with_out_of_place_op(torch::jit::Node* node) {
    PT_EAGER_TRACE;
    torch::jit::WithInsertPoint insert_point(node);

    std::string_view inplace_name = node->kind().toQualString();
    std::string_view new_name = inplace_name;
    new_name.remove_suffix(1);

    PT_EAGER_DEBUG(
        "[replace_with_out_of_place_op] Op Name: ",
        inplace_name,
        " is replaced by: ",
        new_name);

    auto new_node =
        m_graph->create(c10::Symbol::fromQualString(std::string(new_name)));
    for (size_t i = 0; i < node->inputs().size(); ++i) {
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
    m_graph->insertNode(new_node);
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->output(i)->replaceAllUsesWith(new_node->output(i));
    }
    node->destroy();
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::map<int64_t, std::vector<int64_t>> m_input_base_sizes_to_set;
};

bool HandleInputViews(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    std::map<int64_t, std::vector<int64_t>>& input_base_sizes_map,
    std::vector<habana_helpers::RangeInfo>& range_infos) {
  PT_EAGER_TRACE;
  HandleInputViewsPass pass{graph};
  bool changed{pass.run(example_inputs, range_infos)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  input_base_sizes_map = pass.get_base_sizes_to_set_during_launch();
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana