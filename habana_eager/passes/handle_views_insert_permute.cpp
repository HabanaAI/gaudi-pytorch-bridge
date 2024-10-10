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
 *******************************************************************************/

#include <algorithm>
#include <numeric>

#include "habana_eager/eager_exec.h"
#include "habana_eager/eager_view.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "handle_views_insert_permute.h"

namespace habana {
namespace graph {
namespace pass {

struct StridedViewAndPermuteInfo {
  StridedViewAndPermuteInfo(
      torch::jit::Node* node,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<int64_t> shapes,
      std::vector<int64_t> strides)
      : node(node),
        dtype(dtype),
        device(std::move(device)),
        shapes(std::move(shapes)),
        strides(std::move(strides)) {}

  torch::jit::Node* const node;
  const c10::ScalarType dtype;
  const c10::Device device;
  const std::vector<int64_t> shapes;
  const std::vector<int64_t> strides;
  std::vector<int64_t> newShapes;
  std::vector<int64_t> newStrides;
  std::vector<int64_t> permutation;
};

static inline bool isPermuted(const std::vector<int64_t>& strides) {
  bool monotic_decreasing = true;
  // check that strides are monotonically decreasing
  for (size_t i = 0; i < (strides.size() - 1); ++i) {
    monotic_decreasing &= (strides[i] >= strides[i + 1]);
  }
  return !monotic_decreasing;
}

// ToDo: optimize this logic
static inline uint8_t getPermuteOrder(std::vector<int64_t>& permutation) {
  // base permute order map
  std::map<uint8_t, uint8_t> base_permute_order_map;
  for (size_t i = 0; i < permutation.size(); ++i) {
    base_permute_order_map.insert({permutation[i], i});
  }
  // create copy of base permute order map for dot product
  std::map<uint8_t, uint8_t> permute_order_map{base_permute_order_map};

  // function to check the identity permutation
  auto check_identity{[&]() -> bool {
    for (const auto& p : permute_order_map) {
      if (p.first != p.second) {
        return false;
      }
    }
    return true;
  }};

  uint8_t order = 0;
  while (!check_identity()) {
    // calculate dot product w.r.t. base order map and update
    for (auto& p : permute_order_map) {
      auto key = p.second;
      auto val = base_permute_order_map[key];
      p.second = val;
    }
    order++;
  }
  return order;
}

// Check if strided view node can be permuted
static bool isStridedViewPermuted(StridedViewAndPermuteInfo& info) {
  constexpr int64_t fcd_stride = 1;
  if (info.strides.back() == fcd_stride) {
    return false;
  }

  const auto& shapes = info.shapes;
  const auto& strides = info.strides;
  auto& new_shapes = info.newShapes;
  auto& new_strides = info.newStrides;

  // strides detected at fcd dimension i.e. 'x', get permutation values
  auto& permutation = info.permutation;
  permutation.resize(strides.size());
  std::iota(permutation.begin(), permutation.end(), 0);

  // sort permutation indices by the stable order of the strides
  std::stable_sort(
      permutation.begin(), permutation.end(), [&strides](auto i, auto j) {
        return strides[i] > strides[j];
      });

  // update new shapes and new strides w.r.t permutation values
  new_shapes.resize(permutation.size());
  new_strides.resize(permutation.size());
  std::transform(
      permutation.begin(),
      permutation.end(),
      new_shapes.begin(),
      [&shapes](auto i) { return shapes[i]; });
  std::transform(
      permutation.begin(),
      permutation.end(),
      new_strides.begin(),
      [&strides](auto i) { return strides[i]; });

  // check if permuted True on strides and permuted False on permuted strides
  // get permute order and update permutation i.e. combine multiple permutes
  if (isPermuted(strides) && (!isPermuted(new_strides))) {
    auto permute_order = getPermuteOrder(permutation);
    HABANA_ASSERT(
        permute_order >= 1,
        "Expected min permute order 1, but got ",
        permute_order);
    PT_EAGER_DEBUG("permute order: ", permute_order);
    PT_EAGER_DEBUG("orginal permutation: ", permutation);

    // Update permutation w.r.t org permutation for order > 1
    std::vector<int64_t> orgPermutation{permutation};
    std::vector<int64_t> prevPermutation(permutation.size());
    for (auto i = 1; i < permute_order; ++i) {
      prevPermutation.swap(permutation);
      std::transform(
          prevPermutation.begin(),
          prevPermutation.end(),
          permutation.begin(),
          [&orgPermutation](auto i) { return orgPermutation[i]; });
      PT_EAGER_DEBUG("updated permutation: ", permutation);
    }
    return true;
  }
  return false;
}

void HandleStridedViewsAndInsertPermute(
    std::shared_ptr<torch::jit::Graph>& graph) {
  PT_EAGER_TRACE;
  static const std::set<c10::Symbol> strided_view_symbols{
      c10::Symbol::fromQualString("aten::as_strided"),
      c10::Symbol::fromQualString("hpu::strided_view")};

  // Step 1
  // Pass to collect strided_view nodes and their info with strides on FCD
  // if such a node can be represented using contiguous view and permute(s)
  std::vector<StridedViewAndPermuteInfo> strided_view_permute_info_vec;
  for (auto* node : graph->nodes()) {
    if (strided_view_symbols.find(node->kind()) != strided_view_symbols.end()) {
      // Check for meta atrribute for strided view
      auto meta = torch::jit::attr::arg1;
      if (node->hasAttribute(meta)) {
        continue;
      }

      const auto node_output_tensor =
          node->output(0)->type()->cast<c10::TensorType>();
      TORCH_CHECK(
          node_output_tensor != nullptr, "Expected node output as a tensor");
      if (!node_output_tensor->scalarType().has_value()) {
        // skip op if dtype is not propagated
        //
        // It can happen when we create JIT graph out of FX graph, the converter
        // does not propagate dtypes from FX graph. It occurs when user will
        // explicitly use as_strided in its script. The case addressed by this
        // pass is when GraphExec adds as_strided for all non-contiguous view
        // inputs. When they are added, dtype is correctly set. Those are the
        // ones that we want to decompose to permutes.
        continue;
      }

      // Check node output data type if permute op can be supported
      TORCH_CHECK(
          node->output(0)->isCompleteTensor() == true,
          "Expected node output as a complete tensor");

      const auto dtype = node_output_tensor->scalarType().value();
      const auto device = node_output_tensor->device().value();
      const auto shapes = toIValue(node->input(1)).value().toIntVector();
      const auto strides = toIValue(node->input(2)).value().toIntVector();
      StridedViewAndPermuteInfo info(node, dtype, device, shapes, strides);

      PT_EAGER_DEBUG("strided_view shapes - ", info.shapes);
      PT_EAGER_DEBUG("strided_view strides - ", info.strides);

      if (isStridedViewPermuted(info)) {
        strided_view_permute_info_vec.emplace_back(info);
        PT_EAGER_DEBUG("strided_view new shapes - ", info.newShapes);
        PT_EAGER_DEBUG("strided_view new strides - ", info.newStrides);
        PT_EAGER_DEBUG("permutation - ", info.permutation);
      }
    }
  }

  // Step 2
  // Pass to replace collected strided_view nodes if any
  // with new strided view nodes with new shape and strides
  // and insert a permute node afterwards with the old shape
  if (!strided_view_permute_info_vec.empty()) {
    PT_EAGER_DEBUG(
        "\nPermute node insertion:=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        "[Before]",
        '\n',
        graph->toString(),
        "JIT_IR_Graph_END\n");
    const auto& op_strided_view =
        c10::Symbol::fromQualString("aten::as_strided");
    const auto& op_permute = c10::Symbol::fromQualString("aten::permute");
    for (auto info : strided_view_permute_info_vec) {
      auto node = info.node;
      torch::jit::WithInsertPoint insert_point(node);

      // Create and insert strided_view node with new shapes and new strides
      auto value_new_shapes =
          graph->insertConstant(torch::jit::IValue(info.newShapes));
      auto value_new_strides =
          graph->insertConstant(torch::jit::IValue(info.newStrides));

      auto new_node = graph->create(
          op_strided_view,
          {node->input(0), value_new_shapes, value_new_strides, node->input(3)},
          1);
      new_node->output(0)->setType(c10::TensorType::createContiguous(
          info.dtype, info.device, info.newShapes));

      habana::eager::set_deterministic(new_node);
      graph->insertNode(new_node);
      node->output(0)->replaceAllUsesWith(new_node->output(0));

      // Create permute node w.r.t permute order with old shape
      // and insert it after new strided_view node
      auto value_input = new_node->output(0);
      auto value_permutation =
          graph->insertConstant(torch::jit::IValue(info.permutation));

      auto permute_node =
          graph->create(op_permute, {value_input, value_permutation}, 1);
      permute_node->output(0)->setType(c10::TensorType::createContiguous(
          info.dtype, info.device, info.shapes));

      habana::eager::set_deterministic(permute_node);
      graph->insertNode(permute_node);
      value_input->replaceAllUsesAfterNodeWith(
          permute_node, permute_node->output(0));

      // destroy old strided_view node
      // and its data i.e. old shapes and strides nodes
      auto old_shapes_node = node->input(1)->node();
      auto old_strides_node = node->input(2)->node();
      node->destroy();
      old_shapes_node->destroy();
      old_strides_node->destroy();
    }
    PT_EAGER_DEBUG(
        "\nPermute node insertion:=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        "[After]",
        '\n',
        graph->toString(),
        "JIT_IR_Graph_END\n");
  } else {
    PT_EAGER_DEBUG(
        "\nPermute node insertion not required.=====================\n");
  }
}

} // namespace pass
} // namespace graph
} // namespace habana
