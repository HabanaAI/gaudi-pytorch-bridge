/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_lazy_test_infra.h"

namespace habana_lazy_test {

// Create a 3 Node vector from first level IR
// This is what is expected after a post order traversal
// of the first level IR
PostOrderTestStruct GetPostOrderNodes(bool jumbled) {
  PostOrderTestStruct post_order_struct;

  auto* add_node = new habana_lazy::ir::Node(torch::jit::aten::add);
  auto* sub_node = new habana_lazy::ir::Node(torch::jit::aten::sub);
  auto* mul_node = new habana_lazy::ir::Node(torch::jit::aten::mul);

  if (!jumbled) {
    post_order_struct.post_order_nodes.emplace_back(add_node);
    post_order_struct.post_order_nodes.emplace_back(sub_node);
    post_order_struct.post_order_nodes.emplace_back(mul_node);

    post_order_struct.post_order_str =
        "%0 = hpu::input()"
        "%1 = hpu::input()"
        "%2 = aten::add(%0, %1)"
        "%3 = aten::sub(%2, %0)"
        "%4 = aten::mul(%3, %0)";
  } else {
    post_order_struct.post_order_nodes.emplace_back(mul_node);
    post_order_struct.post_order_nodes.emplace_back(sub_node);
    post_order_struct.post_order_nodes.emplace_back(add_node);

    post_order_struct.post_order_str =
        "%0 = hpu::input()"
        "%1 = hpu::input()"
        "%2 = aten::mul(%0, %1)"
        "%3 = aten::sub(%2, %0)"
        "%4 = aten::add(%3, %0)";
  }

  return post_order_struct;
}

// Create input IValues.
// tensor_shapes creates n tensors with given shapes.
// scalars creates m scalars with given value
std::vector<torch::jit::IValue> CreateInputs(
    std::vector<std::vector<int64_t>> tensor_shapes,
    std::vector<float> scalars) {
  std::vector<torch::jit::IValue> input_ivalues;

  for (const auto shape : tensor_shapes) {
    torch::Tensor t = torch::randn(shape);
    input_ivalues.emplace_back(at::IValue{t});
  }

  for (const auto val : scalars) {
    input_ivalues.emplace_back(at::IValue{at::Scalar(val)});
  }

  return input_ivalues;
}

std::shared_ptr<torch::jit::Graph> CreateJITGraph() {
  // Create a JIT IR graph corresponding to the 3 nodes
  // This is similar to the JT graph that will be created
  // first time from the post order nodes.
  auto g = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor):
      %2 : int = prim::Constant[value=1]()
      %c : Tensor = aten::add(%a, %b, %2)
      %d : Tensor = aten::sub(%c, %b, %2)
      %6 : Tensor = aten::mul(%d, %a)
      return (%6))IR";
  // Create a JIT graph
  torch::jit::parseIR(graph_string, g.get());
  return g;
}

} // namespace habana_lazy_test
