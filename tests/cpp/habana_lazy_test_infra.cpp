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
#include <absl/strings/match.h>

#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"

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

    std::string post_order_str =
        "%0 = hpu::input()"
        "%1 = hpu::input()"
        "%2 = aten::add(%0, %1)"
        "%3 = aten::sub(%2, %0)"
        "%4 = aten::mul(%3, %0)";
    post_order_struct.post_order_nodes_hash =
        std::hash<std::string>{}(post_order_str);
  } else {
    post_order_struct.post_order_nodes.emplace_back(mul_node);
    post_order_struct.post_order_nodes.emplace_back(sub_node);
    post_order_struct.post_order_nodes.emplace_back(add_node);

    std::string post_order_str =
        "%0 = hpu::input()"
        "%1 = hpu::input()"
        "%2 = aten::mul(%0, %1)"
        "%3 = aten::sub(%2, %0)"
        "%4 = aten::add(%3, %0)";
    post_order_struct.post_order_nodes_hash =
        std::hash<std::string>{}(post_order_str);
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

torch::jit::Stack createStack(std::vector<at::Tensor>&& list) {
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

uint64_t EnvHelper::InitSeed() {
  // Fix seed as 0 by default
  uint64_t seed = 0;
  const char* s = std::getenv("PT_HPU_TEST_SEED");

  if (s) {
    int base = absl::StartsWith(s, "0x") ? 16 : 10;
    uint64_t val = std::stoul(s, nullptr, base);
    // Magic code to use random seed
    if (val == 0xDEADBEEF) {
      auto gen = at::detail::getDefaultCPUGenerator();
      return gen.seed();
    }
    return val;
  }

  return seed;
}
} // namespace habana_lazy_test

namespace jit_ir_test {
nlohmannV340::json read_json(std::string input_json) {
  std::ifstream infile(input_json);
  std::stringstream is;
  is << infile.rdbuf();
  infile.close();
  auto json_file_ = nlohmannV340::json::parse(is);
  return json_file_;
}

std::string get_jit_graph(nlohmannV340::json json_) {
  auto jit_json_vec = json_[0]["000000000"]["compilations"][0]["jit ir graph"];
  std::stringstream graph_ss;
  std::string graph_str;
  for (auto& jit_json : jit_json_vec) {
    std::string str = jit_json.get<std::string>();
    size_t pos = str.find(", scope");
    if (pos != std::string::npos) {
      str = str.substr(0, pos);
    }
    std::replace(str.begin(), str.end(), '/', '_');
    graph_str += str + '\n';
  }
  return graph_str;
}

at::Tensor create_empty_tensor(
    const std::vector<int64_t>& tshape,
    c10::TensorOptions& tensor_options,
    bool is_shape_tensor) {
  if (is_shape_tensor) {
    auto pt_tensor = habana_lazy::empty_hpu_lazy(
        tshape, tensor_options, c10::nullopt, false, SHAPE_TENSOR);
    return pt_tensor;
  }
  auto pt_tensor = at::empty(tshape, tensor_options);
  return pt_tensor;
}

std::map<std::string, c10::ScalarType> create_tensor_dtype_map(
    const at::ArrayRef<torch::jit::Value*>& inputs) {
  std::map<std::string, c10::ScalarType> tensor_dtype_map;
  for (int i = 0; i < inputs.size(); i++) {
    auto tp = inputs[i]->type()->cast<torch::jit::TensorType>();
    std::string name = std::to_string(i) + "_" + inputs[i]->debugName();
    tensor_dtype_map[name] = *tp->scalarType();
  }
  return tensor_dtype_map;
}

std::vector<at::Tensor> get_input_tensors(
    const std::map<std::string, std::string>& shapes_map,
    std::map<std::string, c10::ScalarType> tensor_dtype_map) {
  std::vector<at::Tensor> input_tensors;
  std::map<int, at::Tensor> idx_tensor_map;
  for (auto& name_shape_pair : shapes_map) {
    bool is_shape_tensor = false;
    std::string input_name = name_shape_pair.first;
    std::replace(input_name.begin(), input_name.end(), '/', '_');

    std::string input_shape(name_shape_pair.second);
    std::string shape_str = " shape tensor";
    std::string::size_type shape_tensor_pos = (input_shape).find(shape_str);

    size_t pos = input_name.find('_');
    int idx = std::stoi(input_name.substr(0, pos));

    if (shape_tensor_pos != std::string::npos) {
      input_shape.erase(shape_tensor_pos, shape_str.length());
      is_shape_tensor = true;
    }
    input_shape.erase(
        remove(input_shape.begin(), input_shape.end(), '['), input_shape.end());
    input_shape.erase(
        remove(input_shape.begin(), input_shape.end(), ']'), input_shape.end());

    std::vector<int64_t> tensor_shape;
    std::stringstream ss(input_shape);
    std::string item;
    while (std::getline(ss, item, ',')) {
      tensor_shape.push_back(stoi(item));
    }

    auto tensor_options = torch::TensorOptions().device("hpu").dtype(
        tensor_dtype_map[input_name]);
    auto pt_tensor =
        create_empty_tensor(tensor_shape, tensor_options, is_shape_tensor);
    PT_TEST_DEBUG(
        "Creating tensor for input[",
        idx,
        "]: ",
        input_name,
        habana_helpers::DebugString(pt_tensor));
    idx_tensor_map.emplace(idx, pt_tensor);
  }

  for (auto p : idx_tensor_map) {
    PT_TEST_DEBUG(
        "Adding tensor for input[",
        p.first,
        "]: ",
        habana_helpers::DebugString(p.second));
    input_tensors.push_back(p.second);
  }

  return input_tensors;
}
} // namespace jit_ir_test
