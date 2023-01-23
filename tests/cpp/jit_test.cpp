#include <tests/cpp/habana_lazy_test_infra.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"

using namespace habana_lazy;

class JIT_IR_test : public habana_lazy_test::LazyTest {};

TEST_F(JIT_IR_test, RerunJITGraph) {
  std::string json_name = GET_ENV_FLAG_NEW(PT_HPU_RERUN_JSON_FILE);
  if (json_name == "") {
    return;
  }

  habana_helpers::EnableRefineDynamicShape();
  LazyExecutionMode exec_mode{habana_lazy_executor.getExecutionMode()};
  habana_lazy_executor.setExecutionMode(LazyExecutionMode::kLOWERING);

  nlohmannV340::json json_file_ = jit_ir_test::read_json(json_name);
  std::string graph_string = jit_ir_test::get_jit_graph(json_file_);

  auto jit_ir_graph_ptr = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, jit_ir_graph_ptr.get());
  size_t num_inputs = jit_ir_graph_ptr->inputs().size();

  auto tensor_dtype_map =
      jit_ir_test::create_tensor_dtype_map(jit_ir_graph_ptr->inputs());
  for (const auto& index : json_file_.items()) {
    uint64_t key = atoi(index.key().c_str());
    std::string step = absl::StrFormat("%0*d", 9, key);
    std::vector<at::Tensor> input_tensors = jit_ir_test::get_input_tensors(
        index.value()[step]["shapes"], tensor_dtype_map);

    torch::jit::Stack input_stack =
        habana_lazy_test::createStack(std::move(input_tensors));
    TORCH_CHECK(
        num_inputs == input_stack.size(),
        "Input stack size=",
        input_stack.size(),
        " is not matching with #graph_inputs=",
        num_inputs);

    auto input_refs = torch::jit::last(input_stack, num_inputs);
    auto g_and_m_data = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
        jit_ir_graph_ptr, input_refs);
    habana::HabanaLaunchOpPT habanaLoweringOp{g_and_m_data};
    habanaLoweringOp.run(input_stack);
  }
  tensor_dtype_map.clear();
  habana_lazy_executor.setExecutionMode(exec_mode);
  habana_helpers::DisableRefineDynamicShape();
}
