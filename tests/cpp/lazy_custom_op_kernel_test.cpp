#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/habana_lazy_custom.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyCustomKernelKernelTest : public habana_lazy_test::LazyTest {
 public:
  LazyCustomKernelKernelTest() {
    register_custom_add();
    register_custom_gelu();
  }

 private:
  void register_custom_add() {
    // Registering ustom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};
    // output desc
    habana::custom_op::OutputDesc output_desc{0};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_add", "add_fwd_f32", inputs_desc, outputs_desc);
  }

  void register_custom_gelu() {
    // Registering ustom_op::custom_gelu
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    std::vector<habana::custom_op::InputDesc> inputs_desc{input_a_desc};
    // output desc
    habana::custom_op::OutputDesc output_desc{0};
    habana::custom_op::OutputDesc output_desc_2{1};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc, output_desc_2};
    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_gelu", "gelu_fwd_f32", inputs_desc, outputs_desc);
  }
};

std::vector<at::Tensor> custom_add_execute(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  std::vector<c10::IValue> inputs{input_a, input_b};
  auto op_desc =
      habana::KernelRegistry().get_custom_op_desc("custom_op::custom_add");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return output;
}

std::vector<at::Tensor> custom_gelu_execute(torch::Tensor input_a) {
  std::vector<c10::IValue> inputs{input_a};
  auto op_desc =
      habana::KernelRegistry().get_custom_op_desc("custom_op::custom_gelu");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return output;
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_add(Tensor self, Tensor other) -> Tensor[]");
  m.def("custom_gelu(Tensor self) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_add", custom_add_execute);
  m.impl("custom_gelu", custom_gelu_execute);
}

TEST_F(LazyCustomKernelKernelTest, BinaryOp) {
  SetSeed();
  torch::Tensor input_a_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  torch::Tensor input_b_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  torch::Tensor results_cpu = input_a_cpu.add(input_b_cpu);

  torch::Tensor input_a = input_a_cpu.to(torch::kHPU);
  torch::Tensor input_b = input_b_cpu.to(torch::kHPU);

  std::vector<at::Tensor> result = custom_add_execute(input_a, input_b);
  auto hl_result = GetHbLazyTensor(result.at(0));

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  std::vector<at::Tensor> input_list{input_a, input_b};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("custom_op::custom_add")
      ->run(*hlexec->get_graph());

  bool equal = results_cpu.allclose(result.at(0).to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyCustomKernelKernelTest, MultipleOutputs) {
  SetSeed();
  torch::Tensor input_a_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  torch::Tensor input_a = input_a_cpu.to(torch::kHPU);

  torch::Tensor results_cpu = torch::nn::functional::gelu(input_a);

  std::vector<at::Tensor> result = custom_gelu_execute(input_a);
  auto hl_result = GetHbLazyTensor(result.at(0));

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  std::vector<at::Tensor> input_list{input_a};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("custom_op::custom_gelu")
      ->run(*hlexec->get_graph());

  bool equal = results_cpu.allclose(result.at(0).to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}