#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using json = nlohmannV340::json;

using namespace habana_lazy;
using namespace at;

class GraphOptimizeTest : public habana_lazy_test::LazyTest {
 protected:
  void SetUp() override {
    ForceMode(1); // This test suite expects to run only with lazy=1
  }
};

TEST_F(GraphOptimizeTest, PeepholeOptimTest) {
  torch::Tensor tensor_in = torch::randn({2, 3});

  torch::Tensor hl_tensor_in = tensor_in.to(torch::kHABANA);

  auto result = torch::sigmoid(hl_tensor_in);
  auto result_t = torch::t(result);
  auto result_t_t = torch::t(result_t);
  auto hl_result = GetHbLazyTensor(result_t_t);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_peephole_optimization = true;

  std::vector<at::Tensor> input_list{hl_tensor_in};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck().check_not("aten::t")->run(
      *hlexec->get_graph());
  exec::OptPassCfg::GetInstance()->enable_peephole_optimization = false;
}

TEST_F(GraphOptimizeTest, SubGraphRewriteTest) {
  setenv("HABANA_TRANSFORM_GRAPH_FILE", "pattern.json", 1);

  // write to .json file patterens
  std::string patterns =
      "{\n"
      " \"MmReluPattern\" :\n"
      " {\n"
      "   \"Pattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %c = aten::mm(%a, %b)\",\n"
      "                   \" %r = aten::relu(%c)\",\n"
      "                   \" return (%r)\"\n"
      "                 ],\n"
      "   \"ReplacePattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %r = hpu::mmrelu(%a, %b)\",\n"
      "                   \" return (%r)\"\n"
      "                 ]\n"
      " }\n"
      "}\n";

  std::ofstream out("pattern.json");
  out << patterns;
  out.close();

  // Add the new kernel so that it does not assert while looking up in the pass
  static auto& KernelRegistry = habana::KernelRegistry().add(
      "hpu::mmrelu", [](const int device_id, c10::ScalarType node_type) {
        static_cast<void>(node_type);
        return std::make_shared<habana::HabanaOperator>("hpu::mmrelu");
      });

  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor outHabana1 = torch::mm(hA, hB);
  torch::Tensor outHabana = torch::relu(outHabana1);

  auto hl_result = GetHbLazyTensor(outHabana);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_not("aten::mm")
      ->check_not("aten::relu")
      ->check_count("hpu::mmrelu", 1)
      ->run(*hlexec->get_graph());
  unsetenv("HABANA_TRANSFORM_GRAPH_FILE");
  remove("pattern.json");
}

TEST_F(GraphOptimizeTest, FuseMmTransposeTest) {
  torch::Tensor tensor_in1 = torch::randn({4, 4});
  torch::Tensor tensor_in2 = torch::randn({4, 4});
  torch::Tensor out_t = torch::t(tensor_in1);
  torch::Tensor out_mm_1 = torch::mm(out_t, tensor_in2);
  torch::Tensor out_1 = torch::t(out_mm_1);

  torch::Tensor out_mm_2 = torch::mm(tensor_in2, out_t);
  torch::Tensor out_2 = torch::t(out_mm_2);
  torch::Tensor out_cpu = torch::add(out_1, out_2);

  torch::Tensor hl_tensor_in1 = tensor_in1.to(torch::kHABANA);
  torch::Tensor hl_tensor_in2 = tensor_in2.to(torch::kHABANA);
  auto result_t = torch::t(hl_tensor_in1);
  auto result_mm_1 = torch::mm(result_t, hl_tensor_in2);
  auto result_1 = torch::t(result_mm_1);

  auto result_mm_2 = torch::mm(hl_tensor_in2, result_t);
  auto result_2 = torch::t(result_mm_2);
  auto result = torch::add(result_1, result_2);

  auto hl_result = GetHbLazyTensor(result);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization = true;

  std::vector<at::Tensor> input_list{hl_tensor_in1, hl_tensor_in2};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("hpu::mm_t", 2)
      ->check_not("aten::t")
      ->check_not("aten::mm")
      ->run(*hlexec->get_graph());

  torch::Tensor out_hpu = result.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, out_hpu), true);
  exec::OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization = false;
}

TEST_F(GraphOptimizeTest, BnReluOptTest) {
  torch::Tensor tensor_in1 = torch::randn({4, 4});
  torch::Tensor tensor_in2 = torch::randn({4, 4});
  torch::Tensor out_t = torch::t(tensor_in1);
  torch::Tensor out_mm_1 = torch::mm(out_t, tensor_in2);
  torch::Tensor out_1 = torch::t(out_mm_1);

  torch::Tensor out_mm_2 = torch::mm(tensor_in2, out_t);
  torch::Tensor out_2 = torch::t(out_mm_2);
  torch::Tensor out_cpu = torch::add(out_1, out_2);

  torch::Tensor hl_tensor_in1 = tensor_in1.to(torch::kHABANA);
  torch::Tensor hl_tensor_in2 = tensor_in2.to(torch::kHABANA);
  auto result_t = torch::t(hl_tensor_in1);
  auto result_mm_1 = torch::mm(result_t, hl_tensor_in2);
  auto result_1 = torch::t(result_mm_1);

  auto result_mm_2 = torch::mm(hl_tensor_in2, result_t);
  auto result_2 = torch::t(result_mm_2);
  auto result = torch::add(result_1, result_2);

  auto hl_result = GetHbLazyTensor(result);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization = true;

  std::vector<at::Tensor> input_list{hl_tensor_in1, hl_tensor_in2};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("hpu::mm_t", 2)
      ->check_not("aten::t")
      ->check_not("aten::mm")
      ->run(*hlexec->get_graph());

  torch::Tensor out_hpu = result.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, out_hpu), true);
  exec::OptPassCfg::GetInstance()->enable_fuse_t_mm_optimization = false;
}

TEST_F(GraphOptimizeTest, PermutePassTest_CL) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
  auto exp1 = torch::conv2d(in, wt, {}, 1, 0, 1, 1);
  auto exp = torch::relu(exp1);

  auto h_in = in.to(torch::kHABANA);
  // add permute-cl
  auto h_in_cl = permute_cl_hpu_lazy(h_in, {0, 2, 3, 1});
  auto wt_hwck = wt.permute({2, 3, 1, 0}).contiguous();
  auto h_wt = wt_hwck.to(torch::kHABANA);

  auto result1 = torch::conv2d(h_in_cl, h_wt, {}, 1, 0, 1, 1);
  auto result = torch::relu(result1);

  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

TEST_F(GraphOptimizeTest, PermutePassTest_Contig) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
  auto exp1 = torch::conv2d(in, wt, {}, 1, 0, 1, 1);
  auto exp = torch::relu(exp1);

  auto h_in = in.to(torch::kHABANA);
  auto h_in_cl = permute_cl_hpu_lazy(h_in, {0, 2, 3, 1});
  Tensor h_in_cl_out = h_in_cl.to(kCPU);

  auto h_in1 = h_in_cl_out.to(torch::kHABANA);
  auto wt_hwck = wt.permute({2, 3, 1, 0}).contiguous();
  auto h_wt = wt_hwck.to(torch::kHABANA);

  auto result1 = torch::conv2d(h_in1, h_wt, {}, 1, 0, 1, 1);
  auto result = torch::relu(result1);

  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass1) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);

  auto hA_relu = torch::relu(hA);
  auto hB_relu = torch::relu(hB);
  hA_relu += hB_relu;
  auto h_Out = torch::relu(hA_relu);

  auto hl_result = GetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = true;

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_not("aten::add_")
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = false;
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass2) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);

  auto hB_relu = torch::relu(hB);
  hA += hB_relu;
  auto h_Out = torch::relu(hA);

  auto hl_result = GetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = true;

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("aten::add_", 1)
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = false;
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass3) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);

  auto h_Out = torch::relu(hA);
  auto hB_relu = torch::relu(hB);
  h_Out += hB_relu;

  auto hl_result = GetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = true;

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("aten::add_", 1)
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
  exec::OptPassCfg::GetInstance()->enable_replace_inplace_ops = false;
}

TEST_F(GraphOptimizeTest, PermutePassReshapeHandling) {
  auto A = torch::randn({16});
  auto B = torch::randn({2, 3, 16, 8});
  auto wt = torch::randn({4, 4, 3, 16});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto hwt = wt.to(torch::kHABANA);
  auto hC = hA.reshape({1, -1, 1, 1});
  auto hConv = torch::conv2d(hB, hwt, {}, 1, 0, 1, 1);
  auto hRelu = hConv.relu();
  auto hOut = hC * hRelu;
  auto out = hOut.to(torch::kCPU);
}

TEST_F(GraphOptimizeTest, PermutePassIndexHandling) {
  auto A = torch::randn({2, 13, 5});
  auto B = torch::randn({2, 3, 16, 8});
  auto wt = torch::randn({4, 4, 3, 16});
  auto indices1 = torch::arange(2).to(torch::kHABANA);
  auto indices2 = torch::arange(2).to(torch::kHABANA);
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto hwt = wt.to(torch::kHABANA);
  auto hConv = torch::conv2d(hB, hwt, {}, 1, 0, 1, 1);
  auto hRelu = hConv.relu();
  auto hIndex = torch::index(hRelu, {indices1, indices2});
  auto hOut = hA + hIndex;
  auto out = hOut.to(torch::kCPU);
}