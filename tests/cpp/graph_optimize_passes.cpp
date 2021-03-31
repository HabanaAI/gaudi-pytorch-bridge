#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using json = nlohmannV340::json;
using namespace habana_lazy;

using namespace habana_lazy;

class GraphOptimizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
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
      "                   \" %r = aten::mmrelu(%a, %b)\",\n"
      "                   \" return (%r)\"\n"
      "                 ]\n"
      " }\n"
      "}\n";

  std::ofstream out("pattern.json");
  out << patterns;
  out.close();

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
  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  exec::OptPassCfg::GetInstance()->enable_permute_pass = false;

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_not("aten::mm")
      ->check_not("aten::relu")
      ->check_count("aten::mmrelu", 1)
      ->run(*hlexec->get_graph());

  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
  exec::OptPassCfg::GetInstance()->enable_permute_pass = true;
}

TEST_F(GraphOptimizeTest, FuseMmTransposeTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(GraphOptimizeTest, BnReluOptTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(GraphOptimizeTest, PermutePassTest_CL) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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

  exec::OptPassCfg::GetInstance()->enable_permute_pass = true;
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  exec::OptPassCfg::GetInstance()->enable_permute_pass = false;
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(GraphOptimizeTest, PermutePassTest_Contig) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
  auto exp1 = torch::conv2d(in, wt, {}, 1, 0, 1, 1);
  auto exp = torch::relu(exp1);

  auto h_in = in.to(torch::kHABANA);
  auto h_in_cl = permute_cl_hpu_lazy(h_in, {0, 2, 3, 1});
  exec::OptPassCfg::GetInstance()->enable_permute_pass = true;
  Tensor h_in_cl_out = h_in_cl.to(kCPU);

  auto h_in1 = h_in_cl_out.to(torch::kHABANA);
  auto wt_hwck = wt.permute({2, 3, 1, 0}).contiguous();
  auto h_wt = wt_hwck.to(torch::kHABANA);

  auto result1 = torch::conv2d(h_in1, h_wt, {}, 1, 0, 1, 1);
  auto result = torch::relu(result1);

  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  exec::OptPassCfg::GetInstance()->enable_permute_pass = false;
  unsetenv("PT_HPU_LAZY_MODE");
}