/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_operator.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy_test_infra.h"

using json = nlohmannV340::json;

using namespace habana_lazy;
using namespace at;

class GraphOptimizeTest : public habana_lazy_test::LazyTest {
 protected:
  void SetUp() override {
    ForceMode(1); // This test suite expects to run only with lazy=1

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
};

TEST_F(GraphOptimizeTest, PeepholeOptimTest) {
  torch::Tensor tensor_in = torch::randn({2, 3});

  torch::Tensor hl_tensor_in = tensor_in.to(torch::kHPU);

  auto result = torch::sigmoid(hl_tensor_in);
  auto result_t = torch::t(result);
  auto result_t_t = torch::t(result_t);
  result_t_t = torch::add(result_t_t, 1.0);
  auto hl_result = SyncAndGetHbLazyTensor(result_t_t);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetPeepholeOpt(true);

  std::vector<at::Tensor> input_list{hl_tensor_in};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_not("= aten::t")
      ->run(*hlexec->get_graph());
}

TEST_F(GraphOptimizeTest, SubGraphRewriteTest) {
  const char* fpath = "/tmp/pattern.json";
  setenv("HABANA_TRANSFORM_GRAPH_FILE", fpath, 1);

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
      "                   \" %r = aten::matmul(%a, %b)\",\n"
      "                   \" return (%r)\"\n"
      "                 ]\n"
      " }\n"
      "}\n";

  std::string marker_begin = "{\n";

  std::string patterns0 =
      " \"MmReluPattern0\" :\n"
      " {\n"
      "   \"Pattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %c = aten::mm[deterministic=0](%a, %b)\",\n"
      "                   \" %r = aten::relu[deterministic=0](%c)\",\n"
      "                   \" return (%r)\"\n"
      "                 ],\n"
      "   \"ReplacePattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %r = aten::matmul[deterministic=0](%a, %b)\",\n"
      "                   \" return (%r)\"\n"
      "                 ]\n"
      "},\n";

  std::string patterns1 =
      " \"MmReluPattern1\" :\n"
      " {\n"
      "   \"Pattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %c = aten::mm[deterministic=1](%a, %b)\",\n"
      "                   \" %r = aten::relu[deterministic=1](%c)\",\n"
      "                   \" return (%r)\"\n"
      "                 ],\n"
      "   \"ReplacePattern\" : [\n"
      "                   \"graph(%a, %b):\",\n"
      "                   \" %r = aten::matmul[deterministic=1](%a, %b)\",\n"
      "                   \" return (%r)\"\n"
      "                 ]\n"
      "}\n";

  std::string marker_end = "\n}";

  std::ofstream out(fpath, std::ofstream::out);
  out << marker_begin;
  out << patterns0;
  out << patterns1;
  out << marker_end;
  out.close();

  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor outHabana1 = torch::mm(hA, hB);
  torch::Tensor outHabana = torch::relu(outHabana1);

  auto hl_result = SyncAndGetHbLazyTensor(outHabana);
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
      .check_not("= aten::mm")
      ->check_not("= aten::relu")
      ->run(*hlexec->get_graph());

  unsetenv("HABANA_TRANSFORM_GRAPH_FILE");
  remove(fpath);
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

  torch::Tensor hl_tensor_in1 = tensor_in1.to(torch::kHPU);
  torch::Tensor hl_tensor_in2 = tensor_in2.to(torch::kHPU);
  auto result_t = torch::t(hl_tensor_in1);
  auto result_mm_1 = torch::mm(result_t, hl_tensor_in2);
  auto result_1 = torch::t(result_mm_1);

  auto result_mm_2 = torch::mm(hl_tensor_in2, result_t);
  auto result_2 = torch::t(result_mm_2);
  auto result = torch::add(result_1, result_2);

  auto hl_result = SyncAndGetHbLazyTensor(result);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetFuseTMM(true);

  std::vector<at::Tensor> input_list{hl_tensor_in1, hl_tensor_in2};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));
  // if running hash enabled set the hash
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
    auto& graph_hash_builder = GraphHashBuilder::getInstance();
    uint64_t fwd_running_hash = graph_hash_builder.getFwdRunningHash();
    hlexec->set_fwd_graph_hash(fwd_running_hash);
  }
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= hpu::mm_t", 2)
      ->check_not("= aten::t")
      ->check_not("= aten::mm")
      ->run(*hlexec->get_graph());

  torch::Tensor out_hpu = result.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, out_hpu), true);
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

  torch::Tensor hl_tensor_in1 = tensor_in1.to(torch::kHPU);
  torch::Tensor hl_tensor_in2 = tensor_in2.to(torch::kHPU);
  auto result_t = torch::t(hl_tensor_in1);
  auto result_mm_1 = torch::mm(result_t, hl_tensor_in2);
  auto result_1 = torch::t(result_mm_1);

  auto result_mm_2 = torch::mm(hl_tensor_in2, result_t);
  auto result_2 = torch::t(result_mm_2);
  auto result = torch::add(result_1, result_2);

  auto hl_result = SyncAndGetHbLazyTensor(result);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetFuseTMM(true);

  std::vector<at::Tensor> input_list{hl_tensor_in1, hl_tensor_in2};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));
  // if running hash enabled set the hash
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
    auto& graph_hash_builder = GraphHashBuilder::getInstance();
    uint64_t fwd_running_hash = graph_hash_builder.getFwdRunningHash();
    hlexec->set_fwd_graph_hash(fwd_running_hash);
  }
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= hpu::mm_t", 2)
      ->check_not("= aten::t")
      ->check_not("= aten::mm")
      ->run(*hlexec->get_graph());

  torch::Tensor out_hpu = result.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, out_hpu), true);
}

// input(NCHW) -> permute_cl -> conv2d -> relu
TEST_F(GraphOptimizeTest, PermutePassTest_CL) {
  // TODO: Removed once make sure removed from all tests lists
  return;
}

// input(CL) -> conv2d -> relu
TEST_F(GraphOptimizeTest, PermutePassTest_Contig) {
  // TODO: Removed once make sure removed from all tests lists
}

// input(NCHW) -> conv2d -> relu
TEST_F(GraphOptimizeTest, DISABLED_PermutePassTest_NCHW) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
  // HPU graph input in nchw format, permut gets added, weights need permute
  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);
  auto result1 = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto result = torch::relu(result1);
  Tensor out = result.to(kCPU);
  // CPU graph
  auto exp1 = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto exp = torch::relu(exp1);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// input(NCHW) -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_NCHW_InplaceLeaky) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
  // HPU graph input in nchw format, permut gets added, weights need permute
  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);
  auto result = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(result);
  Tensor out = result.to(kCPU);
  // CPU graph
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(exp);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// input(CL) -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_InplaceCL) {
  // TODO: Removed once make sure removed from all tests lists
}

// input(NCHW) -> permute_cl_hpu -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_Permute_Inplace) {
  // TODO: Removed once make sure removed from all tests lists
}

// input0(NCHW) -> conv2d -> leaky_relu_ -> abs_
TEST_F(GraphOptimizeTest, DISABLED_PermutePassTest_DoubleInplace) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto h_wt = wt.to(torch::kHPU);
  auto h_in = in.to(torch::kHPU);
  // HPU graph
  auto result = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(result);
  torch::abs_(result);
  Tensor out = result.to(kCPU);
  // CPU graph
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(exp);
  torch::abs_(exp);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// Input(NCHW) -> Add_(input1(CL)) -> conv2D -> leakyRelu_
TEST_F(GraphOptimizeTest, PermutePassTest_Add_Inplace) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto in1 = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto h_wt = wt.to(torch::kHPU);
  auto h_in = in.to(torch::kHPU);
  auto h_in1 = in1.to(torch::kHPU);
  // HPU graph
  h_in = torch::add(h_in, h_in1, 1.0);
  auto result = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(result);
  Tensor out = result.to(kCPU);
  // CPU graph
  in = torch::add(in, in1, 1.0);
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(exp);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// Input(NCHW) -> Add_(input1(CL)) -> conv2D -> leakyRelu_
TEST_F(GraphOptimizeTest, PermutePassTest_Add_Inplace_MF) {
  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto in1 = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto in2 = torch::zeros({1, 4, 28, 28}); // nchw

  auto h_wt = wt.to(torch::kHPU);
  auto h_in1 = in1.to(torch::kHPU);
  auto h_in2 = in2.to(torch::kHPU);

  auto h_in_permuted = permute_cl_hpu_lazy(h_in1, {0, 2, 3, 1});
  auto in_permuted = h_in_permuted.to(kCPU);

  // HPU graph
  auto h_out = torch::add(h_in_permuted, h_in2, 1.0);
  auto result = torch::conv2d(h_out, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(result);
  Tensor result_cpu = result.to(kCPU);

  // CPU graph
  auto out = torch::add(in_permuted, in2, 1.0);
  auto exp = torch::conv2d(out, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::leaky_relu_(exp);
  EXPECT_EQ(allclose(result_cpu, exp, 0.01, 0.01), true);
}

TEST_F(GraphOptimizeTest, PermutePassTestInplace_Debug) {
  auto in = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw

  auto wt = torch::randn(
      {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto h_wt = wt.to(torch::kHPU);
  auto h_in = in.to(torch::kHPU);
  // Input permute taken care by permute pass (PASS scenerio)
  torch::leaky_relu_(h_in);
  auto result1 = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto result = torch::relu(result1);
  Tensor out = result.to(kCPU);

  // CPU graph
  torch::leaky_relu_(in);
  auto exp1 = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto exp = torch::relu(exp1);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// Above test but runs in loop for caching behaviour

// input(NCHW) -> permute_cl -> conv2d -> relu
TEST_F(GraphOptimizeTest, PermutePassTest_CL_cache) {
  // TODO: Removed once make sure removed from all tests lists
}

// input(CL) -> conv2d -> relu
TEST_F(GraphOptimizeTest, PermutePassTest_Contig_cache) {
  // TODO: Removed once make sure removed from all tests lists
}

// input(NCHW) -> conv2d -> relu
TEST_F(GraphOptimizeTest, PermutePassTest_NCHW_cache) {
  for (int i = 0; i < 2; i++) {
    auto in = torch::randn(
        {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
    auto wt = torch::randn(
        {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
    // HPU graph input in nchw format, permut gets added, weights need permute
    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);
    auto result1 =
        torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    auto result = torch::relu(result1);
    Tensor out = result.to(kCPU);
    // CPU graph
    auto exp1 = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    auto exp = torch::relu(exp1);

    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

// input(NCHW) -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_NCHW_InplaceLeaky_cache) {
  for (int i = 0; i < 2; i++) {
    auto in = torch::randn(
        {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false));
    auto wt = torch::randn(
        {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
    // HPU graph input in nchw format, permut gets added, weights need permute
    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);
    auto result =
        torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(result);
    Tensor out = result.to(kCPU);
    // CPU graph
    auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(exp);

    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

// input(CL) -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_InplaceCL_cache) {
  // TODO: Removed once make sure removed from all tests lists
}

// input(NCHW) -> permute_cl_hpu -> conv2d -> leaky_relu_
TEST_F(GraphOptimizeTest, PermutePassTest_Permute_Inplace_cache) {
  // TODO: Removed once make sure removed from all tests lists
}

// input0(NCHW) -> conv2d -> leaky_relu_ -> abs_
TEST_F(GraphOptimizeTest, PermutePassTest_DoubleInplace_cache) {
  // TODO: Removed once make sure removed from all tests lists
}

// Input(NCHW) -> Add_(input1(CL)) -> conv2D -> leakyRelu_
TEST_F(GraphOptimizeTest, PermutePassTest_Add_Inplace_cache) {
  for (int i = 0; i < 2; i++) {
    auto in = torch::randn(
        {6, 4, 28, 28},
        torch::dtype(torch::kFloat).requires_grad(false)); // nchw
    auto in1 = torch::randn(
        {6, 4, 28, 28},
        torch::dtype(torch::kFloat).requires_grad(false)); // nchw
    auto wt = torch::randn(
        {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
    auto h_wt = wt.to(torch::kHPU);
    auto h_in = in.to(torch::kHPU);
    auto h_in1 = in1.to(torch::kHPU);
    // HPU graph
    h_in = torch::add(h_in, h_in1, 1.0);
    auto result =
        torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(result);
    Tensor out = result.to(kCPU);
    // CPU graph
    in = torch::add(in, in1, 1.0);
    auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(exp);

    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

// Input(NCHW) -> Add_(input1(CL)) -> conv2D -> leakyRelu_
TEST_F(GraphOptimizeTest, PermutePassTest_Add_Inplace_MF_cache) {
  for (int i = 0; i < 2; i++) {
    auto wt = torch::randn(
        {5, 4, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
    auto in1 = torch::randn(
        {6, 4, 28, 28},
        torch::dtype(torch::kFloat).requires_grad(false)); // nchw
    auto in2 = torch::zeros({1, 4, 28, 28}); // nchw

    auto h_wt = wt.to(torch::kHPU);
    auto h_in1 = in1.to(torch::kHPU);
    auto h_in2 = in2.to(torch::kHPU);

    auto h_in_permuted = permute_cl_hpu_lazy(h_in1, {0, 2, 3, 1});
    auto in_permuted = h_in_permuted.to(kCPU);

    // HPU graph
    auto h_out = torch::add(h_in_permuted, h_in2, 1.0);
    auto result =
        torch::conv2d(h_out, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(result);
    Tensor result_cpu = result.to(kCPU);

    // CPU graph
    auto out = torch::add(in_permuted, in2, 1.0);
    auto exp = torch::conv2d(out, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::leaky_relu_(exp);
    EXPECT_EQ(allclose(result_cpu, exp, 0.01, 0.01), true);
  }
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass1) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto hA_relu = torch::relu(hA);
  auto hB_relu = torch::relu(hB);
  hA_relu += hB_relu;
  auto h_Out = torch::relu(hA_relu);

  auto hl_result = SyncAndGetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetReplaceInplaceOps(true);

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_not("= aten::add_")
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass2) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto hB_relu = torch::relu(hB);
  hA += hB_relu;
  auto h_Out = torch::relu(hA);

  auto hl_result = SyncAndGetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetReplaceInplaceOps(true);

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));
  // if running hash enabled set the hash
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
    auto& graph_hash_builder = GraphHashBuilder::getInstance();
    uint64_t fwd_running_hash = graph_hash_builder.getFwdRunningHash();
    hlexec->set_fwd_graph_hash(fwd_running_hash);
  }
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= hpu::add_", 1)
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
}

TEST_F(GraphOptimizeTest, RemoveInplaceOps_pass3) {
  torch::Tensor A = torch::randn({4, 4});
  torch::Tensor B = torch::randn({4, 4});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto h_Out = torch::relu(hA);
  auto hB_relu = torch::relu(hB);
  h_Out += hB_relu;

  auto hl_result = SyncAndGetHbLazyTensor(h_Out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::OptPassCfg::GetInstance()->SetReplaceInplaceOps(true);

  std::vector<at::Tensor> input_list{hA, hB};
  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));
  // if running hash enabled set the hash
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
    auto& graph_hash_builder = GraphHashBuilder::getInstance();
    uint64_t fwd_running_hash = graph_hash_builder.getFwdRunningHash();
    hlexec->set_fwd_graph_hash(fwd_running_hash);
  }
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= hpu::add", 1)
      ->run(*hlexec->get_graph());

  Tensor Out = h_Out.to(kCPU);
}

TEST_F(GraphOptimizeTest, PermutePassReshapeHandling) {
  auto A = torch::randn({16});
  auto B = torch::randn({2, 3, 16, 8});
  auto wt = torch::randn({16, 3, 4, 4});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hwt = wt.to(torch::kHPU);
  auto hC = hA.reshape({1, -1, 1, 1});
  auto hConv = torch::conv2d(hB, hwt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto hRelu = hConv.relu();
  auto hOut = hC * hRelu;
  auto out = hOut.to(torch::kCPU);
}

TEST_F(GraphOptimizeTest, PermutePassIndexHandling) {
  auto A = torch::randn({2, 13, 5});
  auto B = torch::randn({2, 3, 16, 8});
  auto wt = torch::randn({16, 3, 4, 4});
  auto indices1 = torch::arange(2).to(torch::kHPU);
  auto indices2 = torch::arange(2).to(torch::kHPU);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hwt = wt.to(torch::kHPU);
  auto hConv = torch::conv2d(hB, hwt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto hRelu = hConv.relu();
  auto hIndex = torch::index(hRelu, {indices1, indices2});
  auto hOut = hA + hIndex;
  auto out = hOut.to(torch::kCPU);
}

TEST_F(GraphOptimizeTest, ConvCatConv) {
  auto input_tensor =
      torch::arange(90, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 6, 5}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  auto weight_tensor =
      torch::arange(36, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 2, 2}); // hwck
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
  auto input_tensor2 =
      torch::arange(90, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 6, 5}); // nchw
  torch::Tensor tHabanaX2 = input_tensor2.to(torch::kHPU);

  torch::Tensor outConv =
      torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor outConv2 =
      torch::conv2d(tHabanaX2, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor catOut = torch::cat({outConv, outConv2}, 1);

  auto weight_tensor2 =
      torch::arange(192, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({8, 6, 2, 2}); // hwck
  torch::Tensor tHabanaW2 = weight_tensor2.to(torch::kHPU);
  torch::Tensor outConv3 =
      torch::conv2d(catOut, tHabanaW2, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto out = outConv3.to(torch::kCPU);
}

TEST_F(GraphOptimizeTest, CatTest) {
  auto input_tensor0 = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  torch::Tensor tHabanaX0 = input_tensor0.to(
      torch::kHPU,
      c10::ScalarType::Float,
      false,
      false,
      c10::MemoryFormat::ChannelsLast);

  auto input_tensor1 = torch::randn(
      {6, 4, 28, 28}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  torch::Tensor tHabanaX1 = input_tensor1.to(
      torch::kHPU,
      c10::ScalarType::Float,
      false,
      false,
      c10::MemoryFormat::ChannelsLast);

  torch::Tensor catOut = torch::cat({tHabanaX0, tHabanaX1}, 1);
  auto out = catOut.to(torch::kCPU);
}

TEST_F(GraphOptimizeTest, WeightExpandViewTest) {
  auto weight0 = torch::randn(
      {32, 32, 2}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw
  auto in = torch::randn(
      {8, 32, 1, 63}, torch::dtype(torch::kFloat).requires_grad(false));
  at::Tensor wt = weight0.view({32, 32, 1, 2});
  auto exp1 = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0, 0}, {1}, 1);
  auto exp = torch::relu(exp1);
  auto h_in = in.to(torch::kHPU);
  auto h_weight0 = weight0.to(torch::kHPU);
  auto h_wt = torch::as_strided(h_weight0, wt.sizes(), wt.strides(), 0);
  auto result1 = torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto result = torch::relu(result1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}
