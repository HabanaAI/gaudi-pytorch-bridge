#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;

class LazyKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyKernelTest, LazyDoATest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor I = torch::add(hA, hB);
  torch::Tensor out = torch::add(hC, I);
  EXPECT_EQ(out.dim(), 2);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, BasicCopyTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, ConvReluTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 3, 3}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHABANA);

  auto bias_tensor =
      torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 1, 1, 1});
  torch::Tensor tHabanaB = bias_tensor.to(torch::kHABANA);

  torch::Tensor outConv = torch::conv2d(tHabanaX, tHabanaW, {}, 1, 0, 1, 1);
  torch::Tensor outHabana = torch::relu(outConv);

  // Match lazy IR graph
  auto hl_result = GetHbLazyTensor(outHabana);
  auto ir_value = hl_result.CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("prim::Constant()")
      ->check_count("prim::Constant[value=[1, 1]]", 2)
      ->check("prim::Constant[value=0]")
      ->check("prim::Constant[value=1]")
      ->check("aten::convolution_overidable")
      ->run(*exec.get_graph());

  torch::jit::testing::FileCheck()
      .check_count("prim::Constant[value=[0, 0]]", 2)
      ->run(*exec.get_graph());

  auto out_string = IrGraphDumpUtil::ToText(a);
  EXPECT_EQ(
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = hpu::input()\n"
          "  %2 = aten::convolution_overidable(%1, %0), stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, output_padding=[0, 0], groups=1\n"
          "  %3 = aten::relu(%2), ROOT=0\n"
          "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({5265}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, MmMulTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto x = torch::randn({2, 3});
  auto y = torch::randn({3, 3});
  auto z = torch::randn({2, 3});
  torch::Tensor hx = x.to(torch::kHABANA);
  torch::Tensor hy = y.to(torch::kHABANA);
  torch::Tensor hz = z.to(torch::kHABANA);

  auto hy_exp = torch::mm(hx, hy);
  auto hz_exp = torch::mul(hy_exp, hz);
  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hz_exp));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = hpu::input()\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = hpu::input()\n"
                      "  %3 = aten::mm(%2, %1)\n"
                      "  %4 = aten::mul(%3, %0), ROOT=0\n"
                      "}"),
      0);

  // Match expectd output
  // ASSERT_TRUE(torch::allclose(hz_exp, hz_exp));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, AddMmTest) {
  torch::Tensor A = torch::randn({2});
  torch::Tensor B = torch::randn({2, 2});
  torch::Tensor C = torch::randn({2, 2});
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor hA = A.to(kHABANA);
  torch::Tensor hB = B.to(kHABANA);
  torch::Tensor hC = C.to(kHABANA);
  torch::Tensor O = torch::addmm(hA, hB, hC, 1, 1);
  std::string out =
      IrGraphDumpUtil::ToText({GetHbLazyTensor(O).CurrentIrValue().mp_node});
  EXPECT_EQ(
      out.find("IR {\n"
               "  %0 = prim::constant(), value=1\n"
               "  %1 = prim::constant(), value=1\n"
               "  %2 = hpu::input()\n"
               "  %3 = hpu::input()\n"
               "  %4 = hpu::input()\n"
               "  %5 = aten::addmm(%4, %3, %2, %1, %0), ROOT=0\n"
               "}"),
      !std::string::npos);
  unsetenv("PT_HPU_LAZY_MODE");
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(O)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto computed = O.to(torch::kCPU);
  auto expected = torch::addmm(A, B, C, 1, 1);

  EXPECT_EQ(allclose(expected, computed), true);
}

TEST_F(LazyKernelTest, CatTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor out = torch::cat({hA, hB, hC});

  auto hl_result = GetHbLazyTensor(out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("Tensor[] = prim::ListConstruct")
      ->check("int = prim::Constant[value=0]")
      ->check("Tensor = aten::cat")
      ->run(*exec.get_graph());
  // ASSERT_TRUE(torch::allclose(hz_exp, hz_exp));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, LocalScalarDenseTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({1}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);

  auto hl_result = GetOrCreateHbLazyTensor(A, A.device());

  // .item() invokes local scalar dense
  auto s = hA.item();
  auto s_cpu = A.item();

  EXPECT_EQ(s.to<float>(), s_cpu.to<float>());

  unsetenv("PT_HPU_LAZY_MODE");
}
TEST_F(LazyKernelTest, ConvMaxPoolTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHABANA);

  auto bias_tensor =
      torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 1, 1, 1});
  torch::Tensor tHabanaB = bias_tensor.to(torch::kHABANA);

  torch::Tensor outConv = torch::conv2d(tHabanaX, tHabanaW, {}, 1, 0, 1, 1);
  torch::Tensor outHabana = torch::max_pool2d(outConv, 2, 1);

  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(outHabana));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = hpu::input()\n"
          "  %2 = aten::convolution_overidable(%1, %0), stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, output_padding=[0, 0], groups=1\n"
          "  %3 = aten::maxpool2d_overidable(%2), kernel_size=[2], stride=[1], padding=[0], dilation=[1], transposed=[0], ROOT=0\n"
          "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({11952}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  // ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, MaxPoolBWDTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  // fwd propga
  torch::Tensor outHabana1 = torch::max_pool2d(tHabanaX, 2, 1);
  torch::Tensor outHabana = torch::relu(outHabana1);

  // bwd propga with dummy grad tensor
  auto grad_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 3, 3, 3});
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
  outHabana.backward({tHabanaG}, false, true);

  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(outHabana));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = aten::maxpool2d_overidable(%0), kernel_size=[2], stride=[1], padding=[0], dilation=[1], transposed=[0]\n"
          "  %2 = aten::relu(%1.0), ROOT=0\n"
          "}"),
      0);

  // auto expected = torch::tensor({11952}, torch::kFloat);
  // EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  // ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, OptSgdCustomOp) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt, 0.1, false);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = GetHbLazyTensor(out1);
  auto hl_moment = GetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::LazyValueToJitValueMap input_map, output_map;
  std::tie(input_map, output_map) =
      hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=0.10000000149011612]")
      ->check("prim::Constant[value=0]")
      ->check_count("habanaOptimizerSparseSgd", 1)
      ->run(*hlexec->get_graph());
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyKernelTest, OptAdagradCustomOp) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = GetHbLazyTensor(out1);
  auto hl_moment = GetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  exec::HlExec* hlexec = new exec::HlExec();
  exec::LazyValueToJitValueMap input_map, output_map;
  std::tie(input_map, output_map) =
      hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check_count("habanaOptimizerSparseAdagrad", 1)
      ->run(*hlexec->get_graph());
  unsetenv("PT_HPU_LAZY_MODE");
}