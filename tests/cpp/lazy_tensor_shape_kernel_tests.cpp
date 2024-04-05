#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazyTensorShapeKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyTensorShapeKernelTest, CatExecTest1) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto C = torch::relu(A);
  auto D = torch::relu(B);
  auto exp = torch::cat({C, D});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);

  auto hC = torch::relu(hA);
  auto hD = torch::relu(hB);

  torch::Tensor out = torch::cat({hC, hD});
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest2) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto exp = torch::cat({A, B});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);

  torch::Tensor out = torch::cat({hA, hB});
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecOutTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  torch::Tensor output = torch::empty({0});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hout = output.to(torch::kHPU);

  torch::cat_outf({A, B}, 0, output);
  torch::cat_outf({hA, hB}, 0, hout);
  auto result = hout.to(torch::kCPU);
  EXPECT_EQ(allclose(result, output), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest3) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 4}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));

  auto tempc1 = torch::cat({A, B}, 1);
  auto exp = torch::cat({A, tempc1}, 1);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hC = B.to(torch::kHPU);

  torch::Tensor temp1 = torch::cat({hA, hB}, 1);
  auto out = torch::cat({hA, temp1}, 1);

  auto result = out.to(torch::kCPU);

  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest4) {
  torch::Tensor A = torch::randn({10, 2}, torch::requires_grad(false));

  auto B = torch::relu(A);
  auto C = torch::cat({B, B});
  auto exp = torch::relu(C);

  torch::Tensor hA = A.to(torch::kHPU);
  auto hB = torch::relu(hA);
  auto hC = torch::cat({hB, hB});
  auto out = torch::relu(hC);
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
}

// Also Validates InferOutputMeta for concat
TEST_F(LazyTensorShapeKernelTest, CatTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto A = torch::randn({1, 8, 8}).to(torch::kBFloat16);
  auto B = torch::randn({1, 8, 8}).to(torch::kBFloat16);
  auto C = torch::randn({1, 8, 8}).to(torch::kBFloat16);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  auto out_cpu = torch::cat({A, B, C});
  auto out_hpu = torch::cat({hA, hB, hC});

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyTensorShapeKernelTest, CatOutViewTest) {
  auto A = torch::randn({1, 8, 8}).to(torch::kBFloat16);
  auto B = torch::randn({1, 8, 8}).to(torch::kBFloat16);
  auto C = torch::randn({1, 8, 8}).to(torch::kBFloat16);

  auto aV = A.view({1, 64});
  auto bV = B.view({1, 64});
  auto cV = C.view({1, 64});

  auto hAV = A.to(torch::kHPU).view({1, 64});
  auto hBV = B.to(torch::kHPU).view({1, 64});
  auto hCV = C.to(torch::kHPU).view({1, 64});

  auto out_cpu = torch::zeros({3 * 8 * 8}).to(torch::kBFloat16).view({3, 64});
  auto out_hpu = torch::zeros({3 * 8 * 8})
                     .to(torch::kBFloat16)
                     .to(torch::kHPU)
                     .view({3, 64});

  torch::cat_out(out_cpu, {aV, bV, cV});
  torch::cat_out(out_hpu, {hAV, hBV, hCV});

  out_cpu.add_(1.0);
  out_hpu.add_(1.0);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
}

TEST_F(LazyTensorShapeKernelTest, IndexTest) {
  torch::Tensor input_cpu = torch::arange(4.0).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({{0, 1}, {1, 1}})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  EXPECT_TRUE(torch::equal(out_cpu, out_hpu.cpu()));
};

TEST_F(LazyTensorShapeKernelTest, PermuteTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({1, 0});
  torch::Tensor Out = A.permute({1, 0});

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, PermuteInplaceTest) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_PERMUTE_WITH_STRIDED_VIEW)) {
    torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = hA.permute({1, 0});
    hOut.add_(1.0);
    torch::Tensor Out = A.permute({1, 0});
    Out.add_(1.0);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
    EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
  }
}

TEST_F(LazyTensorShapeKernelTest, Permute6DTest) {
  torch::Tensor A =
      torch::randn({2, 3, 4, 3, 2, 6}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({1, 0, 2, 5, 3, 4});
  torch::Tensor Out = A.permute({1, 0, 2, 5, 3, 4});

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, PermuteTest7D) {
  torch::Tensor A =
      torch::randn({3, 2, 2, 2, 2, 3, 1}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({2, 0, 1, 6, 3, 4, 5});
  torch::Tensor Out = A.permute({2, 0, 1, 6, 3, 4, 5});

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, PermuteTest8D) {
  torch::Tensor A =
      torch::randn({3, 2, 2, 2, 2, 3, 1, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({1, 6, 7, 0, 5, 2, 3, 4});
  torch::Tensor Out = A.permute({1, 6, 7, 0, 5, 2, 3, 4});

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::t(hA);
  torch::Tensor Out = torch::t(A);

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);

  int64_t dim = 1;
  int64_t index = 0;

  Tensor h_out = torch::select(h_a, dim, index);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, index);

  EXPECT_EQ(equal(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectBackwardTest) {
  std::array<int64_t, 4> input_sizes = {8, 3, 28, 28};
  torch::Tensor a = torch::randn({8, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);

  int64_t dim = 1;
  int64_t index = 0;

  Tensor h_out = torch::select_backward(h_a, input_sizes, dim, index);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select_backward(a, input_sizes, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectBackwardTest_2) {
  double rtol = 0; // 1e-03;
  double atol = 0; // 1e-03;

  std::array<int64_t, 3> input_sizes = {5, 4, 2};

  torch::Tensor a = torch::randn({5, 2}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);

  int64_t dim = 1;
  int64_t index = 1;

  Tensor h_out = torch::select_backward(h_a, input_sizes, dim, index);
  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select_backward(a, input_sizes, dim, index);

  EXPECT_EQ(allclose(h_cout, cout, rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectBackwardTest_3) {
  double rtol = 0; // 1e-03;
  double atol = 0; // 1e-03;

  std::array<int64_t, 3> input_sizes = {5, 4, 2};

  torch::Tensor t = torch::randn({5, 4, 2}, torch::requires_grad(true));
  torch::Tensor h_t = t.to(torch::kHPU);

  torch::Tensor a = torch::randn({5, 2}, torch::requires_grad(true));
  torch::Tensor h_a = a.to(torch::kHPU);

  int64_t dim = 1;
  int64_t index = 1;

  torch::Tensor sel_h_out = torch::select(h_t, dim, index);

  auto sel_h_cout = sel_h_out.to(torch::kCPU);
  auto sel_cout = torch::select(t, dim, index);

  EXPECT_EQ(allclose(sel_h_cout, sel_cout), true);

  torch::Tensor h_out =
      torch::select_backward(sel_h_out, input_sizes, dim, index);
  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select_backward(sel_cout, input_sizes, dim, index);

  EXPECT_EQ(allclose(h_cout, cout, rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, SliceTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 8;
  int64_t step = 1;

  Tensor h_out = torch::slice(h_a, dim, start_index, end, step);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::slice(a, dim, start_index, end, step);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, SliceTestZeroDimSize) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 0;
  int64_t step = 1;

  auto aa = torch::add(a, a);
  auto cout = torch::slice(aa, dim, start_index, end, step);

  Tensor h_aa = torch::add(h_a, h_a);
  Tensor h_out = torch::slice(h_aa, dim, start_index, end, step);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, ViewExecute) {
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanain = input_tensor.to(torch::kHPU);
  std::array<int64_t, 2> size_array = {-1, 48};
  c10::IntArrayRef new_size = size_array;
  auto result = torch::_unsafe_view(tHabanain, new_size);
  auto hl_result =
      std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(result));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  at::Tensor result_lazy = result.to(torch::kCPU);
  auto result_cpu = torch::_unsafe_view(input_tensor, new_size);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

// Also Validates InferOutputMeta for transpose
TEST_F(LazyTensorShapeKernelTest, TransposeTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 1, 0);
  torch::Tensor Out = torch::transpose(A, 1, 0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also Validates InferOutputMeta for transpose
TEST_F(LazyTensorShapeKernelTest, TransposeTest2) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({8, 224, 224, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 0, 3);
  torch::Tensor Out = torch::transpose(A, 0, 3);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTestCL) {
  torch::Tensor A = torch::randn({8, 224, 224, 3}, torch::requires_grad(false))
                        .contiguous(c10::MemoryFormat::ChannelsLast);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 0, 3);
  torch::Tensor Out = torch::transpose(A, 0, 3);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeCloneTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 5}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 2, 3).clone();
  torch::Tensor Out = torch::transpose(A, 2, 3).clone();
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeInPlaceAddTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 5}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 2, 3);
  torch::Tensor Out = torch::transpose(A, 2, 3);
  hOut.add_(1);
  Out.add_(1);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest3) {
  torch::Tensor A =
      torch::randint(1, 24, {2, 3, 4}, torch::dtype(torch::kInt64));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 1, 2);
  torch::Tensor Out = torch::transpose(A, 1, 2);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest4) {
  torch::Tensor A =
      torch::randint(1, 24, {1, 3, 2}, torch::dtype(torch::kInt64));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, -1, -3);
  torch::Tensor Out = torch::transpose(A, -1, -3);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest5) {
  torch::Tensor A = torch::randint(0, 2, {2, 3, 4}, torch::dtype(torch::kBool));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 1, 2);
  torch::Tensor Out = torch::transpose(A, 1, 2);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest6) {
  torch::Tensor A = torch::randint(0, 2, {1, 3, 2}, torch::dtype(torch::kBool));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, -1, -3);
  torch::Tensor Out = torch::transpose(A, -1, -3);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest6D) {
  torch::Tensor A =
      torch::randn({3, 2, 2, 2, 2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 1, 2);
  torch::Tensor Out = torch::transpose(A, 1, 2);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest7D) {
  torch::Tensor A =
      torch::randn({3, 2, 2, 2, 2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 3, 2);
  torch::Tensor Out = torch::transpose(A, 3, 2);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest8D) {
  torch::Tensor A =
      torch::randn({3, 2, 2, 2, 2, 3, 4, 5}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 4, 1);
  torch::Tensor Out = torch::transpose(A, 4, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TTest2d) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::t(hA);
  torch::Tensor Out = torch::t(A);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TTestAddInplace) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::t(hA);
  hOut.add_(1.0);
  torch::Tensor Out = torch::t(A);
  Out.add_(1.0);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest) {
  torch::Tensor A = torch::randn({3, 1}, torch::requires_grad(false));

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.expand({3, 4}, false);
  torch::Tensor Out = A.expand({3, 4}, false);

  std::vector<HbLazyTensor> hl_tensors = {SyncAndGetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, Repeat) {
  torch::Tensor A = torch::randn({4, 5});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.repeat({2, 3});
  torch::Tensor Out = A.repeat({2, 3});

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

TEST_F(LazyTensorShapeKernelTest, RepeatInlv) {
  auto test = [](std::vector<int64_t> rpt_vals, int64_t dim, int64_t out_sz) {
    auto A = torch::randn({4, 5});
    auto rpt = torch::tensor(rpt_vals);
    auto hrpt = rpt.to(torch::kHPU);
    auto hA = A.to(torch::kHPU);
    if (dim != -1 && out_sz != -1) {
      auto hOut = hA.repeat_interleave(hrpt, dim, out_sz);
      auto Out = A.repeat_interleave(rpt, dim, out_sz);
      EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
    } else if (dim != -1) {
      auto hOut = hA.repeat_interleave(hrpt, dim);
      auto Out = A.repeat_interleave(rpt, dim);
      EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
    } else {
      auto hOut = hA.repeat_interleave(hrpt);
      auto Out = A.repeat_interleave(rpt);
      EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
    }
  };
  test({2}, -1, -1);
  test({2, 3, 1, 2}, 0, -1);
  test({2, 3, 1, 2, 3}, 1, -1);
  test({2, 1, 3, 1}, 0, 7);
}

TEST_F(LazyTensorShapeKernelTest, SplitWithSizesTest) {
  auto split_with_size = [](auto split_sizes, auto dim) {
    auto input = torch::randn({8, 3, 24, 12});
    auto h_input = input.to(torch::kHPU);

    auto result = at::native::split_with_sizes(h_input, split_sizes, dim);
    auto cpu_out = at::native::split_with_sizes(input, split_sizes, dim);

    std::vector<at::Tensor> hpu_out;
    hpu_out.reserve(result.size());
    for (const auto& ht : result) {
      hpu_out.push_back(ht.to(torch::kCPU));
    }
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_EQ(allclose(cpu_out[i], hpu_out[i]), true);
    }
  };
  std::array<int64_t, 3> size1 = {2, 4, 2};
  c10::IntArrayRef split_sizes = size1;
  int64_t dim = 0;
  split_with_size(split_sizes, dim);

  std::array<int64_t, 3> size2 = {12, 6, 6};
  split_sizes = c10::IntArrayRef(size2);
  dim = 2;
  split_with_size(split_sizes, dim);
}

TEST_F(LazyTensorShapeKernelTest, SplitTest) {
  auto input = torch::randn({2, 3, 4, 5});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  std::vector<at::Tensor> hpu_out;
  hpu_out.reserve(result.size());
  for (const auto& ht : result) {
    hpu_out.push_back(ht.to(torch::kCPU));
  }

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(allclose(cpu_out[i], hpu_out[i]), true);
  }
}

// Test case to check the basic split with a 6D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitTest6D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({4, 3, 4, 5, 2, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  std::vector<at::Tensor> hpu_out;
  hpu_out.reserve(result.size());
  for (const auto& ht : result) {
    hpu_out.push_back(ht.to(torch::kCPU));
  }

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(allclose(cpu_out[i], hpu_out[i], rtol, atol), true);
  }
}

// Test case to check the split with ChannelLast mem format.
TEST_F(LazyTensorShapeKernelTest, SplitViewContgCLTest) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input =
      torch::randn({4, 3, 2, 5}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add(1);
  splitCout1.add(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with a add op - on a view, created out
// of a 2D tensor with 3 splits along axis 1.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({4, 2});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with a add op - on a view(2nd), created out
// of a 2D tensor with 3 splits.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with a add op - on a view, created out
// of a 2D tensor with 2 splits.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_3) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 3, 1);
  auto cpu_out = torch::split(input, 3, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with a mul op - on a view (3rd), created out
// of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_4) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[2];
  auto splitCout1 = cpu_out[2];

  splitRes1.mul_(2);
  splitCout1.mul_(2);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with add and mul combination ops - on a view
// (2nd), created out of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_5) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.add_(1);
  splitRes1.mul_(4);
  splitCout1.add_(1);
  splitCout1.mul_(4);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with div op - on a view (2nd), created out
// of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_6) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.div_(2);
  splitCout1.div_(2);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with sub op - on a view (2nd), created out
// of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_7) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.sub_(1);
  splitCout1.sub_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates with add op - on views (1st 2nd), created out
// of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_8) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(2);
  splitCout1.add_(2);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto splitRes2 = result[1];
  auto splitCout2 = cpu_out[1];

  splitRes2.add_(2);
  splitCout2.add_(2);

  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes2.cpu(), splitCout2, rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

// Test case to check the updates with add, sub, mul ops - on views (1st 2nd and
// 3rd), created out of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_9) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(2);
  splitCout1.add_(2);

  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);

  auto splitRes2 = result[1];
  auto splitCout2 = cpu_out[1];

  splitRes2.sub_(1);
  splitCout2.sub_(1);

  EXPECT_EQ(allclose(splitRes2.cpu(), splitCout2, rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);

  auto splitRes3 = result[2];
  auto splitCout3 = cpu_out[2];

  splitRes3.mul_(2);
  splitCout3.mul_(2);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes3.cpu(), splitCout3, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes2.cpu(), splitCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views (2nd and 1st), created out of 2D
// tensor with two different (2 and 3) splits in axis 0 and 1.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest2D_10) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 4});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto result1 = torch::split(h_input, 3, 1);
  auto cpu_out1 = torch::split(input, 3, 1);

  auto splitRes2 = result1[0];
  auto splitCout2 = cpu_out1[0];

  splitRes2.add_(2);
  splitCout2.add_(2);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes2.cpu(), splitCout2, rtol, atol), true);
}

// Test case to check the updates on a view (1st), created out of 4D tensor
// 3 splits in axis 1.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest4D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({4, 2, 3, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on a view, created out of 4D tensor
// 2 splits in axis 1.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest4D_2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({4, 2, 3, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 3, 1);
  auto cpu_out = torch::split(input, 3, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on a view, created out of 6D tensor split.
TEST_F(LazyTensorShapeKernelTest, SplitViewTest6D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({4, 2, 3, 2, 2, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 1);
  auto cpu_out = torch::split(input, 2, 1);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views resulting from split the tensor
// in to 3 parts and cat (which is inverse of split) all of them
// resulting in original tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewCatTest) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  auto splitRes2 = result[1];
  auto splitCout2 = cpu_out[1];

  auto splitRes3 = result[2];
  auto splitCout3 = cpu_out[2];

  auto exp = torch::cat({splitCout1, splitCout2, splitCout3});

  torch::Tensor cat_out = torch::cat({splitRes1, splitRes2, splitRes3});
  auto cat_result = cat_out.to(torch::kCPU);
  EXPECT_EQ(allclose(cat_result, exp), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

// Test case to check the updates on views resulting from split and
// cat (which is inverse of split) on two different 2D tensor
// created out of a same parent 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewCatTest_2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  auto exp = torch::cat({splitCout1, input});

  torch::Tensor cat_out = torch::cat({splitRes1, h_input});
  auto cat_result = cat_out.to(torch::kCPU);
  EXPECT_EQ(allclose(cat_result, exp), true);

  exp.add_(2);
  cat_out.add_(2);

  auto exp1 = torch::cat({input, splitCout1});

  torch::Tensor cat_out1 = torch::cat({h_input, splitRes1});
  auto cat_result1 = cat_out1.to(torch::kCPU);

  EXPECT_EQ(allclose(cat_result1, exp1), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views resulting from split and
// slice created out of a 2D tensor with same shape.
TEST_F(LazyTensorShapeKernelTest, SplitViewSliceTest2D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto cout = torch::slice(input, 0, 0, 2);
  Tensor h_out = torch::slice(h_input, 0, 0, 2);

  auto sliceCout2 = cout;
  auto sliceRes2 = h_out;

  sliceRes2.add_(2);
  sliceCout2.add_(2);

  EXPECT_EQ(allclose(h_out.to(torch::kCPU), cout), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(sliceRes2.cpu(), sliceCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views with single elements(slice view) of a
// 2D tensor resulting from split and slice created out of a 2D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewSliceTest2D_2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto cout = torch::slice(input, 0, 0, 3);
  Tensor h_out = torch::slice(h_input, 0, 0, 3);

  auto sliceCout2 = cout[1, 1];
  auto sliceRes2 = h_out[1, 1];

  sliceRes2.add_(2);
  sliceCout2.add_(2);

  EXPECT_EQ(allclose(h_out.to(torch::kCPU), cout), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(sliceRes2.cpu(), sliceCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views resulting from split and a column
// selected (different shape) from the slice of a 2D tensor operations.
TEST_F(LazyTensorShapeKernelTest, SplitViewSliceTest2D_3) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[0];
  auto splitCout1 = cpu_out[0];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto cout = torch::slice(input, 0, 0, 5);
  Tensor h_out = torch::slice(h_input, 0, 0, 5);

  auto sliceCout2 = cout;
  auto sliceRes2 = h_out;

  sliceCout2 = torch::select(sliceCout2, 1, 1);
  sliceRes2 = torch::select(sliceRes2, 1, 1);

  sliceRes2.add_(2);
  sliceCout2.add_(2);

  EXPECT_EQ(allclose(h_out.to(torch::kCPU), cout), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(sliceRes2.cpu(), sliceCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views resulting from split and
// slice out of a 4D tensor operations.
TEST_F(LazyTensorShapeKernelTest, SplitViewSliceTest4D) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto cout = torch::slice(input, 0, 0, 2);
  Tensor h_out = torch::slice(h_input, 0, 0, 2);

  auto sliceCout2 = cout;
  auto sliceRes2 = h_out;

  sliceRes2.add_(2);
  sliceCout2.add_(2);

  EXPECT_EQ(allclose(h_out.to(torch::kCPU), cout), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(sliceRes2.cpu(), sliceCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

// Test case to check the updates on views resulting from split and
// slice with single elements (different shape) of a 4D tensor.
TEST_F(LazyTensorShapeKernelTest, SplitViewSliceTest4D_2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  auto input = torch::randn({6, 2, 3, 1});
  auto h_input = input.to(torch::kHPU);

  auto result = torch::split(h_input, 2, 0);
  auto cpu_out = torch::split(input, 2, 0);

  auto splitRes1 = result[1];
  auto splitCout1 = cpu_out[1];

  splitRes1.add_(1);
  splitCout1.add_(1);

  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);

  auto cout = torch::slice(input, 0, 0, 3);
  Tensor h_out = torch::slice(h_input, 0, 0, 3);

  auto sliceCout2 = cout[0, 1, 2];
  auto sliceRes2 = h_out[0, 1, 2];

  sliceRes2.add_(2);
  sliceCout2.add_(2);

  EXPECT_EQ(allclose(h_out.to(torch::kCPU), cout), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(sliceRes2.cpu(), sliceCout2, rtol, atol), true);
  EXPECT_EQ(allclose(splitRes1.cpu(), splitCout1, rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, Resize) {
  auto h_input = torch::arange(10).to(torch::kHPU);

  std::vector<int64_t> sizes({1, 2, 3, 2});
  h_input.resize_(sizes);
  EXPECT_EQ(
      h_input.nbytes(), at::multiply_integers(sizes) * h_input.itemsize());
  EXPECT_TRUE(equal(h_input.cpu().view(-1).slice(0, 0, 10), torch::arange(10)));

  std::vector<int64_t> sizes2({2, 2});
  h_input.resize_(sizes2);
  EXPECT_EQ(
      h_input.nbytes(), at::multiply_integers(sizes2) * h_input.itemsize());
  EXPECT_TRUE(equal(h_input.cpu().view(-1).slice(0, 0, 4), torch::arange(4)));

  std::vector<int64_t> sizes3({3, 2, 2});
  h_input.resize_(sizes3);
  EXPECT_EQ(
      h_input.nbytes(), at::multiply_integers(sizes3) * h_input.itemsize());
  EXPECT_TRUE(equal(h_input.cpu().view(-1).slice(0, 0, 4), torch::arange(4)));
}

TEST_F(LazyTensorShapeKernelTest, Set) {
  int size = 20;
  auto h_input = torch::arange(size, torch::kFloat).to(torch::kHPU);
  auto h_zeros = torch::zeros(10).to(torch::kHPU);
  h_zeros.set_(h_input.storage(), 0, size);
  EXPECT_TRUE(equal(h_zeros.cpu(), h_input.cpu()));

  size = 4;
  auto h_ones = torch::ones(5).to(torch::kHPU);
  h_zeros.set_(h_ones.storage(), 2, size);
  EXPECT_TRUE(equal(h_zeros.cpu().view(-1).slice(0, 0, 2), torch::ones(2)));
  EXPECT_EQ(h_zeros.nbytes(), size * h_zeros.itemsize());
}

TEST_F(LazyTensorShapeKernelTest, FlipTest) {
  torch::Tensor tensor = torch::rand({2, 3, 2});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::flip(tHabana, {0, 1, 2});
  auto out = torch::flip(tensor, {0, 1, 2});
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, FlipNegativeTest) {
  torch::Tensor tensor = torch::rand({4, 2, 2});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::flip(tHabana, {-1, 1});
  auto out = torch::flip(tensor, {-1, 1});
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, Diag2DTest) {
  torch::Tensor tensor = torch::randn({3, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::diag(tHabana, -1);
  auto out = torch::diag(tensor, -1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, Diag1DTest) {
  torch::Tensor tensor = torch::randn({3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::diag(tHabana, -1);
  auto out = torch::diag(tensor, -1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_GT_1_R_LT_C) {
  torch::Tensor tensor = torch::randn({3, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 3);
  auto out = torch::diag_out(out_tensor, tensor, 3);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_GT_1_R_EQ_C) {
  torch::Tensor tensor = torch::randn({4, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 3);
  auto out = torch::diag_out(out_tensor, tensor, 3);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_GT_1_R_GT_C) {
  torch::Tensor tensor = torch::randn({4, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 3);
  auto out = torch::diag_out(out_tensor, tensor, 3);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_0_R_LT_C) {
  torch::Tensor tensor = torch::randn({3, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 0);
  auto out = torch::diag_out(out_tensor, tensor, 0);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_0_R_EQ_C) {
  torch::Tensor tensor = torch::randn({4, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 0);
  auto out = torch::diag_out(out_tensor, tensor, 0);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_0_R_GT_C) {
  torch::Tensor tensor = torch::randn({4, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 0);
  auto out = torch::diag_out(out_tensor, tensor, 0);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestNegativeDiagonalTest_R_LT_C) {
  torch::Tensor tensor = torch::randn({3, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, -1);
  auto out = torch::diag_out(out_tensor, tensor, -1);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestNegativeDiagonalTest_R_GT_C) {
  torch::Tensor tensor = torch::randn({4, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, -1);
  auto out = torch::diag_out(out_tensor, tensor, -1);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestNegativeDiagonalTest_R_EQ_C) {
  torch::Tensor tensor = torch::randn({4, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, -1);
  auto out = torch::diag_out(out_tensor, tensor, -1);

  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_1_R_LT_C) {
  torch::Tensor tensor = torch::randn({64, 128});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);
  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 1);
  auto out = torch::diag_out(out_tensor, tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_1_R_EQ_C) {
  torch::Tensor tensor = torch::randn({64, 64});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);
  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 1);
  auto out = torch::diag_out(out_tensor, tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTestDiagonal_1_R_GT_C) {
  torch::Tensor tensor = torch::randn({128, 64});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);
  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 1);
  auto out = torch::diag_out(out_tensor, tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, DiagOut1DTest) {
  torch::Tensor tensor = torch::randn({3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);

  torch::Tensor out_tensor = torch::randn({4, 4});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 1);
  auto out = torch::diag_out(out_tensor, tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyTensorShapeKernelTest, TriuTrilTest) {
  auto typetest = [](at::Tensor (*op)(const at::Tensor&, int64_t),
                     int64_t diagonal,
                     c10::ScalarType dtype,
                     c10::IntArrayRef size) {
    auto a = torch::randn(size).to(dtype);
    int64_t diag = diagonal;
    auto out = op(a, diag);

    auto ha = a.to("hpu");
    auto hout = op(ha, diag);

    EXPECT_TRUE(
        allclose(out, hout.to("cpu"), 0.001, 0.001, /*equal_nan*/ true));
  };
  typetest(&torch::triu, 1, torch::kFloat, {1, 4, 3});
  typetest(&torch::triu, 0, torch::kFloat, {1, 3, 3});
  typetest(&torch::triu, -1, torch::kFloat, {1, 3, 2});
  typetest(&torch::triu, 1, torch::kFloat, {5, 8});
  typetest(&torch::triu, 0, torch::kFloat, {5, 7});
  typetest(&torch::triu, -1, torch::kFloat, {7, 8});
  typetest(&torch::tril, 1, torch::kFloat, {1, 3, 3});
  typetest(&torch::tril, 0, torch::kFloat, {1, 3, 3});
  typetest(&torch::tril, -1, torch::kFloat, {6, 6});
  typetest(&torch::tril, 1, torch::kFloat, {8, 5});
  typetest(&torch::tril, 0, torch::kFloat, {7, 5});
  typetest(&torch::tril, -1, torch::kFloat, {8, 7});
}

TEST_F(LazyTensorShapeKernelTest, TriuTrilOutTest) {
  auto typetest = [](at::Tensor& (*op)(const at::Tensor&, int64_t, at::Tensor&),
                     int64_t diagonal,
                     c10::ScalarType dtype,
                     c10::IntArrayRef size) {
    auto a = torch::randn(size).to(dtype);
    auto out = torch::randn(size).to(dtype);
    auto ha = a.to("hpu");
    auto hout = out.to("hpu");
    int64_t diag = diagonal;

    op(a, diag, out);
    op(ha, diag, hout);
    EXPECT_TRUE(
        allclose(out, hout.to("cpu"), 0.001, 0.001, /*equal_nan*/ true));
  };
  typetest(&torch::triu_outf, 1, torch::kFloat, {3, 3});
  typetest(&torch::triu_outf, 0, torch::kFloat, {4, 4});
  typetest(&torch::triu_outf, -1, torch::kFloat, {2, 2});
  typetest(&torch::triu_outf, 1, torch::kFloat, {5, 8});
  typetest(&torch::triu_outf, 0, torch::kFloat, {5, 7});
  typetest(&torch::triu_outf, -1, torch::kFloat, {7, 8});
  typetest(&torch::tril_outf, 1, torch::kFloat, {4, 4});
  typetest(&torch::tril_outf, 0, torch::kFloat, {5, 5});
  typetest(&torch::tril_outf, -1, torch::kFloat, {6, 6});
  typetest(&torch::tril_outf, 1, torch::kFloat, {8, 5});
  typetest(&torch::tril_outf, 0, torch::kFloat, {7, 5});
  typetest(&torch::tril_outf, -1, torch::kFloat, {8, 7});
}

TEST_F(LazyTensorShapeKernelTest, TrilInplaceTest) {
  torch::Tensor A = torch::randn({3, 3});
  int64_t diagonal = 0;

  auto hA = A.to(torch::kHPU);

  A.tril_(diagonal);
  auto exp = A;

  hA.tril_(diagonal);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyTensorShapeKernelTest, TriuInplaceTest) {
  torch::Tensor A = torch::randn({3, 3});
  int64_t diagonal = 0;

  auto hA = A.to(torch::kHPU);

  A.triu_(diagonal);
  auto exp = A;

  hA.triu_(diagonal);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest1) {
  double rtol = 1e-03;
  double atol = 1e-03;

  torch::Tensor input = torch::randn({3, 1});
  torch::Tensor h_input = input.to(torch::kHPU);

  torch::Tensor A = input.expand({3, 4});
  torch::Tensor AT = torch::transpose(A, 1, 0);
  auto Asplit = torch::split(A, 2, 1);
  torch::Tensor AView = input.view(-1);
  torch::Tensor A1 = AView.add_(2.0);

  torch::Tensor hA = h_input.expand({3, 4});
  torch::Tensor hAT = torch::transpose(hA, 1, 0);
  auto hAsplit = torch::split(hA, 2, 1);
  torch::Tensor hAView = h_input.view(-1);
  torch::Tensor hA1 = hAView.add_(2.0);

  std::vector<at::Tensor> hso;
  hso.reserve(hAsplit.size());
  for (const auto& ht : hAsplit) {
    hso.push_back(ht.to(torch::kCPU));
  }

  for (size_t i = 0; i < hAsplit.size(); i++) {
    EXPECT_EQ(allclose(Asplit[i], hso[i]), true);
  }

  EXPECT_EQ(allclose(A, hA.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(AT, hAT.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(A1, hA1.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest2) {
  double rtol = 1e-03;
  double atol = 1e-03;

  torch::Tensor input = torch::randn({4, 1});
  torch::Tensor h_input = input.to(torch::kHPU);

  torch::Tensor Aexp = input.expand({4, 4});
  torch::Tensor Asel = torch::select(Aexp, 1, 0);
  torch::Tensor Aadd = Asel.add_(2.0);

  torch::Tensor hAexp = h_input.expand({4, 4});
  torch::Tensor hAsel = torch::select(hAexp, 1, 0);
  torch::Tensor hAadd = hAsel.add_(2.0);

  EXPECT_EQ(allclose(Aexp, hAexp.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(Asel, hAsel.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(Aadd, hAadd.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest3) {
  double rtol = 1e-03;
  double atol = 1e-03;

  torch::Tensor input = torch::randn({4, 2, 2, 2, 1});
  torch::Tensor h_input = input.to(torch::kHPU);

  torch::Tensor Aexp = input.expand({4, 2, 2, 2, 4});
  torch::Tensor Asel = torch::select(Aexp, -1, -1);
  torch::Tensor Aadd = Asel.add_(10.0);

  torch::Tensor hAexp = h_input.expand({4, 2, 2, 2, 4});
  torch::Tensor hAsel = torch::select(hAexp, -1, -1);
  torch::Tensor hAadd = hAsel.add_(10.0);

  EXPECT_EQ(allclose(Aexp, hAexp.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(Asel, hAsel.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(Aadd, hAadd.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest4) {
  double rtol = 1e-03;
  double atol = 1e-03;

  torch::Tensor input = torch::randn({1});
  torch::Tensor h_input = input.to(torch::kHPU);

  torch::Tensor Aexp = input.expand({0});

  torch::Tensor hAexp = h_input.expand({0});

  EXPECT_EQ(allclose(Aexp, hAexp.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest5) {
  double rtol = 1e-03;
  double atol = 1e-03;

  torch::Tensor input = torch::randn({1});
  torch::Tensor h_input = input.to(torch::kHPU);

  torch::Tensor Aexp = input.expand({0});
  torch::Tensor Aview = Aexp.view({-1});
  torch::Tensor Aadd = Aview.add_(10);

  torch::Tensor hAexp = h_input.expand({0});
  torch::Tensor hAview = hAexp.view({-1});
  torch::Tensor hAadd = hAview.add_(10);

  EXPECT_EQ(allclose(Aexp, hAexp.to(torch::kCPU), rtol, atol), true);
  EXPECT_EQ(allclose(input, h_input.to(torch::kCPU), rtol, atol), true);
}
