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

TEST_F(LazyTensorShapeKernelTest, PermuteTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({1, 0});
  torch::Tensor Out = A.permute({1, 0});

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::t(hA);
  torch::Tensor Out = torch::t(A);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);

  int64_t dim = 1;
  int64_t index = 0;

  Tensor h_out = torch::select(h_a, dim, index);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, SliceTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 8;
  int64_t step = 1;

  Tensor h_out = torch::slice(h_a, dim, start_index, end, step);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::slice(a, dim, start_index, end, step);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, DISABLED_SliceTestZeroDimSize) {
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

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
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
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(result));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  at::Tensor result_lazy = result.to(torch::kCPU);
  auto result_cpu = torch::_unsafe_view(input_tensor, new_size);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::transpose(hA, 1, 0);
  torch::Tensor Out = torch::transpose(A, 1, 0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest) {
  torch::Tensor A = torch::randn({3, 1}, torch::requires_grad(false));

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.expand({3, 4}, false);
  torch::Tensor Out = A.expand({3, 4}, false);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
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

TEST_F(LazyTensorShapeKernelTest, DiagOut2DTest) {
  torch::Tensor tensor = torch::randn({3, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  torch::Tensor out_tensor = torch::randn({1});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 3);
  auto out = torch::diag_out(out_tensor, tensor, 3);

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
