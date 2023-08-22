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

class LazyBinaryInplaceKernelTest : public habana_lazy_test::LazyTest {};

// Also validates InferOutputMeta for MulInplace
TEST_F(LazyBinaryInplaceKernelTest, MulInplaceTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  A = A.mul_(B);
  auto exp = torch::add(A, C);

  hA = hA.mul_(hB);
  auto result = torch::add(hA, hC);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for MulInplaceScalar
TEST_F(LazyBinaryInplaceKernelTest, MulInplaceScalarTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  torch::Scalar s = 0.3;

  A = A.mul_(s);
  auto exp = torch::add(A, B);

  hA = hA.mul_(s);
  auto result = torch::add(hA, hB);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for MulInplaceScalarBf16
TEST_F(LazyBinaryInplaceKernelTest, MulInplaceScalarBfloat16Test) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 3});
  auto A_bf16 = A.to(torch::kBFloat16);
  torch::Tensor B = torch::randn({2, 3});
  auto B_bf16 = B.to(torch::kBFloat16);
  auto hA = A_bf16.to(torch::kHPU);
  auto hB = B_bf16.to(torch::kHPU);

  torch::Scalar s = 0.3;

  A = A.mul_(s);
  auto exp = torch::add(A, B);

  hA = hA.mul_(s);
  auto result = torch::add(hA, hB);
  Tensor out = result.to(torch::kFloat).to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.1, 0.1), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for AddInplaceScalar
TEST_F(LazyBinaryInplaceKernelTest, AddInplaceScalarTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  torch::Scalar s = 0.3;

  A = A.add_(s);
  auto exp = torch::add(A, B);

  hA = hA.add_(s);
  auto result = torch::add(hA, hB);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for SubInplaceScalar
TEST_F(LazyBinaryInplaceKernelTest, SubInplaceScalarTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  torch::Scalar s = 0.3;

  A = A.sub_(s);
  auto exp = torch::add(A, B);

  hA = hA.sub_(s);
  auto result = torch::add(hA, hB);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyBinaryInplaceKernelTest, SqrtAddInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::ones({2, 3});
  torch::Tensor B = torch::ones({2, 3});
  torch::Tensor C = torch::ones({2, 3});
  torch::Tensor D = torch::ones({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hD = D.to(torch::kHPU);

  auto tempc1 = torch::sqrt(A + D);
  tempc1.add_(B);
  auto result_cpu = torch::add(tempc1, C);

  auto temp1 = torch::sqrt(hA + hD);
  hB.add_(temp1);
  auto result_hpu = torch::add(hB, hC);
  Tensor out_hpu = result_hpu.to(kCPU);

  EXPECT_EQ(allclose(result_cpu, out_hpu, 0.001, 0.001), true);
}

TEST_F(LazyBinaryInplaceKernelTest, AddcmulInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor D = torch::zeros({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hD = D.to(torch::kHPU);

  A = A.addcmul_(B, C);
  auto exp = A;

  hA = hA.addcmul_(hB, hC);
  // Dummy add to avoid hA as output node
  auto result = torch::add(hA, hD);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryInplaceKernelTest, AddcmulInplaceTest2) {
  // Same input for tensor 1 and tensor 2
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::zeros({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  A = A.addcmul_(B, B);
  auto exp = A;

  hA = hA.addcmul_(hB, hB);
  // Dummy add to avoid hA as output node
  auto result = torch::add(hA, hC);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryInplaceKernelTest, AddcdivInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor D = torch::zeros({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hD = D.to(torch::kHPU);

  A = A.addcdiv_(B, C);
  auto exp = A;

  hA = hA.addcdiv_(hB, hC);
  // Dummy add to avoid hA as output node
  auto result = torch::add(hA, hD);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryInplaceKernelTest, AddcdivInplaceTest2) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor D = torch::zeros({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hD = D.to(torch::kHPU);

  A = A.addcdiv_(B, C, 3.5);
  auto exp = A;

  hA = hA.addcdiv_(hB, hC, 3.5);
  // Dummy add to avoid hA as output node
  auto result = torch::add(hA, hD);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

// Also validates InferOutputMeta for DivInplace
TEST_F(LazyBinaryInplaceKernelTest, DivInplaceTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  A.div_(B);
  auto exp = A;

  hA.div_(hB);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyBinaryInplaceKernelTest, DivInplaceIntermediateTest) {
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});

  exec::HlExec* hlexec = new exec::HlExec();

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto C = torch::relu(A);

  C.div_(B);
  auto D = torch::relu(C);
  auto exp = D;

  auto hC = torch::relu(hA);
  hC.div_(hB);
  auto hD = torch::relu(hC);
  Tensor out = hD.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}
