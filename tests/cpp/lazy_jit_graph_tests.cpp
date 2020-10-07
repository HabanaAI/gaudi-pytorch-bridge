#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include <torch/csrc/jit/testing/file_check.h>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/hlexec.h"

using namespace habana_lazy;

/**
 * Create a JIT graph and check the nodes created within it.
 */
TEST(LazyJITTest, CreateGraph) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHABANA);
  Scalar alpha = 4.0f, beta = 99.5f;
  auto result = add_tensor_hpu_lazy(tensor_in1, tensor_in2, alpha);

  torch::Tensor tensor_in3 = torch::randn({2, 3}).to(torch::kHABANA);
  auto result2 = add_tensor_hpu_lazy(result, tensor_in3, beta);
  auto hl_result = GetHbLazyTensor(result2);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec *hlexec = new exec::HlExec();
  exec::LazyValueToJitValueMap input_map, output_map;
  std::tie(input_map, output_map) = hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=99.5]")
      ->check("prim::Constant[value=4.]")
      ->check_count("aten::add", 2)
      ->run(*hlexec->get_graph());
}

TEST(LazyJITTest, ExecuteGraph) {
  Scalar alpha = 10.0f;
  Tensor tensor_in1 = torch::rand({2, 3});
  Tensor tensor_in2 = torch::rand({2, 3});
  Tensor exp1 = add(tensor_in1, tensor_in2, alpha);
  Tensor exp2 = add(exp1, tensor_in1, alpha);

  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHABANA);
  torch::Tensor htensor_in2 = tensor_in2.to(torch::kHABANA);
  auto result1 = add_tensor_hpu_lazy(htensor_in1, htensor_in2, alpha);
  auto result2 = add_tensor_hpu_lazy(result1, htensor_in1, alpha);
  unsetenv("PT_HPU_LAZY_MODE");

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result1),
                                       GetHbLazyTensor(result2)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor out1 = result1.to(kCPU);
  Tensor out2 = result2.to(kCPU);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
}
