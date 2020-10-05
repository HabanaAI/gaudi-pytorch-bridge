#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"
#include "habana_kernels/eager_kernels_declarations.h"

using namespace habana_lazy;
using namespace torch;

TEST(PostOrderTest, poTest1) {
  /*
  // test case for result = relu(tensor1)
  torch::Tensor tensor_cpu = torch::randn({2, 3});
  torch::Tensor tensor_in = tensor_cpu.to(torch::kHABANA);

  auto p_hl_tensor_in = std::make_shared<HbLazyTensor>(
      GetOrCreateHbLazyTensor(tensor_in, c10::kHABANA));

  setenv("PT_HPU_LAZY_MODE", "1", 1);

  // Shape inference using eager mode kernel. In addition, invoke lazy tensor
  // impl
  at::Tensor result = relu_hpu(tensor_in);

  auto p_hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(result));
  assert(p_hl_result != nullptr);

  auto p_node = Node::Create(Symbol::fromQualString("aten::relu"));
  auto p_value = std::make_shared<Value>(p_hl_tensor_in, p_node_result, 0);

  // TODO handle cyclic dependency

  std::vector<HbLazyTensor> tensors = {*(p_hl_result.get())};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  unsetenv("PT_HPU_LAZY_MODE");
  // auto result_cpu = result.to(torch::kCPU);
  */
}
