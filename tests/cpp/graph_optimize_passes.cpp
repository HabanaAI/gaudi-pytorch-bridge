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
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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

  exec::LazyValueToJitValueMap input_map, output_map;
  std::tie(input_map, output_map) =
      hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check_not("aten::t")
      ->run(*hlexec->get_graph());

  unsetenv("PT_HPU_LAZY_MODE");
}