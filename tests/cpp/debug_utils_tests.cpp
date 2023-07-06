#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class DebugUtilsTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    ForceMode(1); // This test suite expects to run only with lazy=1
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
};

TEST_F(DebugUtilsTest, DISABLED_GraphTextDump1) {
  auto A = torch::randn({2, 2}, torch::requires_grad(false));
  auto B = torch::randn({2, 2}, torch::requires_grad(false));
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto I = torch::add(hA, hB, 1.0);
  I = torch::relu(I);
  auto out = torch::relu(I);

  auto hl_result = std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(out));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = prim::constant(), value=1.\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = hpu::input()\n"
                      "  %3 = aten::add(%2, %1, %0)\n"
                      "  %4 = aten::relu(%3)\n"
                      "  %5 = aten::relu(%4), ROOT=0\n"
                      "}"),
      0);
}

TEST_F(DebugUtilsTest, GraphDotDump1) {
  auto A = torch::randn({2, 2}, torch::requires_grad(false));
  auto B = torch::randn({2, 2}, torch::requires_grad(false));
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto I = torch::add(hA, hB, 1.0);
  I = torch::relu(I);
  auto out = torch::relu(I);

  auto hl_result = std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(out));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToDot(a);
  EXPECT_EQ(
      out_string.find("digraph G {\n"
                      "  node0 [label=\"prim::constant\\n\\nvalue=1.\"]\n"
                      "  node1 [label=\"hpu::input\\n\"]\n"
                      "  node2 [label=\"hpu::input\\n\"]\n"
                      "  node3 [label=\"hpu::add\\n\"]\n"
                      "  node4 [label=\"aten::relu\\n\"]\n"
                      "  node5 [label=\"aten::relu\\n\\nROOT=0\"]\n"
                      "  node4 -> node5\n"
                      "  node3 -> node4\n"
                      "  node2 -> node3 [label=\"i=0\"]\n"
                      "  node1 -> node3 [label=\"i=1\"]\n"
                      "  node0 -> node3 [label=\"i=2\"]\n"
                      "}"),
      0);
}

TEST_F(DebugUtilsTest, DebugCustomOp3) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE, true, 1);

  auto S = torch::randn({2, 4}, torch::requires_grad(false));
  auto C = torch::relu(S);
  auto hS = S.to(torch::kHPU);
  HbLazyTensor::StepMarker({});

  auto hT = torch::relu(hS);
  HbLazyTensor::StepMarker({});

  at::Tensor hR = at::empty({8}, hS.options());

  at::TensorImpl* impl = hR.unsafeGetTensorImpl();
  auto storage = hS.storage();
  impl->set_storage_keep_dtype(storage);
  impl->set_storage_offset(0);
  impl->set_sizes_and_strides(
      c10::IntArrayRef({2, 4}), c10::IntArrayRef({4, 1}));

  auto hO = torch::relu(hR);
  auto O = hO.to(torch::kCPU);

  EXPECT_EQ(allclose(C, O), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE);
}

TEST_F(DebugUtilsTest, DebugSetStorageAndSizeStride) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE, true, 1);

  auto S = torch::randn({2, 4}, torch::requires_grad(false));
  auto C = torch::relu(S);
  auto hS = S.to(torch::kHPU);
  HbLazyTensor::StepMarker({});

  auto hT = torch::relu(hS);
  HbLazyTensor::StepMarker({});

  at::Tensor hR = at::empty({0}, hS.options());

  at::TensorImpl* impl = hR.unsafeGetTensorImpl();
  auto storage = hS.storage();
  impl->set_storage_keep_dtype(storage);
  impl->set_storage_offset(0);
  impl->set_sizes_and_strides(
      c10::IntArrayRef({2, 4}), c10::IntArrayRef({4, 1}));

  auto hO = torch::relu(hR);
  auto O = hO.to(torch::kCPU);

  EXPECT_EQ(allclose(C, O), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE);
}

TEST_F(DebugUtilsTest, CheckTensorSizeForLongType) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE, true, 1);

  auto S = torch::randn({1}, torch::requires_grad(false)).to(torch::kLong);
  auto hS = S.to(torch::kHPU);
  HbLazyTensor::StepMarker({});

  auto total_size = hS.storage().nbytes();
  auto element_size = hS.element_size();

  auto num_elements = total_size / element_size;
  bool equal = (num_elements == 1);
  EXPECT_EQ(equal, true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
}

TEST_F(DebugUtilsTest, CheckTensorSizeForDoubleType) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE, true, 1);

  auto S = torch::randn({1}, torch::dtype(torch::kDouble));
  auto hS = S.to(torch::kHPU);
  HbLazyTensor::StepMarker({});

  auto total_size = hS.storage().nbytes();
  auto element_size = hS.element_size();

  auto num_elements = total_size / element_size;
  bool equal = (num_elements == 1);
  EXPECT_EQ(equal, true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
}
