#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

class DebugUtilsTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    ForceMode(1); // This test suite expects to run only with lazy=1
  }
};

TEST_F(DebugUtilsTest, GraphTextDump1) {
  auto A = torch::randn({2, 2}, torch::requires_grad(false));
  auto B = torch::randn({2, 2}, torch::requires_grad(false));
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto I = torch::add(hA, hB, 1.0);
  I = torch::relu(I);
  auto out = torch::relu(I);

  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out));
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
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto I = torch::add(hA, hB, 1.0);
  I = torch::relu(I);
  auto out = torch::relu(I);

  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToDot(a);
  EXPECT_EQ(
      out_string.find("digraph G {\n"
                      "  node0 [label=\"prim::constant\\n\\nvalue=1.\"]\n"
                      "  node1 [label=\"hpu::input\\n\"]\n"
                      "  node2 [label=\"hpu::input\\n\"]\n"
                      "  node3 [label=\"aten::add\\n\"]\n"
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
