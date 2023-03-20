
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyExecutionTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyExecutionTest, testmultipleLaunch) {
  for (int i = 0; i < 50; i++) {
    auto in =
        torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
    auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
    auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    HbLazyTensor::StepMarker({});
  }
}

TEST_F(LazyExecutionTest, testmultipleLaunchwithAsync) {
  for (int i = 0; i < 50; i++) {
    auto in =
        torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
    auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
    auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    HbLazyTensor::StepMarker({}, nullptr, {}, true);
    HbLazyTensor::StepMarkerFinish();
  }
}
