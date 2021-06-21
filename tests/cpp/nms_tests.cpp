#include <ATen/ExpandUtils.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"

using namespace habana_lazy;

TEST(NMSEagerTest, NmsSmall) {
  torch::manual_seed(0);
  // Generate random scores for each box
  auto num_boxes = 10;
  torch::Tensor scores = torch::rand({num_boxes});
  torch::Tensor hscores = scores.to(torch::kHABANA);

  // Generate boxes of random sizes
  torch::Tensor boxes = torch::rand({num_boxes, 4}) * 256;
  // ensure x2 > x1 and y2 > y1
  auto tlist = boxes.split(2, 1);
  tlist[1] = tlist[1] + tlist[0];
  auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1);
  torch::Tensor hboxes = new_boxes.to(torch::kHABANA);

  auto nms_boxid = std::getenv("PT_HPU_LAZY_MODE")
      ? habana_nms_hpu_lazy(hboxes, hscores, 0.2, 0.0)
      : habana_nms_hpu(hboxes, hscores, 0.2, 0.0);
  auto ref = torch::tensor({7, 1, 5, 0, 6, 8, 4}).to(torch::kInt);
  bool equal = ref.allclose(nms_boxid.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}
