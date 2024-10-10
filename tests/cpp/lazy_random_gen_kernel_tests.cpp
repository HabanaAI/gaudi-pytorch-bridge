#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/random_gen_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyRandomGenKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyRandomGenKernelTest, RandpermOutTest) {
  constexpr int n = 10;

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);

  torch::manual_seed(0);
  habana::getDefaultHPUGenerator().set_current_seed(0);
  auto eager = torch::randperm(n, hb_options);
  auto eager_cpu = eager.to(torch::kCPU);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 1, 0);

  torch::manual_seed(0);
  habana::getDefaultHPUGenerator().set_current_seed(0);
  auto lazy = torch::randperm(n, hb_options);
  auto lazy_cpu = lazy.to(torch::kCPU);

  auto equal = eager_cpu.equal(lazy_cpu);
  EXPECT_EQ(equal, true);

  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
}
