#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, gt_scalar) {
  GenerateInputs(1);
  int other = 0;

  print(GetCpuInput(0));
  GetCpuInput(0).gt_(other);
  GetHpuInput(0).gt_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gt_tensor) {
  GenerateInputs(2);
  int other = 1;

  GetCpuInput(0).gt_(GetCpuInput(1));
  GetHpuInput(0).gt_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}
