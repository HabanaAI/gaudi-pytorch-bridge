#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, lerp_scalar) {
  GenerateInputs(2, {{1}, {4, 8}});
  int weight = 1;

  auto expected = torch::lerp(GetCpuInput(0), GetCpuInput(1), weight);
  auto result = torch::lerp(GetHpuInput(0), GetHpuInput(1), weight);

  Compare(expected, result);
}

TEST_F(HpuOpTest, lerp_scalar_out) {
  GenerateInputs(2);
  int weight = 1;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::lerp_out(expected, GetCpuInput(0), GetCpuInput(1), weight);
  torch::lerp_out(result, GetHpuInput(0), GetHpuInput(1), weight);

  Compare(expected, result);
}

TEST_F(HpuOpTest, lerp_tensor) {
  GenerateInputs(3, {{1}, {1, 4}, {2, 4}});

  auto expected = torch::lerp(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  auto result = torch::lerp(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

  Compare(expected, result);
}

TEST_F(HpuOpTest, lerp_tensor_out) {
  GenerateInputs(3, {{1, 4}, {1}, {2, 4}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::lerp_out(expected, GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  torch::lerp_out(result, GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

  Compare(expected, result);
}

TEST_F(HpuOpTest, lerp_scalar_) {
  GenerateInputs(2);
  const int weight = 1;

  GetCpuInput(0).lerp_(GetCpuInput(1), weight);
  GetHpuInput(0).lerp_(GetHpuInput(1), weight);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lerp_tensor_) {
  GenerateInputs(3);

  GetCpuInput(0).lerp_(GetCpuInput(1), GetCpuInput(2));
  GetHpuInput(0).lerp_(GetHpuInput(1), GetHpuInput(2));

  Compare(GetCpuInput(0), GetHpuInput(0));
}
