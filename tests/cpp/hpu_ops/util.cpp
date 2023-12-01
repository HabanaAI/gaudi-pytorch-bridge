/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

// Default rtol/atol per dtype as defined in
// https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close
static std::tuple<double, double> get_default_tolerances(at::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return {0.001, 1e-5};
    case torch::kBFloat16:
      return {0.016, 1e-5};
    case torch::kFloat32:
    case torch::kFloat64:
      // return {1.3e-6, 1e-5};
      // Why dont hpu use the default tol defined by torch?
      return {1e-3, 1e-3};
    default:
      return {0, 0};
  }
}

void HpuOpTestUtilBase::Compare(
    const torch::Tensor& cpu_result,
    const torch::Tensor& hpu_result,
    c10::optional<double> opt_rtol,
    c10::optional<double> opt_atol) const {
  EXPECT_TRUE(hpu_result.is_hpu());

  EXPECT_EQ(cpu_result.scalar_type(), hpu_result.scalar_type())
      << "expected dtype=" << cpu_result.scalar_type() << std::endl
      << "actual dtype=" << hpu_result.scalar_type() << std::endl;

  torch::Tensor hpu_result_on_cpu = hpu_result.cpu();

  auto [default_rtol, default_atol] =
      get_default_tolerances(cpu_result.scalar_type());
  auto rtol = opt_rtol.value_or(default_rtol);
  auto atol = opt_atol.value_or(default_atol);

  auto matches = torch::isclose(
      cpu_result, hpu_result_on_cpu, rtol, atol, /*equal_nan=*/true);

  if (matches.all().item().toBool()) {
    return;
  }

  std::ostringstream oss;
  oss << "seed: " << GetSeed() << "\n";

  auto mismatches = ~matches;
  auto number_of_elements = mismatches.numel();
  auto total_mismatches = torch::sum(mismatches).item().toLong();

  oss << "Mismatched elements: " << total_mismatches << " / "
      << number_of_elements << " ("
      << total_mismatches * 100. / number_of_elements << "%)\n";

  auto hpu_flat = hpu_result_on_cpu.flatten();
  auto cpu_flat = cpu_result.flatten();
  auto matches_flat = ~mismatches.flatten();

  auto abs_diff = torch::abs(hpu_flat - cpu_flat);
  // Ensure that only mismatches are used for the max_abs_diff computation
  torch::index_put_(abs_diff, {matches_flat}, torch::tensor(0));
  auto [max_abs_diff, max_abs_diff_flat_idx] = torch::max(abs_diff, 0);
  oss << "Greatest absolute difference: " << max_abs_diff.item() << " at index "
      << max_abs_diff_flat_idx.item() << " (up to " << atol << " allowed)\n";

  auto rel_diff = abs_diff / torch::abs(cpu_flat);
  // Ensure that only mismatches are used for the max_rel_diff computation
  torch::index_put_(rel_diff, {matches_flat}, torch::tensor(0));
  auto [max_rel_diff, max_rel_diff_flat_idx] = torch::max(rel_diff, 0);
  oss << "Greatest relative difference: " << max_rel_diff.item() << " at index "
      << max_rel_diff_flat_idx.item() << " (up to " << rtol << " allowed)\n";

  FAIL() << oss.str();
}

void HpuOpTestUtilBase::GenerateInputs(
    int num_inputs,
    torch::ArrayRef<torch::IntArrayRef> sizes_,
    torch::ArrayRef<torch::ScalarType> dtypes_) {
  SetSeed();

  std::vector<at::IntArrayRef> sizes = sizes_.vec();
  if (sizes.empty()) {
    sizes.resize(num_inputs, m_dims);
  } else if (sizes.size() == 1) {
    sizes.resize(num_inputs, sizes_[0]);
  }

  std::vector<torch::ScalarType> dtypes = dtypes_.vec();
  if (dtypes.empty()) {
    dtypes.resize(num_inputs, torch::kFloat);
  } else if (dtypes.size() == 1) {
    dtypes.resize(num_inputs, dtypes_[0]);
  }

  ASSERT_EQ(num_inputs, sizes.size())
      << "num_inputs(" << num_inputs << ") != num sizes(" << sizes.size()
      << ")";
  ASSERT_EQ(num_inputs, dtypes.size())
      << "num_inputs(" << num_inputs << ") != num dtypes(" << dtypes.size()
      << ")";

  m_cpu_inputs.resize(num_inputs);
  m_hpu_inputs.resize(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    if (torch::isIntegralType(dtypes[i], false)) {
      // Fixed min and max for now, change when required.
      m_cpu_inputs[i] = torch::randint(-127, 128, sizes.at(i)).to(dtypes[i]);
    } else {
      m_cpu_inputs[i] = dtypes[i] == torch::kBool
          ? torch::randn(sizes.at(i)) > 0
          : torch::randn(sizes.at(i)).to(dtypes[i]);
    }
    m_hpu_inputs[i] = m_cpu_inputs[i].to("hpu");
  }
}

void HpuOpTestUtilBase::GenerateIntInputs(
    int num_inputs,
    torch::ArrayRef<torch::IntArrayRef> sizes,
    int low,
    int high) {
  SetSeed();
  ASSERT_EQ(num_inputs, sizes.size());

  m_cpu_inputs.resize(num_inputs);
  m_hpu_inputs.resize(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    m_cpu_inputs[i] = torch::randint(low, high, sizes.at(i), torch::kInt);
    m_hpu_inputs[i] = m_cpu_inputs[i].to("hpu");
  }
}

template <>
int HpuOpTestUtilBase::GenerateScalar(
    c10::optional<int> min,
    c10::optional<int> max) const {
  std::uniform_int_distribution<> dist(min.value_or(-127), max.value_or(128));
  return dist(m_mt);
}

template <>
bool HpuOpTestUtilBase::GenerateScalar(
    c10::optional<bool> min,
    c10::optional<bool> max) const {
  std::bernoulli_distribution dist;
  return dist(m_mt);
}

std::string HpuOpTestUtil::FixTestName(std::string name) {
  std::replace(name.begin(), name.end(), '-', '_');
  std::replace(name.begin(), name.end(), '.', 'p');
  return name;
}
