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

void HpuOpTestUtil::Compare(
    const torch::Tensor& cpu_result,
    const torch::Tensor& hpu_result,
    double rtol,
    double atol) const {
  EXPECT_TRUE(hpu_result.is_hpu());

  EXPECT_EQ(cpu_result.scalar_type(), hpu_result.scalar_type())
      << "exp dtype=" << cpu_result.scalar_type() << std::endl
      << "actual dtype=" << hpu_result.scalar_type() << std::endl;

  torch::Tensor habana_result_on_cpu = hpu_result.cpu();

  if (c10::isIntegralType(cpu_result.scalar_type(), /*includeBool=*/true)) {
    EXPECT_TRUE(torch::equal(cpu_result, habana_result_on_cpu))
        << "seed=" << GetSeed() << std::endl
        << "exp=" << std::endl
        << cpu_result << std::endl
        << "actual=" << std::endl
        << habana_result_on_cpu << std::endl;
  } else {
    EXPECT_TRUE(
        torch::allclose(cpu_result, habana_result_on_cpu, rtol, atol, true))
        << "seed=" << GetSeed() << std::endl
        << "exp=" << std::endl
        << cpu_result << std::endl
        << "actual=" << std::endl
        << habana_result_on_cpu << std::endl;
  }
}

void HpuOpTestUtil::GenerateInputs(
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

void HpuOpTestUtil::GenerateIntInputs(
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
int HpuOpTestUtil::GenerateScalar(
    c10::optional<int> min,
    c10::optional<int> max) const {
  std::uniform_int_distribution<> dist(min.value_or(-127), max.value_or(128));
  return dist(m_mt);
}

template <>
bool HpuOpTestUtil::GenerateScalar(
    c10::optional<bool> min,
    c10::optional<bool> max) const {
  std::bernoulli_distribution dist;
  return dist(m_mt);
}
