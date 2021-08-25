/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/torch.h>

class HpuOpTestUtil : public habana_lazy_test::LazyTest {
 public:
  torch::Tensor& GetCpuInput(int index) {
    return m_cpu_inputs.at(index);
  }

  torch::Tensor& GetHpuInput(int index) {
    return m_hpu_inputs.at(index);
  }

  void Compare(
      const torch::Tensor& cpu_result,
      const torch::Tensor& hpu_result,
      double rtol = 1e-03,
      double atol = 1e-03) const {
    EXPECT_TRUE(hpu_result.is_habana());
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

  void GenerateInputs(int num_inputs, torch::ScalarType dtype = torch::kFloat) {
    SetSeed();
    m_cpu_inputs.resize(num_inputs);
    m_hpu_inputs.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      m_cpu_inputs[i] = dtype == torch::kBool ? torch::randn(m_dims) > 0
                                              : torch::randn(m_dims).to(dtype);
      m_hpu_inputs[i] = m_cpu_inputs[i].to("hpu");
    }
  }

  // Generate inputs with different dtypes/sizes per input
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::IntArrayRef> sizes,
      std::vector<torch::ScalarType> dtypes = {}) {
    SetSeed();
    ASSERT_EQ(num_inputs, sizes.size());
    if (dtypes.empty()) {
      dtypes.resize(num_inputs, torch::kFloat);
    }

    m_cpu_inputs.resize(num_inputs);
    m_hpu_inputs.resize(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      m_cpu_inputs[i] = dtypes[i] == torch::kBool
          ? torch::randn(sizes.at(i)) > 0
          : torch::randn(sizes.at(i)).to(dtypes[i]);
      m_hpu_inputs[i] = m_cpu_inputs[i].to("hpu");
    }
  }

 private:
  const torch::IntArrayRef m_dims = torch::IntArrayRef({2, 3, 2});
  std::vector<torch::Tensor> m_cpu_inputs;
  std::vector<torch::Tensor> m_hpu_inputs;
};
