/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include <torch/torch.h>

namespace habana_lazy {

class PermuteTensors {
 public:
  PermuteTensors() = default;
  virtual ~PermuteTensors() = default;

  static void permuteWeight(torch::Tensor& weight);

 private:
  static void permuteWeightByDim(torch::Tensor& weight);
  static void permuteWeightToRSCKInMemory(torch::Tensor& weight);
  static void permuteWeightToQRSCKInMemory(torch::Tensor& weight);
  template <typename T>
  static void permuteWeightTensorDataToRSCK(const torch::Tensor& weight);
  template <typename T>
  static void restrideWeightTensorDataToQRSCK(const torch::Tensor& weight);
  static bool shouldPermuteWeight(const torch::Tensor& weight);
  static bool shouldPermutePreCastedWeight(const torch::Tensor& weight);
  static const torch::Tensor getPreCastedWeight(const torch::Tensor& weight);

  static unsigned m_permute_counter;
};

} // namespace habana_lazy