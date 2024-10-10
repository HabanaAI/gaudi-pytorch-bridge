/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "hpu_ops/op_logger.h"
#include "util.h"

TEST(HpuOpLogTest, logger) {
  // Tensor
  EXPECT_EQ(
      "HPUFloatType[1, 2, 3, 4]",
      habana::to_string(
          at::empty({1, 2, 3, 4}, at::kHPU, c10::MemoryFormat::ChannelsLast)));
  EXPECT_EQ(
      "CPUHalfType[5, 20]", habana::to_string(at::ones({5, 20}, at::kHalf)));

  // primitive types
  EXPECT_EQ("1.000000", habana::to_string(1.));
  EXPECT_EQ("2", habana::to_string(2));

  // optional int64_t with none
  c10::optional<int64_t> opt = c10::nullopt;
  EXPECT_EQ("None", habana::to_string(opt));

  // Scalar
  at::Scalar f = 10.;
  EXPECT_EQ("10.000000", habana::to_string(f));
  at::Scalar i = 20;
  EXPECT_EQ("20", habana::to_string(i));

  // ScalarType
  EXPECT_EQ("BFloat16", habana::to_string(at::kBFloat16));

  // c10::optional<ScalarType>
  EXPECT_EQ("Float", habana::to_string(c10::make_optional(at::kFloat)));

  // array
  const auto& arr = std::array{true, false};
  EXPECT_EQ("true false", habana::to_string(arr));

  // IntArrayRef
  std::vector<long> long_vec = {2, 10};
  c10::IntArrayRef intarrayref{long_vec};
  EXPECT_EQ("[2, 10]", habana::to_string(intarrayref));

  // SymIntArrayRef
  std::vector<at::SymInt> symint_vec{64, 128};
  c10::SymIntArrayRef symintarrayref{symint_vec};
  EXPECT_EQ("[64, 128]", habana::to_string(symintarrayref));

  // TensorList and friends
  const auto& t1 = at::Tensor();
  const auto& t2 = at::tensor(std::vector<float>{3.14, 42}, at::kHPU);
  const auto& t3 = at::zeros({5, 3, 2, 1, 3, 4});

  at::TensorList tlist{t1, t2};
  EXPECT_EQ("[UndefinedTensor] HPUFloatType[2]", habana::to_string(tlist));

  at::List<c10::optional<at::Tensor>> listopt{{}, t2};
  EXPECT_EQ("None HPUFloatType[2]", habana::to_string(listopt));

  at::IListRef<at::Tensor> ilistref{t3, t2};
  EXPECT_EQ(
      "CPUFloatType[5, 3, 2, 1, 3, 4] HPUFloatType[2]",
      habana::to_string(ilistref));
}
