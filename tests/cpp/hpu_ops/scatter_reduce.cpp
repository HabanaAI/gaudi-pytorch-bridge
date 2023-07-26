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

#include "util.h"

struct ScatterReduceShapeInfoParams {
  int dim;
  std::vector<int64_t> inputShape;
  std::vector<int64_t> indexShape;
  std::vector<int64_t> sourceShape;
  bool deterministic;
};

std::ostream& operator<<(
    std::ostream& os,
    const ScatterReduceShapeInfoParams& params) {
  os << "dim_" << params.dim;
  os << "_input_";
  for (auto&& dim : params.inputShape)
    os << "x" << dim;
  os << "_index_";
  for (auto&& dim : params.indexShape)
    os << "x" << dim;
  os << "_source_";
  for (auto&& dim : params.sourceShape)
    os << "x" << dim;
  os << "_deterministic_" << (params.deterministic ? "true" : "false");

  return os;
}
class ScatterReduceOpTest : public HpuOpTestUtil,
                            public testing::WithParamInterface<std::tuple<
                                ScatterReduceShapeInfoParams, // input params
                                std::string, // reduce
                                c10::ScalarType, // dtype
                                bool>> // include self
{
 public:
  bool verbose = false;
  struct GetName {
    template <class ParamType>
    std::string operator()(
        const ::testing::TestParamInfo<ParamType>& info) const {
      const ScatterReduceShapeInfoParams& params = std::get<0>(info.param);
      std::stringstream ss;
      ss << "params_" << params << "_mode_" << std::get<1>(info.param)
         << "_dtype_" << std::get<2>(info.param) << "_includeSelf_"
         << (std::get<3>(info.param) ? "true" : "false");
      auto name = ss.str();
      std::replace_if(
          name.begin(),
          name.end(),
          [](auto c) { return (c == '-' || c == '.'); },
          '_');
      return name;
    }
  };

 private:
  bool deterministic_;
  void SetUp() override {
    deterministic_ = GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE);
    DisableCpuFallback();
    TearDownBridge();
  }
  void TearDown() override {
    SET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE, deterministic_, 1);
    RestoreMode();
  }
};

TEST_P(ScatterReduceOpTest, scatter_reduce) {
  const auto& testParams = GetParam();
  auto shapeInfo = std::get<0>(testParams);
  auto reduce = std::get<1>(testParams);
  auto dtype = std::get<2>(testParams);
  auto includeSelf = std::get<3>(testParams);

  if (!includeSelf)
    GTEST_SKIP() << "include_self=False is not supported yet";
  if (reduce == "sum" || reduce == "prod" || reduce == "mean")
    GTEST_SKIP() << "reduce=" << reduce << " is not supported yet";
  if (shapeInfo.deterministic) {
    GTEST_SKIP()
        << "Setting environment variables causes that in subsequent test cases variables are not reloaded. This causes sporadic failures. To test deterministic_mode, remove the skip macro and run the test filtering deterministic tests only";
  }
  SET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE, shapeInfo.deterministic, 1);
  GenerateInputs(
      2, {shapeInfo.inputShape, shapeInfo.sourceShape}, {dtype, dtype});
  auto selfCpu = GetCpuInput(0);
  auto selfHpu = GetHpuInput(0);
  auto srcCpu = GetCpuInput(1);
  auto srcHpu = GetHpuInput(1);
  constexpr int minRange = 0;
  int positiveDim = shapeInfo.dim < 0
      ? shapeInfo.inputShape.size() + shapeInfo.dim
      : shapeInfo.dim;
  int maxRange = shapeInfo.inputShape[positiveDim];
  at::Tensor indexCpu;

  if (shapeInfo.deterministic) {
    GenerateIntInputs(1, {shapeInfo.indexShape}, minRange, maxRange);
    indexCpu = GetCpuInput(0).to(torch::kInt64);
  } else {
    indexCpu =
        torch::arange(
            minRange, shapeInfo.indexShape[positiveDim], 1, torch::kInt64)
            .reshape(shapeInfo.indexShape);
  }

  auto indexHpu = indexCpu.to(torch::kHPU);
  auto hpuResult = torch::scatter_reduce(
      selfHpu, shapeInfo.dim, indexHpu, srcHpu, reduce, includeSelf);
  auto cpuResult = torch::scatter_reduce(
      selfCpu, shapeInfo.dim, indexCpu, srcCpu, reduce, includeSelf);
  Compare(cpuResult, hpuResult);

  if (verbose) {
    std::cout << "Self: " << selfCpu << "\n";
    std::cout << "Indexes: " << indexCpu << "\n";
    std::cout << "Source: " << srcCpu << "\n";
    std::cout << "Dim and positive dim: " << shapeInfo.dim << ", "
              << positiveDim << "\n";
    std::cout << "CPU Result: " << cpuResult << "\n";
    std::cout << "HPU Result: " << hpuResult.cpu() << "\n";
  }
}

INSTANTIATE_TEST_SUITE_P(
    sanity,
    ScatterReduceOpTest,
    ::testing::Combine(
        ::testing::Values(
            ScatterReduceShapeInfoParams{0, {3}, {2}, {2}, false},
            ScatterReduceShapeInfoParams{
                0,
                {1, 2, 2},
                {1, 2, 2},
                {1, 2, 2},
                true},
            ScatterReduceShapeInfoParams{
                0,
                {3, 4, 3},
                {2, 3, 2},
                {2, 6, 4},
                true},
            ScatterReduceShapeInfoParams{
                2,
                {3, 4, 3},
                {2, 3, 2},
                {2, 6, 4},
                true},
            ScatterReduceShapeInfoParams{
                -2,
                {3, 4, 3},
                {2, 3, 2},
                {2, 6, 4},
                true},
            ScatterReduceShapeInfoParams{
                0,
                {3, 4, 3},
                {2, 1, 1},
                {2, 6, 4},
                false},
            ScatterReduceShapeInfoParams{
                -2,
                {3, 4, 3},
                {1, 4, 1},
                {2, 6, 4},
                false}),
        ::testing::Values<std::string>("sum", "prod", "mean", "amax", "amin"),
        ::testing::Values<c10::ScalarType>(torch::kFloat, torch::kBFloat16),
        ::testing::Values<bool>(true, false)),
    ScatterReduceOpTest::GetName());
