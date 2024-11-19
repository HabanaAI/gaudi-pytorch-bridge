/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "backend/habana_device/hpu_cached_devices.h"
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
  os << HpuOpTestUtil::SerializeShape(params.inputShape, "_input_");
  os << HpuOpTestUtil::SerializeShape(params.indexShape, "_index_");
  os << HpuOpTestUtil::SerializeShape(params.sourceShape, "_source_");
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
  ScatterReduceShapeInfoParams shapeInfo;
  struct GetName {
    template <class ParamType>
    std::string operator()(
        const ::testing::TestParamInfo<ParamType>& info) const {
      const ScatterReduceShapeInfoParams& params = std::get<0>(info.param);
      std::stringstream ss;
      ss << "params_" << params << "_mode_" << std::get<1>(info.param)
         << "_dtype_" << std::get<2>(info.param) << "_includeSelf_"
         << (std::get<3>(info.param) ? "true" : "false");
      return FixTestName(ss.str());
    }
  };

  ScatterReduceOpTest() {
    shapeInfo = std::get<0>(GetParam());
  }

 private:
  bool deterministicTorchOldValue = false;

  void SetUp() override {
    DisableCpuFallback();
    TearDownBridge();
    auto& hpuGConfig = habana::HPUGlobalConfig::get();
    auto& torchGConfig = at::globalContext();
    deterministicTorchOldValue = torchGConfig.deterministicAlgorithms();
    hpuGConfig.setDeterministic(shapeInfo.deterministic);
    torchGConfig.setDeterministicAlgorithms(shapeInfo.deterministic, false);
  }
  void TearDown() override {
    at::globalContext().setDeterministicAlgorithms(
        deterministicTorchOldValue, false);
    RestoreMode();
  }
};

TEST_P(ScatterReduceOpTest, scatter_reduce) {
  const auto& testParams = GetParam();
  auto reduce = std::get<1>(testParams);
  auto dtype = std::get<2>(testParams);
  auto includeSelf = std::get<3>(testParams);

  if (reduce == "mean" & shapeInfo.deterministic == true) {
    GTEST_SKIP() << "Test sporadically failing - SW-171740";
  }

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
