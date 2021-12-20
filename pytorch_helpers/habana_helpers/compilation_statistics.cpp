/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include "compilation_statistics.h"
#include <fstream>
#include <utility>
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_helpers;
using json = nlohmannV340::json;
namespace {
std::string stringify(DynamicDimsPolicy policy) {
  switch (policy) {
    case DynamicDimsPolicy::CALCULATED:
      return "CALCULATED";
    case DynamicDimsPolicy::CURRENT:
      return "CURRENT";
    case DynamicDimsPolicy::DEFAULT:
      return "DEFAULT";
    case DynamicDimsPolicy::FLATTENED:
      return "FLATTENED";
    case DynamicDimsPolicy::HISTORIC:
      return "HISTORIC";
    default:
      LOG(FATAL) << "Unknown compilation policy";
  }
  return "";
}

std::string stringify(CompilationPass compilation_pass) {
  switch (compilation_pass) {
    case CompilationPass::DYNAMIC_CURRENT:
      return "DYNAMIC CURRENT";
    case CompilationPass::DYNAMIC_MAX:
      return "DYNAMIC MIN + DYNAMIC MAX";
    case CompilationPass::DYNAMIC_MIN:
      return "DYNAMIC MIN";
    case CompilationPass::STATIC:
      return "STATIC";
    default:
      LOG(FATAL) << "Unknown compilation pass";
  }
  return "";
}
}; // namespace
namespace habana_helpers {
class CompilationStatisticsNoOp : public CompilationStatistics {
  using CompilationStatistics::CompilationStatistics;
  ~CompilationStatisticsNoOp() override = default;
  void LogShape(
      int,
      const habana_helpers::TensorShape&,
      const std::string&,
      uint64_t) override{};
  void LogCompilation(
      DynamicDimsPolicy,
      DynamicDimsPolicy,
      DynamicBucketInfo::ResultShapes,
      uint64_t,
      const std::string&,
      CompilationPass,
      uint64_t) override{};
  void LogUsedBucket(int, DynamicBucketInfo::ResultShapes, bool, uint64_t)
      override{};
  void LogSelectedRecipe(uint64_t, uint64_t) override{};
  void LogLaunch(uint64_t, uint64_t) override{};
  void LogRefineCompilation(
      DynamicBucketInfo::ResultShapes,
      uint64_t,
      uint64_t,
      uint64_t) override{};
  void LogRefineResult(const std::string&, uint64_t) override{};
  uint64_t GetCurrentStep() override {
    return 0;
  };
  void DumpAndNextStep() override{};
};

std::unique_ptr<CompilationStatistics> CompilationStatistics::Create(
    size_t id,
    uint64_t global_count) {
  std::unique_ptr<CompilationStatistics> result;
  // https://jira.habana-labs.com/browse/SW-69265
  std::string path_ = GET_ENV_FLAG(PT_COMPILATION_STATS_PATH);
  if (path_ != "") {
    path_ += std::string("/JIT_IR_") + std::to_string(id) + ".json";
    result = std::unique_ptr<CompilationStatistics>(
        new CompilationStatistics{path_, global_count});
  } else {
    result = std::unique_ptr<CompilationStatistics>(
        new CompilationStatisticsNoOp{"", global_count});
  }
  return result;
}

CompilationStatistics::~CompilationStatistics() {
  DumpAndNextStep();
}

CompilationStatistics::CompilationStatistics(
    absl::string_view path,
    uint64_t global_count)
    : path_{path}, step_{global_count} {}
void CompilationStatistics::LogShape(
    int index,
    const habana_helpers::TensorShape& shape,
    const std::string& kind,
    uint64_t step) {
  json_file_[GetStep(step)]["shapes"][std::to_string(index)] =
      shape.DebugString() + (kind.empty() ? "" : " " + kind);
}
void CompilationStatistics::LogCompilation(
    DynamicDimsPolicy min_policy,
    DynamicDimsPolicy max_policy,
    DynamicBucketInfo::ResultShapes ranges,
    uint64_t signature,
    const std::string& result,
    CompilationPass last_compilation_pass,
    uint64_t step) {
  json compilation;
  compilation["min policy"] = stringify(min_policy);
  compilation["max policy"] = stringify(max_policy);
  compilation["ranges"] = GetRanges(ranges);
  compilation["recipe"] = signature;
  compilation["result"] = result;
  compilation["scope"] = stringify(last_compilation_pass);
  const auto kCompilations = "compilations";
  auto& json_step = json_file_[GetStep(step)];
  if (json_step.find(kCompilations) == json_step.end()) {
    json_step[kCompilations] = json::array();
  }
  json_step[kCompilations].push_back(compilation);
}

void CompilationStatistics::LogUsedBucket(
    int id,
    DynamicBucketInfo::ResultShapes ranges,
    bool refine_candidate,
    uint64_t step) {
  json json_bucket;
  json_bucket["id"] = id;
  json_bucket["ranges"] = GetRanges(std::move(ranges));
  json_bucket["refine candidate"] = refine_candidate;
  json_file_[GetStep(step)]["selected bucket"] = json_bucket;
}

void CompilationStatistics::LogSelectedRecipe(
    uint64_t signature,
    uint64_t step) {
  json_file_[GetStep(step)]["selected recipe"] = signature;
}

void CompilationStatistics::LogLaunch(uint64_t ms, uint64_t step) {
  json_file_[GetStep(step)]["synLaunch time"] = ms;
}

void CompilationStatistics::LogRefineCompilation(
    DynamicBucketInfo::ResultShapes ranges,
    uint64_t signature,
    uint64_t bucket,
    uint64_t step) {
  json json_refine;
  json_refine["recipe"] = signature;
  json_refine["ranges"] = GetRanges(std::move(ranges));
  json_refine["bucket id"] = bucket;
  json_file_[GetStep(step)]["refine"] = json_refine;
}

void CompilationStatistics::LogRefineResult(
    const std::string& result,
    uint64_t step) {
  auto& json_refine = json_file_[GetStep(step)]["refine"]["result"];
  json_refine["status"] = result;
  json_refine["step"] = GetCurrentStep();
}

uint64_t CompilationStatistics::GetCurrentStep() {
  return step_;
}
void CompilationStatistics::DumpAndNextStep() {
  std::ofstream out(path_, std::ofstream::app);
  out << std::setw(4) << json_file_ << '\n';
  out.flush();
  step_++;
}

std::string CompilationStatistics::GetStep(uint64_t step) {
  const size_t leading_zeros = 9;
  return absl::StrFormat(
      "%0*d", leading_zeros, step > 0 ? step : GetCurrentStep());
}

nlohmannV340::json CompilationStatistics::GetRanges(
    DynamicBucketInfo::ResultShapes ranges) {
  json result;
  for (auto range : ranges.min_shapes) {
    auto index = range.first;
    result[std::to_string(index)] = ranges.min_shapes[index].DebugString() +
        "-" + ranges.max_shapes[index].DebugString();
  }
  return result;
}
} // namespace habana_helpers