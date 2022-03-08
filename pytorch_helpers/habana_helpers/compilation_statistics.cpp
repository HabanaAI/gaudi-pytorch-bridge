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
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_helpers;
using json = nlohmannV340::json;
namespace fs = std::filesystem;
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
    case DynamicDimsPolicy::LOCAL_HISTORIC:
      return "LOCAL_HISTORIC";
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
  void LogShapes(habana_helpers::InpTensorShapes&, uint64_t) override{};
  void LogCompilation(
      const std::string&,
      DynamicDimsPolicy,
      DynamicDimsPolicy,
      ResultShapes,
      uint64_t,
      const std::string&,
      CompilationPass,
      uint64_t) override{};
  void LogUsedBucket(int, ResultShapes, bool, uint64_t) override{};
  void LogSelectedRecipe(uint64_t, uint64_t) override{};
  void LogLaunch(uint64_t, uint64_t) override{};
  void LogRefineCompilation(ResultShapes, uint64_t, uint64_t, uint64_t)
      override{};
  void LogRefineResult(const std::string&, uint64_t) override{};
  uint64_t GetCurrentStep() override {
    return 0;
  };
  void DumpAndNextStep() override{};
};

std::unique_ptr<CompilationStatistics> CompilationStatistics::Create(
    const std::string& id,
    uint64_t global_count) {
  std::unique_ptr<CompilationStatistics> result;
  std::string path = GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
  if (path != "") {
    if (fs::exists(fs::path(path)) == false) {
      try {
        fs::create_directories(path);
      } catch (std::filesystem::filesystem_error const& ex) {
        std::cerr << ex.what() << std::endl;
        UNSET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
      }
    }
    path += std::string("/") + std::string(id) + ".json";
    result = std::unique_ptr<CompilationStatistics>(
        new CompilationStatistics{path, global_count});
  } else {
    result = std::unique_ptr<CompilationStatistics>(
        new CompilationStatisticsNoOp{"", global_count});
  }
  return result;
}

CompilationStatistics::~CompilationStatistics() {
  file_handle << "\n]";
  file_handle.flush();
  file_handle.close();
}

CompilationStatistics::CompilationStatistics(
    const std::string path,
    uint64_t global_count)
    : path_{path}, step_{global_count}, file_handle(path_, std::ios::trunc) {
  if (file_handle.is_open()) {
    file_handle << "[\n";
    file_handle.flush();
  }
}

void CompilationStatistics::LogShape(
    int index,
    const habana_helpers::TensorShape& shape,
    const std::string& kind,
    uint64_t step) {
  json_file_[GetStep(step)]["shapes"][std::to_string(index)] =
      shape.DebugString() + (kind.empty() ? "" : " " + kind);
}

void CompilationStatistics::LogShapes(
    habana_helpers::InpTensorShapes& shape_map,
    uint64_t step) {
  for (size_t i = 0; i < shape_map.size(); i++) {
    LogShape(i, shape_map[i], "", step);
  }
}
void CompilationStatistics::LogCompilation(
    const std::string& jit_ir,
    DynamicDimsPolicy min_policy,
    DynamicDimsPolicy max_policy,
    ResultShapes ranges,
    uint64_t signature,
    const std::string& result,
    CompilationPass last_compilation_pass,
    uint64_t step) {
  std::stringstream ss(jit_ir);
  std::vector<std::string> ir_vector;

  while (ss.good()) {
    std::string substr;
    getline(ss, substr, '\n');
    ir_vector.push_back(substr);
  }
  json compilation;
  compilation["jit ir graph"] = ir_vector;
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
    ResultShapes ranges,
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
    ResultShapes ranges,
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
  if (step_) {
    file_handle << ",\n";
  }
  file_handle << std::setw(4) << json_file_;
  file_handle.flush();
  json_file_.clear();
  step_++;
}

std::string CompilationStatistics::GetStep(uint64_t step) {
  const size_t leading_zeros = 9;
  return absl::StrFormat(
      "%0*d", leading_zeros, step > 0 ? step : GetCurrentStep());
}

nlohmannV340::json CompilationStatistics::GetRanges(ResultShapes ranges) {
  json result;
  for (auto range : ranges.min_shapes) {
    auto index = range.first;
    result[std::to_string(index)] = ranges.min_shapes[index].DebugString() +
        "-" + ranges.max_shapes[index].DebugString();
  }
  return result;
}
} // namespace habana_helpers
