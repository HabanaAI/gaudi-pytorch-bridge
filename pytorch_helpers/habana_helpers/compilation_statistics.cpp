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
#include <experimental/filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_helpers;
using json = nlohmannV340::json;
namespace fs = std::experimental::filesystem;
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
    case DynamicDimsPolicy::LOCAL_HIST_PER_TSR:
      return "LOCAL_HIST_PER_TSR";
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
      std::string,
      const habana_helpers::TensorShape&,
      const std::string&,
      uint64_t) override{};
  void LogShapes(std::shared_ptr<torch::jit::Graph>, InpTensorShapes&, uint64_t)
      override{};
  void LogCompilation(
      const std::string&,
      std::shared_ptr<torch::jit::Graph>,
      DynamicDimsPolicy,
      DynamicDimsPolicy,
      ResultShapes,
      uint64_t,
      const std::string&,
      CompilationPass,
      uint64_t) override{};
  void LogUsedBucket(
      int,
      std::shared_ptr<torch::jit::Graph>,
      ResultShapes,
      bool,
      uint64_t) override{};
  virtual void LogFallback(
      std::string,
      DynamicDimsPolicy,
      std::string,
      uint64_t) override{};
  void LogSelectedRecipe(uint64_t, uint64_t) override{};
  void LogRecipeMemory(std::shared_ptr<habana::RecipeValueSpec>, uint64_t)
      override{};
  void LogLaunchBase(uint64_t, uint64_t) override{};
  void LogLaunch(uint64_t, uint64_t) override{};
  void LogLaunchPerf(uint64_t, uint64_t, uint64_t) override{};
  void LogRefineCompilation(
      ResultShapes,
      std::shared_ptr<torch::jit::Graph>,
      uint64_t,
      uint64_t,
      const std::string&,
      uint64_t) override{};
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
      } catch (std::experimental::filesystem::filesystem_error const& ex) {
        std::cerr << ex.what() << std::endl;
        UNSET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
      }
    }
    std::string node_id = "";
    if (std::getenv("ID") != nullptr) {
      node_id = std::getenv("ID") ? (std::string("_") + std::getenv("ID")) : "";
    } else {
      node_id =
          std::getenv("RANK") ? (std::string("_") + std::getenv("RANK")) : "";
    }

    path += std::string("/") + std::string(id) + node_id + ".json";
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
    std::string index,
    const habana_helpers::TensorShape& shape,
    const std::string& kind,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)]["shapes"][index] =
      shape.DebugString() + (kind.empty() ? "" : " " + kind);
}

void CompilationStatistics::LogShapes(
    std::shared_ptr<torch::jit::Graph> jit_ir_graph,
    InpTensorShapes& shapes,
    uint64_t step) {
  for (size_t j = 0; j < jit_ir_graph->inputs().size(); j++) {
    auto value_input = jit_ir_graph->inputs().at(j);
    std::string tensor_name =
        std::to_string(j) + std::string("_") + value_input->debugName();
    auto tensor_shape = shapes.at(j);
    bool kind = habana_helpers::is_shape_tensor(tensor_shape.get_tensor_type());
    LogShape(tensor_name, tensor_shape, kind ? "shape tensor" : "", step);
  }
}
void CompilationStatistics::LogCompilation(
    const std::string& jit_ir,
    std::shared_ptr<torch::jit::Graph> jit_ir_graph,
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
  compilation["ranges"] = GetRanges(ranges, jit_ir_graph);
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
    std::shared_ptr<torch::jit::Graph> jit_ir_graph,
    ResultShapes ranges,
    bool refine_candidate,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json json_bucket;
  json_bucket["id"] = id;
  json_bucket["ranges"] = GetRanges(std::move(ranges), jit_ir_graph);
  json_bucket["refine candidate"] = refine_candidate;
  json_file_[GetStep(step)]["selected bucket"] = json_bucket;
}

void CompilationStatistics::LogFallback(
    std::string pass,
    DynamicDimsPolicy policy,
    std::string error,
    uint64_t step) {
  std::string key = pass + "_fallback_" + stringify(policy);
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)][key] = error;
}

void CompilationStatistics::LogSelectedRecipe(
    uint64_t signature,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)]["selected recipe"] = signature;
}

void CompilationStatistics::LogRecipeMemory(
    std::shared_ptr<habana::RecipeValueSpec> cur_rvalpsh,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)]["recipe memory size"] =
      cur_rvalpsh->recipe->get_recipe_host_mem_size();
}

void CompilationStatistics::LogLaunchBase(uint64_t ns, uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)]["synLaunch time base"] = ns;
}

void CompilationStatistics::LogLaunch(uint64_t ns, uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json_file_[GetStep(step)]["synLaunch time"] = ns;
}

void CompilationStatistics::LogLaunchPerf(
    uint64_t base_ns,
    uint64_t ns,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  if (base_ns == 0 || ns == 0) {
    json_file_[GetStep(step)]["synLaunch time performance"] = "-";
  } else {
    json_file_[GetStep(step)]["synLaunch time performance"] =
        (base_ns > ns ? "increased" : "decreased");
  }
}

void CompilationStatistics::LogRefineCompilation(
    ResultShapes ranges,
    std::shared_ptr<torch::jit::Graph> jit_ir_graph,
    uint64_t signature,
    uint64_t bucket,
    const std::string& result_str,
    uint64_t step) {
  std::lock_guard<std::mutex> lg(json_file_mutex_);
  json json_refine;
  json_refine["recipe"] = signature;
  json_refine["ranges"] = GetRanges(std::move(ranges), jit_ir_graph);
  json_refine["bucket id"] = bucket;
  json_refine["start iter"] = GetRefineInitStep();
  auto& json_refine_result = json_refine["result"];
  json_refine_result["status"] = result_str;
  json_refine_result["step"] = GetCurrentStep();
  json_file_[GetStep(step)]["refine"] = json_refine;
}

uint64_t CompilationStatistics::GetCurrentStep() {
  return step_;
}

void CompilationStatistics::SetRefineInitStep(size_t step) {
  refine_init_step_ = step;
}

size_t CompilationStatistics::GetRefineInitStep() {
  return refine_init_step_;
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

nlohmannV340::json CompilationStatistics::GetRanges(
    ResultShapes ranges,
    std::shared_ptr<torch::jit::Graph> jit_ir_graph) {
  json result;
  for (size_t j = 0; j < jit_ir_graph->inputs().size(); j++) {
    if (ranges.min_shapes.find(j) != ranges.min_shapes.end()) {
      auto value_input = jit_ir_graph->inputs().at(j);
      std::string tensor_name =
          std::to_string(j) + std::string("_") + value_input->debugName();
      result[tensor_name] = ranges.min_shapes[j].DebugString() + "-" +
          ranges.max_shapes[j].DebugString();
    }
  }
  return result;
}
} // namespace habana_helpers
