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

#pragma once
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/random.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/ir.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#define COMMON_ATOL_FLOAT 0.001
#define COMMON_RTOL_FLOAT 0.001

namespace habana_lazy_test {

class EnvHelper {
  bool m_defined = false;
  unsigned m_saved = 0;
  unsigned m_dynamic = 0;
  unsigned m_inference = 0;
  unsigned m_fallback_pass = 1;
  uint64_t m_seed = InitSeed();
  bool m_recipe_cache_enable = true;
  bool m_eager_gc_enable = false;
  bool m_eager_view_handling_enable = false;
  std::optional<bool> m_shape_agnostic_enable{};
  bool m_acc_par_mode_enable = true;
  std::string place_on_cpu_env = GET_ENV_FLAG_NEW(PT_HPU_PLACE_ON_CPU);

 private:
  uint64_t InitSeed();

 protected:
  void TearDownBridge() {
    habana::HabanaLaunchOpUtils::cleanUp();
  }

  void SetMode(unsigned mode = 1, int force = 0) {
    m_defined = IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE);
    if (m_defined) {
      m_saved = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }
    if (mode) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, mode, force);
    } else {
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }
  }

  void SetDynamicMode() {
    m_dynamic = habana_helpers::GetRefineDynamicShapeStatus();
    if (!m_dynamic) {
      habana_helpers::EnableRefineDynamicShape();
    }
  }

  void UnsetDynamicMode() {
    if (!m_dynamic) {
      habana_helpers::DisableRefineDynamicShape();
    }
  }

  void SetInferenceMode() {
    m_inference = GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
    if (!m_inference) {
      SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE, true, 1);
    }
  }

  void UnsetInferenceMode() {
    if (!m_inference) {
      UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
    }
  }

  void DisableDynamicPassFallback() {
    m_fallback_pass = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK);
    if (m_fallback_pass) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK, false, 1);
    }
  }

  void RestoreDynamicPassFallback() {
    if (m_fallback_pass) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK, true, 1);
    }
  }

  void DisableCpuFallback() {
    if (place_on_cpu_env.find("none") == std::string::npos) {
      SET_ENV_FLAG_NEW(PT_HPU_PLACE_ON_CPU, "none", 1);
      habana::HpuFallbackHelper::get()->enumerate_fallback();
    }
  }

  void EnableCpuFallback() {
    if (!place_on_cpu_env.empty()) {
      SET_ENV_FLAG_NEW(PT_HPU_PLACE_ON_CPU, "", 1);
      habana::HpuFallbackHelper::get()->enumerate_fallback();
    }
  }

  void DisableRecipeCache() {
    m_recipe_cache_enable = GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE);
    if (m_recipe_cache_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, false, 1);
    }
  }

  void RestoreRecipeCache() {
    if (m_recipe_cache_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, true, 1);
    }
  }

  void DisableAccParMode() {
    m_acc_par_mode_enable = GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE);
    if (m_acc_par_mode_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE, false, 1);
    }
  }

  void RestoreAccParMode() {
    if (m_acc_par_mode_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE, true, 1);
    }
  }

  void EnableEagerViewHandling() {
    m_eager_view_handling_enable = GET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING);
    if (!m_eager_view_handling_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING, true, 1);
    }
  }

  void RestoreEagerViewHandling() {
    if (!m_eager_view_handling_enable) {
      SET_ENV_FLAG_NEW(PT_HPU_EAGER_VIEW_HANDLING, false, 1);
    }
  }

  void EnableShapeAgnostic(bool enable = true) {
    if (enable != GET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH)) {
      SET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH, enable, 1);
      m_shape_agnostic_enable = !enable;
    }
  }

  void RestoreShapeAgnostic() {
    if (m_shape_agnostic_enable.has_value()) {
      SET_ENV_FLAG_NEW(
          PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH,
          m_shape_agnostic_enable.value(),
          1);
    }
  }

  void RestoreMode() {
    if (m_defined) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, m_saved, 1);
    } else {
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }

    SET_ENV_FLAG_NEW(PT_HPU_PLACE_ON_CPU, place_on_cpu_env.c_str(), 1);
    habana::HpuFallbackHelper::get()->enumerate_fallback();
  }

  // Wrappers with convenient names
  void SetLazyMode(unsigned mode = 1) {
    // mode can be 1, 2 or 3
    SetMode(mode);
  }

  void SetEagerMode() {
    SetMode(0);
  }

  uint64_t GetSeed() const {
    return m_seed;
  }

  void SetSeed() const {
    torch::manual_seed(m_seed);
    habana::detail::getDefaultHPUGenerator().set_current_seed(m_seed);
  }

 public:
  template <typename F>
  void ExecuteEager(F&& fn) {
    unsigned old_mode;
    bool is_defined = IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE);
    if (is_defined) {
      old_mode = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 0, 1);

    std::forward<F>(fn)();
    if (is_defined)
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, old_mode, 1);
    else
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  }
};

class LazyTest : public ::testing::Test, public EnvHelper {
 protected:
  void SetUp() override {
    // Save the original value
    SetLazyMode();
    SetSeed();
    DisableCpuFallback();
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
    TearDownBridge();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    // Restore the original value back
    RestoreMode();
  }

  void ForceMode(unsigned mode) {
    SetMode(mode, 1);
  }
};

class LazyDynamicTest : public LazyTest {
 protected:
  void SetUp() override {
    SetDynamicMode();
    DisableDynamicPassFallback();
    habana::HPUDeviceContext::recipe_cache().clear();
    LazyTest::SetUp();
  }

  void TearDown() override {
    UnsetDynamicMode();
    RestoreDynamicPassFallback();
    LazyTest::TearDown();
  }
};

typedef struct {
  habana_lazy::ir::NodePtrList post_order_nodes;
  size_t post_order_nodes_hash;
} PostOrderTestStruct;

// Create a 3 Node vector from first level IR
// This is what is expected after a post order traversal
// of the first level IR
PostOrderTestStruct GetPostOrderNodes(bool jumbld = false);
// Create input IValues.
// tensor_shapes creates n tensors with given shapes.
// scalars creates m scalars with given value
std::vector<torch::jit::IValue> CreateInputs(
    std::vector<std::vector<int64_t>> tensor_shapes,
    std::vector<float> scalars);

std::shared_ptr<torch::jit::Graph> CreateJITGraph();
torch::jit::Stack createStack(std::vector<at::Tensor>&& list);

} // namespace habana_lazy_test

namespace jit_ir_test {
nlohmannV340::json read_json(std::string input_json);
std::string get_jit_graph(nlohmannV340::json json_);
at::Tensor create_empty_tensor(
    const std::vector<int64_t>& tshape,
    c10::TensorOptions& tensor_options,
    bool is_shape_tensor = false);
std::map<std::string, c10::ScalarType> create_tensor_dtype_map(
    const at::ArrayRef<torch::jit::Value*>& inputs);
std::vector<at::Tensor> get_input_tensors(
    const std::map<std::string, std::string>& shapes_map,
    std::map<std::string, c10::ScalarType> tensor_dtype_map);
} // namespace jit_ir_test
