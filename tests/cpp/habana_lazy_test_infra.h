/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_cache.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#define COMMON_ATOL_FLOAT 0.001
#define COMMON_RTOL_FLOAT 0.001

namespace habana_lazy_test {

const char* const place_on_cpu_env = getenv("PT_HPU_PLACE_ON_CPU");

class EnvHelper {
  bool m_defined = false;
  unsigned m_saved = 0;
  unsigned m_dynamic = 0;
  unsigned m_fallback_pass = 1;
  unsigned m_fallback_launch = 0;
  uint64_t m_seed = InitSeed();

 private:
  uint64_t InitSeed();

 protected:
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
    m_dynamic = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
    if (!m_dynamic) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
    }
  }

  void UnsetDynamicMode() {
    if (!m_dynamic) {
      UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
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

  void EnableDynamicLaunchFallback() {
    m_fallback_launch = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK);
    if (!m_fallback_launch) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK, true, 1);
    }
  }

  void RestoreDynamicLaunchFallback() {
    if (!m_fallback_launch) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK, false, 1);
    }
  }

  void DisableCpuFallback() {
    if (!place_on_cpu_env) {
      setenv("PT_HPU_PLACE_ON_CPU", "none", 0);
      habana::HpuFallbackHelper::get()->enumerate_fallback();
    }
  }

  void EnableCpuFallback() {
    if (!place_on_cpu_env) {
      unsetenv("PT_HPU_PLACE_ON_CPU");
      habana::HpuFallbackHelper::get()->enumerate_fallback();
    }
  }

  void RestoreMode() {
    if (m_defined) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, m_saved, 1);
    } else {
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }

    if (place_on_cpu_env) {
      setenv("PT_HPU_PLACE_ON_CPU", place_on_cpu_env, 1);
    } else {
      unsetenv("PT_HPU_PLACE_ON_CPU");
    }
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
  void SetUp() override {
    // Save the original value
    SetLazyMode();

    SetSeed();

    DisableCpuFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    // Restore the original value back
    RestoreMode();
  }

 protected:
  void ForceMode(unsigned mode) {
    SetMode(mode, 1);
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
