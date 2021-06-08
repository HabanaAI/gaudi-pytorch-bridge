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
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_cache.h"

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

namespace habana_lazy_test {

class EnvHelper {
  char* m_saved = nullptr;
  int m_seed = InitSeed();

 private:
  int InitSeed() {
    const char* s = std::getenv("PT_HPU_TEST_SEED");
    srand(time(nullptr));
    return s ? std::stoi(s) : rand();
  }

 protected:
  void SetMode(const char* mode = "1", int force = 0) {
    m_saved = std::getenv("PT_HPU_LAZY_MODE");

    if (mode) {
      setenv("PT_HPU_LAZY_MODE", mode, force);
    } else {
      unsetenv("PT_HPU_LAZY_MODE");
    }
  }

  void RestoreMode() {
    if (m_saved) {
      setenv("PT_HPU_LAZY_MODE", m_saved, 1);
    } else {
      unsetenv("PT_HPU_LAZY_MODE");
    }
  }

  // Wrappers with convenient names
  void SetLazyMode(const char* mode = "1") {
    // mode can be 1, 2 or 3
    SetMode(mode);
  }

  void SetEagerMode() {
    SetMode(nullptr);
  }

  int GetSeed() const {
    return m_seed;
  }

  void SetSeed() const {
    torch::manual_seed(m_seed);
  }

 public:
  template <typename F>
  void ExecuteEager(F&& fn) {
    const char* old = std::getenv("PT_HPU_LAZY_MODE");
    unsetenv("PT_HPU_LAZY_MODE");

    std::forward<F>(fn)();

    if (old) {
      setenv("PT_HPU_LAZY_MODE", old, 1);
    }
  }
};

class LazyTest : public ::testing::Test, public EnvHelper {
  void SetUp() override {
    // Save the original value
    SetLazyMode();

    SetSeed();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    // Restore the original value back
    RestoreMode();
  }

 protected:
  void ForceMode(int mode) {
    SetMode(std::to_string(mode).c_str(), 1);
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

} // namespace habana_lazy_test
