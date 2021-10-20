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

#define COMMON_ATOL_FLOAT 0.001
#define COMMON_RTOL_FLOAT 0.001

namespace habana_lazy_test {
void print_tensor_details(torch::Tensor& t, std::string tname);
}

#define PRINT_TENSOR_DETAILS(T) \
  habana_lazy_test::print_tensor_details(T, std::string(#T))

namespace habana_lazy_test {

class EnvHelper {
  bool m_defined = false;
  unsigned m_saved = 0;
  int m_seed = InitSeed();

 private:
  int InitSeed() {
    // Fix seed as 0
    const char* s = std::getenv("PT_HPU_TEST_SEED");
    return s ? std::stoi(s) : 0;
  }

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

  void RestoreMode() {
    if (m_defined) {
      SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, m_saved, 1);
    } else {
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
    }
  }

  // Wrappers with convenient names
  void SetLazyMode(unsigned mode = 1) {
    // mode can be 1, 2 or 3
    SetMode(mode);
  }

  void SetEagerMode() {
    SetMode(0);
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
