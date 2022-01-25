/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_lazy/ir.h"

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: sbs_runner.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool

namespace habana_lazy {

enum SBSModes : unsigned {
  SBS_MODE_DISABLED = 0,
  SBS_MODE_STANDALONE = 1,
  SBS_MODE_USE_CPU_INPUT = 2,
  SBS_MODE_USE_HPU_INPUT = 3
};

class SBSInterface {
 public:
  static std::shared_ptr<SBSInterface> getSBSHandler(std::string op_type);

  virtual void populateInputForCPUOp(
      const std::vector<at::IValue>& inputs,
      const ir::MetaData& metadata,
      std::vector<at::IValue>& stack) = 0;

  virtual void setCPUInputs(const std::vector<at::IValue>& inputs) = 0;

  virtual void run(
      at::TensorList results,
      const std::vector<at::IValue>& inputs,
      const std::vector<at::IValue>& prealloc_stack =
          std::vector<at::IValue>()) = 0;

  bool LogError(
      const std::string& op_name,
      const std::string& message_short,
      const std::string& message_detailed = "");

 private:
  static std::map<std::string, std::shared_ptr<SBSInterface>> m_special_sbs_ops;
};

class SBSDisabledOp : public SBSInterface {
 public:
  void populateInputForCPUOp(
      UNUSED const std::vector<at::IValue>& inputs,
      UNUSED const ir::MetaData& metadata,
      UNUSED std::vector<at::IValue>& stack) override {}

  void setCPUInputs(UNUSED const std::vector<at::IValue>& inputs) override {}

  void run(
      at::TensorList results,
      UNUSED const std::vector<at::IValue>& inputs,
      UNUSED const std::vector<at::IValue>& prealloc_stack) override;
};

class SBSRunner : public SBSInterface {
 public:
  void populateInputForCPUOp(
      const std::vector<at::IValue>& inputs,
      const ir::MetaData& metadata,
      std::vector<at::IValue>& stack) override;

  void setCPUInputs(const std::vector<at::IValue>& inputs) override;

  void run(
      at::TensorList results,
      const std::vector<at::IValue>& inputs,
      const std::vector<at::IValue>& prealloc_stack =
          std::vector<at::IValue>()) override;

 private:
  at::IValue gatherInputForCPUOp(const at::Tensor& input, size_t index);
  virtual at::Tensor prepareCPUTensor(const at::Tensor& tensor, size_t index);
  virtual c10::Symbol buildCPUOpSymbol(const c10::Symbol& hpu_op);

  std::shared_ptr<torch::jit::Operator> createCPUOperator(
      std::string ir_name,
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs);
};
} // namespace habana_lazy