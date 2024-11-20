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
#include "habana_lazy/ir.h"

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: debug_utils.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool

namespace habana_lazy {

class HbLazyTensor;
class SBSInterface;
typedef std::map<std::string, std::shared_ptr<SBSInterface>> SBSInterfaceMap;
typedef std::set<size_t> SBSTensorIndexSet;

class SBSInterface {
 public:
  static std::shared_ptr<SBSInterface> getSBSHandler(std::string op_type);
  static size_t getNumberOfHandledOps();
  static size_t getNumberOfOpTries();
  static size_t getNumberOfHandledOpTensors();
  static size_t getNumberOfErrors();
  static size_t getNumberOfRuns();
  static size_t getNumberOfTensorCopies();
  static void reset();

  virtual ~SBSInterface() = default;

  virtual void populateInputForCPUOp(
      const std::vector<at::IValue>& inputs,
      const ir::MetaData& metadata,
      std::vector<at::IValue>& stack) = 0;

  virtual void setCPUInputs(const std::vector<at::IValue>& inputs) = 0;

  virtual void run(
      at::TensorList results,
      const std::vector<at::IValue>& inputs,
      const std::vector<at::IValue>& prealloc_stack = std::vector<at::IValue>(),
      const ir::NodePtr& node = nullptr) = 0;

  bool LogError(
      const std::string& op_name,
      const std::string& message_short,
      const std::string& message_detailed = "");

 protected:
  static size_t m_number_of_handled_ops;
  static size_t m_number_of_op_tries;
  static size_t m_number_of_tensor_runs;
  static size_t m_number_of_errors;
  static size_t m_number_of_tensor_copies;

 private:
  static SBSInterfaceMap m_special_sbs_ops;
};

class SBSDisabledOp : public SBSInterface {
 public:
  void populateInputForCPUOp(
      [[maybe_unused]] const std::vector<at::IValue>& inputs,
      [[maybe_unused]] const ir::MetaData& metadata,
      [[maybe_unused]] std::vector<at::IValue>& stack) override {}

  void setCPUInputs([
      [maybe_unused]] const std::vector<at::IValue>& inputs) override {}

  void run(
      at::TensorList results,
      [[maybe_unused]] const std::vector<at::IValue>& inputs,
      [[maybe_unused]] const std::vector<at::IValue>& prealloc_stack,
      [[maybe_unused]] const ir::NodePtr& node) override;
};

class SBSRunner : public SBSInterface {
 public:
  explicit SBSRunner(SBSTensorIndexSet disabled_output_tensors = {})
      : m_disabled_output_tensors(disabled_output_tensors) {}

  void populateInputForCPUOp(
      const std::vector<at::IValue>& inputs,
      const ir::MetaData& metadata,
      std::vector<at::IValue>& stack) override;

  void setCPUInputs(const std::vector<at::IValue>& inputs) override;

  void run(
      at::TensorList results,
      const std::vector<at::IValue>& inputs,
      const std::vector<at::IValue>& prealloc_stack = std::vector<at::IValue>(),
      const ir::NodePtr& node = nullptr) override;

 protected:
  void handleTensorForCPUInput(
      const at::Tensor& input,
      std::vector<at::IValue>& inputs_modified);
  virtual at::Tensor prepareTensorToCPU(const at::Tensor& tensor, size_t index);
  virtual void processOutputCPUTensor(
      HbLazyTensor hl_result,
      at::Tensor& cpu_tensor,
      size_t index);

 private:
  at::IValue gatherInputForCPUOp(const at::Tensor& input, size_t index);
  virtual c10::Symbol buildCPUOpSymbol(const c10::Symbol& hpu_op);

  virtual bool getNodeInfo(
      const at::Tensor& result,
      ir::NodePtr& node,
      std::string& ir_name);

  std::shared_ptr<torch::jit::Operator> createCPUOperator(
      const std::string& ir_name,
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs);

  // used in special cases when the CPU tensor value is knowingly different than
  // the HPU
  SBSTensorIndexSet m_disabled_output_tensors;
};

class SBSPermutable : public SBSRunner {
 public:
  explicit SBSPermutable(size_t index_to_permute)
      : SBSRunner(), m_index_to_permute(index_to_permute) {}

 protected:
  at::Tensor prepareTensorToCPU(const at::Tensor& tensor, size_t index)
      override;

 private:
  size_t m_index_to_permute;

  c10::Symbol buildCPUOpSymbol(const c10::Symbol& hpu_op) override;
};

class SBSViews : public SBSRunner {
  bool getNodeInfo(
      const at::Tensor& result,
      ir::NodePtr& node,
      std::string& ir_name) override;
};

} // namespace habana_lazy