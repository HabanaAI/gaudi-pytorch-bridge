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
#include "sbs_runner.h"
#include <exception>
#include <memory>
#include <sstream>
#include "aten_lazy_bridge.h"
#include "debug_utils.h"
#include "lazy_executor.h"
#include "passes/pass_utils.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "sbs_debug.h"

namespace habana_lazy {

SBSInterfaceMap SBSInterface::m_special_sbs_ops = {
    {"aten::convolution_overrideable",
     std::static_pointer_cast<SBSInterface>(
         std::make_shared<SBSPermutable>(-1))},
    {"hpu::nonzero",
     std::static_pointer_cast<SBSInterface>(std::make_shared<SBSDisabledOp>())},
    // Failed in LazyIndexKernelTest.IndexTest
    {"hpu::index",
     std::static_pointer_cast<SBSInterface>(std::make_shared<SBSDisabledOp>())},
};

size_t SBSInterface::m_number_of_handled_ops = 0;
size_t SBSInterface::m_number_of_op_tries = 0;
size_t SBSInterface::m_number_of_tensor_runs = 0;
size_t SBSInterface::m_number_of_errors = 0;
size_t SBSInterface::m_number_of_tensor_copies = 0;

std::shared_ptr<SBSInterface> SBSInterface::getSBSHandler(std::string op_type) {
  auto iter = m_special_sbs_ops.find(op_type);
  if (iter != m_special_sbs_ops.end()) {
    return iter->second;
  }
  return std::make_shared<SBSRunner>();
}

size_t SBSInterface::getNumberOfHandledOps() {
  return m_number_of_handled_ops;
}
size_t SBSInterface::getNumberOfOpTries() {
  return m_number_of_op_tries;
}
size_t SBSInterface::getNumberOfHandledOpTensors() {
  return m_number_of_errors + m_number_of_tensor_runs;
}
size_t SBSInterface::getNumberOfErrors() {
  return m_number_of_errors;
}
size_t SBSInterface::getNumberOfRuns() {
  return m_number_of_tensor_runs;
}
size_t SBSInterface::getNumberOfTensorCopies() {
  return m_number_of_tensor_copies;
}

void SBSInterface::reset() {
  PT_LAZY_DEBUG("SBS: Resetting runner");
  m_number_of_handled_ops = 0;
  m_number_of_op_tries = 0;
  m_number_of_tensor_runs = 0;
  m_number_of_errors = 0;
  m_number_of_tensor_copies = 0;
}

bool SBSInterface::LogError(
    const std::string& op_name,
    const std::string& message_short,
    const std::string& message_detailed) {
  ++m_number_of_errors;
  ++m_number_of_op_tries;
  PT_LAZY_DEBUG(
      __FUNCTION__, " SBS: Current number of op errors: ", m_number_of_errors);
  return SBSDebug::getInstance().LogError(
      op_name, message_short, message_detailed);
}

void SBSDisabledOp::run(
    at::TensorList results,
    [[maybe_unused]] const std::vector<at::IValue>& inputs,
    [[maybe_unused]] const std::vector<at::IValue>& prealloc_stack,
    [[maybe_unused]] const ir::NodePtr& prealloc_node) {
  auto hl_result = GetHbLazyTensor(results[0]);
  LogError(hl_result.FetchSBSTensorName(), "SBS is disabled for op");
}

at::IValue SBSRunner::gatherInputForCPUOp(
    const at::Tensor& input,
    size_t index) {
  auto hl_input = GetHbLazyTensor(input);
  c10::optional<at::Tensor> pTensor = hl_input.GetCPUTensorData();
  if (pTensor != c10::nullopt) {
    PT_LAZY_DEBUG("SBS: pTensor != c10::nullopt");
    // Decision point to use HPU data
    if ((GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_USE_HPU_INPUT) &&
        (hl_input.GetSBSLiveTensorIndication())) {
      PT_LAZY_DEBUG(
          "SBS: HPU input, decision point. Type: ", input.scalar_type());
      return std::move(prepareTensorToCPU(input, index));
    } else {
      const auto& cpu_tensor = pTensor.value();
      PT_LAZY_DEBUG(
          "SBS: CPU input, or not decision point. Type: ",
          cpu_tensor.scalar_type());
      return std::move(cpu_tensor);
    }
  } else {
    PT_LAZY_DEBUG("SBS: pTensor == c10::nullopt");
    return std::move(prepareTensorToCPU(input, index));
  }
}

static torch::jit::Value* addInputToGraph(
    const at::Tensor& input,
    std::shared_ptr<torch::jit::Graph>& graph) {
  auto hl_input = GetHbLazyTensor(input);
  auto inp = hl_input.GetIrValue();

  auto t = graph->addInput(inp.ToString());
  HABANA_ASSERT(!inp.m_data_ptr.expired());
  std::shared_ptr<Data> d = inp.m_data_ptr.lock();

  t->setType(c10::TensorType::create(
      d->logical_element_type, d->device, d->sizes.size(), false));
  t->setDebugName(inp.ToString());

  return t;
}

void SBSRunner::populateInputForCPUOp(
    const std::vector<at::IValue>& inputs,
    const ir::MetaData& metadata,
    std::vector<at::IValue>& stack) {
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    return;
  }
  PT_LAZY_DEBUG("SBS: populateInputForCPUOp: inputs size=", inputs.size());
  stack.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (metadata.count(i)) {
      if (input.isTensor()) {
        PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata, tensor");
        stack[i] = at::IValue();
      } else if (input.isDevice()) {
        PT_LAZY_DEBUG(
            "SBS: input index: ",
            i,
            " is metadata device: ",
            input.toDevice().str(),
            " setting to CPU");
        stack[i] = c10::Device(c10::kCPU);
      } else {
        PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata, not tensor");
        stack[i] = std::move(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is tensor");
      stack[i] = gatherInputForCPUOp(input.toTensor(), i);
    } else if (input.isTensorList()) {
      TORCH_CHECK(false, "SBS: Got tensor list: i=", i, " this is unhandled");
    } else if (input.isScalar()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is scalar");
      stack[i] = std::move(input.toScalar());
    } else {
      PT_LAZY_DEBUG(
          "SBS: input index: ", i, " is something else, this is unhandled");
    }

    if (stack[i].isTensor()) {
      TORCH_CHECK(
          stack[i].toTensor().device().type() == c10::DeviceType::CPU,
          "SBS: Input tensor to CPU Op is not in CPU. Stack index: ",
          i);
    }
  }

  PT_LAZY_DEBUG("SBS: populateInputForCPUOp: input stack size=", stack.size());
}

void SBSRunner::handleTensorForCPUInput(
    const at::Tensor& input,
    std::vector<at::IValue>& inputs_modified) {
  if (!input.defined()) {
    // setting this undefined tensor, it's meant to be that way
    inputs_modified.push_back(input);
    return;
  }
  if (input.device().type() != c10::DeviceType::HPU) {
    // a special case when tensor is still on CPU - see set_inputs()
    inputs_modified.push_back(std::move(input.detach().to(c10::kHPU)));
    if (!habana_lazy::AccThread::IsAccThreadEnabled() ||
        habana_lazy::AccThread::Get().CanUseAccThread()) {
      ++m_number_of_tensor_copies; // we'll increase number of tensor copies to
      // validate sbs run in test
    }
    return;
  }
  auto hl_input = GetHbLazyTensor(input);
  c10::optional<at::Tensor> pTensor = hl_input.GetCPUTensorData();
  if ((pTensor != c10::nullopt) && hl_input.GetSBSLiveTensorIndication()) {
    inputs_modified.push_back(
        std::move(pTensor.value().detach().to(c10::kHPU)));
    if (!habana_lazy::AccThread::IsAccThreadEnabled() ||
        habana_lazy::AccThread::Get().CanUseAccThread()) {
      ++m_number_of_tensor_copies; // we'll increase number of tensor copies to
      // validate sbs run in test
    }
  } else {
    if (hl_input.GetSBSLiveTensorIndication()) {
      PT_LAZY_WARN(
          "SBS: Tensor is live (decision point), but has no CPU (SBS is not supported). Name: ",
          hl_input.CurrentIrValue().ToString(),
          " sbs name: ",
          hl_input.FetchSBSTensorName());
    }
    // There's no CPU input or this is not a decision point
    // >> we'll take the HPU data
    inputs_modified.push_back(std::move(input));
  }
}

void SBSRunner::setCPUInputs(const std::vector<at::IValue>& inputs) {
  // PT_SBS=2 means inject CPU inputs to HPU
  if (GET_ENV_FLAG_NEW(PT_SBS) != SBSModes::SBS_MODE_USE_CPU_INPUT) {
    return;
  }
  std::vector<at::IValue> inputs_modified;
  for (auto& input : inputs) {
    if (input.isTensor()) {
      handleTensorForCPUInput(input.toTensor(), inputs_modified);
    } else if (input.isTensorList()) {
      for (const at::Tensor& tensor : input.toTensorList()) {
        handleTensorForCPUInput(tensor, inputs_modified);
      }
    } else {
      inputs_modified.push_back(std::move(input));
    }
  }
}

void SBSRunner::run(
    at::TensorList results,
    const std::vector<at::IValue>& inputs,
    const std::vector<at::IValue>& prealloc_stack,
    const ir::NodePtr& prealloc_node) {
  PT_LAZY_TRACE;
  // Get the CPU Op
  // getting ir node, we'll need it to make the jit node
  ir::NodePtr node = prealloc_node;
  std::string ir_name;

  if (!node && (!getNodeInfo(results[0], node, ir_name))) {
    return;
  }

  PT_LAZY_DEBUG("Checking node. IR name: ", ir_name, " node=", *node);

  if (!node) {
    LogError(ir_name, "IR Node doesn't exist (runSBS)");
    return;
  }
  PT_LAZY_DEBUG("SBS: Trying SBS for op tensor ", node->GetName());
  auto jit_op = createCPUOperator(ir_name, node, inputs);
  if (!jit_op) {
    return;
  }

  std::vector<at::IValue> stack = prealloc_stack;
  if (stack.empty()) {
    PT_LAZY_DEBUG("runSBS calling populateInputForCPUOp. IR name: ", ir_name);
    populateInputForCPUOp(inputs, node->GetMetaData(), stack);
  }

  TORCH_CHECK(
      stack.size() == inputs.size(),
      "SBS: HPU and CPU inputs size should equal");

  try {
    PT_LAZY_DEBUG(
        "SBS: Running CPU Op Schema: ",
        torch::jit::canonicalSchemaString(jit_op->schema()));
    // Run CPU kernel
    jit_op->getOperation()(stack);
  } catch (std::exception& e) {
    std::string error_str = e.what();
    std::stringstream ss;
    ss << "Failed to run CPU Op. Details :\n" << error_str;
    std::string first_error_line = error_str.substr(0, error_str.find('\n'));
    std::string error_short = std::string("Failed to run CPU Op: ") +
        first_error_line + " (Check lazy log for call stack)";
    LogError(ir_name, error_short, ss.str());
    return;
  }

  std::string op_name = node->op().toQualString();
  PT_LAZY_DEBUG("SBS: CPU Op output stack size: ", stack.size());
  TORCH_CHECK(
      stack.size() == results.size(),
      "SBS: HPU and CPU output size should equal");

  // Connect CPU result to HPU result
  for (size_t i = 0; i < stack.size(); ++i) {
    auto& output = stack[i];
    auto& result = results[i];
    if (output.isTensor()) {
      TORCH_CHECK(
          output.toTensor().device().type() == c10::DeviceType::CPU,
          "SBS: CPU output tensor is not in cpu. stack index =",
          i);
      PT_LAZY_DEBUG("SBS: calling process CPU tensor (single)");
      processOutputCPUTensor(GetHbLazyTensor(result), output.toTensor(), i);
    } else if (output.isTensorList()) {
      for (at::Tensor tensor : output.toTensorList()) {
        // TODO: Are we sure this is the right thing?
        PT_LAZY_DEBUG("SBS: calling process CPU tensor (from list)");
        processOutputCPUTensor(GetHbLazyTensor(result), tensor, i);
      }
    }
  }
  ++m_number_of_handled_ops;
  PT_LAZY_DEBUG(
      "SBS: finished run for op: ", op_name, " results[0] name: ", ir_name);
}

std::shared_ptr<torch::jit::Operator> SBSRunner::createCPUOperator(
    const std::string& ir_name,
    ir::NodePtr node,
    const std::vector<at::IValue>& inputs) {
  std::vector<torch::jit::Value*> node_inputs(
      inputs.size()); // includes input and metadata
  auto graph = std::make_shared<torch::jit::Graph>();
  auto nodeMetadata = node->GetMetaData();

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (nodeMetadata.count(i)) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata");
      if (input.isTensor()) // undefined tensor as decided in
                            // lazy_kernels.h::create_node()
      {
        node_inputs[i] = graph->addInput();
      } else {
        node_inputs[i] = graph->insertConstant(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is tensor");
      node_inputs[i] = addInputToGraph(input.toTensor(), graph);
    } else if (input.isTensorList()) {
      TORCH_CHECK(
          false, "SBS: Got tensor list: i=", i, " this is unhandled now");
    } // scalar or something else
    // logic taken from LazyOp::create_node()
    else {
      PT_LAZY_DEBUG(
          "SBS: input index: ",
          i,
          " is something else. Treating as scalar. type: ",
          input.toScalar().type());
      node_inputs[i] = graph->insertConstant(input.toScalar());
    }
  }

  auto cpu_op = buildCPUOpSymbol(node->op());

  std::shared_ptr<torch::jit::WithCurrentScope> scope_context;
  if (node->GetScope()) {
    scope_context = std::make_shared<torch::jit::WithCurrentScope>(
        *graph,
        c10::make_intrusive<torch::jit::Scope>(
            torch::jit::ScopePtr(),
            c10::Symbol::fromQualString("debug::" + *node->GetScope())));
  }
  at::ArrayRef<torch::jit::Value*> args(node_inputs);
  auto jit_node = graph->create(cpu_op, args, node->GetNumOutputs());
  if (!jit_node) {
    LogError(ir_name, "CPU Op was not found (jit node is null)");
    return nullptr;
  }
  graph->insertNode(jit_node);
  PT_LAZY_DEBUG("SBS: Candidate CPU JIT graph: ", graph->toString());

  // PT_LAZY_DEBUG("creating jit op");
  std::shared_ptr<torch::jit::Operator> op = nullptr;
  try {
    op = std::make_shared<torch::jit::Operator>(jit_node->getOperator());
  } catch (std::exception& e) {
    std::string error_str = e.what();
    std::stringstream ss;
    ss << "Failed to create CPU Op. Details :\n" << error_str;
    LogError(
        ir_name,
        "Failed to create CPU Op. Check lazy log for details and call stack",
        ss.str());
    return nullptr;
  }
  if (op) {
    PT_LAZY_DEBUG(
        "SBS: CPU Op was found for: " + std::string(node->op().toQualString()));
  } else {
    LogError(ir_name, "CPU Op was not found (jit op is null)");
  }

  return op;
}

c10::Symbol SBSRunner::buildCPUOpSymbol(const c10::Symbol& hpu_op) {
  return hpu_op;
}

c10::Symbol SBSPermutable::buildCPUOpSymbol(const c10::Symbol& hpu_op) {
  c10::Symbol cpu_op = hpu_op;
  std::string qualstring(cpu_op.toQualString());
  std::map<std::string, std::string> to_replace_map = {
      {"aten::convolution_overrideable", "aten::convolution"},
  };
  for (auto& to_replace : to_replace_map) {
    size_t pos = qualstring.find(to_replace.first);
    if (pos != std::string::npos) {
      std::string qualstring_old = qualstring;
      qualstring.replace(pos, to_replace.first.length(), to_replace.second);
      PT_LAZY_DEBUG(
          "SBS: Replacing hpu op ", qualstring_old, " with ", qualstring);
      cpu_op = at::Symbol::fromQualString(qualstring);
    }
  }
  return cpu_op;
}

at::Tensor SBSRunner::prepareTensorToCPU(
    const at::Tensor& tensor,
    [[maybe_unused]] size_t index) {
  PT_LAZY_DEBUG("SBSRunner::", __FUNCTION__, " index=", index);
  auto hb_tensor = GetHbLazyTensor(tensor);
  // if we are moving a tensor to CPU, it means that it has valid storage
  // attached and therefore its execution status can be unconditionally marked
  // complete
  hb_tensor.getDataPtr()->execution_status = kEXECUTION_COMPLETE;

  // Saving current tensor name, to be used later
  PT_LAZY_DEBUG(
      __FUNCTION__,
      " Setting sbs tensor name: ",
      hb_tensor.CurrentIrValue().ToString());
  hb_tensor.SetSBSTensorName(hb_tensor.CurrentIrValue().ToString());

  // Special handling for view tensors - we need to sync before we copy to CPU
  auto& params_opt = hb_tensor.getDataPtr()->stride_params;
  if (params_opt.has_value()) {
    PT_LAZY_DEBUG(
        "SBSRunner::",
        __FUNCTION__,
        " calling sync for view tensor id=",
        hb_tensor.getTensorUniqueId(),
        " name: ",
        hb_tensor.CurrentIrValue().ToString(),
        " version: ",
        hb_tensor.GetSBSTensorVersion());
    std::vector<habana_lazy::HbLazyTensor> tens = {hb_tensor};
    HbLazyTensor::SyncTensorsGraph(&tens);
  }
  PT_LAZY_DEBUG(
      "SBS: Copying tensor to CPU. Name: ",
      hb_tensor.CurrentIrValue().ToString());
  auto tens_cpu = tensor.to(c10::kCPU).detach();

  return tens_cpu;
}

at::Tensor SBSPermutable::prepareTensorToCPU(
    const at::Tensor& tensor,
    size_t index) {
  PT_LAZY_DEBUG("SBSPermutable::", __FUNCTION__);
  at::Tensor out = SBSRunner::prepareTensorToCPU(tensor, index);
  if (index == m_index_to_permute) {
    PT_LAZY_DEBUG(
        "SBS: ",
        __FUNCTION__,
        " Replacing HPU tensor sizes: ",
        tensor.sizes(),
        " strides: ",
        tensor.strides(),
        " index: ",
        index);
    // taken from convolution_hpu_lazy()
    auto is_5d_layout = out.dim() == 5;

    // function call taken from habana_lazy/passes/permute_graph.cpp
    auto dims = is_5d_layout
        ? getDimsForLayout5d(
              habana::LayoutFormat::NCHW, habana::LayoutFormat::HWCK)
        : getDimsForLayout(
              habana::LayoutFormat::NCHW, habana::LayoutFormat::HWCK);
    PT_LAZY_DEBUG("SBS: ", __FUNCTION__, " Dims to permute: ", dims);

    out = out.permute(dims);
    PT_LAZY_DEBUG("SBS: ", __FUNCTION__, " New tensor sizes: ", out.sizes());
  }
  return out;
}

void SBSRunner::processOutputCPUTensor(
    habana_lazy::HbLazyTensor hl_result,
    at::Tensor& cpu_tensor,
    size_t index) {
  PT_LAZY_DEBUG("SBSRunner::", __FUNCTION__);
  PT_LAZY_DEBUG(
      "SBS: Setting CPU tensor to HPU[",
      index,
      "]. id=",
      hl_result.getTensorUniqueId(),
      " ir name=",
      hl_result.CurrentIrValue().ToString(),
      " sbs name: ",
      hl_result.FetchSBSTensorName(),
      " version: ",
      hl_result.GetSBSTensorVersion());
  hl_result.SetCPUTensorData(cpu_tensor);
  // Treating an inplace tensor as non-live until set
  // otherwise, to save graph flushes when the HPU tensor is not
  // yet available
  hl_result.SetSBSLiveTensorIndication(false);
  if (m_disabled_output_tensors.count(index)) {
    PT_LAZY_DEBUG(
        "SBS: disabling compare of tensor[",
        index,
        "], name: ",
        hl_result.CurrentIrValue().ToString(),
        " sbs name: ",
        hl_result.FetchSBSTensorName());
    hl_result.SetSBSCompareIndication(false);
  }
  hl_result.UpdateSBSTensorVersion();
  ++m_number_of_tensor_runs;
  PT_LAZY_DEBUG(
      __FUNCTION__,
      " SBS: Current number of op tensors runs: ",
      m_number_of_tensor_runs,
      " current tensor name: ",
      hl_result.CurrentIrValue().ToString(),
      " sbs name: ",
      hl_result.FetchSBSTensorName(),
      " id=",
      hl_result.getTensorUniqueId(),
      " version: ",
      hl_result.GetSBSTensorVersion(),
      " scalar type: ",
      cpu_tensor.scalar_type(),
      " dtype: ",
      cpu_tensor.dtype());
  if (hl_result.CurrentTensorData() != c10::nullopt) {
    PT_LAZY_DEBUG("SBS: HPU tensor is available, calling CompareTensors");
    std::vector<habana_lazy::HbLazyTensor> resultVec = {hl_result};
    SBSDebug::getInstance().CompareTensors(resultVec);
  }
}

bool SBSRunner::getNodeInfo(
    const at::Tensor& result,
    ir::NodePtr& node,
    std::string& ir_name) {
  auto hl_result = GetHbLazyTensor(result);
  node = hl_result.CurrentIrValue().mp_node;
  ir_name = hl_result.CurrentIrValue().ToString();
  if (!ir_name.empty()) {
    PT_LAZY_DEBUG(__FUNCTION__, " Setting sbs tensor name: ", ir_name);
    hl_result.SetSBSTensorName(ir_name);
  } else {
    // ir_name = std::string("Op Name N/A. ID ") +
    ir_name = std::string("Op Name N/A. id=") +
        std::to_string(hl_result.getTensorUniqueId());
  }

  PT_LAZY_DEBUG(
      "Tensor id=",
      hl_result.getTensorUniqueId(),
      " ir name ",
      ir_name,
      " version: ",
      hl_result.GetSBSTensorVersion());
  if (!node) {
    LogError(ir_name, "IR Node doesn't exist (getNodeInfo)");
    return false;
  }

  return true;
}

// logic taken from strided_insert_hpu_lazy()
bool SBSViews::getNodeInfo(
    const at::Tensor& result,
    [[maybe_unused]] ir::NodePtr& node,
    [[maybe_unused]] std::string& ir_name) {
  auto hl_result = GetHbLazyTensor(result);
  auto id = hl_result.getTensorUniqueId();
  PT_LAZY_DEBUG(
      __FUNCTION__,
      " result id=",
      id,
      " name: ",
      hl_result.CurrentIrValue().ToString(),
      " sbs name: ",
      hl_result.FetchSBSTensorName(),
      " version: ",
      hl_result.GetSBSTensorVersion());

  return false;
}

} // namespace habana_lazy
