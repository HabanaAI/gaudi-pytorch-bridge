/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_lazy_tensors.h"
#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_bridge/kernel/ds_graph_recompile.h"
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_cache.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ops/hpu_input.h"

#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

using ValueList = std::vector<ir::Value>;

bool HbLazyTensor::switch_dynamic_mode = false;
std::future<bool> HbLazyTensor::refinement_handle_{};

HbContextArena* HbContextArena::Get() {
  static HbContextArena* arena = new HbContextArena();
  return arena;
};

void HbContextArena::RegisterTensor(std::shared_ptr<Data> data) {
  std::lock_guard<std::recursive_mutex> lock(m_mtx);
  HbContext* devctx = GetHbContext(data->device);
  devctx->tensors_data.emplace(data->unique_id, data);
  // Register to execution context as well, we can merge these two contexts
  // later
  auto device_id = data->device.index();
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_id);
  context->RegisterTensor(data);
}

void HbContextArena::UnregisterTensor(Data* data) {
  std::lock_guard<std::recursive_mutex> lock(m_mtx);
  HbContext* devctx = GetHbContext(data->device);
  devctx->tensors_data.erase(data->unique_id);
  // UnRegister from execution context as well, we can merge these two contexts
  // later
  auto device_id = data->device.index();
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_id);

  // clear the entry in view tables
  auto it = context->orig_tensor_map.find(data->unique_id);
  if (it != context->orig_tensor_map.end()) {
    context->orig_tensor_map.erase(it);
  }
  auto view_it = context->view_table.find(data->unique_id);
  if (view_it != context->view_table.end()) {
    context->view_table.erase(view_it);
  }
  context->UnregisterTensor(data);
}

std::vector<HbLazyTensor> HbContextArena::GetLiveTensors(
    const c10::Device* device) {
  PT_LAZY_TRACE;
  std::vector<HbLazyTensor> tensors;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto fn = [&](HbContext* devctx) {
    for (auto& uid_wptr : devctx->tensors_data) {
      std::shared_ptr<Data> data = uid_wptr.second.lock();
      if (data != nullptr) {
        auto hl_t = HbLazyTensor(std::move(data));
        auto id = hl_t.getTensorUniqueId();
        // exclude the views
        auto ir_value = hl_t.CurrentIrValue();
        if ((ir_value && ir_value.mp_node->is_input() == false) &&
            (context->view_table.find(id) != context->view_table.end() ||
             (context->orig_tensor_map.find(id) !=
              context->orig_tensor_map.end()))) {
          // book keep view tensors to clear the ir nodes after mark step
          context->hb_tensors_out_view.emplace_back(hl_t);
        } else {
          // TODO: SW-69618 JIT optimization passes are failing for
          // habanaOptimizerLambPhase1 and habanaOptimizerLambPhase2 because we
          // dont support tensorlist in lowering that matches kernel schema.
          // Adding unpack will return TensorList, which is not supported as
          // graph output.
          if ((ir_value &&
               (std::string(ir_value.mp_node->op().toQualString())
                    .find("hpu::habanaOptimizerLambPhase") !=
                std::string::npos))) {
            exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();
          }
          tensors.emplace_back(hl_t);
        }
      } // if (data != nullptr)
    } // for (auto& uid_wptr : devctx->tensors_data)
  };
  ForAllHbContexts(fn, device);
  return tensors;
}

void HbContextArena::MarkStep(const c10::Device& device) {
  PT_LAZY_TRACE;
  HbContext* devctx = GetHbContext(device);
  devctx->seed_ir_value = ir::Value();
}

std::vector<HbContext*> HbContextArena::GetAllHbContexts() {
  std::vector<HbContext*> all_device_contexts;
  all_device_contexts.reserve(mp_device_contexts.size());
  for (auto& device_contexts : mp_device_contexts) {
    all_device_contexts.push_back(device_contexts.second);
  }
  return all_device_contexts;
}

void HbContextArena::ForAllHbContexts(
    const std::function<void(HbContext*)>& fn,
    const c10::Device* device) {
  if (device == nullptr) {
    for (auto devctx : GetAllHbContexts()) {
      fn(devctx);
    }
  } else {
    fn(GetHbContext(*device));
  }
}

HbContext* HbContextArena::GetHbContext(const c10::Device& device) {
  auto it = mp_device_contexts.find(device);
  if (it == mp_device_contexts.end()) {
    it = mp_device_contexts.emplace(device, new HbContext()).first;
  }
  return it->second;
}

Data::~Data() {
  auto context = HbContextArena::Get();
  context->UnregisterTensor(this);
  data_ptr = nullptr;
}

HbLazyTensor::HbLazyTensor(const at::Tensor& tensor, const c10::Device& device)
    : mp_data(std::make_shared<Data>(tensor, device)) {}
HbLazyTensor::HbLazyTensor(const c10::Device& device)
    : mp_data(std::make_shared<Data>(device)) {}

HbLazyTensor::HbLazyTensor(
    ir::Value ir_value,
    const at::Device& device,
    c10::optional<at::ScalarType> logical_element_type)
    : mp_data(std::make_shared<Data>(
          std::move(ir_value),
          device,
          logical_element_type)) {
  // TODO : TryLimitGraphSize();
}

HbLazyTensor::HbLazyTensor(std::shared_ptr<Data> data)
    : mp_data(std::move(data)) {}

HbLazyTensor HbLazyTensor::Create(
    const at::Tensor& tensor,
    const c10::Device& device) {
  HbLazyTensor habana_tensor(tensor, device);
  HbContextArena::Get()->RegisterTensor(habana_tensor.data_ptr());
  return habana_tensor;
}

void HbLazyTensor::setTensorSize(std::vector<int64_t> sizes) {
  data()->sizes = sizes;
}
HbLazyTensor HbLazyTensor::Create(
    ir::Value ir_value,
    const at::Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  HbLazyTensor hb_tensor(std::move(ir_value), device, logical_element_type);
  HbContextArena::Get()->RegisterTensor(hb_tensor.data_ptr());
  return hb_tensor;
}
at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor HbLazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    // TODO:: Will need to check if we need to activate this path
    // We arent allocation any new memory to tensors which isnt coming via At
    // calls
    // so this case shouldnt arise
    return tensor;
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = c10::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void HbLazyTensor::AssignIrValue(ir::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
}

ir::Value& HbLazyTensor::CurrentIrValue() const {
  return data()->ir_value;
}

void* HbLazyTensor::CurrentHabanaData() const {
  return data()->data_ptr;
}

ir::Value HbLazyTensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  void* device_data = CurrentHabanaData();
  if (device_data != nullptr) {
    // In case of tensor node, we do not clear the device data when we set the
    // IR node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single Habana parameter op). So call
    // which wants the device data will still find it, w/out having to fetch it
    // via a computation on device
    AssignIrValue(CreateTensorNode());
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (tensor_data)
    AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  else {
    at::Tensor tensor_dummy;
    AssignIrValue(GetIrValueForTensor(tensor_dummy, GetDevice()));
  }

  return data()->ir_value;
}

void HbLazyTensor::MarkStep(const c10::Device& device) {
  HbContextArena::Get()->MarkStep(device);
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      device.index());
  context->MarkAllTensorsExecuted();
}

bool HbLazyTensor::isStorageAttached() {
  if (data()->tensor_data) {
    if (data()->tensor_data.value().unsafeGetTensorImpl())
      return true;
    else
      return false;
  } else {
    return false;
  }
}
void HbLazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

void HbLazyTensor::SetCPUTensorData(at::Tensor cpu_tensor_data) {
  PT_BRIDGE_DEBUG(
      "Type is ", c10::DeviceTypeName(cpu_tensor_data.device().type()));
  HABANA_ASSERT(cpu_tensor_data.device().type() == c10::DeviceType::CPU);
  data()->cpu_tensor_data = std::move(cpu_tensor_data);
}

void HbLazyTensor::SetSBSLiveTensorIndication() {
  data()->sbs_live_tensor = true;
}

bool HbLazyTensor::GetSBSLiveTensorIndication() const {
  return data()->sbs_live_tensor;
}

const c10::optional<at::Tensor>& HbLazyTensor::GetCPUTensorData() const {
  const auto& tens = data()->cpu_tensor_data;
  if (tens != c10::nullopt) {
    bool isCPU = tens.value().device().type() == c10::DeviceType::CPU;
    HABANA_ASSERT(isCPU);
  }
  return tens;
}

c10::TensorImpl* HbLazyTensor::getAttachedTensorImpl() const {
  if (data()->tensor_data) {
    return (data()->tensor_data.value().unsafeGetTensorImpl());
  } else {
    return nullptr;
  }
}
c10::optional<at::Tensor> HbLazyTensor::CurrentTensorData() const {
  auto device_id = GetDevice().index();
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_id);
  if (context != nullptr) {
    auto status = context->getTensorExecutionStatus(data()->unique_id);
    if (status == kEXECUTION_COMPLETE || status == kINPUT) {
      return data()->tensor_data;
    } else {
      return c10::nullopt;
    }
  }
  return c10::nullopt;
}

const c10::Device& HbLazyTensor::GetDevice() const {
  return data()->device;
}

const std::vector<int64_t>& HbLazyTensor::GetSizes() const {
  return data()->sizes;
}

void HbLazyTensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void HbLazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  AssignIrValue(ir::Value());
  setPtrDataIrToData();
}

Data* HbLazyTensor::data() const {
  return mp_data.get();
}

at::ScalarType HbLazyTensor::dtype() const {
  if (data()->logical_element_type) {
    return *data()->logical_element_type;
  } else {
    return c10::ScalarType::Float;
  }
}

c10::optional<at::ScalarType> HbLazyTensor::dtype_optional() const {
  return data()->logical_element_type;
}

ir::Value HbLazyTensor::CreateTensorNode() const {
  setTensorAsInputNode(*this);
  return CurrentIrValue();
}

void HbLazyTensor::setPtrDataIrToData() {
  if (mp_data.get())
    mp_data->ir_value.m_data_ptr = mp_data;
}

ir::Value HbLazyTensor::createIrValueFromData() {
  ir::Value v{data_ptr()};
  return v;
}

ir::Value HbLazyTensor::GetIrValueForTensor(
    const at::Tensor& tensor,
    const c10::Device& device) const {
  static_cast<void>(device);
  static_cast<void>(tensor);
  return CreateTensorNode();
}

void HbLazyTensor::ClearAndAssignNewIrValue() {
  // Reset the ir_value with the following content -
  // - The m_data_ptr should continue to point to the
  //   same lazy tensor data_ptr()
  // - New hpu::input Tensor node within the ir_value as
  //   the output tensors are obtained after computing the
  //   graph associated with it and can be used as an input
  //   tensor to further ops using this tensor.

  ir::Value val = createIrValueFromData();
  if (GET_ENV_FLAG_NEW(PT_HPU_AVOID_RE_EXECUTE_GRAPHS)) {
    ir::Value& currentIrVal = CurrentIrValue();
    // Check if any other node uses this node, if used, then replace its irval
    // with the new one.
    if (currentIrVal.mp_node) {
      auto node = currentIrVal.mp_node.get();
      auto& uses = node->GetUses();
      if (uses.size()) {
        // Set the value ptr as input node, this will make sure the mp_node in
        // value is proper.
        ir::NodePtr inp_node = std::make_shared<ir::Input>(*this);
        val.SetNode(inp_node, GetDevice(), GetSizes(), dtype_optional());
        auto tensor = AtenFromHbLazyTensor(
            *this, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
        for (ir::Use use : uses) {
          if (use.mp_node) {
            use.mp_node->ReplaceInput(val, use.m_operand_index, tensor);
          }
        }
      }
    }
  }
  // The version of lazy tensors is maintained per graph execution
  // reset the counter for use in next graph
  resetVersionCounter();
  AssignIrValue(val);
}

HbLazyTensor HbLazyTensor::CreateHbLazyTensor(
    c10::IntArrayRef size,
    at::Scalar fill_value,
    const at::Device& device,
    at::ScalarType scalar_type) {
  PT_LAZY_TRACE;
  static_cast<void>(fill_value);
  ir::Value val;
  // Creating a dummy IR::Value right now
  // After Vaibhav's update, we should plug in utility to create IR
  // from metadata(commented line)
  HbLazyTensor hb_tensor = Create(
      // GetIrValueForScalar(fill_value, shape, device)
      val,
      device,
      scalar_type);

  // We keep a weak pointer in our IR back to data pointer of lazy tensor
  // This needs to be updated here or we can push it in the constructor
  // Keeping it here for now so that its not implicitly set
  hb_tensor.setPtrDataIrToData();
  // Setup the size information in the data of Lazy tensor
  hb_tensor.setTensorSize(size.vec());
  return hb_tensor;
}

/************************************************************************
 * @brief Returns indices of tensors corresponding to tensors with valid IR
 * values which feeds in to RunPostOrder
 ************************************************************************/
std::vector<int> HbLazyTensor::CollectSyncTensors(
    const std::vector<HbLazyTensor>& tensors) {
  PT_LAZY_TRACE;
  std::vector<int> indices = {};
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto ir_value = tensors[i].CurrentIrValue();
    // Skip the tensors which don't have any node to evaluate and points
    // to hpu::input node.
    if (ir_value && ir_value.mp_node->is_input() == false) {
      indices.push_back(i);
    }
  }
  return indices;
}

/************************************************************************
 * @brief Computes post order list of NodePtrs. This function should be
 *executed at the trigger points. The output would be consumed during the
 *conversion of lazy IR to JIT IR.
 * @param[in] tensors - vector Hb Lazy Tensors
 * @param[in] indices - vector of indices corresponding to input tensors with
 *valid IR values
 * @param[out] po_data - Post Ordered vector of tensors
 ************************************************************************/
habana_lazy::ir::PostOrderData HbLazyTensor::RunPostOrder(
    const std::vector<HbLazyTensor>& tensors,
    std::vector<int> indices) {
  PT_LAZY_TRACE;
  habana_lazy::ir::PostOrderData po_data;
  std::vector<ir::NodePtr> p_roots;
  p_roots.reserve(indices.size());
  for (auto index : indices) {
    auto ir_value = tensors.at(index).CurrentIrValue();
    if (ir_value) {
      p_roots.push_back(ir_value.mp_node);
      // update output list
      po_data.outputs.push_back(ir_value);
    }
  }

  ir::Utils::ComputePostOrder(p_roots, po_data);
  if (!GET_ENV_FLAG_NEW(PT_HPU_DUMP_IR_DOT_GRAPH)) {
    PT_LAZY_DEBUG(
        "Lazy_IR_Graph_BEGIN\n",
        IrGraphDumpUtil::PostOrderToText(po_data.post_order, p_roots),
        "Lazy_IR_Graph_END");
    PT_IRGRAPH_DEBUG(IrGraphDumpUtil::PostOrderToText(
        po_data.post_order, p_roots, true, true));
  } else {
    PT_LAZY_DEBUG(IrGraphDumpUtil::PostOrderToDot(po_data.post_order, p_roots));
  }
  return po_data;
}

c10::optional<at::Tensor> HbLazyTensor::GetHbLazyTensorData() {
  // Generate the tensor data if its not been generated yet
  if (CurrentIrValue() && !CurrentTensorData()) {
    if (GET_ENV_FLAG_NEW(PT_USE_MARKSTEP)) {
      HbLazyTensor::StepMarker({});
    } else {
      std::lock_guard<std::recursive_mutex> lock(
          HbContextArena::Get()->GetMutex());
      applyPendingGraph();
    }
  }
  return data()->tensor_data;
}

void HbLazyTensor::applyPendingGraph() {
  PT_LAZY_TRACE;
  // Ensure that the graph execution has taken place so taht the tensors
  // requested have the data required updated in them. This is usually done
  // before sync points in execution
  if (!CurrentTensorData()) {
    std::vector<HbLazyTensor> tensors;
    auto node = data()->ir_value.mp_node.get();
    auto live_tensors = GetLiveTensors(&GetDevice());
    for (auto& tensor : live_tensors) {
      if (tensor.data()->ir_value.mp_node.get() == node) {
        tensors.emplace_back(tensor);
      }
    }
    SyncTensorsGraph(&tensors);
  }
}

std::vector<HbLazyTensor> HbLazyTensor::GetLiveTensors(
    const c10::Device* device) {
  return HbContextArena::Get()->GetLiveTensors(device);
}

void HbLazyTensor::SyncTensorsGraph(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  SyncTensorsGraphInternal(tensors, lazyFrontEndInfo);
}

void HbLazyTensor::SyncTensorsGraphFast(
    std::vector<HbLazyTensor>* tensors,
    std::vector<ir::Value>& input_values,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  SyncTensorsGraphInternalFast(tensors, input_values, lazyFrontEndInfo);
}

void HbLazyTensor::SyncLiveTensorsGraph(
    const c10::Device* device,
    bool use_cached_graph = false,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info =
        nullptr) {
  PT_LAZY_TRACE;
  DebugHelper::getInstance().resetCurrentAccumulatedOps();
  if (use_cached_graph) {
    ExecuteCachedGraph();
  } else {
    auto tensors = GetLiveTensors(device);
    SyncTensorsGraph(&tensors, lazy_front_end_info);
  }
}

at::Tensor HbLazyTensor::Process0DTensor(std::shared_ptr<Data>& d) {
  TORCH_CHECK(d->tensor_data.has_value(), "Empty tensor optional");
  at::Tensor pt_tensor = d->tensor_data.value();

  // Make regular 0D tensors 1D
  auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
  bool is_shape_tensor = impl && impl->isShapeTensor();
  if (pt_tensor.dim() == 0) {
    TORCH_CHECK(is_shape_tensor == false, "0D shape tensor encountered");
    pt_tensor.unsafeGetTensorImpl()->set_sizes_contiguous({1});
  }

  return pt_tensor;
}

void HbLazyTensor::SyncTensorsGraphInternal(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo) {
  PT_LAZY_TRACE;
  std::vector<int> indices = CollectSyncTensors(*tensors);
  if (indices.empty()) {
    // Nothing to do, return without trying to execute an empty graph
    return;
  }

  auto context = habana_lazy_executor.getDeviceExecutionContext(
      (*tensors)[0].GetDevice().index());

  auto po_data = HbLazyTensor::RunPostOrder(*tensors, indices);

  exec::HlExec hlexec{};

  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  stack.reserve(std::max(po_data.inputs.size(), po_data.outputs.size()));
  std::vector<uint64_t> executing_indices;
  for (const auto& in : po_data.inputs) {
    // PT_LAZY_DEBUG(std::string("Lowering - ") + in.ToString());
    if (!in.DataPtrValidAndNotExpired()) {
      std::vector<ir::NodePtr> p_roots;
      p_roots.reserve(indices.size());
      for (auto index : indices) {
        auto ir_value = tensors->at(index).CurrentIrValue();
        if (ir_value) {
          p_roots.push_back(ir_value.mp_node);
        }
      }
      PT_LAZY_DEBUG(
          " Node = ",
          in.ToString(),
          "\n Failing IR graph = ",
          IrGraphDumpUtil::PostOrderToText(po_data.post_order, p_roots));
      HABANA_ASSERT(in.DataPtrValidAndNotExpired());
    }
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    auto pt_tensor = Process0DTensor(d);
    stack.emplace_back(pt_tensor);
    // We dont get the correct lazy tensor back from internal tensor
    // So marking for execution here
    context->MarkTensorExecuting(d->unique_id);
    executing_indices.push_back(d->unique_id);
  }

  hlexec.set_lazy_front_end_info(lazyFrontEndInfo);

  hlexec.GetOrCreate(po_data, stack);
  // This is the logic to remove outputs of control edges that are dangling from
  // the outputs of JIT graph We dont want to alter graph execution, so removing
  // after graph is already prepared. Also we DO want that the tensors of this
  // node are marked processed, as they would have through output stack So we do
  // all the markings before entering execution
  bool remove_control_edge_outputs = true;
  if (remove_control_edge_outputs) {
    int vec_index = 0;
    int num_outputs = po_data.outputs.size();
    for (int i = 0; i < num_outputs; i++) {
      auto ir_value = po_data.outputs[vec_index];
      auto data = ir_value.m_data_ptr.lock();
      if (ir_value.mp_node->is_control_edge() && data->version == 0) {
        auto& tensor = (*tensors)[i];

        ir::Value val = tensor.createIrValueFromData();
        tensor.AssignIrValue(val);
        context->MarkTensorExecuted(data->unique_id);
        executing_indices.erase(
            std::remove(
                executing_indices.begin(),
                executing_indices.end(),
                data->unique_id),
            executing_indices.end());

        po_data.outputs.erase(po_data.outputs.begin() + vec_index);
        indices.erase(indices.begin() + vec_index);
        hlexec.get_graph()->eraseOutput(vec_index);
      } else {
        vec_index++;
      }
    }
  }

  // Dump the JIT graph with PT_IRGRAPH_DEBUG
  PT_IRGRAPH_DEBUG(hlexec.DumpGraph());

  // Remove any tensor_data held at output, this will reduce the memory
  // pressure
  for (size_t idx = 0; idx < indices.size();) {
    auto out_tensor = (*tensors)[indices[idx++]];
    out_tensor.SetTensorData(at::Tensor());
  }

  // Launch the execution
  hlexec.Launch(stack);
  HABANA_ASSERT(stack.size() == indices.size());

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    auto st = v.toTensor();
    auto out_tensor = (*tensors)[indices[i++]];
    executing_indices.push_back(out_tensor.getTensorUniqueId());
    out_tensor.SetTensorData(st);
  }
  context->MarkTensorsExecuted(executing_indices);

  // Compare Tensors
  if (GET_ENV_FLAG_NEW(PT_SBS) != SBSModes::SBS_MODE_DISABLED) {
    SBSDebug::getInstance().CompareTensors(*tensors);
  }

  // Graph executed, clear IR values corresponding to sync tensors
  for (auto idx : indices) {
    auto& i = (*tensors)[idx];
    i.ClearAndAssignNewIrValue();
  }

  // clear IR values corresponding to unexecuted view outputs
  for (auto& t : context->hb_tensors_out_view) {
    ir::Value val = t.createIrValueFromData();
    t.resetVersionCounter();
    t.AssignIrValue(val);
  }

  // Save po_data input and output to context for perf mode
  if (context->m_is_cached == false) {
    context->saveInputsAndOutputs(
        po_data.inputs, po_data.outputs, *tensors, indices);
    context->m_is_cached = true;
  }

  // clear the context
  context->clear();
  // Restore the optimizations which are cleared forcefully in getlivetensors
  exec::OptPassCfg::GetInstance()->RestoreOptPass();
}

void HbLazyTensor::SyncTensorsGraphInternalFast(
    std::vector<HbLazyTensor>* tensors,
    std::vector<ir::Value>& input_values,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo) {
  PT_LAZY_TRACE;
  std::vector<int> indices;
  for (size_t i = 0; i < tensors->size(); ++i) {
    indices.push_back(i);
  }

  auto context = habana_lazy_executor.getDeviceExecutionContext(
      (*tensors)[0].GetDevice().index());

  torch::jit::Stack stack;
  stack.reserve(std::max(input_values.size(), indices.size()));
  std::vector<uint64_t> executing_indices;
  for (const auto& in : input_values) {
    HABANA_ASSERT(in.DataPtrValidAndNotExpired());
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    auto pt_tensor = Process0DTensor(d);
    stack.emplace_back(pt_tensor);
    // We dont get the correct lazy tensor back from internal tensor
    // So marking for execution here
    context->MarkTensorExecuting(d->unique_id);
    executing_indices.push_back(d->unique_id);
  }

  HABANA_ASSERT(lazyFrontEndInfo != nullptr);

  size_t optimized_lazy_eager_key =
      lazyFrontEndInfo->get_optimized_lazy_eager_key();
  std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
      fast_path_jit_ir_and_mdata =
          habana_lazy::FastLazyGraphCache::GetFastLazyCache()
              .GetOptimizedJITGraphAndMetaData(optimized_lazy_eager_key);
  PT_LAZY_DEBUG("Fast Path JIT Cache hit :: key ", optimized_lazy_eager_key);

  // Dump the JIT graph with PT_LAZY_DEBUG
  // PT_LAZY_DEBUG(hlexec.DumpGraph());

  // Remove any tensor_data held at output, this will reduce the memory
  // pressure
  for (size_t idx = 0; idx < indices.size();) {
    auto out_tensor = (*tensors)[indices[idx++]];
    out_tensor.SetTensorData(at::Tensor());
  }

  // TODO : remove this env variable use
  // This is temporarily done to deactivate code in synapse helpers for lazy
  // mode kernel registration We will move to using shape utilities instead and
  // not do env variable based check anymore
  // We have short-circuited certain utilities in synapse helpers, we need to
  // remove that code
  // TODO : Not seting this would cause the synapse graph creation set to
  // dry run. Hence not setting this would cause synpase graph to be not be
  // created. This needs to be optimized.
  SET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING, 1, 1);
  context->setExecutionMode(kLOWERING);

  habana::HabanaLaunchOpPT launch{
      fast_path_jit_ir_and_mdata->get_cached_graph(),
      std::make_shared<habana::HabanaMetaDataToLowering>(
          false,
          0,
          lazyFrontEndInfo->get_lazy_op_name(),
          fast_path_jit_ir_and_mdata->get_cached_opstrs(),
          fast_path_jit_ir_and_mdata->get_cached_graph_key(),
          true)};
  try {
    launch.run(stack);
  } catch (std::exception& e) {
    PT_BRIDGE_DEBUG("HabanaLaunchOpPT Run returned exception ", e.what());
    context->setExecutionMode(kLAZY);
    UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING);
    throw;
  }

  context->setExecutionMode(kLAZY);
  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING);
  HABANA_ASSERT(stack.size() == indices.size());

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    auto st = v.toTensor();
    auto out_tensor = (*tensors)[indices[i++]];
    executing_indices.push_back(out_tensor.getTensorUniqueId());
    out_tensor.SetTensorData(st);
  }
  context->MarkTensorsExecuted(executing_indices);

  // Graph executed, clear IR values corresponding to sync tensors
  for (auto idx : indices) {
    auto& i = (*tensors)[idx];
    // Reset the ir_value with the following content -
    // - The m_data_ptr should continue to point to the
    //   same lazy tensor data_ptr()
    // - New hpu::input Tensor node within the ir_value as
    //   the output tensors are obtained after computing the
    //   graph associated with it and can be used as an input
    //   tensor to further ops using this tensor.
    ir::Value val = i.createIrValueFromData();
    // The version of lazy tensors is maintained per graph execution
    // reset the counter for use in next graph
    i.resetVersionCounter();
    i.AssignIrValue(val);
  }

  // clear the scalar to tensor cache
  context->scalar_to_tensor_map.clear();

  // clear retained tensor list
  context->m_retained_tensor_list.clear();
}

void HbLazyTensor::ExecuteCachedGraph() {
  PT_LAZY_TRACE;
  exec::HlExec hlexec{};

  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  HbExecutionContext* context =
      habana_lazy_executor.getDeviceExecutionContext(0);

  HABANA_ASSERT(context->m_is_cached == true);

  auto& input_vals = context->getInputs();
  auto& output_vals = context->getOutputs();
  auto hb_lazy_tensors = context->getHbLazyTensors();

  stack.reserve(std::max(input_vals.size(), output_vals.size()));

  for (const auto& in : input_vals) {
    PT_LAZY_DEBUG(std::string("Lowering - ") + in.ToString());
    HABANA_ASSERT(in.DataPtrValidAndNotExpired());
    if (in.mp_node) {
      PT_LAZY_DEBUG(std::string("    Node ") + in.mp_node->ToString());
    }
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    stack.emplace_back(d->tensor_data);
  }

  // Fetch graph from device context
  hlexec.set_graph(context->getGraph());

  // Launch the execution
  hlexec.Launch(stack);

  HABANA_ASSERT(stack.size() == hb_lazy_tensors.size());

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    auto st = v.toTensor();
    HbLazyTensor out_tensor = hb_lazy_tensors[i++];
    out_tensor.SetTensorData(st);
  }
}

void HbLazyTensor::setTensorOriginalType(c10::ScalarType type) {
  data_ptr()->original_element_type = type;
}

c10::ScalarType HbLazyTensor::getTensorOriginalType() const {
  return data_ptr()->original_element_type;
}

void HbLazyTensor::ShallowCopyTo(HbLazyTensor* dest) const {
  // We can add stuff related to view tensors later
  // SW-43241: The shallow copy copies the ir_value etc from one tensor
  // to another. If the same ir_value is used in both the tensors, then
  // they will have a weak pointer to the same Data pointer from the
  // first lazy tensor.
  // If the first lazy tensor is destroyed, the associated Data will
  // also get removed, making the weak pointer to the Data in the
  // second lazy tensor ir_value to be expired.
  // To avoid this, create a new ir_value with data pointer from the dest
  // tensor. This ensures that the ir_value within each tensor points to
  // its own Data pointer. Additionally, copy the ir node from the source
  // ir_value so that the dest ir_value also has the same ir node parent.
  // However, prevent adding this ir_value as another output to the ir node.
  // Since we do the post order traversal from output values backward to its
  // ir nodes, both the ir_value will reach the same ir node.
  habana_lazy::ir::Value val{dest->GetIrValue().m_data_ptr.lock()};
  val.SetNodeForShallowCopy(GetIrValue().mp_node);
  dest->AssignIrValue(val);
  // If the src tensor has an evaluated tensor internally on the device, then
  // the lazy tensor shallow copy needs to ensure the desc lazy tensor also
  // points to the same internal device tensor.
  auto data_tensor = CurrentTensorData();
  if (data_tensor.has_value()) {
    dest->SetTensorData(*data_tensor);
  }
}

void HbLazyTensor::StepMarkerBind(const std::string& device_str) {
  PT_LAZY_TRACE;
  PT_IRGRAPH_DEBUG("step marker due to host step marker");
  PT_LAZY_DEBUG("step marker due to host step marker");
  StepMarker(device_str);
}

void HbLazyTensor::StepMarker(
    const std::string& device_str,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info) {
  PT_LAZY_TRACE;

  // Entry point of bucket refinement thread
  InitiateBucketRefinement();

  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  c10::Device device = GetDeviceOrCurrent(device_str);
  HbLazyTensor::SyncLiveTensorsGraph(
      &device, /* is_cached*/ false, lazy_front_end_info);
  HbLazyTensor::MarkStep(device);
  if (switch_dynamic_mode) {
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
    switch_dynamic_mode = false;
  }
}

void HbLazyTensor::InitiateBucketRefinement() {
  PT_LAZY_TRACE;
  if (GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD)) {
    // Start the separate compile thread
    if (!HbLazyTensor::refinement_handle_.valid()) {
      PT_TEST_DEBUG_TH(
          "Bucket refine thread is not started. Starting a new thread ...");
      HbLazyTensor::refinement_handle_ =
          std::async(habana::RefineBucketDS, 0.9);
    } else {
      std::chrono::milliseconds span(0);
      auto compile_status = HbLazyTensor::refinement_handle_.wait_for(span);
      if (std::future_status::ready != compile_status) {
        PT_TEST_DEBUG_TH("Bucket refine thread is running ...");
      } else {
        PT_TEST_DEBUG(
            "Bucket refine thread is completed. Starting a new thread ...");
        HbLazyTensor::refinement_handle_ =
            std::async(habana::RefineBucketDS, 0.9);
      }
    }
  }
}

void HbLazyTensor::SetDynamicMode() {
  bool dynamic_env = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  switch_dynamic_mode = dynamic_env ? false : true;
  if (switch_dynamic_mode) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
  }
}

void HbLazyTensor::RunSavedGraph(const std::string& device_str) {
  c10::Device device = GetDeviceOrCurrent(device_str);
  HbLazyTensor::SyncLiveTensorsGraph(&device, true);
  HbLazyTensor::MarkStep(device);
}

void* HbLazyTensor::lazyTensorDataPtr(const at::Tensor& t) {
  return GetLazyTensorDataPtr(t);
}
