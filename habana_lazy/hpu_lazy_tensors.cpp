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

#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_cache.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ops/hpu_input.h"
#include "habana_lazy/sbs_debug.h"
#include "habana_lazy/view_utils.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

bool HbLazyTensor::switch_dynamic_mode = false;

HbContextArena* HbContextArena::Get() {
  static HbContextArena* arena = new HbContextArena();
  return arena;
};

void HbContextArena::RegisterTensor(std::shared_ptr<Data> data) {
  std::lock_guard<std::recursive_mutex> lock(GetMutex());
  HbContext* devctx = GetHbContext(data->device);
  devctx->tensors_data.emplace(data->unique_id, data);
  // Register to execution context as well, we can merge these two contexts
  // later
  auto device_id = data->device.index();
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_id);
  context->RegisterTensor(data);
}

std::weak_ptr<Data>& HbContextArena::GetTensorDataPtrFromHbContext(Data* data) {
  HbContext* devctx = GetHbContext(data->device);
  std::lock_guard<std::recursive_mutex> lock(GetMutex());
  return devctx->tensors_data[data->unique_id];
}

void HbContextArena::UnregisterTensor(Data* data) {
  HbContext* devctx = GetHbContext(data->device);
  // UnRegister from execution context as well, we can merge these two contexts
  // later
  auto device_id = data->device.index();
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_id);

  context->UnregisterTensor(data);
  // The weak ptr in tensors_data is reset before acquiring the m_mtx,
  // release_resources will acquire GIL and it may conflict with m_mtx. So first
  // free the resources and then acquire m_mtx and then free erase from
  // tensors_data. tensors holded in viewEntryTensor/strideParams will be erased
  // once lock scope is over.
  auto tData = GetTensorDataPtrFromHbContext(data);
  auto unique_id = data->unique_id;
  tData.reset();
  {
    std::lock_guard<std::recursive_mutex> lock(GetMutex());
    devctx->tensors_data.erase(unique_id);
  }

  c10::optional<HbLazyTensor> shallowCopyHbTensor;
  c10::optional<at::Tensor> viewEntryTensor;
  StrideParams strideParams;
  {
    // clear the entry in view tables
    std::lock_guard<std::recursive_mutex> view_table_lock(
        context->viewContext.GetViewTableMutex());
    shallowCopyHbTensor =
        context->viewContext.GetShallowCopyMapEntry(unique_id);
    context->viewContext.DelShallowCopyMapEntry(unique_id);
    viewEntryTensor = context->viewContext.GetOrigTensorMapEntry(unique_id);
    context->viewContext.DelOrigTensorMapEntry(unique_id);
    auto* params_ptr = context->viewContext.GetViewTableEntry(unique_id);
    if (params_ptr != nullptr) {
      strideParams = *params_ptr;
      context->viewContext.DelViewTableEntry(unique_id);
    }
  }
}

std::vector<HbLazyTensor> HbContextArena::GetLiveTensors(
    const c10::Device* device,
    bool is_allreduce,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;
  std::vector<HbLazyTensor> tensors;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  // Live tensor collection is not allowed if the launch thread execution is  in
  // progeress.
  if (!(GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
        (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2))) {
    HABANA_ASSERT(context->m_launch_thread_handle.valid() == false);
  }

  std::lock_guard<std::recursive_mutex> view_table_lock(
      context->viewContext.GetViewTableMutex());
  HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(*device);

  HbLazyTensorViews::HandleViewsLiveTensors(
      devctx, is_allreduce, bucket_recent_id);

  for (auto& uid_wptr : devctx->tensors_data) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();
    if (data != nullptr) {
      auto id = data->unique_id;
      if (data->ir_value && data->ir_value.mp_node->is_input() == false) {
        auto hl_t = HbLazyTensor(std::move(data));

        auto params_ptr = context->viewContext.GetViewTableEntry(id);
        auto is_view = (params_ptr != nullptr);

        if (bucket_recent_id.count(id)) {
          context->viewContext.updated_bucket_list.emplace_back(hl_t);
        }
        auto is_view_out = context->viewContext.view_outputs.count(id);

        if (is_view_out ||
            ((bucket_recent_id.count(id) == 0) && (!is_view) &&
             (context->viewContext.GetOrigTensorMapEntry(id) ==
              c10::nullopt))) {
          tensors.emplace_back(hl_t);
        }
      } else { // if (data != nullptr)
        data->execution_status = kEXECUTION_COMPLETE;
      }
    } // for (auto& uid_wptr : devctx->tensors_data)
  }
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

HbContext* HbContextArena::GetHbContext() {
  HABANA_ASSERT(mp_device_contexts.size() == 1);
  return mp_device_contexts.begin()->second;
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
  HbContextArena::Get()->RegisterTensor(habana_tensor.getDataPtr());
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
  HbContextArena::Get()->RegisterTensor(hb_tensor.getDataPtr());
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

bool HbLazyTensor::IsExecutionInProgress() const {
  return data()->is_executing;
}

void HbLazyTensor::SetExecutionInProgress() const {
  data()->is_executing = true;
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

c10::optional<at::Tensor>& HbLazyTensor::GetTensorData() {
  auto& tens = data()->tensor_data;
  if (tens != c10::nullopt) {
    bool isHPU = tens.value().device().type() == c10::DeviceType::HPU;
    HABANA_ASSERT(isHPU);
  }
  return tens;
}

void HbLazyTensor::SetCPUTensorData(at::Tensor cpu_tensor_data) {
  PT_BRIDGE_DEBUG(
      "Type is ", c10::DeviceTypeName(cpu_tensor_data.device().type()));
  HABANA_ASSERT(cpu_tensor_data.device().type() == c10::DeviceType::CPU);
  data()->cpu_tensor_data = std::move(cpu_tensor_data);
}

void HbLazyTensor::SetSBSLiveTensorIndication(bool live) {
  data()->sbs_live_tensor = live;
}

bool HbLazyTensor::GetSBSLiveTensorIndication() const {
  return data()->sbs_live_tensor;
}

void HbLazyTensor::SetSBSCompareIndication(bool compare) {
  data()->sbs_compare_tensor = compare;
}

bool HbLazyTensor::GetSBSCompareIndication() const {
  return data()->sbs_compare_tensor;
}

void HbLazyTensor::UpdateSBSTensorVersion() {
  data()->sbs_tensor_version++;
  PT_LAZY_DEBUG(
      "SBS: Updated tensor version to ",
      data()->sbs_tensor_version,
      " name ",
      CurrentIrValue().ToString(),
      " id=",
      getTensorUniqueId());
}
int HbLazyTensor::GetSBSTensorVersion() const {
  return data()->sbs_tensor_version;
}

void HbLazyTensor::SetSBSTensorName(const std::string& name) {
  data()->sbs_tensor_name = name;
}
std::string HbLazyTensor::FetchSBSTensorName() const {
  auto name = data()->sbs_tensor_name;
  if (name.empty() && CurrentIrValue()) {
    name = CurrentIrValue().ToString();
  }
  return name;
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
    auto status = context->getTensorExecutionStatus(getDataPtr());
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
      // Set the value ptr as input node, this will make sure the mp_node in
      // value is proper.
      ir::NodePtr inp_node = std::make_shared<ir::Input>(*this);
      val.SetNode(inp_node, GetDevice(), GetSizes(), dtype_optional());
      if (uses.size()) {
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
  static int idx{1};
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
        "Graph ",
        idx,
        '\n',
        IrGraphDumpUtil::PostOrderToText(po_data.post_order, p_roots),
        "Lazy_IR_Graph_END");
    idx += 1;
    PT_IRGRAPH_DEBUG(IrGraphDumpUtil::PostOrderToText(
        po_data.post_order, p_roots, true, true));
  } else {
    PT_LAZY_DEBUG(IrGraphDumpUtil::PostOrderToDot(po_data.post_order, p_roots));
  }
  return po_data;
}

c10::optional<at::Tensor> HbLazyTensor::GetHbLazyTensorData() {
  PT_LAZY_TRACE;
  // Generate the tensor data if its not been generated yet
  // Forced for finishing the pending execution here
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      GetDevice().index());

  // Check if in-flight execution thread has data, then wait for its completion.
  if (IsExecutionInProgress()) {
    context->JoinPendingLaunchThread();
  }

  // If data isn't available then do step marker to get data.
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

/*
 * Method for getting tensor data for Media data loader
 *
 * This API is not thread safe as it might call Step marker in other threads of
 * Media application or data loader. This can conflict with backward passes i.e.
 * autograd thread doing LazyOp Accumulation as Step marker breaks the graph and
 * executes accumulated Ops and accumulating Ops IR values might change after
 * current graph execution.
 *
 * Media data loader generally calls htcore.data_ptr(tensor) [mapped to
 * GetHbLazyTensorDataForMedia()] after creating empty HPU tensors and fills
 * with data. These tensors come as graph input later and do not need full
 * StepMarker instead, tensor data ptr is returned. But StepMarker can be called
 * for output tensors.
 */
c10::optional<at::Tensor> HbLazyTensor::GetHbLazyTensorDataForMedia() {
  auto currentIrValue = CurrentIrValue();
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      GetDevice().index());

  if (CurrentIrValue() && !CurrentTensorData()) {
    context->JoinPendingLaunchThread();
  }
  if (currentIrValue && !CurrentTensorData()) {
    // Return tensor_data if it is graph input
    if (currentIrValue.mp_node->is_input() == true) {
      return data()->tensor_data;
    } else if (GET_ENV_FLAG_NEW(PT_USE_MARKSTEP)) {
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
    auto live_tensors = HbContextArena::Get()->GetLiveTensors(&GetDevice());
    for (auto& tensor : live_tensors) {
      if (tensor.data()->ir_value.mp_node.get() == node) {
        tensors.emplace_back(tensor);
      }
    }
    SyncTensorsGraph(&tensors);
  }
}

void HbLazyTensor::SyncTensorsGraph(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    bool async,
    bool collect_sync_tensors,
    synEventHandle event_handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag) {
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  context->executing_tids.clear();
  context->executing_tids.reserve(tensors->size());
  // Add all the tensor ids to list to update the execution
  // status after launch.
  for (size_t i = 0; i < tensors->size(); i++) {
    HbLazyTensor& t = (*tensors)[i];
    context->executing_tids.emplace_back(t.getTensorUniqueId());
  }
  SyncTensorsGraphInternal(
      tensors,
      lazyFrontEndInfo,
      async,
      collect_sync_tensors,
      event_handle,
      event_stream,
      event_flag);
}

void HbLazyTensor::SyncLiveTensorsGraph(
    const c10::Device* device,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info = nullptr,
    std::vector<HbLazyTensor> out_hb_lazy_tensor,
    bool async,
    synEventHandle event_handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag,
    bool is_allreduce,
    std::set<int64_t> bucket_id,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;
  if (StageSubmission::getInstance().getCurrentAccumulatedOps() == 0) {
    // if the accumulated op is empty, then just record the event
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    if (event_handle) {
      auto status =
          synEventRecord(event_handle, dev.get_compute_stream(event_stream));
      if (synStatus::synSuccess != status) {
        PT_LAZY_FATAL("synEventRecord failed ", status);
      }
    }
    return;
  }
  StageSubmission::getInstance().resetCurrentAccumulatedOps();
  // For optimized lazy eager, use the output tensors as it is while
  // for normal eager and Lazy, prepare tensors from live tensors
  std::vector<HbLazyTensor> tensors = out_hb_lazy_tensor;
  if (!(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 && lazy_front_end_info &&
        lazy_front_end_info->get_optimized_lazy_eager_key())) {
    tensors = HbContextArena::Get()->GetLiveTensors(
        device, is_allreduce, bucket_recent_id);
  }
  if (tensors.size()) {
    SyncTensorsGraph(
        &tensors,
        lazy_front_end_info,
        async,
        false,
        event_handle,
        event_stream,
        event_flag);
  } else {
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    if (event_handle) {
      auto status =
          synEventRecord(event_handle, dev.get_compute_stream(event_stream));
      if (synStatus::synSuccess != status) {
        PT_LAZY_FATAL("synEventRecord failed ", status);
      }
    }
  }

  {
    auto context =
        habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
    std::lock_guard<std::recursive_mutex> view_table_lock(
        context->viewContext.GetViewTableMutex());

    if (context->viewContext.view_outputs.size()) {
      // delete the origtensor map entry only when view outputs are present
      // ex: megatron has all reduce on embedding tables which doesnt involve
      // strided view output. Such cases should be excluded from deletion

      for (auto id : bucket_id) {
        context->viewContext.DelOrigTensorMapEntry(id);
      }
      context->viewContext.view_outputs.clear();
    }
  }
}

std::string DumpGraph(std::shared_ptr<torch::jit::Graph> jit_graph) {
  std::stringstream strbuff;
  std::streambuf* oldbuff = std::cout.rdbuf(strbuff.rdbuf());
  std::cout << "JIT IR graph\n";
  jit_graph->dump();
  std::string str = strbuff.str();
  std::cout.rdbuf(oldbuff);
  return str;
}

at::Tensor Process0DTensor(std::shared_ptr<Data>& d) {
  TORCH_CHECK(d->tensor_data.has_value(), "Empty tensor optional");
  at::Tensor pt_tensor = d->tensor_data.value();

  // Make regular 0D tensors 1D
  auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
  bool is_shape_tensor = impl && impl->isShapeTensor();
  if (pt_tensor.dim() == 0 &&
      !pt_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    TORCH_CHECK(is_shape_tensor == false, "0D shape tensor encountered");
    pt_tensor.unsafeGetTensorImpl()->set_sizes_contiguous({1});
  }

  return pt_tensor;
}

torch::jit::Stack PrepareInputStack(
    std::vector<HbLazyTensor>* tensors,
    std::vector<int>& indices,
    habana_lazy::ir::ValueList& inputs,
    bool is_OptimizedLazyEager UNUSED,
    habana_lazy::ir::NodePtrList* ptr_post_order = nullptr) {
  auto device = (*tensors)[0].GetDevice();
  auto context = habana_lazy_executor.getDeviceExecutionContext(device.index());
  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  stack.reserve(std::max(inputs.size(), indices.size()));

  // Initiate non-blocking copy to device for all scalar inputs
  if (!context->copy_scalar_to_hpu_tensor_list.empty()) {
    habana_helpers::copy_scalars_to_device(
        context->copy_scalar_to_hpu_tensor_list);
    context->copy_scalar_to_hpu_tensor_list.clear();
  }

  for (const auto& in : inputs) {
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
      if (ptr_post_order != nullptr) {
        PT_LAZY_DEBUG(
            " Node = ",
            in.ToString(),
            "\n Failing IR graph = ",
            IrGraphDumpUtil::PostOrderToText(*ptr_post_order, p_roots));
      }
      HABANA_ASSERT(in.DataPtrValidAndNotExpired());
    }
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    auto pt_tensor = Process0DTensor(d);
    stack.emplace_back(pt_tensor);
    // We dont get the correct lazy tensor back from internal tensor
    // So marking for execution here
    context->MarkTensorExecuting(d);
    d->is_executing = true;
    context->executing_tids.emplace_back(d->unique_id);
  }

  return stack;
}

void PostLaunch(
    std::vector<HbLazyTensor>* tensors,
    torch::jit::Stack& stack,
    std::vector<int>& indices,
    std::vector<at::Tensor>& retained_tensor_list,
    bool is_exception,
    UNUSED bool is_OptimizedLazyEager = false) {
  auto device = (*tensors)[0].GetDevice();
  auto context = habana_lazy_executor.getDeviceExecutionContext(device.index());
  HABANA_ASSERT(is_exception || (stack.size() == indices.size()));

  std::vector<int64_t> executing_indices;
  executing_indices.reserve(stack.size());
  if (!is_exception) {
    size_t i = 0;
    for (const torch::IValue& v : stack) {
      auto& out_tensor = (*tensors)[indices[i++]];
      auto st = v.toTensor();
      executing_indices.push_back(out_tensor.getTensorUniqueId());
      out_tensor.SetTensorData(st);
    }
  }

  context->MarkTensorsExecuted(device, executing_indices);
  context->MarkTensorsExecuted(device, context->executing_tids);
  context->executing_tids.clear();

  SBSDebug::getInstance().CompareTensors(*tensors);

  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
    PT_LAZY_DEBUG("retained_tensor_list : ", retained_tensor_list);
    // TODO: To clear at the right place: retained_tensor_list.clear()
  } else {
    retained_tensor_list.clear();
  }

  // Restore the optimizations which are cleared forcefully in getlivetensors
  exec::OptPassCfg::GetInstance()->RestoreOptPass();
}

void LaunchSyncTensorsGraph(
    std::vector<HbLazyTensor> tensors_ptr,
    std::vector<int> indices,
    exec::HlExec hlexec,
    torch::jit::Stack stack,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    std::vector<at::Tensor> retained_tensor_list,
    bool async,
    std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
        optimized_path_jit_ir_and_mdata,
    std::string lazyOpName,
    size_t optimizedLazyEagerKey,
    bool isOptimizedLazyEager,
    const c10::hpu::HPUStream& stream,
    synEventHandle event_handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag) {
  PT_LAZY_TRACE;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  context->m_launch_thread_context = true;
  std::vector<HbLazyTensor>* tensors = &tensors_ptr;

  // Launch the execution
  std::exception_ptr launch_except = nullptr;
  bool exception = false;
  if (isOptimizedLazyEager) {
    habana_lazy_executor.setExecutionMode(LazyExecutionMode::kLOWERING);
    optimized_path_jit_ir_and_mdata->SetOpName(lazyOpName);
    optimized_path_jit_ir_and_mdata->SetOptimizedLazyEagerFlag(true);
    optimized_path_jit_ir_and_mdata->SetHPUStream(stream);
    optimized_path_jit_ir_and_mdata->SetEventHandle(event_handle);
    optimized_path_jit_ir_and_mdata->SetEventRecordStream(event_stream);
    habana::HabanaLaunchOpPT habanaLoweringOp{optimized_path_jit_ir_and_mdata};
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
      auto& input_values = lazyFrontEndInfo->get_input_values();
      stack = PrepareInputStack(tensors, indices, input_values, true);
    }

    try {
      habanaLoweringOp.run(stack);
      if (optimized_path_jit_ir_and_mdata->get_syn_graph_empty_flag() == true) {
        // The graph was not compiled. Remove the JIT graph from the cache
        // To Do - To incorporate the Graph index change
        PT_LAZY_DEBUG(
            "Removing Optimized JIT IR Graph with :: key ",
            optimizedLazyEagerKey,
            " from the Optimized JIT Cache");
        OptimizedLazyGraphCache::GetOptimizedLazyCache().RemoveGraph(
            optimizedLazyEagerKey);
      }
    } catch (...) {
      launch_except = std::current_exception();
      exception = true;
    }
    habana_lazy_executor.setExecutionMode(LazyExecutionMode::kLAZY);
  } else {
    try {
      hlexec.Launch(stack, stream, event_handle, event_stream, event_flag);
      if (hlexec.GetJITGraphMetaDataPtr()->get_syn_graph_empty_flag() == true) {
        // The graph was not compiled. Remove the JIT graph from the cache
        PT_LAZY_DEBUG(
            "Removing JIT IR Graph with :: key ",
            hlexec.GetGraphHash(),
            ", graph_index ",
            visualize::GetGraphIndex(hlexec.GetGraphHash()),
            " from  the JIT Cache");
        LazyGraphCache::GetLazyCache().RemoveGraph(hlexec.GetGraphHash());
      }
    } catch (...) {
      launch_except = std::current_exception();
      exception = true;
    }
  }

  PostLaunch(tensors, stack, indices, retained_tensor_list, exception);

  // Rethrow exception in case exception occuured during launch
  if (exception) {
    if (async) {
      context->m_launch_thread_exception_handler = launch_except;
    }
    std::rethrow_exception(launch_except);
  }
  context->m_launch_thread_context = false;
}

void HbLazyTensor::SyncTensorsGraphInternal(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    bool async,
    bool collect_sync_tensors,
    synEventHandle event_handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag) {
  PT_LAZY_TRACE;
  if (!(*tensors).size())
    return;

  auto device = (*tensors)[0].GetDevice();
  auto context = habana_lazy_executor.getDeviceExecutionContext(device.index());
  bool isOptimizedLazyEager = false;
  size_t optimized_lazy_eager_key = 0;
  if (lazyFrontEndInfo) {
    optimized_lazy_eager_key = lazyFrontEndInfo->get_optimized_lazy_eager_key();
    // To check if it is Optimized Lazy Cached graph
    isOptimizedLazyEager = lazyFrontEndInfo->get_is_optimized_lazy_eager();
  }
  if (!isOptimizedLazyEager) {
    context->JoinPendingLaunchThread();
  }

  std::vector<int> indices = {};
  // collect_sync_tensors will be true when the markstep is invoked and the live
  // tensors are collected. In this scenario tensors list wont contain any input
  // tensors.
  if (!collect_sync_tensors ||
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 && lazyFrontEndInfo &&
       lazyFrontEndInfo->get_optimized_lazy_eager_key())) {
    indices.resize((*tensors).size());
    std::iota(indices.begin(), indices.end(), 0);
  } else {
    indices = CollectSyncTensors(*tensors);
  }

  if (indices.empty()) {
    // Nothing to do, return without trying to execute an empty graph
    context->MarkTensorsExecuted(device, context->executing_tids);
    context->executing_tids.clear();
    // if the accumulated op is empty, then just record the event
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    if (event_handle) {
      auto status =
          synEventRecord(event_handle, dev.get_compute_stream(event_stream));
      if (synStatus::synSuccess != status) {
        PT_LAZY_FATAL("synEventRecord failed ", status);
      }
    }
    return;
  }

  std::vector<std::vector<int64_t>> out_shapes{};
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH)) {
    for (auto idx : indices) {
      auto& out_tensor = (*tensors)[idx];
      out_shapes.push_back(out_tensor.GetSizes());
      PT_SHAPE_AGNOSTIC_DEBUG(
          "[LAZY EAGER SHAPE AGNOSTIC] output idx : ",
          idx,
          " shape : ",
          out_tensor.GetSizes());
    }
  }

  torch::jit::Stack stack;
  habana_lazy::ir::PostOrderData po_data;
  std::vector<uint64_t> executing_indices{};
  exec::HlExec hlexec{};
  std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
      optimized_path_jit_ir_and_mdata;
  std::string lazy_op_name{};

  if (isOptimizedLazyEager) {
    HABANA_ASSERT(lazyFrontEndInfo != nullptr);
    lazy_op_name = lazyFrontEndInfo->get_lazy_op_name();
    optimized_path_jit_ir_and_mdata =
        habana_lazy::OptimizedLazyGraphCache::GetOptimizedLazyCache()
            .GetOptimizedJITGraphAndMetaData(optimized_lazy_eager_key);
    PT_LAZY_DEBUG(
        "Optimized Path JIT Cache hit :: key ", optimized_lazy_eager_key);
    PT_IRGRAPH_DEBUG(
        DumpGraph(optimized_path_jit_ir_and_mdata->get_cached_graph()));
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH)) {
      // Setting output shapes for the lazy eager shape agnostic graph
      optimized_path_jit_ir_and_mdata->set_output_shapes(out_shapes);
    }
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
      context->JoinPendingLaunchThread();
      std::vector<ir::Value>& input_values =
          lazyFrontEndInfo->get_input_values();
      stack = PrepareInputStack(tensors, indices, input_values, true);
    }
  } else {
    po_data = HbLazyTensor::RunPostOrder(*tensors, indices);
    stack = PrepareInputStack(
        tensors, indices, po_data.inputs, false, &po_data.post_order);

    hlexec.set_lazy_front_end_info(lazyFrontEndInfo);

    hlexec.GetOrCreate(po_data, stack);

    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH)) {
      // Setting output shapes for the lazy eager shape agnostic graph
      hlexec.GetJITGraphMetaDataPtr()->set_output_shapes(out_shapes);
    }

    // This is the logic to remove outputs of control edges that are dangling
    // from
    // the outputs of JIT graph We dont want to alter graph execution, so
    // removing after graph is already prepared. Also we DO want that the
    // tensors of this node are marked processed, as they would have through
    // output stack So we do all the markings before entering execution
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
          context->MarkTensorExecuted(data);
          po_data.outputs.erase(po_data.outputs.begin() + vec_index);
          indices.erase(indices.begin() + vec_index);
          hlexec.get_graph()->eraseOutput(vec_index);
        } else {
          vec_index++;
        }
      }
    }
    // Dump the JIT graph with PT_IRGRAPH_DEBUG
    PT_IRGRAPH_DEBUG(DumpGraph(hlexec.get_graph()));
  }

  // Remove any tensor_data held at output, this will reduce the memory
  // pressure
  for (auto idx : indices) {
    auto& out_tensor = (*tensors)[idx];
    out_tensor.SetExecutionInProgress();
    if (!async) {
      out_tensor.SetTensorData(at::Tensor());
    }
    // clear IR values corresponding to sync tensors
    out_tensor.ClearAndAssignNewIrValue();
  }

  // clear IR values corresponding to unexecuted view outputs
  for (auto& t : context->viewContext.hb_tensors_exclude_out_view) {
    t.SetExecutionInProgress();
    context->executing_tids.emplace_back(t.getTensorUniqueId());
    ir::Value val = t.createIrValueFromData();
    t.resetVersionCounter();
    t.AssignIrValue(val);
  }

  // Save po_data input and output to context for perf mode
  if (context->getCapturing()) {
    context->saveInputsAndOutputs(
        po_data.inputs, po_data.outputs, *tensors, indices);
  }

  if (async) {
    // Use threadpool if the hosttracing is enabled or if its eager mode
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAUNCHTHREAD_USE_THREADPOOL) ||
        (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) ||
        GET_ENV_FLAG_NEW(TRACE_POINT_ENABLE)) {
      context->m_launch_thread_handle =
          SingleTonExecThreadPool::getInstance().enqueue(
              LaunchSyncTensorsGraph,
              *tensors,
              std::vector<int>(indices),
              exec::HlExec(hlexec),
              torch::jit::Stack(stack),
              lazyFrontEndInfo,
              context->m_retained_tensor_list,
              async,
              optimized_path_jit_ir_and_mdata,
              lazy_op_name,
              optimized_lazy_eager_key,
              isOptimizedLazyEager,
              c10::hpu::getCurrentHPUStream(),
              event_handle,
              event_stream,
              event_flag);
    } else {
      context->m_launch_thread_handle = std::async(
          std::launch::async,
          LaunchSyncTensorsGraph,
          *tensors,
          std::vector<int>(indices),
          exec::HlExec(hlexec),
          torch::jit::Stack(stack),
          lazyFrontEndInfo,
          context->m_retained_tensor_list,
          async,
          optimized_path_jit_ir_and_mdata,
          lazy_op_name,
          optimized_lazy_eager_key,
          isOptimizedLazyEager,
          c10::hpu::getCurrentHPUStream(),
          event_handle,
          event_stream,
          event_flag);
    }
  } else {
    LaunchSyncTensorsGraph(
        *tensors,
        std::vector<int>(indices),
        exec::HlExec(hlexec),
        torch::jit::Stack(stack),
        lazyFrontEndInfo,
        context->m_retained_tensor_list,
        async,
        optimized_path_jit_ir_and_mdata,
        lazy_op_name,
        optimized_lazy_eager_key,
        isOptimizedLazyEager,
        c10::hpu::getCurrentHPUStream(),
        event_handle,
        event_stream,
        event_flag);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRADIENT_BUCKET_VIEW)) {
    for (auto t : context->viewContext.updated_bucket_list) {
      // clear IR values corresponding to sync tensors
      t.ClearAndAssignNewIrValue();
    }
  }

  // clear the context
  context->clear();
}

void HbLazyTensor::ExecuteCachedGraph(
    GraphPtr graph,
    size_t hash,
    size_t graphKey,
    std::string opStrs,
    ir::ValueList& input_vals,
    ir::ValueList& output_vals,
    std::vector<habana_lazy::HbLazyTensor> hblazy_tensors,
    bool is_cached) {
  PT_LAZY_TRACE;
  exec::HlExec hlexec{};

  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  HABANA_ASSERT(is_cached == true);

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
  hlexec.set_graph(graph);

  // Set the graph hash
  hlexec.set_hash(hash);

  // Set the graph key
  hlexec.set_graph_key(graphKey);

  // Set the op strs
  hlexec.set_opstrs(opStrs);

  // Launch the execution
  hlexec.Launch(stack, c10::hpu::getCurrentHPUStream(), {}, 0, 0);

  HABANA_ASSERT(stack.size() == hblazy_tensors.size());

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    auto st = v.toTensor();
    HbLazyTensor out_tensor = hblazy_tensors[i++];
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
  PT_LAZY_TRACE;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto hl_t = *this;
  if (dest->IsExecutionInProgress() || hl_t.IsExecutionInProgress()) {
    context->JoinPendingLaunchThread();
  }

  auto src_id = this->getTensorUniqueId();
  auto dst_id = dest->getTensorUniqueId();

  auto hl_opt = context->viewContext.GetShallowCopyMapEntry(src_id);
  if (hl_opt != c10::nullopt) {
    hl_t = hl_opt.value();
  }

  context->viewContext.AddShallowCopyMapEntry(dst_id, hl_t);
}

void HbLazyTensor::StepMarkerBind(const std::string& device_str) {
  PT_LAZY_TRACE;
  PT_IRGRAPH_DEBUG("step marker due to host step marker");
  PT_LAZY_DEBUG("step marker due to host step marker");
  StageSubmission::getInstance().resetStageSubmissionFlow();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1)) {
    StepMarker(device_str, nullptr, {}, true, nullptr, false);
  } else {
    StepMarker(device_str);
  }
}

void HbLazyTensor::StepMarkerFinish() {
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  context->m_launch_thread_context = false;
  context->JoinPendingLaunchThread();
}

void HbLazyTensor::IterStepMarker() {
  habana_lazy_executor.resetUniqueGraphCntr();
}

void HbLazyTensor::StepMarker(
    const std::string& device_str,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
    std::vector<HbLazyTensor> out_hb_lazy_tensor,
    bool async,
    synEventHandle handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag,
    bool is_allreduce,
    std::set<int64_t> bucket_id,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();

  if (!synapse_helpers::HPURegistrar::isInitialized()) {
    // Nothing to do
    PT_LAZY_DEBUG("StepMarker called before device was initialized, skipping");
    return;
  }
  c10::Device device = GetDeviceOrCurrent(device_str);
  if (!device.is_hpu()) {
    PT_LAZY_DEBUG(
        "Could get hpu device device_str = \"",
        device_str,
        "\", skipping StepMarker");
    return;
  }
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  context->m_launch_thread_context = false;
  if (!(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 && lazy_front_end_info &&
        lazy_front_end_info->get_optimized_lazy_eager_key())) {
    context->JoinPendingLaunchThread();
  }

  HbLazyTensor::SyncLiveTensorsGraph(
      &device,
      lazy_front_end_info,
      out_hb_lazy_tensor,
      async,
      handle,
      event_stream,
      event_flag,
      is_allreduce,
      bucket_id,
      bucket_recent_id);
  if (!async) {
    context->JoinPendingLaunchThread();
  }
  HbLazyTensor::MarkStep(device);
  if (switch_dynamic_mode) {
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
    switch_dynamic_mode = false;
  }
}

void HbLazyTensor::SetDynamicMode() {
  bool dynamic_env = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  switch_dynamic_mode = dynamic_env ? false : true;
  if (switch_dynamic_mode) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
  }
}

void* HbLazyTensor::lazyTensorDataPtr(const at::Tensor& t) {
  return GetLazyTensorDataPtr(t);
}
