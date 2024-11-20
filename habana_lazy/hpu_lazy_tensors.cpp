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
#include "hpu_lazy_tensors.h"
#include <torch/csrc/jit/ir/ir.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/random.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "common/utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy/lazy_graph_hash_disabler.h"
#include "habana_lazy/ops/hpu_input.h"
#include "habana_lazy/sbs_debug.h"
#include "habana_lazy/view_utils.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/visualize/visualize.h"

using namespace habana_lazy;

std::shared_ptr<Snapshot> StaleLazyTensorKeeper::extract_snapshot() {
  auto snapshot = std::make_shared<Snapshot>();
  std::lock_guard<std::mutex> lock(mutex);
  while (!kept_alive.empty()) {
    auto& front = kept_alive.front();
    if (front.marker) {
      kept_alive.pop_front();

      // pop all stopers until next valid tensor
      while (!kept_alive.empty()) {
        auto& maybe_stopper = kept_alive.front();
        if (maybe_stopper.marker) {
          kept_alive.pop_front();
        } else {
          break;
        }
      }

      return snapshot;
    } else {
      if (!front.tensor.is_null() &&
          !front.tensor.IsOpAccumulationInProgress()) {
        snapshot->tensors.emplace_back(std::move(front.tensor));
      } else if (front.tensor.IsOpAccumulationInProgress()) {
        Holder holder(std::move(front.tensor), false);
        Holder stopper(HbLazyTensor(), true);
        kept_alive.push_back(std::move(holder));
        kept_alive.push_back(std::move(stopper));
      } else {
        // nothing to do
      }
      kept_alive.pop_front();
    }
  }
  return snapshot;
}

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
  auto context = habana_lazy::get_device_lazy_execution_context();
  context->RegisterTensor(data);
  if (synapse_helpers::memory_reporter_enable()) {
    auto& device = habana::HPUDeviceContext::get_device();
    synapse_helpers::MemoryReporter* reporter =
        device.get_device_memory().get_memory_reporter();
    reporter->getTensorStats()->createTensor(data->unique_id);
  }
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
  auto context = habana_lazy::get_device_lazy_execution_context();

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
    devctx->erase(unique_id);
    if (synapse_helpers::memory_reporter_enable() &&
        habana::HPUDeviceContext::is_device_acquired()) {
      auto& device = habana::HPUDeviceContext::get_device();
      synapse_helpers::MemoryReporter* reporter =
          device.get_device_memory().get_memory_reporter();
      reporter->getTensorStats()->removeTensor(unique_id);
    }
  }
}

std::vector<HbLazyTensor> HbContextArena::GetLiveTensors(
    const c10::Device* device,
    bool is_allreduce,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;
  std::vector<HbLazyTensor> tensors;
  auto context = get_device_lazy_execution_context();

  // Live tensor collection is not allowed if the launch thread execution is  in
  // progeress.
  // if (!(GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
  //       (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2))) {
  //   HABANA_ASSERT(context->m_launch_thread_handle.valid() == false);
  // }

  HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(*device);

  HbLazyTensorViews::HandleViewsLiveTensors(
      devctx, is_allreduce, bucket_recent_id);
  for (auto& uid : devctx->tensors_data_opt_order) {
    std::shared_ptr<Data> data = devctx->getDataPtr(uid);
    if (data) {
      auto id = data->unique_id;
      auto hl_t = HbLazyTensor(std::move(data));

      auto is_view = hl_t.getDataPtr()->stride_params.has_value();

      if (bucket_recent_id.count(id)) {
        context->viewContext.updated_bucket_list.emplace_back(hl_t);
      }
      auto is_view_out = context->viewContext.view_outputs.count(id);

      if (is_view_out ||
          ((bucket_recent_id.count(id) == 0) && (!is_view) &&
           (hl_t.getDataPtr()->recent_base == c10::nullopt))) {
        tensors.emplace_back(hl_t);
      }
    }
  }
  {
    std::lock_guard<std::recursive_mutex> lock(GetMutex());
    devctx->clear_tensors_data();
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

int64_t Data::GetNextTensorId() {
  // Tensors are created concurrently in accumulation and main threads.
  // Tensor ids order is strictly determining the PostOrder graph.
  // In order to avoid nondeterministic tensor ids list,
  // tensors created inside acc thread get ids starting in the middle of (0,max)
  // of int64 range to avoid collisions with tensors created in main thread.
  static auto id_generator = std::atomic<int64_t>(1);
  static auto acc_id_generator =
      std::atomic<int64_t>(std::numeric_limits<int64_t>::max() / 2);
  auto id = AccThread::Get().inAccThreadContext()
      ? acc_id_generator.fetch_add(1)
      : id_generator.fetch_add(1);
  return id;
}

HbLazyTensor::HbLazyTensor(const at::Tensor& tensor, const c10::Device& device)
    : mp_data(std::make_shared<Data>(tensor, device)) {}
HbLazyTensor::HbLazyTensor(const c10::Device& device)
    : mp_data(std::make_shared<Data>(device)) {}

HbLazyTensor::HbLazyTensor(
    ir::Value&& ir_value,
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
  auto is_tensor_const = habana::is_tensor_const(tensor);
  auto tensor_const_id = habana::get_tensor_const_id(tensor);
  HbLazyTensor habana_tensor(tensor, device);
  habana_tensor.SetIsConstTensor(is_tensor_const, tensor_const_id);
  HbContextArena::Get()->RegisterTensor(habana_tensor.getDataPtr());
  return habana_tensor;
}

void HbLazyTensor::setTensorSize(c10::IntArrayRef sizes) {
  // create smallvector from intarrayref to avoid heap allocation, using
  // sizes.vec() to create std::vector is costly due to heap allocation
  data()->sizes.insert(data()->sizes.begin(), sizes.begin(), sizes.end());
}
HbLazyTensor HbLazyTensor::Create(
    ir::Value&& ir_value,
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
  auto context = habana_lazy::get_device_lazy_execution_context();
  context->JoinPendingLaunchThread();

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

void HbLazyTensor::ResetExecutionInProgress() const {
  data()->is_executing = false;
}

bool HbLazyTensor::IsOpAccumulationInProgress() const {
  return data()->is_op_acc;
}

void HbLazyTensor::SetOpAccumulationInProgress() const {
  auto context = get_device_lazy_execution_context();
  std::lock_guard<std::mutex> lock(context->GetOpAccTidsMutex());
  // note: we need to increment only once per op accmulation pphase
  if (context->op_acc_tids.find(data()->unique_id) ==
      context->op_acc_tids.end()) {
    data()->is_op_acc++;
    context->op_acc_tids.emplace(data()->unique_id);
  }
}

const ir::Value& HbLazyTensor::GetIrValue() const {
  ir::Value& ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  // In case of tensor node, we do not clear the device data when we set the
  // IR node. This because we want further calls to GetIrValue() to fetch the
  // same IR node, and not create new ones (even though the lowering context
  // will still collapse them all into a single Habana parameter op). So call
  // which wants the device data will still find it, w/out having to fetch it
  // via a computation on device
  IrReconnectAsInputNode();
  return CurrentIrValue();
}

ir::Value& HbLazyTensor::IrSetNode(ir::NodePtr node, size_t index) const {
  auto& v{CurrentIrValue()};
  v.SetNode(node, GetDevice(), GetSizes(), dtype_optional(), index);
  return v;
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
void HbLazyTensor::SetTensorDataNullOpt() {
  data()->tensor_data = c10::nullopt;
}

void HbLazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
  if (synapse_helpers::memory_reporter_enable()) {
    auto& device = habana::HPUDeviceContext::get_device();
    synapse_helpers::MemoryReporter* reporter =
        device.get_device_memory().get_memory_reporter();
    reporter->getTensorStats()->setTensorAddressData(
        data()->unique_id,
        (tensor_data.has_storage() ? tensor_data.storage().data_ptr().get()
                                   : nullptr));
  }
}

c10::optional<at::Tensor> HbLazyTensor::GetTensorData() {
  auto tens = data()->tensor_data;
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

void HbLazyTensor::SetCollective() {
  data()->collective = true;
}

void HbLazyTensor::ClearCollective() {
  data()->collective = false;
}

bool HbLazyTensor::IsCollective() const {
  return data()->collective;
}

void HbLazyTensor::ClearStrideParams() {
  data()->stride_params.reset();
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
  auto context = habana_lazy::get_device_lazy_execution_context();
  if (context != nullptr) {
    auto status = context->getTensorExecutionStatus(getDataPtr());
    if (status == kEXECUTION_COMPLETE || status == kINPUT) {
      // brave assert that when kInput then IR is Input
      // actually we can assert
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

const SmallSizeVec& HbLazyTensor::GetSizes() const {
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

void HbLazyTensor::IrInitAsInputNode() const {
  HABANA_ASSERT(
      (!CurrentIrValue()),
      " Habana Lazy Trying to set a tensor as leaf input node but IR value"
      " is set already");
  IrReconnectAsInputNode();
}

void HbLazyTensor::IrReconnectAsInputNode() const {
  ir::Value val = createIrValueFromData();
  ir::NodePtr node = std::make_shared<ir::Input>(*this);
  val.SetNode(node, GetDevice(), GetSizes(), dtype_optional());
  AssignIrValue(val);
}

void HbLazyTensor::setPtrDataIrToData() {
  if (mp_data.get())
    mp_data->ir_value.m_data_ptr = mp_data;
}

// cannot add constant support here as tensor not present
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
      std::move(val),
      device,
      scalar_type);

  // We keep a weak pointer in our IR back to data pointer of lazy tensor
  // This needs to be updated here or we can push it in the constructor
  // Keeping it here for now so that its not implicitly set
  hb_tensor.setPtrDataIrToData();
  // Setup the size information in the data of Lazy tensor
  hb_tensor.setTensorSize(size);
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

void HbLazyTensor::ValidateTensorData() const {
  auto tensor_data{data()->tensor_data};
  HABANA_ASSERT(
      tensor_data,
      "data_ptr()->unique_id: ",
      data_ptr()->unique_id,
      " Habana Lazy: no storage tensor attached to lazy tensor");
  HABANA_ASSERT(
      tensor_data->has_storage(),
      "data_ptr()->unique_id: ",
      data_ptr()->unique_id,
      "Habana Lazy: lazy tensor doesn't has a storage");
}

at::Tensor HbLazyTensor::EvaluateTensorData(bool sync_acc_thread) {
  PT_LAZY_TRACE;
  // Generate the tensor data if its not been generated yet
  // Forced for finishing the pending execution here
  auto context = habana_lazy::get_device_lazy_execution_context();

  // Check if in-flight execution thread has data, then wait for its completion.
  if (IsExecutionInProgress()) {
    context->JoinPendingLaunchThread();
  }

  // When acc thread is present, IR value may not be available so the next check
  // may fail and skip StepMarker
  if (sync_acc_thread) {
    habana_lazy::AccThread::Get().SyncAccThreadPool();
  }

  // If data isn't available then do step marker to get data.
  if (CurrentIrValue() && !CurrentTensorData()) {
    if (GET_ENV_FLAG_NEW(PT_USE_MARKSTEP)) {
      PT_IRGRAPH_DEBUG("step marker due to EvaluateTensorData-PT_USE_MARKSTEP");
      HbLazyTensor::StepMarker({});
    } else {
      std::lock_guard<std::recursive_mutex> lock(
          HbContextArena::Get()->GetMutex());
      applyPendingGraph();
    }
  }
  ValidateTensorData();
  return data()->tensor_data.value();
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
  // When acc thread is present, IR value may not be available so the next check
  // may fail and skip StepMarker
  habana_lazy::AccThread::Get().SyncAccThreadPool();

  auto currentIrValue = CurrentIrValue();
  auto context = habana_lazy::get_device_lazy_execution_context();

  if (CurrentIrValue() && !CurrentTensorData()) {
    context->JoinPendingLaunchThread();
  }

  if (currentIrValue && !CurrentTensorData()) {
    // Return tensor_data if it is graph input
    if (currentIrValue.mp_node->is_input() == true) {
      return data()->tensor_data;
    } else if (GET_ENV_FLAG_NEW(PT_USE_MARKSTEP)) {
      PT_IRGRAPH_DEBUG(
          "step marker due to GetHbLazyTensorDataForMedia-PT_USE_MARKSTEP");
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
    StaleLazyTensorKeeper::getInstance().mark_end_of_accumulation();
    SyncTensorsGraph(&tensors);
  }
}

namespace {
inline c10::Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return habana::HPUDeviceContext::aten_device();
  }

  return c10::Device(device_str);
}
} // namespace

void HbLazyTensor::SyncTensorsGraph(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    bool async,
    bool collect_sync_tensors) {
  habana_lazy::NoAccThread no_acc_thread;
  HbContext* devctx =
      habana_lazy::HbContextArena::Get()->GetHbContext(GetDeviceOrCurrent(""));
  // If we reach here with anything in tensors_data_opt, it means its a direct
  // call to SyncTensorsGraph and partial execution may happen => clear data
  // for tensors that will be evaluated as part of current execution. For all
  // other cases tensors_data_opt is already completely cleared in
  // GetLiveTensors.
  if (devctx->tensors_data_size()) {
    for (auto& t : *tensors) {
      devctx->erase(t.getTensorUniqueId());
      if (devctx->tensors_data_size() == 0) {
        break;
      }
    }
  }

  auto context = habana_lazy::get_device_lazy_execution_context();
  context->executing_tids.clear();
  context->executing_tids.reserve(tensors->size());
  // Add all the tensor ids to list to update the execution
  // status after launch.
  for (size_t i = 0; i < tensors->size(); i++) {
    HbLazyTensor& t = (*tensors)[i];
    context->executing_tids.emplace_back(t.getTensorUniqueId());

    // clear collective flag for all tensors so they won't cause insertion of
    // StepMarker
    t.ClearCollective();
  }
  SyncTensorsGraphInternal(
      tensors, lazyFrontEndInfo, async, collect_sync_tensors);
}

void HbLazyTensor::SyncLiveTensorsGraph(
    const c10::Device* device,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info = nullptr,
    std::vector<HbLazyTensor> out_hb_lazy_tensor,
    bool async,
    bool is_allreduce,
    std::vector<HbLazyTensor> bucket_hl_t,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;
  if (StageSubmission::getInstance().getCurrentOpCount() == 0) {
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
    StaleLazyTensorKeeper::getInstance().mark_end_of_accumulation();
    SyncTensorsGraph(&tensors, lazy_front_end_info, async, false);
  }

  {
    auto context = habana_lazy::get_device_lazy_execution_context();

    if (context->viewContext.view_outputs.size()) {
      // delete the origtensor map entry only when view outputs are present
      // ex: megatron has all reduce on embedding tables which doesnt involve
      // strided view output. Such cases should be excluded from deletion

      for (auto& hl_t : bucket_hl_t) {
        hl_t.getDataPtr()->recent_base = c10::nullopt;
      }
      context->viewContext.view_outputs.clear();
    }
  }
}

std::string DumpGraph(std::shared_ptr<torch::jit::Graph> jit_graph) {
  std::stringstream strbuff;
  std::streambuf* oldbuff = std::cout.rdbuf(strbuff.rdbuf());
  jit_graph->dump();
  std::string str = strbuff.str();
  std::cout.rdbuf(oldbuff);
  return str;
}

void ValidateSyncInputTensors(habana_lazy::ir::ValueList& inputs) {
  for (const auto& in : inputs) {
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    if (d == nullptr) {
      PT_LAZY_FATAL(
          "Error, ValidateSyncInputTensors m_data_ptr expired. irValue:",
          in.ToString());
    }
    if (d && (!d->tensor_data.has_value())) {
      PT_LAZY_FATAL(
          "Error, ValidateSyncInputTensors tensor_data is empty. Tensorid:",
          d->unique_id,
          " QueueStatus:",
          SingleTonExecThreadPool::getInstance().ToString(),
          " irValue:",
          in.ToString());
    }
  }
}

void SetLaunchContextFlags(
    habana_lazy::ir::ValueList& inputs,
    std::vector<int64_t>& executing_tids) {
  auto context = get_device_lazy_execution_context();
  for (const auto& in : inputs) {
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    context->MarkTensorExecuting(d);
    d->is_executing = true;
    executing_tids.emplace_back(d->unique_id);
  }
}

torch::jit::Stack PrepareInputStack(
    std::vector<HbLazyTensor>* tensors,
    std::vector<int>& indices,
    ir::ValueList& inputs,
    bool is_OptimizedLazyEager [[maybe_unused]],
    habana_lazy::ir::NodePtrList* ptr_post_order = nullptr,
    bool copy_scalar_to_hpu = true) {
  auto context = get_device_lazy_execution_context();
  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  stack.reserve(std::max(inputs.size(), indices.size()));

  // Initiate non-blocking copy to device for all scalar inputs
  if (copy_scalar_to_hpu && !context->copy_scalar_to_hpu_tensor_list.empty()) {
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
    HABANA_ASSERT(d->tensor_data.has_value(), "Empty tensor optional");
    at::Tensor pt_tensor = d->tensor_data.value();
    auto is_const_tensor = habana::is_tensor_const(pt_tensor);

    if (d->is_const_tensor && !is_const_tensor) {
      habana::set_tensor_const(pt_tensor, d->is_const_tensor, d->const_id);
    }

    stack.emplace_back(pt_tensor);
    // We dont get the correct lazy tensor back from internal tensor
    // So marking for execution here
    context->MarkTensorExecuting(d);
  }

  return stack;
}

void PostLaunch(
    std::vector<HbLazyTensor>* tensors,
    torch::jit::Stack& stack,
    std::vector<int>& indices,
    std::vector<int64_t>& executing_indices,
    std::set<int64_t>& accumulated_indices,
    std::vector<at::Tensor>& retained_tensor_list,
    bool is_exception,
    [[maybe_unused]] bool is_OptimizedLazyEager = false) {
  auto device = (*tensors)[0].GetDevice();
  auto context = get_device_lazy_execution_context();

  HABANA_ASSERT(is_exception || (stack.size() == indices.size()));

  if (!is_exception) {
    size_t i = 0;
    for (const torch::IValue& v : stack) {
      auto& out_tensor = (*tensors)[indices[i++]];
      auto st = v.toTensor();
      executing_indices.push_back(out_tensor.getTensorUniqueId());
      if ((!context->getDryRun()) ||
          (context->getDryRun() && !out_tensor.isStorageAttached())) {
        out_tensor.SetTensorData(st);
      }
    }
  }

  context->MarkTensorsExecuted(device, executing_indices);
  context->ClearOpAccmulationFlag(device, accumulated_indices);

  // release those launched stale lazy tensor
  auto snapshot = StaleLazyTensorKeeper::getInstance().extract_snapshot();
  snapshot.reset();

  SBSDebug::getInstance().CompareTensors(*tensors);

  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
    PT_LAZY_EAGER_DEBUG(
        "[LAZY EAGER MT] retained_tensor_list, size : ",
        retained_tensor_list.size());
    // TODO: To clear at the right place: retained_tensor_list.clear()
  } else {
    PT_LAZY_DEBUG(
        " Clearing retained_tensor_list, size : ", retained_tensor_list.size());
    retained_tensor_list.clear();
  }

  // Restore the optimizations which are cleared forcefully in getlivetensors
  exec::OptPassCfg::GetInstance()->RestoreOptPass();
}

struct LaunchTensorsInfo {
  std::vector<HbLazyTensor> tensors_ptr;
  std::vector<std::shared_ptr<Data>> input_list;
  std::vector<int> indices;
  // Tensorids list which is part of current exec thread
  std::vector<int64_t> executing_tids;
  // Tensorids list that are marked in accumulation phase
  std::set<int64_t> op_acc_tids;
  habana_lazy::ir::PostOrderData po_data;
  exec::HlExec hlexec;
  torch::jit::Stack stack;
  bool async;
  bool has_queued;
  uint64_t launch_jobid;
  bool dynamic_shape;
  bool dry_run;
};

struct LaunchEagerInfo {
  std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo;
  std::vector<at::Tensor> retained_tensor_list;
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
      optimized_path_jit_ir_and_mdata;
  std::string lazyOpName;
  std::vector<std::vector<int64_t>> out_shapes;
  size_t optimizedLazyEagerKey;
  bool isOptimizedLazyEager;
};

struct LaunchStreamInfo {
  const c10::hpu::HPUStream stream;
};

void LaunchSyncTensorsGraph(
    LaunchTensorsInfo&& launch_info,
    LaunchEagerInfo&& lazy_eager_info,
    LaunchStreamInfo&& stream_info) {
  PT_LAZY_TRACE;
  PT_LAZY_EXEC_THREAD(
      "Launch started async:",
      launch_info.async,
      " has_queued:",
      launch_info.has_queued,
      " launch_jobid:",
      launch_info.launch_jobid,
      " dynamic:",
      launch_info.dynamic_shape);
  get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLOWERING);
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  habana_helpers::SetRefineDynamicShape(launch_info.dynamic_shape);
  habana_lazy::NoAccThread no_acc_thread(
      false); // disable acc thread during launch, but do not sync the acc
              // thread
  auto context = get_device_lazy_execution_context();
  context->HandleException();
  context->m_launch_thread_context = true;
  std::vector<HbLazyTensor>* tensors = &launch_info.tensors_ptr;
  // Launch the execution
  std::exception_ptr launch_except = nullptr;
  bool exception = false;
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>&
      optimized_path_jit_ir_and_mdata =
          lazy_eager_info.optimized_path_jit_ir_and_mdata;
  if (lazy_eager_info.isOptimizedLazyEager) {
    optimized_path_jit_ir_and_mdata->SetOpName(lazy_eager_info.lazyOpName);
    optimized_path_jit_ir_and_mdata->SetOptimizedLazyEagerFlag(true);
    optimized_path_jit_ir_and_mdata->SetHPUStream(stream_info.stream);
    bool isDynamic = habana_helpers::GetRefineDynamicShapeStatus();
    optimized_path_jit_ir_and_mdata->SetDynamicGraph(isDynamic);

    habana::HabanaLaunchOpPT habanaLoweringOp{optimized_path_jit_ir_and_mdata};
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
      auto& input_values = lazy_eager_info.lazyFrontEndInfo->get_input_values();
      launch_info.stack =
          PrepareInputStack(tensors, launch_info.indices, input_values, true);
      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] PrepareInputStack in Launch for key: ",
          lazy_eager_info.optimizedLazyEagerKey);
    }

    try {
      habanaLoweringOp.run(launch_info.stack);
      if (optimized_path_jit_ir_and_mdata->get_syn_graph_empty_flag() == true) {
        // The graph was not compiled. Remove the JIT graph from the cache
        // To Do - To incorporate the Graph index change
        PT_LAZY_DEBUG(
            "Removing Optimized JIT IR Graph with :: key ",
            lazy_eager_info.optimizedLazyEagerKey,
            " from the Optimized JIT Cache");
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().RemoveGraph(
            lazy_eager_info.optimizedLazyEagerKey);
      }
    } catch (...) {
      launch_except = std::current_exception();
      exception = true;
      get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLAZY);
    }
  } else {
    try {
      if (launch_info.has_queued) {
        ValidateSyncInputTensors(launch_info.po_data.inputs);
        launch_info.stack = PrepareInputStack(
            tensors,
            launch_info.indices,
            launch_info.po_data.inputs,
            false,
            &launch_info.po_data.post_order,
            false);

        launch_info.hlexec.set_lazy_front_end_info(
            lazy_eager_info.lazyFrontEndInfo);
        launch_info.hlexec.GetOrCreate(launch_info.po_data, launch_info.stack);
      }
      // Dump the JIT graph with PT_IRGRAPH_DEBUG
      PT_IRGRAPH_DEBUG(launch_info.hlexec.get_graph()->toString());

      launch_info.hlexec.Launch(
          launch_info.stack, stream_info.stream, launch_info.dry_run);
      if (launch_info.hlexec.GetJITGraphMetaDataPtr()
              ->get_syn_graph_empty_flag() == true) {
        // The graph was not compiled. Remove the JIT graph from the cache
        PT_LAZY_DEBUG(
            "Removing JIT IR Graph with :: key ",
            launch_info.hlexec.GetGraphHash(),
            ", graph_index ",
            visualize::GetGraphIndex(launch_info.hlexec.GetGraphHash()),
            " from  the JIT Cache");
        habana::JitGraphCache::GetJitCache().RemoveGraph(
            launch_info.hlexec.GetGraphHash());
      }
    } catch (...) {
      launch_except = std::current_exception();
      exception = true;
    }
  }

  PostLaunch(
      tensors,
      launch_info.stack,
      launch_info.indices,
      launch_info.executing_tids,
      launch_info.op_acc_tids,
      lazy_eager_info.retained_tensor_list,
      exception);

  // Rethrow exception in case exception occuured during launch
  if (exception) {
    if (launch_info.async) {
      context->m_launch_thread_exception_handler = launch_except;
    }
    std::rethrow_exception(launch_except);
  }
  context->m_launch_thread_context = false;
  habana_helpers::SetRefineDynamicShape(dynamic_env_);
  get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLAZY);
  launch_info.input_list.clear();
  PT_LAZY_EXEC_THREAD(
      "Launch completed async:",
      launch_info.async,
      " has_queued:",
      launch_info.has_queued,
      " launch_jobid:",
      launch_info.launch_jobid);
  context->DelFromJobidStreamidMap(launch_info.launch_jobid);
}

void SetupExecutionFromRunningHash(
    c10::Device& device,
    exec::HlExec& hlexec,
    const std::vector<HbLazyTensor>& tensors,
    const std::vector<int>& indices,
    habana_lazy::ir::PostOrderData& po_data) {
  auto& graph_hash_builder = GraphHashBuilder::getInstance();
  uint64_t fwd_running_hash =
      graph_hash_builder.combineSyncData(tensors, indices);

  hlexec.set_fwd_graph_hash(fwd_running_hash);

  auto mp_g_and_meta_data_ =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          fwd_running_hash);

  if (mp_g_and_meta_data_ != nullptr) {
    PT_IRGRAPH_DEBUG("Fwd Graph Hash Cache Hit", fwd_running_hash);
    if (GET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING)) {
      habana_lazy::AccThread::Get().discardPendingTasks();
    }
    // prepare po_data inputs and outputs
    po_data.outputs.reserve(indices.size());
    for (auto index : indices) {
      auto ir_value = tensors.at(index).CurrentIrValue();
      if (ir_value) {
        // update output list
        po_data.outputs.push_back(ir_value);
      }
    }

    auto stack_input_map =
        mp_g_and_meta_data_->get_fwd_graph_builder_stack_map();
    graph_hash_builder.prepareInputs(stack_input_map, po_data.inputs);
    graph_hash_builder.invalidateDeviceTids(device);
    graph_hash_builder.reset();
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VALIDATE_GRAPH_RUNNING_HASH)) {
      po_data = HbLazyTensor::RunPostOrder(tensors, indices);
    }
  } else {
    DisableRunningHashUpdates disable(true);
    PT_IRGRAPH_DEBUG("Fwd Graph Hash Cache Miss ", fwd_running_hash);
    po_data = HbLazyTensor::RunPostOrder(tensors, indices);

    // Prepare Input Stack map from post order for cache Miss case
    graph_hash_builder.prepareInputsStackMap(po_data.inputs);
    hlexec.set_fwd_graph_stack_map(graph_hash_builder.getInputStackMap());

    graph_hash_builder.invalidateDeviceTids(device);
    graph_hash_builder.reset();
  }
}

void CorrectInputOrder(
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    const std::vector<uint64_t>& input_map) {
  std::vector<ir::Value>& input_values_in_orig_order =
      lazyFrontEndInfo->get_input_values();
  assert(input_map.size());
  std::vector<ir::Value> input_values_in_post_order{};
  input_values_in_post_order.reserve(input_map.size());
  for (auto idx : input_map) {
    input_values_in_post_order.emplace_back(input_values_in_orig_order.at(idx));
  }
  lazyFrontEndInfo->set_input_values(std::move(input_values_in_post_order));
}

void PrepareInputOrderMap(
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    const std::vector<ir::Value>& inputs,
    exec::HlExec& hlexec) {
  auto graph_input_stack_uids =
      lazyFrontEndInfo->get_lazy_eager_op_input_uids();
  HABANA_ASSERT(graph_input_stack_uids.size() > 0, " Input uids not prepared!");
  auto& graph_hash_builder = GraphHashBuilder::getInstance();
  graph_hash_builder.set_graph_input_stack_uids(
      std::move(graph_input_stack_uids));
  graph_hash_builder.prepareInputsStackMap(inputs);
  hlexec.set_fwd_graph_stack_map(graph_hash_builder.getInputStackMap());
  graph_hash_builder.reset();
}

void HbLazyTensor::SyncTensorsGraphInternal(
    std::vector<HbLazyTensor>* tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo,
    bool async,
    bool collect_sync_tensors) {
  PT_LAZY_TRACE;
  if (!(*tensors).size())
    return;

  LaunchStreamInfo stream_info = {c10::hpu::getCurrentHPUStream()};

  auto device = (*tensors)[0].GetDevice();
  auto context = get_device_lazy_execution_context();
  bool isOptimizedLazyEager = false;
  size_t optimized_lazy_eager_key = 0;
  if (lazyFrontEndInfo) {
    optimized_lazy_eager_key = lazyFrontEndInfo->get_optimized_lazy_eager_key();
    // To check if it is Optimized Lazy Cached graph
    isOptimizedLazyEager = lazyFrontEndInfo->get_is_optimized_lazy_eager();
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
    return;
  }

  std::vector<std::vector<int64_t>> out_shapes{};
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH)) {
    if (lazyFrontEndInfo && lazyFrontEndInfo->get_out_shapes().size()) {
      out_shapes = lazyFrontEndInfo->get_out_shapes();
      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER SHAPE AGNOSTIC] output shapes : ", out_shapes);
    } else {
      for (auto idx : indices) {
        auto& out_tensor = (*tensors)[idx];
        // conversion from smallvector to std::vector, impact on lazy eager perf
        // should be negligible since graphs in lazy eager are small with only
        // few outputs. This conversion can be removed if synapse lowering code
        // also starts using SmallSizeVec in future.
        auto& t = out_tensor.GetSizes();
        std::vector<int64_t> outtensor{};
        outtensor.reserve(t.size());
        outtensor.insert(outtensor.begin(), t.begin(), t.end());
        out_shapes.push_back(std::move(outtensor));
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER SHAPE AGNOSTIC] output idx : ",
            idx,
            " shape : ",
            out_tensor.GetSizes());
      }
    }
  }

  torch::jit::Stack stack;
  habana_lazy::ir::PostOrderData po_data;
  std::vector<uint64_t> executing_indices{};
  exec::HlExec hlexec{};
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
      optimized_path_jit_ir_and_mdata;
  std::string lazy_op_name{};
  bool has_queued = false;
  std::vector<std::shared_ptr<Data>> input_list;

  if (isOptimizedLazyEager) {
    HABANA_ASSERT(lazyFrontEndInfo != nullptr);
    lazy_op_name = lazyFrontEndInfo->get_lazy_op_name();
    optimized_path_jit_ir_and_mdata =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache()
            .GetOptimizedJITGraphAndMetaData(optimized_lazy_eager_key);
    PT_LAZY_DEBUG(
        "Optimized Path JIT Cache hit :: key ", optimized_lazy_eager_key);
    PT_IRGRAPH_DEBUG(
        optimized_path_jit_ir_and_mdata->get_cached_graph()->toString());

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      auto input_map =
          optimized_path_jit_ir_and_mdata->get_fwd_graph_builder_stack_map();
      if (input_map.size() > 1) {
        CorrectInputOrder(lazyFrontEndInfo, input_map);
      }
    }

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) ||
        !(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2)) {
      context->JoinPendingLaunchThread();
      std::vector<ir::Value>& input_values =
          lazyFrontEndInfo->get_input_values();
      stack = PrepareInputStack(tensors, indices, input_values, true);
      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] JoinPendingLaunchThread in Sync for key: ",
          optimized_lazy_eager_key);
    }
  } else {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VALIDATE_GRAPH_RUNNING_HASH) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
      SetupExecutionFromRunningHash(device, hlexec, *tensors, indices, po_data);
    } else {
      po_data = HbLazyTensor::RunPostOrder(*tensors, indices);
    }

    // When queuing synlaunches is enabled, we shouldnot do
    // JoinPendingLaunchThread. if the mode is sync/threadpool/eager is not
    // enabled, then JoinPendingLaunchThread must be done
    if (!GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES) ||
        (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) ||
        !context->copy_scalar_to_hpu_tensor_list.empty() || !async) {
      PT_LAZY_EXEC_THREAD(
          "SyncTensors not queueing, async:",
          async,
          " scalar_to_hpu_tensor_list size:",
          context->copy_scalar_to_hpu_tensor_list.size());
      context->JoinPendingLaunchThread();

      if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) && lazyFrontEndInfo) {
        auto num_of_input_uids =
            lazyFrontEndInfo->get_lazy_eager_op_num_of_uids();
        if (num_of_input_uids > 1) {
          PrepareInputOrderMap(lazyFrontEndInfo, po_data.inputs, hlexec);
        }
      }

      // ValidateSyncInputTensors(po_data.inputs);
      stack = PrepareInputStack(
          tensors, indices, po_data.inputs, false, &po_data.post_order);
      hlexec.set_lazy_front_end_info(lazyFrontEndInfo);

      hlexec.GetOrCreate(po_data, stack);
    } else {
      for (const auto& in : po_data.inputs) {
        std::shared_ptr<Data> d = in.m_data_ptr.lock();
        input_list.emplace_back(std::move(d));
      }
      has_queued = true;
    }
  }

  std::vector<int64_t> executing_tids = std::move(context->executing_tids);
  SetLaunchContextFlags(po_data.inputs, executing_tids);

  // Remove any tensor_data held at output, this will reduce the memory
  // pressure
  for (auto idx : indices) {
    auto& out_tensor = (*tensors)[idx];
    out_tensor.SetExecutionInProgress();
    // clear IR values corresponding to sync tensors
    out_tensor.IrReconnectAsInputNode();
  }

  // clear IR values corresponding to unexecuted view outputs
  for (auto& t : context->viewContext.hb_tensors_exclude_out_view) {
    t.SetExecutionInProgress();
    executing_tids.emplace_back(t.getTensorUniqueId());
    t.IrReconnectAsInputNode();
  }

  for (auto t : context->viewContext.updated_bucket_list) {
    // clear IR values corresponding to sync tensors
    t.IrReconnectAsInputNode();
  }

  // Save po_data input and output to context for perf mode
  if (context->getCapturing()) {
    context->saveInputsAndOutputs(
        po_data.inputs, po_data.outputs, *tensors, indices);
  }

  // Get the jobid,
  size_t launch_jobid = context->GetUniqueJobId();
  context->AddToJobidStreamidMap(launch_jobid, stream_info.stream.stream());
  LaunchTensorsInfo launch_info = {
      *tensors,
      std::move(input_list),
      indices,
      std::move(executing_tids),
      std::move(context->op_acc_tids),
      std::move(po_data),
      hlexec,
      stack,
      async,
      has_queued,
      launch_jobid,
      habana_helpers::GetRefineDynamicShapeStatus(),
      context->getDryRun()};

  LaunchEagerInfo lazy_eager_info = {
      lazyFrontEndInfo,
      context->m_retained_tensor_list,
      optimized_path_jit_ir_and_mdata,
      lazy_op_name,
      out_shapes,
      optimized_lazy_eager_key,
      isOptimizedLazyEager};

  if (async) {
    context->m_launch_thread_handle =
        SingleTonExecThreadPool::getInstance().enqueue(
            LaunchSyncTensorsGraph,
            std::move(launch_info),
            std::move(lazy_eager_info),
            std::move(stream_info));
  } else {
    LaunchSyncTensorsGraph(
        std::move(launch_info),
        std::move(lazy_eager_info),
        std::move(stream_info));
  }
  PT_LAZY_EXEC_THREAD(
      "SyncTensorsGraphInternal async:",
      async,
      " has_queued:",
      has_queued,
      " launch_jobid:",
      launch_jobid,
      " QueueStatus:",
      SingleTonExecThreadPool::getInstance().ToString());

  // clear the context
  context->clear();
}

void HbLazyTensor::ExecuteCachedGraph(
    std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh,
    GraphPtr graph,
    size_t hash,
    size_t graphKey,
    std::string opStrs,
    std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_in,
    std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_out,
    std::vector<habana_lazy::HbLazyTensor> hbt_last_out_used_as_inputs,
    const std::unordered_map<int64_t, c10::optional<at::Generator>>&
        seed_tensors_generator_map,
    uint64_t launch_jobid) {
  PT_LAZY_TRACE;
  get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLOWERING);
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  habana_helpers::SetRefineDynamicShape(false);
  habana_lazy::NoAccThread no_acc_thread(
      false); // disable acc thread during launch, but do not sync the acc
              // thread

  exec::HlExec hlexec{};

  torch::jit::Stack stack;
  // stack is used for both inputs to synapse lowering and outputs from
  // synapse lowering, therefore allocate memory which is max of input
  // and output size.
  stack.reserve(std::max(hblazy_tensors_in.size(), hblazy_tensors_out.size()));

  for (const auto& in : hblazy_tensors_in) {
    auto d = in.getDataPtr();
    // TODO: test randomseed
    if (d->is_random_seed_tensor) {
      auto seed_map = seed_tensors_generator_map.find(d->unique_id);
      if (seed_map == seed_tensors_generator_map.end()) {
        PT_LAZY_FATAL("Failed to find seed tensorid:", d->unique_id);
      }
      // Update the tensor data with new seed value tensor
      d->tensor_data = habana::get_seed_tensor_hpu(
          seed_tensors_generator_map.at(d->unique_id));
    }
    HABANA_ASSERT(
        d->tensor_data.has_value(),
        "Tensor Data not present in ExecuteCachedGraph");
    stack.emplace_back(d->tensor_data);
  }

  auto context = get_device_lazy_execution_context();
  if (!context->copy_scalar_to_hpu_tensor_list.empty()) {
    habana_helpers::copy_scalars_to_device(
        context->copy_scalar_to_hpu_tensor_list);
    context->copy_scalar_to_hpu_tensor_list.clear();
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
  hlexec.Launch(stack, cached_rarg_psh, c10::hpu::getCurrentHPUStream());

  HABANA_ASSERT(stack.size() == hblazy_tensors_out.size());
  for (const auto& in : hblazy_tensors_in) {
    in.ResetExecutionInProgress();
  }

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    HbLazyTensor& out_tensor = hblazy_tensors_out[i++];

    // clear the orig tensor map entries corresponding to cached graph outputs
    out_tensor.getDataPtr()->recent_base = c10::nullopt;
    if (out_tensor.IsHpuGraphOutTensor()) {
      auto st = v.toTensor();
      out_tensor.SetTensorData(st);
    } else {
      out_tensor.SetTensorDataNullOpt();
    }
    out_tensor.ResetExecutionInProgress();
  }

  for (auto& hl_t : hbt_last_out_used_as_inputs) {
    hl_t.SetTensorDataNullOpt();
  }

  habana_helpers::SetRefineDynamicShape(dynamic_env_);
  get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLAZY);
  context->DelFromJobidStreamidMap(launch_jobid);
}

void HbLazyTensor::setTensorOriginalType(c10::ScalarType type) {
  data_ptr()->original_element_type = type;
}

c10::ScalarType HbLazyTensor::getTensorOriginalType() const {
  return data_ptr()->original_element_type;
}

void HbLazyTensor::ShallowCopyTo(HbLazyTensor* dest) const {
  PT_LAZY_TRACE;

  // check for shallow copy in src
  auto hl_src_updated = *this;
  auto src_tensor_opt = hl_src_updated.getDataPtr()->tensor_shallow_copy;
  if (src_tensor_opt.has_value()) {
    hl_src_updated = GetHbLazyTensor(src_tensor_opt.value().back());
  }

  if (hl_src_updated.getTensorUniqueId() == dest->getTensorUniqueId()) {
    return;
  }

  auto aten_t = AtenFromHbLazyTensor(
      hl_src_updated, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

  // the original dest data is now stale. Release it if not in op accmulation
  // phase
  if (!dest->IsOpAccumulationInProgress()) {
    dest->data()->tensor_data = c10::nullopt;
  }

  // loop over the shallow copy vectors to collect the ones that are in use, and
  // put those in-used data to the data keeper.
  c10::SmallVector<at::Tensor, 3> tensors_in_use;
  auto& dst_tensor_opt = dest->getDataPtr()->tensor_shallow_copy;
  if (dst_tensor_opt.has_value()) {
    auto& t_vec = dest->getDataPtr()->tensor_shallow_copy.value();
    for (auto& t : t_vec) {
      auto hl_t = GetHbLazyTensor(t);
      if (hl_t.IsOpAccumulationInProgress()) {
        StaleLazyTensorKeeper::getInstance().add(std::move(hl_t));
      }
      auto hl_orig_t = GetHbLazyTensor(t, false);
      if (hl_orig_t.IsOpAccumulationInProgress()) {
        StaleLazyTensorKeeper::getInstance().add(std::move(hl_orig_t));
      }
    }

    // update the vector with the latest shallow copy
    tensors_in_use.emplace_back(aten_t);
    dest->getDataPtr()->tensor_shallow_copy = tensors_in_use;
  } else {
    // Shallow copy field is being created for the first time
    dest->getDataPtr()->tensor_shallow_copy = {aten_t};
  }
}

void HbLazyTensor::StepMarkerBind(const std::string& device_str, bool sync) {
  PT_LAZY_TRACE;
  PT_IRGRAPH_DEBUG("step marker due to host step marker");
  PT_LAZY_DEBUG("step marker due to host step marker");

  habana_helpers::EmitEvent(habana_helpers::EventDispatcher::Topic::MARK_STEP);

  StageSubmission::getInstance().resetStageSubmissionFlow();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1)) {
    StepMarker(device_str, nullptr, {}, not sync);
  } else {
    StepMarker(device_str);
  }
}

void HbLazyTensor::StepMarkerFinish(bool wait_only) {
  PT_LAZY_TRACE;
  auto context = get_device_lazy_execution_context();
  context->m_launch_thread_context = false;
  context->JoinPendingLaunchThread(wait_only);
}

void HbLazyTensor::IterStepMarker() {
  get_habana_lazy_executor().resetUniqueGraphCntr();
}

void HbLazyTensor::StepMarker(
    const std::string& device_str,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
    std::vector<HbLazyTensor> out_hb_lazy_tensor,
    bool async,
    bool is_allreduce,
    std::vector<HbLazyTensor> bucket_hl_t,
    std::set<int64_t> bucket_recent_id) {
  PT_LAZY_TRACE;

  if (!common::IsStepMarkerSupported()) {
    PT_BRIDGE_WARN("StepMarker is invoked, but not supported. Ignoring..");
    return;
  }

  if (!habana::HPUDeviceContext::is_device_acquired()) {
    // Nothing to do
    PT_LAZY_DEBUG("StepMarker called before device was initialized, skipping");
    return;
  }

  auto context = get_device_lazy_execution_context();
  if (context->m_async_d2h_context) {
    PT_LAZY_DEBUG("StepMarker called in D2H async context, skipping");
    HABANA_ASSERT(0, "StepMarker called in D2H async context, skipping");
    return;
  }

  // Sync accumulation thread if needed and clean up all accumulation resources
  habana_lazy::NoAccThread no_acc_thread;
  habana_lazy::AccThread::Get().ExecuteAllCleanupTasks();

  c10::Device device = GetDeviceOrCurrent(device_str);
  if (!device.is_hpu()) {
    PT_LAZY_DEBUG(
        "Could get hpu device device_str = \"",
        device_str,
        "\", skipping StepMarker");
    return;
  }
  context->m_launch_thread_context = false;
  HbLazyTensor::SyncLiveTensorsGraph(
      &device,
      lazy_front_end_info,
      out_hb_lazy_tensor,
      async,
      is_allreduce,
      bucket_hl_t,
      bucket_recent_id);
  if (!async) {
    context->JoinPendingLaunchThread();
  }
  HbLazyTensor::MarkStep(device);
  if (switch_dynamic_mode) {
    habana_helpers::DisableRefineDynamicShape();
    switch_dynamic_mode = false;
  }
  if (context->getCapturing()) {
    context->CaptureGraphMarkStep();
  }
}

void HbLazyTensor::SetDynamicMode() {
  bool dynamic_env = habana_helpers::GetRefineDynamicShapeStatus();
  switch_dynamic_mode = dynamic_env ? false : true;
  if (switch_dynamic_mode) {
    habana_helpers::EnableRefineDynamicShape();
  }
}

void* HbLazyTensor::lazyTensorDataPtr(const at::Tensor& t) {
  return GetLazyTensorDataPtr(t);
}

void habana_lazy::MaybeSyncLaunchBeforeShallowCopy(
    const HbLazyTensor* dest,
    const HbLazyTensor* src) {
  if (src->IsExecutionInProgress() || dest->IsExecutionInProgress()) {
    auto context = get_device_lazy_execution_context();
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
        context->JoinPendingLaunchThread();
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] JoinPendingLaunchThread before ShallowCopyTo");
      }
    } else {
      context->JoinPendingLaunchThread();
    }
  }
}
