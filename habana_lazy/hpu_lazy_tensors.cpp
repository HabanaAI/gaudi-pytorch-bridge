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
#include "debug_utils.h"
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/ir.h"
#include "hlexec.h"

using namespace habana_lazy;

using ValueList = std::vector<ir::Value>;

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
  context->UnregisterTensor(data);
}

std::vector<HbLazyTensor> HbContextArena::GetLiveTensors(
    const c10::Device* device) {
  PT_LAZY_TRACE;
  std::vector<HbLazyTensor> tensors;
  auto fn = [&](HbContext* devctx) {
    for (auto& uid_wptr : devctx->tensors_data) {
      std::shared_ptr<Data> data = uid_wptr.second.lock();
      if (data != nullptr) {
        // Activate this code once the logic is fixed for live tensors
        // auto exec_context =
        //   habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        //       device->index());
        // auto status =
        // exec_context->getTensorExecutionStatus(data->unique_id);
        // if (status != kEXECUTION_COMPLETE && status != kINPUT) {
        tensors.emplace_back(HbLazyTensor(std::move(data)));
        //}
      }
    }
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
  context->MarkTensorsExecuted(false);
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
  return CreateTensorNode();
}

HbLazyTensor HbLazyTensor::CreateHbLazyTensor(
    c10::IntArrayRef size,
    at::Scalar fill_value,
    const at::Device& device,
    at::ScalarType scalar_type) {
  PT_LAZY_TRACE;
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

  PT_LAZY_DEBUG(IrGraphDumpUtil::PostOrderToText(po_data.post_order, p_roots));

  return po_data;
}

c10::optional<at::Tensor> HbLazyTensor::GetHbLazyTensorData() {
  // Generate the tensor data if its not been generated yet
  if (data()->ir_value && !CurrentTensorData()) {
    applyPendingGraph();
  }
  return data()->tensor_data;
}

void HbLazyTensor::applyPendingGraph() {
  PT_LAZY_TRACE;
  // Ensure that the graph execution has taken place so taht the tensors
  // requested have the data required updated in them. This is usually done
  // before sync points in execution
  if (!CurrentTensorData()) {
    std::vector<HbLazyTensor> tensors({*this});
    SyncTensorsGraph(&tensors);
  }
}

std::vector<HbLazyTensor> HbLazyTensor::GetLiveTensors(
    const c10::Device* device) {
  return HbContextArena::Get()->GetLiveTensors(device);
}

void HbLazyTensor::SyncTensorsGraph(std::vector<HbLazyTensor>* tensors) {
  PT_LAZY_TRACE;
  SyncTensorsGraphInternal(tensors);
}

void HbLazyTensor::SyncLiveTensorsGraph(
    const c10::Device* device,
    bool use_cached_graph = false) {
  PT_LAZY_TRACE;
  if (use_cached_graph) {
    ExecuteCachedGraph();
  } else {
    auto tensors = GetLiveTensors(device);
    SyncTensorsGraph(&tensors);
  }
}

void HbLazyTensor::SyncTensorsGraphInternal(
    std::vector<HbLazyTensor>* tensors) {
  PT_LAZY_TRACE;
  const std::vector<int>& indices = CollectSyncTensors(*tensors);
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

  for (const auto& in : po_data.inputs) {
    PT_LAZY_DEBUG(std::string("Lowering - ") + in.ToString());
    HABANA_ASSERT(!in.m_data_ptr.expired());
    if (in.mp_node) {
      PT_LAZY_DEBUG(std::string("    Node ") + in.mp_node->ToString());
    }
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    stack.emplace_back(d->tensor_data);
    // We dont get the correct lazy tensor back from internal tensor
    // So marking for execution here
    context->MarkTensorExecuting(d->unique_id);
  }

  hlexec.GetOrCreate(po_data, stack);

  // Dump the JIT graph with PT_LAZY_DEBUG
  PT_LAZY_DEBUG(hlexec.DumpGraph());

  // Launch the execution
  hlexec.Launch(stack);
  HABANA_ASSERT(stack.size() == indices.size());

  size_t i = 0;
  for (const torch::IValue& v : stack) {
    auto st = v.toTensor();
    auto out_tensor = (*tensors)[indices[i++]];
    context->MarkTensorExecuted(out_tensor.getTensorUniqueId());
    out_tensor.SetTensorData(st);
  }
  context->MarkTensorsExecuted();

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
    i.AssignIrValue(val);
  }

  // Save po_data input and output to context for perf mode
  if (context->m_is_cached == false) {
    context->saveInputsAndOutputs(
        po_data.inputs, po_data.outputs, *tensors, indices);
    context->m_is_cached = true;
  }
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
    HABANA_ASSERT(!in.m_data_ptr.expired());
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
  dest->AssignIrValue(GetIrValue());
}

void HbLazyTensor::StepMarker(const std::string& device_str) {
  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  c10::Device device = GetDeviceOrCurrent(device_str);
  HbLazyTensor::SyncLiveTensorsGraph(&device);
  HbLazyTensor::MarkStep(device);
}

void HbLazyTensor::RunSavedGraph(const std::string& device_str) {
  c10::Device device = GetDeviceOrCurrent(device_str);
  HbLazyTensor::SyncLiveTensorsGraph(&device, true);
  HbLazyTensor::MarkStep(device);
}

extern "C" void mark_step() {
  HbLazyTensor::StepMarker({});
}

extern "C" void run_saved_model() {
  HbLazyTensor::RunSavedGraph({});
}
