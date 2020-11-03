/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "lazy_executor.h"
namespace habana_lazy {

////////////////////////////////////////////////////////////////////////////UTILITIES////////////////////////////////////////////////////////////////////////////////////////
HbExecutionContextArena habana_lazy_executor = HbExecutionContextArena::Get();

bool allocateTensorWithStorage(int device_index) {
  bool allocate = false;
  auto context = habana_lazy_executor.getDeviceExecutionContext(device_index);
  if (context != nullptr) {
    auto exec_mode = context->getExecutionMode();
    allocate = exec_mode == kLOWERING ? true : false;
  } else {
    TORCH_CHECK(false, "Lazy mode execution context not found");
  }
  return allocate;
}

bool isDeviceInLoweringMode(int device_index) {
  bool is_in_lowering_mode = false;
  auto context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device_index);
  if (context != nullptr) {
    auto exec_mode = context->getExecutionMode();
    is_in_lowering_mode = exec_mode == kLOWERING ? true : is_in_lowering_mode;
  }
  return is_in_lowering_mode;
}
////////////////////////////////////////////////////////////////////////////CONTEXT////////////////////////////////////////////////////////////////////////////////////////

void HbExecutionContext::RegisterTensor(std::shared_ptr<Data> data) {
  auto exec_status = m_tensor_execution_status.find(data->unique_id);
  if (exec_status != std::end(m_tensor_execution_status)) {
    // Tensor is already registered, need to check what to set in this case as
    // its a re-execution
    std::printf(
        "\n Habana Lazy execution context, tensor registeration DUPLICATION \n");
  } else {
    m_tensor_execution_status.emplace(data->unique_id, kREGISTERED);
  }
}

void HbExecutionContext::UnregisterTensor(Data* data) {
  auto exec_status = m_tensor_execution_status.find(data->unique_id);
  if (exec_status != std::end(m_tensor_execution_status)) {
    exec_status->second = kUN_REGISTERED;
  }
}

//////////////////////////////////////////////////////////////////////////////ARENA/////////////////////////////////////////////////////////////////////////////////
HbExecutionContext* HbExecutionContextArena::getDeviceExecutionContext(
    int index) {
  auto hbcontext = m_execution_context_list.find(index);
  if (hbcontext != std::end(m_execution_context_list)) {
    return hbcontext->second;
  } else {
    return createExecutionContext(index);
  }
}

HbExecutionContext* HbExecutionContextArena::createExecutionContext(int index) {
  m_execution_context_list[index] = new HbExecutionContext;
  // m_execution_context_list[index]->setDevice(device);
  return m_execution_context_list[index];
}

void HbExecutionContextArena::removeExecutionContext(int index) {
  auto context = m_execution_context_list[index];
  delete context;
  m_execution_context_list.erase(index);
}

HbExecutionContextArena HbExecutionContextArena::Get() {
  return HbExecutionContextArena();
}
} // namespace habana_lazy