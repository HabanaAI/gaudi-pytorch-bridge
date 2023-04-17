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
#include "habana_helpers/python_utils.h"

namespace habana_lazy {

////////////////////////////////////////////////////////////////////////////UTILITIES////////////////////////////////////////////////////////////////////////////////////////
thread_local LazyExecutionMode HbExecutionContextArena::execution_mode{
    LazyExecutionMode::kLAZY};
thread_local bool HbExecutionContext::m_launch_thread_context{false};
thread_local bool HbExecutionContext::m_async_d2h_context{false};
std::atomic_uint64_t habana_lazy::HbExecutionContext::m_unique_jobid_count(0);

HbExecutionContextArena habana_lazy_executor = HbExecutionContextArena::Get();

bool isDeviceInLoweringMode() {
  return (
      habana_lazy_executor.getExecutionMode() == LazyExecutionMode::kLOWERING);
}
////////////////////////////////////////////////////////////////////////////CONTEXT////////////////////////////////////////////////////////////////////////////////////////

void HbExecutionContext::RegisterTensor(std::shared_ptr<Data> data) {
  return MarkTensorStatus(data, kREGISTERED);
}

void HbExecutionContext::UnregisterTensor(Data* data) {
  data->execution_status = kUN_REGISTERED;
}

void HbExecutionContext::MarkTensorStatus(
    std::shared_ptr<Data> data,
    LazyTensorExecutionStatus status) {
  data->execution_status = status;
}

void HbExecutionContext::MarkTensorExecuting(std::shared_ptr<Data> data) {
  TORCH_CHECK(
      data->execution_status != kUN_REGISTERED,
      "Habana Lazy execution : trying to set Executing stage to unregistered tensor");
  if (data->execution_status != kEXECUTION_COMPLETE &&
      data->execution_status != kINPUT) {
    data->execution_status = kEXECUTING;
  }
}

void HbExecutionContext::AddToJobidStreamidMap(
    uint64_t jobId,
    synapse_helpers::hpuStream_t stream) {
  std::lock_guard<std::mutex> lock(m_jobid_streamid_map_mtx);
  m_jobid_streamid_map.insert({jobId, stream});
}

void HbExecutionContext::DelFromJobidStreamidMap(uint64_t jobId) {
  std::lock_guard<std::mutex> lock(m_jobid_streamid_map_mtx);
  m_jobid_streamid_map.erase(jobId);
}

bool HbExecutionContext::HaveJobsInStream(synapse_helpers::hpuStream_t stream) {
  std::lock_guard<std::mutex> lock(m_jobid_streamid_map_mtx);
  for (auto it = m_jobid_streamid_map.begin(); it != m_jobid_streamid_map.end();
       ++it) {
    if (it->second == stream) {
      return true;
    }
  }
  return false;
}

std::uint64_t HbExecutionContext::GetUniqueJobId() {
  return ++m_unique_jobid_count;
}

void HbExecutionContext::JoinPendingLaunchThread(bool wait_only) {
  PT_LAZY_TRACE;

  if (m_launch_thread_handle.valid()) {
    if (!m_launch_thread_context) {
      PT_LAZY_EXEC_THREAD("Waiting for launch thread to finish");
      AutoNoGIL gil_release;
      // If the future is already ready when below line executes, it can
      // create an exception. Ignore the exception as the wait is already
      // over.

      if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2)) {
        SingleTonExecThreadPool::queueStatus();
        m_launch_thread_handle.get();
      } else {
        if (wait_only) {
          m_launch_thread_handle.wait();
        } else {
          m_launch_thread_handle.get();
        }
      }
    }
  }
  HandleException();
}

void HbExecutionContext::MarkTensorExecuted(std::shared_ptr<Data> data) {
  TORCH_CHECK(
      data->execution_status != kUN_REGISTERED,
      "Habana Lazy execution : trying to set executed stage to unregistered tensor");
  data->execution_status = kEXECUTION_COMPLETE;
}

LazyTensorExecutionStatus HbExecutionContext::getTensorExecutionStatus(
    std::shared_ptr<Data> data) {
  return data->execution_status;
}

void HbExecutionContext::saveInputsAndOutputs(
    ir::ValueList inputVals,
    ir::ValueList outputVals,
    std::vector<habana_lazy::HbLazyTensor>& tensors,
    const std::vector<int>& indices) {
  m_input_vals.clear();
  for (auto& val : inputVals) {
    m_input_vals.emplace_back(val);
  }

  m_output_vals.clear();
  for (auto& val : outputVals) {
    m_output_vals.emplace_back(val);
  }

  m_hblazy_tensors.clear();
  for (auto& i : indices) {
    m_hblazy_tensors.emplace_back((tensors)[i]);
  }
}

bool HbExecutionContext::updateInputsRequired(std::vector<size_t>& indices) {
  return (!m_input_vals.empty() && !indices.empty());
}

void HbExecutionContext::updateInputs(ir::ValueList inputVals) {
  m_input_vals.clear();
  for (auto& val : inputVals) {
    m_input_vals.emplace_back(val);
  }
}

//////////////////////////////////////////////////////////////////////////////ARENA/////////////////////////////////////////////////////////////////////////////////

HbExecutionContext* HbExecutionContextArena::getDeviceExecutionContext(
    int index) {
  index = 0;
  // Force everything to use index 0. Need to remove device index as
  // a whole
  index = 0;
  auto hbcontext = m_execution_context_list.find(index);
  if (hbcontext != std::end(m_execution_context_list)) {
    return hbcontext->second;
  } else {
    return createExecutionContext(index);
  }
}

HbExecutionContext* HbExecutionContextArena::createExecutionContext(int index) {
  index = 0;
  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  m_execution_context_list[index] = new HbExecutionContext;
  // m_execution_context_list[index]->setDevice(device);
  return m_execution_context_list[index];
}

void HbExecutionContextArena::removeExecutionContext(int index) {
  std::lock_guard<std::recursive_mutex> lock(HbContextArena::Get()->GetMutex());
  index = 0;
  auto context = m_execution_context_list[index];
  delete context;
  m_execution_context_list.erase(index);
}

HbExecutionContextArena HbExecutionContextArena::Get() {
  return HbExecutionContextArena();
}

const LazyExecutionMode& HbExecutionContextArena::getExecutionMode() {
  return execution_mode;
}

void HbExecutionContextArena::setExecutionMode(LazyExecutionMode m) {
  execution_mode = m;
}
} // namespace habana_lazy
