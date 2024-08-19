/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "lazy_executor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/python_utils.h"

namespace habana_lazy {

////////////////////////////////////////////////////////////////////////////UTILITIES////////////////////////////////////////////////////////////////////////////////////////
thread_local LazyExecutionMode HbExecutionContextArena::execution_mode{
    LazyExecutionMode::kLAZY};
thread_local bool HbExecutionContext::m_launch_thread_context{false};
thread_local bool HbExecutionContext::m_async_d2h_context{false};
std::atomic_uint64_t habana_lazy::HbExecutionContext::m_unique_jobid_count(0);

bool isDeviceInLoweringMode() {
  return (
      get_habana_lazy_executor().getExecutionMode() ==
      LazyExecutionMode::kLOWERING);
}

std::unique_ptr<SingleTonExecThreadPool> SingleTonExecThreadPool::instance_{
    nullptr};
std::once_flag SingleTonExecThreadPool::initialize_once_flag_{};

void SingleTonExecThreadPool::CreateInstance() {
  instance_.reset(new SingleTonExecThreadPool());
  habana::hpu_registrar().register_lazy_exec_thread_pool(
      []() { instance_.reset(nullptr); });
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
      habana_helpers::AutoNoGIL gil_release;
      // If the future is already ready when below line executes, it can
      // create an exception. Ignore the exception as the wait is already
      // over.

      if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2)) {
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
  m_user_input_positions.clear();
  m_user_input_match_index.clear();
  m_input_vals.clear();
  size_t idx = 0;
  std::copy_if(
      inputVals.begin(),
      inputVals.end(),
      std::back_inserter(m_input_vals),
      [this, &idx](habana_lazy::ir::Value input_ir_v_) {
        size_t input_tensor_idx = 0;
        auto input_ir_v_uid = input_ir_v_.m_data_ptr.lock()->unique_id;
        for (auto& m_marked_user_input_t : m_marked_user_inputs) {
          auto marked_input_hbl =
              habana_lazy::GetHbLazyTensor(m_marked_user_input_t, false, false);
          if (input_ir_v_uid == marked_input_hbl.getTensorUniqueId()) {
            m_user_input_positions[idx] = input_tensor_idx;
            m_user_input_match_index.insert(input_tensor_idx);
            break;
          }
          ++input_tensor_idx;
        }
        ++idx;
        return true;
      });

  // Only hold on to the input val if this isn't a user given input
  for (auto& k : m_user_input_positions) {
    inputVals[k.first] = habana_lazy::ir::Value();
  }

  m_output_vals.clear();
  for (auto& val : outputVals) {
    m_output_vals.emplace_back(val);
  }

  m_hblazy_tensors.clear();
  for (auto& i : indices) {
    tensors[i].SetHpuGraphOutTensor(true);
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
std::unique_ptr<HbExecutionContextArena> HbExecutionContextArena::instance_{};
std::once_flag HbExecutionContextArena::initialize_once_flag_{};

HbExecutionContext* HbExecutionContextArena::getDeviceExecutionContext(int) {
  return &execution_context_;
}

void HbExecutionContextArena::CreateInstance() {
  instance_.reset(new HbExecutionContextArena());
  habana::hpu_registrar().register_lazy_execution_arena(
      []() { instance_.reset(nullptr); });
}

const LazyExecutionMode& HbExecutionContextArena::getExecutionMode() {
  return execution_mode;
}

void HbExecutionContextArena::setExecutionMode(LazyExecutionMode m) {
  execution_mode = m;
}
} // namespace habana_lazy
