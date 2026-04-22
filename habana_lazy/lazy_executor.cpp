/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "lazy_executor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/python_utils.h"
#include "habana_kernels/h2d_scales_lazy.h"

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
std::once_flag SingleTonExecThreadPool::initialize_once_flag_;

void SingleTonExecThreadPool::CreateInstance() {
  instance_.reset(
      new SingleTonExecThreadPool()); // NOLINT(cppcoreguidelines-owning-memory)
  habana::hpu_registrar().register_lazy_exec_thread_pool(
      []() { instance_.reset(nullptr); });
}

////////////////////////////////////////////////////////////////////////////CONTEXT////////////////////////////////////////////////////////////////////////////////////////

void HbExecutionContext::RegisterTensor(std::shared_ptr<Data> data) {
  MarkTensorStatus(data, kREGISTERED);
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
  HABANA_ASSERT(
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
      if (wait_only) {
        m_launch_thread_handle.wait();
      } else {
        m_launch_thread_handle.get();
      }
    }
  }
  HandleException();
}

void HbExecutionContext::MarkTensorExecuted(std::shared_ptr<Data> data) {
  HABANA_ASSERT(
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
    const std::vector<size_t>& indices) {
  m_user_input_positions.clear();
  m_user_input_match_index.clear();
  m_input_vals.clear();
  size_t idx = 0;
  for (const auto& input_val : inputVals) {
    const int64_t input_uid = input_val.m_data_ptr.lock()->unique_id;
    bool matched = false;
    size_t tensor_idx = 0;

    for (const auto& user_input : m_marked_user_inputs) {
      if (user_input.device().is_cpu()) {
        auto it = m_cpuDataPtrToH2Dtid.find(user_input.data_ptr());
        if (it != m_cpuDataPtrToH2Dtid.end() && it->second == input_uid) {
          matched = true;
        }
      } else {
        auto hbl_tensor =
            habana_lazy::GetHbLazyTensor(user_input, false, false);
        if (hbl_tensor.getTensorUniqueId() == input_uid) {
          matched = true;
        }
      }

      if (matched) {
        m_user_input_positions[idx] = tensor_idx;
        m_user_input_match_index.insert(tensor_idx);
        break;
      }
      ++tensor_idx;
    }

    m_input_vals.emplace_back(input_val);
    ++idx;
  }

  m_cpuDataPtrToH2Dtid.clear();
  // Only hold on to the input val if this isn't a user given input
  for (const auto& [input_idx, _] : m_user_input_positions) {
    inputVals[input_idx] = habana_lazy::ir::Value();
  }

  m_output_vals = std::move(outputVals);

  m_hblazy_tensors.clear();
  for (const auto& i : indices) {
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

void HbExecutionContext::updateCurrentIndicesOfH2dScales() {
  for (auto& [key, scales] : m_scalar_to_h2d_scales_map) {
    scales.current_idx = scales.scales.size() - 1;
    for (auto& scale : scales.scales) {
      auto opt_value = habana_lazy::GetOrCreateHbLazyTensor(scale, at::kHPU)
                           .CurrentTensorAttached();
      if (opt_value.has_value()) {
        habana::get_tensor_extra_meta(opt_value.value())
            ->set_h2d_not_reciprocal(false);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////ARENA/////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<HbExecutionContextArena> HbExecutionContextArena::instance_;
std::once_flag HbExecutionContextArena::initialize_once_flag_;

HbExecutionContext* HbExecutionContextArena::getDeviceExecutionContext() {
  return &execution_context_;
}

void HbExecutionContextArena::CreateInstance() {
  instance_ = std::make_unique<HbExecutionContextArena>();
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
