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

#include "backend/kernel/refinement_engine.h"

#include "backend/kernel/ds_graph_recompile.h"

#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"

#include "habana_helpers/logging.h"

habana::RefinementEngine& habana::RefinementEngine::GetEngine() {
  static RefinementEngine engine_;
  return engine_;
}

void habana::RefinementEngine::Initialize() {
  if (GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD)) {
    // Currently creating single refine thread
    if (m_threads.empty()) {
      m_refineFlag = true;
      m_threads.emplace_back(&RefinementEngine::Refine, this);
    }
  }
}

void habana::RefinementEngine::Refine() {
  PT_BRIDGE_BEGIN;
  habana_lazy::get_habana_lazy_executor().setExecutionMode(
      LazyExecutionMode::kLOWERING);
  while (m_refineFlag) {
    std::unique_lock<std::mutex> mutex_lock(m_mutex);
    m_refineCV.wait(mutex_lock, [&]() { return !m_readyQueue.empty(); });

    // Check whether shutdown is received
    auto last_qentry{m_readyQueue.back()};
    if (false == last_qentry.has_value()) {
      mutex_lock.unlock();
      break;
    }

    auto qentry{m_readyQueue.front()};
    m_readyQueue.pop_front();
    mutex_lock.unlock();

    auto graph_key{qentry.value()};
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD)) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "Refinement thread : received graph hash ", graph_key);
      habana::RefineBucketDS(graph_key);
    } else {
      PT_DYNAMIC_SHAPE_DEBUG("Refinement disabled");
    }
  }
  PT_BRIDGE_END;
}

void habana::RefinementEngine::Shutdown() {
  if (m_threads.empty()) {
    return;
  }

  {
    m_refineFlag = false;
    std::unique_lock<std::mutex> mutex_lock(m_mutex);
    while (!m_readyQueue.empty()) {
      m_readyQueue.pop_back();
    }

    m_readyQueue.emplace_back(absl::nullopt);
  }

  m_refineCV.notify_all();
  for (auto& th : m_threads) {
    if (th.joinable()) {
      th.join();
    }
  }
}

void habana::RefinementEngine::AddGraphKey(size_t key) {
  PT_BRIDGE_BEGIN;
  if (!m_threads.empty()) {
    std::unique_lock<std::mutex> mutex_lock(m_mutex);
    PT_DYNAMIC_SHAPE_DEBUG("Adding graph key ", key, " for refinement");
    bool key_found{false};
    for (size_t i = 0; i < m_readyQueue.size(); i++) {
      // We should not see a shutdown enqueued if we are within this function
      if (m_readyQueue[i].value() == key) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Graph key ", key, " is already added for refinement");
        key_found = true;
        break;
      }
    }
    if (!key_found) {
      m_readyQueue.emplace_back(key);
    }
    m_refineCV.notify_one();
  }
  PT_BRIDGE_END;
}
