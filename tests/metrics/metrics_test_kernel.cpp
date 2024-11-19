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
#include <thread>
#include "event_dispatcher.h"

void metrics_trigger() {
  auto timestamp_init = std::chrono::high_resolution_clock::now();
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CPU_FALLBACK,
      habana_helpers::EventDispatcher::EventParams(
          {{"op_name", "metrics_trigger_fallback_op"}}));

  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CPU_FALLBACK,
      habana_helpers::EventDispatcher::EventParams(
          {{"op_name", "metrics_trigger_fallback_op_2"}}));

  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      habana_helpers::EventDispatcher::EventParams(
          {{"success", std::to_string(false)}}));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto milliseconds_metric =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - timestamp_init)
          .count();
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      habana_helpers::EventDispatcher::EventParams(
          {{"success", std::to_string(true)},
           {"milliseconds", std::to_string(milliseconds_metric)}}));

  size_t recipe_id_1 = 123;
  size_t recipe_id_2 = 456;
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CACHE_MISS,
      habana_helpers::EventDispatcher::EventParams(
          {{"recipe_id", std::to_string(recipe_id_1)}}));
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CACHE_HIT,
      habana_helpers::EventDispatcher::EventParams(
          {{"recipe_id", std::to_string(recipe_id_1)}}));
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CACHE_HIT,
      habana_helpers::EventDispatcher::EventParams(
          {{"recipe_id", std::to_string(recipe_id_2)}}));
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CACHE_HIT,
      habana_helpers::EventDispatcher::EventParams(
          {{"recipe_id", std::to_string(recipe_id_2)}}));
}
