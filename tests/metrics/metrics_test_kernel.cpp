/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
