/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include "pytorch_helpers/habana_helpers/log_manager.h"

class LogManagerTest : public habana_lazy_test::LazyTest {};

TEST_F(LogManagerTest, simpleExample) {
  // 0 - trace
  LOG_TRACE(
      DEVICE,
      "HI DEVICE module {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);

  // 1 - debug
  LOG_DEBUG(
      DEVICE,
      "HI DEVICE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);
  LOG_DEBUG(
      KERNEL,
      "HI Kernel module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::KERNEL);
  LOG_DEBUG(
      BRIDGE,
      "HI BRIDGE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::BRIDGE);
  LOG_DEBUG(
      SYNHELPER,
      "HI SYNHELPER module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::SYNHELPER);
  LOG_DEBUG(
      DISTRIBUTED,
      "HI DISTRIBUTED module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DISTRIBUTED);
  LOG_DEBUG(
      LAZY,
      "HI LAZY module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::LAZY);
  LOG_DEBUG(
      HABANAHOOKS,
      "Hi HABANAHOOKS module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::HABANAHOOKS);
  LOG_DEBUG(
      FALLBACK,
      "Hi FALLBACK module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::FALLBACK);
  LOG_DEBUG(
      STATS,
      "Hi STATS module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::STATS);
  LOG_DEBUG(
      TEST,
      "Hi TEST module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::TEST);
  LOG_DEBUG(
      DYNAMIC_SHAPE,
      "Hi DYNAMIC_SHAPE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DYNAMIC_SHAPE);
  LOG_DEBUG(
      DEVMEM,
      "Hi DEVMEM module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVMEM);
  LOG_DEBUG(
      HABHELPER,
      "Hi HABHELPER module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::HABHELPER);
  // 2 - info
  LOG_INFO(
      DEVICE,
      "Hi DEVICE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);
  // 3 - warnning
  LOG_WARN(
      DEVICE,
      "Hi DEVICE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);
  // 4 - error
  LOG_ERR(
      DEVICE,
      "Hi DEVICE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);
  // 5 - critical
  LOG_CRITICAL(
      DEVICE,
      "Hi DEVICE module: {}",
      (uint32_t)ptspdlogger::LogManager::LogType::DEVICE);
}