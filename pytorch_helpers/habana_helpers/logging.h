/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <iostream>
class PtLogger {
 private:
  static PtLogger* instance;
  unsigned long module_mask_;
  unsigned long type_mask_;

  PtLogger() {
    char* mask = getenv("PT_HPU_LOG_MOD_MASK");
    if (mask != nullptr) {
      module_mask_ = std::stoul(mask, nullptr, 16);
    } else {
      // enable all modules by default
      module_mask_ = INT64_MAX;
    }

    mask = getenv("PT_HPU_LOG_TYPE_MASK");
    if (mask != nullptr) {
      type_mask_ = std::stoul(mask, nullptr, 16);
    } else {
      // enable fatal errors and warnings by default
      type_mask_ = TypeMask::FATAL + TypeMask::WARNING;
    }
  }

 public:
  PtLogger(const PtLogger&) = delete;
  PtLogger& operator=(const PtLogger&) = delete;

  static PtLogger* getLogger() {
    if (instance == 0) {
      instance = new PtLogger();
    }

    return instance;
  }

  unsigned long getTypeMask() {
    return type_mask_;
  }

  unsigned long getModuleMask() {
    return module_mask_;
  }

  enum TypeMask {
    FATAL = 1,
    WARNING = 2,
    TRACE = 4,
    DEBUG = 8,
  };

  enum ModuleMask {
    DEVICE = 1,
    KERNEL = 2,
    BRIDGE = 4,
    SYNHELPER = 8,
    DISTRIBUTED = 16,
  };
};

/************************CRITICAL MACROS************************/
#define PT_MOD_FATAL(MOD, ...)                                          \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&              \
       (PtLogger::getLogger()->getTypeMask() &                          \
        (PtLogger::TypeMask::FATAL)))) {                                \
    std::cerr << str(__VA_ARGS__) << " " << __FILE__ << ":" << __LINE__ \
              << "\t" << __func__ << "\n";                              \
  }

#define PT_DEVICE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)

#define PT_KERNEL_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)

#define PT_BRIDGE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)

#define PT_SYNHELPER_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)

#define PT_DISTRIBUTED_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

/************************WARNING MACROS************************/
#define PT_MOD_WARN(MOD, ...)                                           \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&              \
       (PtLogger::getLogger()->getTypeMask() &                          \
        (PtLogger::TypeMask::WARNING)))) {                              \
    std::cerr << str(__VA_ARGS__) << " " << __FILE__ << ":" << __LINE__ \
              << "\t" << __func__ << "\n";                              \
  }

#define PT_DEVICE_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)

#define PT_KERNEL_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)

#define PT_BRIDGE_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)

#define PT_SYNHELPER_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)

#define PT_DISTRIBUTED_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

/************************TRACE MACROS************************************/
#define PT_MOD_BEGIN(MOD)                                                \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&               \
       (PtLogger::getLogger()->getTypeMask() &                           \
        (PtLogger::TypeMask::TRACE)))) {                                 \
    std::clog << "HABANA_LOG: begin of " << __PRETTY_FUNCTION__ << "\n"; \
  };

#define PT_DEVICE_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::DEVICE)
#define PT_KERNEL_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::KERNEL)
#define PT_BRIDGE_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::BRIDGE)
#define PT_SYNHELPER_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::SYNHELPER)
#define PT_DISTRIBUTED_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::DISTRIBUTED)

#define PT_MOD_END(MOD)                                                \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&             \
       (PtLogger::getLogger()->getTypeMask() &                         \
        (PtLogger::TypeMask::TRACE)))) {                               \
    std::clog << "HABANA_LOG: end of " << __PRETTY_FUNCTION__ << "\n"; \
  };

#define PT_DEVICE_END PT_MOD_END(PtLogger::ModuleMask::DEVICE)
#define PT_KERNEL_END PT_MOD_END(PtLogger::ModuleMask::KERNEL)
#define PT_BRIDGE_END PT_MOD_END(PtLogger::ModuleMask::BRIDGE)
#define PT_SYNHELPER_END PT_MOD_END(PtLogger::ModuleMask::SYNHELPER)
#define PT_DISTRIBUTED_END PT_MOD_END(PtLogger::ModuleMask::DISTRIBUTED)

/************************DEBUG MACROS************************************/
#define PT_MOD_DEBUG(MOD, ...)                             \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
       (PtLogger::getLogger()->getTypeMask() &             \
        (PtLogger::TypeMask::DEBUG)))) {                   \
    std::clog << str(__VA_ARGS__) << "\n";                 \
  };

#define PT_DEVICE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)
#define PT_KERNEL_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)
#define PT_BRIDGE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)
#define PT_HELPER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::HELPER, __VA_ARGS__)
#define PT_DISTRIBUTED_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

#define LOG_FUNC_BEGIN        \
  std::clog << "DEPRECATED! " \
            << "HABANA_LOG: begin of " << __PRETTY_FUNCTION__ << "\n"
#define LOG_FUNC_END          \
  std::clog << "DEPRECATED! " \
            << "HABANA_LOG: end of " << __PRETTY_FUNCTION__ << "\n"
