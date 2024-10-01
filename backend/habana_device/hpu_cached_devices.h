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
#pragma once

#include <c10/util/Exception.h>
#include <synapse_api_types.h>

#include <array>
#include <memory>
#include <mutex>
#include <thread>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/session.h"

namespace habana {

/** Wrapper of a function that is executed when the wrapper is deleted.
 * This is used to hold arbitrary resources along with a deleter function.
 */
class CallFinally {
 public:
  using FinalFunc = std::function<void()>;

  CallFinally() = default;
  CallFinally(FinalFunc&& final_func) : final_func_{std::move(final_func)} {}
  CallFinally(CallFinally&& other) {
    std::swap(other.final_func_, final_func_);
  }
  CallFinally& operator=(CallFinally&& other) {
    std::swap(other.final_func_, final_func_);
    return *this;
  }

  CallFinally& operator=(const CallFinally&) = delete;
  CallFinally(const CallFinally&) = delete;

  ~CallFinally() {
    reset();
  }

  operator bool() const {
    return bool(final_func_);
  }

  void reset(FinalFunc&& new_final_func = nullptr) {
    if (final_func_) {
      final_func_();
    }
    final_func_ = std::move(new_final_func);
  }

 private:
  FinalFunc final_func_;
};

class HPUGlobalConfig {
 public:
  static HPUGlobalConfig& get() {
    static HPUGlobalConfig instance_;
    return instance_;
  }

  bool getDeterministic() {
    return deterministic_;
  }
  void setDeterministic(bool val) {
    deterministic_ = val;
  }

 private:
  HPUGlobalConfig() = default;
  std::atomic<bool> deterministic_{false};
};

class HPURegistrar {
 public:
  static HPURegistrar& get_hpu_registrar() {
    std::call_once(initialize_once_flag_, create_instance);
    return *instance_;
  }

  virtual ~HPURegistrar();
  HPURegistrar(HPURegistrar const&) = delete;
  HPURegistrar& operator=(HPURegistrar const&) = delete;
  HPURegistrar(HPURegistrar&&) = delete;
  HPURegistrar& operator=(HPURegistrar&&) = delete;

  void register_media_proxy_finalizer(CallFinally&& finalizer) {
    media_proxy_finalizer_ = std::move(finalizer);
  }

  void register_process_group_finalizer(CallFinally&& callf) {
    process_group_finalizer_ = std::move(callf);
  }

  void register_acc_thread(CallFinally::FinalFunc&& acc_thread_cleanup) {
    TORCH_CHECK(!accumulation_thread_cleanup_);
    accumulation_thread_cleanup_.reset(std::move(acc_thread_cleanup));
  }

  void register_lazy_exec_thread_pool(
      CallFinally::FinalFunc&& lazy_exec_thread_pool_cleanup) {
    TORCH_CHECK(!lazy_exec_thread_pool_cleanup_);
    lazy_exec_thread_pool_cleanup_.reset(
        std::move(lazy_exec_thread_pool_cleanup));
  }

  void register_lazy_execution_arena(
      CallFinally::FinalFunc&& lazy_execution_arena_cleanup) {
    TORCH_CHECK(!lazy_execution_arena_cleanup_);
    lazy_execution_arena_cleanup_.reset(
        std::move(lazy_execution_arena_cleanup));
  }

  void register_device_deleter(CallFinally::FinalFunc&& device_deleter) {
    TORCH_CHECK(!device_deleter_);
    device_deleter_.reset(std::move(device_deleter));
  }

  static const std::thread::id& get_main_thread_id();

 private:
  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<HPURegistrar> instance_;
  HPURegistrar();

  static void create_instance();
  static void finalize_instance();

  static const std::thread::id main_thread_id_;

  CallFinally device_deleter_;
  CallFinally lazy_execution_arena_cleanup_{};
  CallFinally lazy_exec_thread_pool_cleanup_{};
  CallFinally process_group_finalizer_;
  CallFinally accumulation_thread_cleanup_{};
  CallFinally media_proxy_finalizer_;
};
inline HPURegistrar& hpu_registrar() {
  return HPURegistrar::get_hpu_registrar();
}

}; // namespace habana
