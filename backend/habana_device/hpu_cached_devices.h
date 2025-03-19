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
#pragma once

#include <synapse_api_types.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include "habana_helpers/logging.h"

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
    HABANA_ASSERT(!accumulation_thread_cleanup_);
    accumulation_thread_cleanup_.reset(std::move(acc_thread_cleanup));
  }

  void register_lazy_exec_thread_pool(
      CallFinally::FinalFunc&& lazy_exec_thread_pool_cleanup) {
    HABANA_ASSERT(!lazy_exec_thread_pool_cleanup_);
    lazy_exec_thread_pool_cleanup_.reset(
        std::move(lazy_exec_thread_pool_cleanup));
  }

  void register_lazy_execution_arena(
      CallFinally::FinalFunc&& lazy_execution_arena_cleanup) {
    HABANA_ASSERT(!lazy_execution_arena_cleanup_);
    lazy_execution_arena_cleanup_.reset(
        std::move(lazy_execution_arena_cleanup));
  }

  void register_thread_deleter(CallFinally::FinalFunc&& thread_deleter) {
    HABANA_ASSERT(!thread_deleter_);
    thread_deleter_.reset(std::move(thread_deleter));
  }

  void register_device_deleter(CallFinally::FinalFunc&& device_deleter) {
    HABANA_ASSERT(!device_deleter_);
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
  CallFinally thread_deleter_;
  CallFinally lazy_exec_thread_pool_cleanup_{};
  CallFinally process_group_finalizer_;
  CallFinally accumulation_thread_cleanup_{};
  CallFinally media_proxy_finalizer_;
};
inline HPURegistrar& hpu_registrar() {
  return HPURegistrar::get_hpu_registrar();
}

}; // namespace habana
