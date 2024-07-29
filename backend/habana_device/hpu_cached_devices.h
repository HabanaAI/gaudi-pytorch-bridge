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
#include "backend/synapse_helpers/session.h"

#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/synapse_helpers/device.h"

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

class HPURegistrar {
 public:
  static HPURegistrar& get_hpu_registrar() {
    std::call_once(initialize_once_flag_, create_instance);
    return *raw_instance_;
  }

  virtual ~HPURegistrar();
  HPURegistrar(HPURegistrar const&) = delete;
  HPURegistrar& operator=(HPURegistrar const&) = delete;
  HPURegistrar(HPURegistrar&&) = delete;
  HPURegistrar& operator=(HPURegistrar&&) = delete;

  class HPUGlobalConfig {
   public:
    HPUGlobalConfig() = default;

    bool getDeterministic() {
      std::lock_guard<std::mutex> lock(config_lock_);
      return deterministic_;
    }
    void setDeterministic(bool val) {
      std::lock_guard<std::mutex> lock(config_lock_);
      deterministic_ = val;
    }

   private:
    bool deterministic_{false};
    std::mutex config_lock_{};
  };

  static synapse_helpers::device_handle try_get_syn_device(int device_id) {
    if (device_id == 0)
      return synapse_helpers::device::device_in_use.lock();
    return {};
  }

  // Return acquired device or die if no device is initialized
  static HPUDevice& get_device(int device_id) {
    auto& instance{get_hpu_registrar()};
    TORCH_CHECK(device_id == 0, "Device ", device_id, " is not initialized");
    return instance.get_active_device();
  }

  static HPUDevice& get_device() {
    auto& instance{get_hpu_registrar()};
    return instance.get_active_device();
  }

  HPUDevice& get_active_device() {
    TORCH_CHECK(active_device_ != nullptr, "Habana device not initialized");
    if (is_closing()) {
      TORCH_WARN("Habana device is accessed while closing");
    }

    return *active_device_;
  }

  HPUDevice& get_or_create_device();

  bool is_initialized() {
    return active_device_ != nullptr;
  }

  // Note: Need to finish execution all performed operations till this point
  // Ensure a synchronous mark_step is invoked before calling this function
  // for device synchronization
  static void synchronize_device() {
    auto& device = get_hpu_registrar().get_device();
    device.synchronize();
  }

  // To finish execution pending in the host multi stage pipeline.
  static void synchronize_host_multistage_pipeline() {
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
      c10::hpu::joinEagerThreadsCB();
    } else {
      PT_BRIDGE_WARN(
          "synchronize_host_multistage_pipeline is invoked, but not supported. Ignoring..");
    }
  }

  static std::string get_device_capability() {
    auto& device = get_hpu_registrar().get_device();
    return device.syn_device().get_device_capability();
  }

  static std::string get_device_properties(unsigned id) {
    return synapse_helpers::device::get_device_properties(id);
  }

  static int get_total_device_count() {
    return synapse_helpers::device::get_total_device_count();
  }

  static const std::thread::id& get_main_thread_id() {
    return main_thread_id_;
  }

  static HPUGlobalConfig& get_hpu_global_config() {
    static HPUGlobalConfig instance_;
    return instance_;
  }

  /**
   * Tells if the registrar is deleting the device right now.
   * In this state no additional requests should be arriving, but it may so
   * happen, that as the device is releasing, there are still some resources
   * (Tensors) being freed that require the allocator to be operational.
   * A typical example are Tensors prolonged for asynchrnous processing on
   * a device stream that are getting released during teardown of streams
   * threads.
   * When the registrar is closing, the device as owned by acquired_device_ is
   * being deleted, yet it is still accessible by get_device() via
   * active_device_.
   */
  bool is_closing() {
    return active_device_ && !acquired_device_;
  }

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

  void register_eager_context(CallFinally::FinalFunc&& eager_context_cleanup) {
    TORCH_CHECK(!eager_context_cleanup_);
    eager_context_cleanup_.reset(std::move(eager_context_cleanup));
  }

 private:
  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<HPURegistrar> instance_;
  static HPURegistrar* raw_instance_;
  static bool finalized_;
  HPURegistrar();

  static void create_instance();
  static void finalize_instance();

  static const std::thread::id main_thread_id_;

  HPUDevice* active_device_{nullptr};
  using device_holder = std::unique_ptr<HPUDevice, void (*)(HPUDevice*)>;
  device_holder acquired_device_;
  CallFinally lazy_exec_thread_pool_cleanup_{};
  CallFinally lazy_execution_arena_cleanup_{};
  CallFinally eager_context_cleanup_{};
  CallFinally process_group_finalizer_;
  CallFinally accumulation_thread_cleanup_{};
  CallFinally media_proxy_finalizer_;

  /**
   * Some unusual operations are possible on the registrar for testability.
   * The mechanism is that a derived test instance of a registrar may
   * temporarily change resolution of get_hpu_registrar to itself.
   */
  friend class HPURegistrarTester;
  std::function<void()> test_inject_late_cleanup_{nullptr};

  static void device_deleter(HPUDevice* device);
  void device_deleter_internal(HPUDevice* device);
};

inline HPURegistrar& hpu_registrar() {
  return HPURegistrar::get_hpu_registrar();
}

} // namespace habana
