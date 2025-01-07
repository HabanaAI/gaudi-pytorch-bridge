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
#include "synapse_shim/synapse_api_shim.h"
#include <dlfcn.h>
#include <link.h>
#include <iostream>
#include <memory>
#include <mutex>
#include "synapse_shim/hccl_api_shim.h"
#include "synapse_shim/lib_synapse_loader.h"
#include "synapse_shim/logging.h"

void LibSynapseLoader::EnsureLoaded() {
  LibSynapseLoader::GetInstance();
}

LibSynapseLoader& LibSynapseLoader::GetInstance() {
  static LibSynapseLoader instance;
  return instance;
}

LibSynapseLoader::LibSynapseLoader() {
  synapse_lib_handle_ = dlopen("libSynapse.so", RTLD_LOCAL | RTLD_NOW);
  CHECK_NULL_MSG(synapse_lib_handle_, dlerror());
  link_map* l_map = nullptr;
  CHECK_TRUE_DL(dlinfo(synapse_lib_handle_, RTLD_DI_LINKMAP, &l_map) == 0);
  synapse_lib_path_ = l_map->l_name;
}

LibSynapseLoader::~LibSynapseLoader() {
  dlclose(synapse_lib_handle_);
}

void* LibSynapseLoader::DlSym(const char* sym) const {
  void* result{dlsym(synapse_lib_handle_, sym)};
  return result;
}

/* A singleton to load synapse and lookup API functions on the first attempt to
 * do an API call. As soon as synapse is loaded the syn_api is updated to
 * directly call function pointers from dynamic lookup: i.e. EnsureLoaded is
 * not called again.
 */
class LazySynapseApi {
 public:
  LazySynapseApi();

  static void EnsureLoaded() {
    instance_.EnsureLoadedImpl();
  }
  static synapse_api_t* GetSynapseApi() {
    EnsureLoaded();
    return &instance_.synapse_api_;
  }
  static hccl_api_t* GetHcclApi() {
    EnsureLoaded();
    return &instance_.hccl_api_;
  }

 private:
  synapse_api_t synapse_api_;
  hccl_api_t hccl_api_;

  std::mutex init_mutex_;
  std::once_flag initialize_once_flag_;

  static LazySynapseApi instance_;
  void EnsureLoadedImpl();
};

template <typename T>
struct LazyLoader;

template <typename Result, typename... Args>
struct LazyLoader<std::function<Result(Args...)>> {
  using FunctionPtr = Result(Args...);
  using Function = std::function<FunctionPtr>;
  template <FunctionPtr ApiFunction>
  static Function LoadAndFwd() {
    return [](Args&&... args) {
      LazySynapseApi::EnsureLoaded();
      return ApiFunction(args...);
    };
  }
};

#define INIT_LAZY_LOADER_API(api, func) \
  api##_api_.func = LazyLoader<api##_api_t::func##_t>::LoadAndFwd<func>();
#define INIT_LAZY_LOADER_SYN_API(func) INIT_LAZY_LOADER_API(synapse, func)
#define INIT_LAZY_LOADER_HCCL_API(func) INIT_LAZY_LOADER_API(hccl, func)

LazySynapseApi::LazySynapseApi() {
  syn_api = &synapse_api_;
  hccl_api = &hccl_api_;
  SYN_API_SYMBOL_VISIT(INIT_LAZY_LOADER_SYN_API);
  HCCL_API_SYMBOL_VISIT(INIT_LAZY_LOADER_HCCL_API);
}

#define INIT_API_FUNC(api, func)                                \
  {                                                             \
    void* sym{loader.DlSym(#func)};                             \
    CHECK_NULL(sym);                                            \
    api##_api_.func = {reinterpret_cast<decltype(func)*>(sym)}; \
  }

#define INIT_SYN_FUNC(func) INIT_API_FUNC(synapse, func)
#define INIT_HCCL_FUNC(func) INIT_API_FUNC(hccl, func)

void LazySynapseApi::EnsureLoadedImpl() {
  // lock to prevent race conditions when two threads attempt to do API call.
  std::lock_guard<std::mutex> lock{init_mutex_};
  // Make sure to load synapse once, and more importantly to only once update
  // syn_api functions.
  // This is for extra safety since INIT_SYNAPSE_API removes LazyLoader api
  // backend that was established in ctor, so there should be no more a code
  // path calling EnsureLoaded.
  std::call_once(initialize_once_flag_, [this]() {
    auto& loader{LibSynapseLoader::GetInstance()};
    SYN_API_SYMBOL_VISIT(INIT_SYN_FUNC);
    HCCL_API_SYMBOL_VISIT(INIT_HCCL_FUNC);
  });
}

LazySynapseApi LazySynapseApi::instance_;
synapse_api_t* syn_api;

synapse_api_t* GetSynapseApi() {
  return LazySynapseApi::GetSynapseApi();
}

hccl_api_t* hccl_api;
hccl_api_t* GetHcclApi() {
  return LazySynapseApi::GetHcclApi();
}

void EnableSynapseApi() {
  syn_api = GetSynapseApi();
  hccl_api = GetHcclApi();
}

/*
 * Synapse API implementation that loggs api calls.
 *
 * Works similarily to LazySynapseApi but uses API symbols from
 * synapse_logger.so
 */
class LoggerSynapseApi {
 public:
  LoggerSynapseApi(synapse_logger::SynapseLoggerObserver* observer)
      : synapse_logger_lib_handle_{nullptr} {
    synapse_logger_lib_handle_ =
        dlopen("pytorch_synapse_logger.so", RTLD_LOCAL | RTLD_NOW);
    CHECK_NULL(synapse_logger_lib_handle_);
    LoggerSynapseApi& loader{*this};
    SYN_API_SYMBOL_VISIT(INIT_SYN_FUNC);
    // TODO: SW-141655 - Currently there's no hccl logger available
    // HCCL_API_SYMBOL_VISIT(INIT_HCCL_FUNC);
    if (observer) {
      using register_synapse_logger_oberver_t =
          void(synapse_logger::SynapseLoggerObserver*);
      auto register_synapse_logger_oberver =
          (register_synapse_logger_oberver_t*)dlsym(
              synapse_logger_lib_handle_, "register_synapse_logger_oberver");
      CHECK_NULL(register_synapse_logger_oberver);
      register_synapse_logger_oberver(observer);
    }
  }
  ~LoggerSynapseApi() {
    if (synapse_logger_lib_handle_ != nullptr) {
      Uninstall();
      dlclose(synapse_logger_lib_handle_);
      synapse_logger_lib_handle_ = nullptr;
    }
  }

  void* DlSym(const char* sym) const {
    void* result{dlsym(synapse_logger_lib_handle_, sym)};
    return result;
  }

  void Uninstall() {
    syn_api = old_syn_api_;
  }

  void Install() {
    old_syn_api_ = syn_api;
    syn_api = &synapse_api_;
    // TODO: SW-141655 - Currently there's no hccl logger available
    // hccl_api = &hccl_api_;
  }

 private:
  void* synapse_logger_lib_handle_;
  synapse_api_t* old_syn_api_ = nullptr;
  synapse_api_t synapse_api_;
  hccl_api_t hccl_api_;
};

void EnableSynapseApiLogger(synapse_logger::SynapseLoggerObserver* observer) {
  static LoggerSynapseApi instance{observer};
  instance.Install();
}
