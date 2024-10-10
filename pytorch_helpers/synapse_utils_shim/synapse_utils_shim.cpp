/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/

#include <dlfcn.h>
#include <syn_sl_api.h>
#include <memory>
#include <mutex>
#include "synapse_shim/logging.h"

#define SLU_API_PTR(func) decltype(::func)* func
#define SLU_INIT_PTR(func) \
  CHECK_NULL(func = (decltype(func))dlsym(lib_handle, #func))

namespace shim_slu {
SLU_API_PTR(synSharedLayerInit);
SLU_API_PTR(synSharedLayerValidateGuidV2);
SLU_API_PTR(synSharedLayerGetGuidNames);
SLU_API_PTR(synSharedLayerFinit);

void LoadSymbols(void* lib_handle) {
  SLU_INIT_PTR(synSharedLayerInit);
  SLU_INIT_PTR(synSharedLayerValidateGuidV2);
  SLU_INIT_PTR(synSharedLayerGetGuidNames);
  SLU_INIT_PTR(synSharedLayerFinit);
}

} // namespace shim_slu

namespace {

class LibSynapseUtilsLoader {
 public:
  static void EnsureLoaded();
  static LibSynapseUtilsLoader& GetInstance();
  LibSynapseUtilsLoader(LibSynapseUtilsLoader const&) = delete;
  void operator=(LibSynapseUtilsLoader const&) = delete;

 private:
  LibSynapseUtilsLoader();
  ~LibSynapseUtilsLoader();
  void* lib_handle_ = nullptr;
};

void LibSynapseUtilsLoader::EnsureLoaded() {
  LibSynapseUtilsLoader::GetInstance();
}

LibSynapseUtilsLoader& LibSynapseUtilsLoader::GetInstance() {
  static LibSynapseUtilsLoader instance;
  return instance;
}

LibSynapseUtilsLoader::LibSynapseUtilsLoader() {
  lib_handle_ = dlopen("libsynapse_utils.so", RTLD_LOCAL | RTLD_NOW);
  CHECK_NULL(lib_handle_);
  shim_slu::LoadSymbols(lib_handle_);
}

LibSynapseUtilsLoader::~LibSynapseUtilsLoader() {
  dlclose(lib_handle_);
}

} // namespace
// Proxy

SharedLayer::Return_t synSharedLayerInit() {
  LibSynapseUtilsLoader::EnsureLoaded();
  return shim_slu::synSharedLayerInit();
}

SharedLayer::Return_t synSharedLayerValidateGuidV2(
    const SharedLayer::ParamsV2_t* const params) {
  LibSynapseUtilsLoader::EnsureLoaded();
  return shim_slu::synSharedLayerValidateGuidV2(params);
}

SharedLayer::Return_t synSharedLayerGetGuidNames(
    char* guidNames[SharedLayer::MAX_NODE_NAME],
    int* guidCount,
    const SharedLayer::DeviceId deviceId) {
  LibSynapseUtilsLoader::EnsureLoaded();
  return shim_slu::synSharedLayerGetGuidNames(guidNames, guidCount, deviceId);
}

SharedLayer::Return_t synSharedLayerFinit() {
  LibSynapseUtilsLoader::EnsureLoaded();
  return shim_slu::synSharedLayerFinit();
}
